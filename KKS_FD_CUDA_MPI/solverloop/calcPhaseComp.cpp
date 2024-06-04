// check for the required headers
void __initMu__helper(double **phi, double **comp, double **phaseComp, double **mu,
                long *thermo_phase, double temperature,
                long NUMPHASES, long NUMCOMPONENTS, long DIMENSION,
                long sizeX, long sizeY, long sizeZ,
                long xStep, long yStep, long padding,int idx)
{

        double y[MAX_NUM_COMP], mu0[MAX_NUM_COMP];
        double sum = 0.0;
// check the correct data clause ( it should be like phi[:NUMPHASES*(rangeof (idx))]
   #pragma acc data present(phi, comp, phaseComp, mu,thermo_phase,temperature, NUMPHASES, NUMCOMPONENTS, DIMENSION,sizeX, sizeY,sizeZ,xStep,yStep,padding) create(y,sum) copyin(idx)     
    {
        #pragma acc parallel loop
        for (long phase = 0; phase < NUMPHASES; phase++)
        {
            // Bulk
            if (phi[phase][idx] == 1.0)
            {
                sum = 0.0;
                #pragma acc parallel loop reduction(+:sum)
                for (long i = 0; i < NUMCOMPONENTS-1; i++)
                {
                    phaseComp[phase + NUMPHASES*i][idx] = comp[i][idx];
                    y[i] = comp[i][idx];
                    sum += y[i];
                }

                y[NUMCOMPONENTS-1] = 1.0 - sum;

                (*Mu_tdb_dev[thermo_phase[phase]])(temperature, y, mu0);
                #pragma acc parallel loop
                for (long i = 0; i < NUMCOMPONENTS-1; i++)
                    mu[i][idx] = mu0[i];
            }
            else
            {
                #pragma acc parallel loop
                for (long i = 0; i < NUMCOMPONENTS-1; i++)
                    phaseComp[phase + NUMPHASES*i][idx] = 0.0;
            }
        }
    }
}


void __initMu__(double **phi, double **comp, double **phaseComp, double **mu,
                long *thermo_phase, double temperature,
                long NUMPHASES, long NUMCOMPONENTS, long DIMENSION,
                long sizeX, long sizeY, long sizeZ,
                long xStep, long yStep, long padding)
{

    #pragma acc parallel loop collapse(3)
    for (long i = padding; i < sizeX - padding; i++)
    {
        for (long j = (DIMENSION >= 2 ? padding : 0); j < (DIMENSION >= 2 ? sizeY - padding : 1); j++)
        {
            for (long k = (DIMENSION == 3 ? padding : 0); k < (DIMENSION == 3 ? sizeZ - padding : 1); k++)
            {
                long idx = i*xStep + j*yStep + k;
                
                __initMu__helper(phi, comp, phaseComp, mu,thermo_phase, temperature,NUMPHASES, NUMCOMPONENTS,DIMENSION,sizeX,sizeY,  sizeZ, xStep, yStep,  padding);
            }
        }
    }
}

void __calcPhaseComp__helper(double **phi, double **comp,
                       double **phaseComp,
                       double *F0_A, double *F0_B, double *F0_C,
                       long NUMPHASES, long NUMCOMPONENTS, long DIMENSION,
                       long sizeX, long sizeY, long sizeZ,
                       long xStep, long yStep, long padding,int idx)
{

    // Number of phase compositions
    long N = NUMPHASES*(NUMCOMPONENTS-1);

    // Iterate to fill jacobian and function vector
    long iter1, iter2, iter3, iter4;
    long index1, index2;

    // Delta vector, function vector, jacobian matrix
    double sol[MAX_NUM_PHASE_COMP], B[MAX_NUM_PHASE_COMP];
    double A[MAX_NUM_PHASE_COMP][MAX_NUM_PHASE_COMP];

    // Permutation matrix required for LUP linear system solver
    int P[MAX_NUM_PHASE_COMP+1];

    // Tolerance for LU solver
    double tol = 1e-10;

 #pragma acc data present(phi, comp,phaseComp,F0_A, F0_B, F0_C,NUMPHASES,  NUMCOMPONENTS,  DIMENSION,sizeX,  sizeY,  sizeZ,xStep,  yStep,  padding)  create(iter1, iter2, iter3, iter4 , index1,index2,sol,B,A,P) copyin(N,idx,tol)  
 {
        #pragma acc parallel loop collapse(2)
        for (iter1 = 0; iter1 < NUMCOMPONENTS-1; iter1++)
        {
            for (iter2 = 0; iter2 < NUMPHASES; iter2++)
            {
                index1 = iter2 + iter1*NUMPHASES;
                if (iter2)
                {
                    // Constant part (linear part of free-energy) taken to the RHS of AX = B
                    // B^{K}_{P+1} - B^{K}_{P}
                    B[index1] = F0_B[iter1 + iter2*(NUMCOMPONENTS-1)] - F0_B[iter1 + (iter2-1)*(NUMCOMPONENTS-1)];
                }
                else
                {
                    // Composition conservation equation
                    B[index1] = comp[iter1][idx];
                }
            }
        }

         // Calculate jacobian using x^n
        #pragma acc parallel loop collapse(2)
        for (iter1 = 0; iter1 < NUMCOMPONENTS-1; iter1++)
        {
            for (iter2 = 0; iter2 < NUMPHASES; iter2++)
            {
                index1 = iter2 + iter1*NUMPHASES;
                if (iter2)
                {
                    #pragma acc parallel loop collapse(2)
                    for (iter3 = 0; iter3 < NUMCOMPONENTS-1; iter3++)
                    {
                        for (iter4 = 0; iter4 < NUMPHASES; iter4++)
                        {
                            index2 = iter4 + iter3*NUMPHASES;

                            if (iter4 == iter2-1)
                            {
                                if (iter3 == iter1)
                                    A[index1][index2] = 2.0*F0_A[(iter3 + iter4*(NUMCOMPONENTS-1))*(NUMCOMPONENTS-1) + iter1];
                                else
                                    A[index1][index2] = F0_A[(iter3 + iter4*(NUMCOMPONENTS-1))*(NUMCOMPONENTS-1) + iter1];
                            }
                            else if (iter4 == iter2)
                            {
                                if (iter3 == iter1)
                                    A[index1][index2] = -2.0*F0_A[(iter3 + iter4*(NUMCOMPONENTS-1))*(NUMCOMPONENTS-1) + iter1];
                                else
                                    A[index1][index2] = -1.0*F0_A[(iter3 + iter4*(NUMCOMPONENTS-1))*(NUMCOMPONENTS-1) + iter1];
                            }
                            else
                                A[index1][index2] = 0.0;
                        }
                    }
                } // if (iter2)
                else
                {
                    #pragma acc parallel loop collapse(2)
                    for (iter3 = 0; iter3 < NUMCOMPONENTS-1; iter3++)
                    {
                        for (iter4 = 0; iter4 < NUMPHASES; iter4++)
                        {
                            index2 = iter4 + iter3*NUMPHASES;

                            if (iter3 == iter1)
                                A[index1][index2] = calcInterp5th(phi, iter4, idx, NUMPHASES);
                            else
                                A[index1][index2] = 0.0;
                        }
                    }
                }
            } 
        }

        LUPDecomposePC1(A, N, tol, P);
        LUPSolvePC1(A, P, B, N, sol);
        #pragma acc parallel loop
        for (iter1 = 0; iter1 < N; iter1++)
        {
            // Update phase composition
            phaseComp[iter1][idx] = sol[iter1];
        }
    }
}

void __calcPhaseComp__(double **phi, double **comp,
                       double **phaseComp,
                       double *F0_A, double *F0_B, double *F0_C,
                       long NUMPHASES, long NUMCOMPONENTS, long DIMENSION,
                       long sizeX, long sizeY, long sizeZ,
                       long xStep, long yStep, long padding)
{   
    #pragma acc parallel loop collapse(3)
    for (long i = padding; i < sizeX - padding; i++)
    {
        for (long j = (DIMENSION >= 2 ? padding : 0); j < (DIMENSION >= 2 ? sizeY - padding : 1); j++)
        {
            for (long k = (DIMENSION == 3 ? padding : 0); k < (DIMENSION == 3 ? sizeZ - padding : 1); k++)
            {
                long idx = i*xStep + j*yStep + k;
                
                 __calcPhaseComp__helper(phi,comp,phaseComp,F0_A,F0_B, F0_C,NUMPHASES, NUMCOMPONENTS,  DIMENSION,sizeX,  sizeY,  sizeZ,xStep,  yStep,  padding);
            }
        }
    }
}

void __calcPhaseComp_02__helper(double **phi, double **comp,
                          double **phaseComp, double **mu, double *cguess,
                          double temperature, long *thermo_phase,
                          long NUMPHASES, long NUMCOMPONENTS, long DIMENSION,
                          long sizeX, long sizeY, long sizeZ,
                          long xStep, long yStep, long padding,int idx)
{
        double fun[MAX_NUM_COMP], jacInv[MAX_NUM_COMP][MAX_NUM_COMP], cn[MAX_NUM_COMP], co[MAX_NUM_COMP];
        double tmp0, norm;
        double retdmuphase[MAX_NUM_COMP*MAX_NUM_COMP], retdmuphase2[MAX_NUM_COMP][MAX_NUM_COMP], y[MAX_NUM_COMP], mu0[MAX_NUM_COMP];

        double tol = 1e-6;

        long interface = 1;
        long bulkphase;

        #pragma acc data present(phi, comp,phaseComp, mu, cguess,temperature,thermo_phase,NUMPHASES,  NUMCOMPONENTS,  DIMENSION,sizeX,  sizeY,  sizeZ,xStep,  yStep,  padding) create(fun,jacInv,cn,co,tmp0,norm,retdmuphase,retdmuphase2,y,mu0,bulkphase) copyin(idx,tol,interface)
        {

            int local = 1;
            #pragma acc parallel loop copyin(local)
            for (long is = 0; is < NUMPHASES; is++)
            {
                if (phi[is][idx] > 0.99999)
                {
                    if(local == 1)
                    {
                        bulkphase = is;
                        interface = 0;
                        local = 0;
                    }
                }
            }

            if(interface)
            {
                // Number of iterations for Newton-Raphson
                long count  ;
                // Number of iterations for diffusion-potential correction
                long count2 ;

                long is, is1, is2;
                long maxCount = 10000;

                // Permutation matrix required by LU decomposition routine
                int P[MAX_NUM_COMP];

                double dmudc[(MAX_NUM_COMP)*(MAX_NUM_COMP)];
                double dcdmu[(MAX_NUM_COMP)*(MAX_NUM_COMP)];
                double Inv[MAX_NUM_COMP][MAX_NUM_COMP];

                double deltac[MAX_NUM_COMP] = {0.0};
                long deltac_flag = 0;

                #pragma acc data create(is,is1,is2,P,dmudc,dcdmu,Inv,count,count2) copyin(count,maxCount,deltac,deltac_flag)
                {
                    
                    count2 = 0;
                    count = 0 ;

                    #pragma acc parallel loop collapse(2)
                    for(count2 ; count2<1000 ; count2++)
                    {
                         for (long phase = 0; phase < NUMPHASES; phase++)
                         {
                            #pragma acc parallel loop
                            for (is = 0; is < NUMCOMPONENTS-1; is++)
                            {
                                cn[is] = cguess[(phase + phase*(NUMPHASES))*(NUMCOMPONENTS-1) + is];
                            }

                            #pragma acc parallel loop
                            for (count; count < maxCount; count++)
                            {
                                tmp0 = 0.0;
                                #pragma acc parallel loop reduction(+:tmp0)
                                for (is = 0; is < NUMCOMPONENTS-1; is++)
                                {
                                    co[is] = cn[is];
                                    y[is]  = co[is];
                                    tmp0  += co[is];
                                }
                                y[NUMCOMPONENTS-1] = 1.0 - tmp0;

                                
                                (*Mu_tdb_dev[thermo_phase[phase]])(temperature, y, mu0);

                                #pragma acc parallel loop
                                for (is = 0; is < NUMCOMPONENTS-1; is++)
                                    fun[is] = (mu0[is] - mu[is][idx]);

                               
                                (*dmudc_tdb_dev[thermo_phase[phase]])(temperature, y, retdmuphase);

                                #pragma acc parallel loop collapse(2)
                                for (is1 = 0; is1 < NUMCOMPONENTS-1; is1++)
                                {
                                    for (is2 = 0; is2 < NUMCOMPONENTS-1; is2++)
                                    {
                                        retdmuphase2[is1][is2] = retdmuphase[is1*(NUMCOMPONENTS-1) + is2];
                                    }
                                }

                                 // Newton-Raphson (-J^{-1}F)
                                #pragma acc parallel loop
                                for (is1 = 0; is1 < NUMCOMPONENTS-1; is1++)
                                {
                                    tmp0 = 0.0;
                                    #pragma acc parallel loop reduction(+:tmp0)
                                    for (is2 = 0; is2 < NUMCOMPONENTS-1; is2++)
                                    {
                                        tmp0 += jacInv[is1][is2] * fun[is2];
                                    }

                                    cn[is1] = co[is1] - tmp0;
                                }

                                // L-inf norm
                                norm = 0.0;
                                #pragma acc parallel loop
                                for (is = 0; is < NUMCOMPONENTS-1; is++)
                                    if (fabs(cn[is] - co[is]) > 1e-6)
                                        norm = 1.0;

                                if(norm == 0)
                                    count = maxCount;
                            }

                            #pragma acc parallel loop
                            for (is = 0; is < NUMCOMPONENTS-1; is++)
                            phaseComp[is*NUMPHASES + phase][idx] = cn[is];
                        }
                    }

                     // Check conservation of comp
                    deltac_flag = 0;
                    
                    #pragma acc parallel loop
                    for (is = 0; is < NUMCOMPONENTS-1; is++)
                    {
                        deltac[is] = 0.0;
                        #pragma acc parallel loop reduction(+:deltac[:MAX_NUM_COMP]) // check size
                        for (int phase = 0; phase < NUMPHASES; phase++)
                        {
                            deltac[is] += phaseComp[is*NUMPHASES + phase][idx]*calcInterp5th(phi, phase, idx, NUMPHASES);
                        }

                        deltac[is] = comp[is][idx] - deltac[is];

                        if (fabs(deltac[is]) > 1e-6)
                            deltac_flag = 1;
                    }

                    if (deltac_flag)
                    {
                        #pragma acc parallel loop collapse(2)
                         for (int component = 0; component < NUMCOMPONENTS-1; component++)
                        {   
                            for (int component2 = 0; component2 < NUMCOMPONENTS-1; component2++)
                            {
                                dcdmu[component*(NUMCOMPONENTS-1) + component2] = 0.0;
                            }
                        }

                        for (long phase = 0; phase < NUMPHASES; phase++)
                        {
                            double sum = 0.0;
                            #pragma acc parallel loop reduction(+:sum)
                            for (long component = 0; component < NUMCOMPONENTS-1; component++)
                            {
                                y[component] = phaseComp[component*NUMPHASES + phase][idx];
                                sum += y[component];
                            }

                            y[NUMCOMPONENTS-1] = 1.0 - sum;

                            (*dmudc_tdb_dev[thermo_phase[phase]])(temperature, y, dmudc);

                            LUPDecomposeC2(dmudc, NUMCOMPONENTS-1, tol, P);
                            LUPInvertC2(dmudc, P, NUMCOMPONENTS-1, Inv);
                            #pragma acc parallel loop collapse(2) reduction(+:dcdmu[:(MAX_NUM_COMP)*(MAX_NUM_COMP)])
                            for (long component = 0; component < NUMCOMPONENTS-1; component++)
                                for (long component2 = 0; component2 < NUMCOMPONENTS-1; component2++)
                                    dcdmu[component*(NUMCOMPONENTS-1) + component2] += calcInterp5th(phi, phase, idx, NUMPHASES)*Inv[component][component2];
                        }

                         LUPDecomposeC2(dcdmu, NUMCOMPONENTS-1, tol, P);
                         LUPInvertC2(dcdmu, P, NUMCOMPONENTS-1, Inv);

                        #pragma acc parallel loop collapse(2) reduction(+:mu) // check size
                        for (int component = 0; component < NUMCOMPONENTS-1; component++)
                        {
                            for (int component2 = 0; component2 < NUMCOMPONENTS-1; component2++)
                            {
                                mu[component][idx] += Inv[component][component2]*deltac[component2];
                            }
                        }
                    }

            }
        }

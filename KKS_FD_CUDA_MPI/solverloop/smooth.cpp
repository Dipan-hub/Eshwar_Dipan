#define phiAniso(phase, x, y, z) (phiAniso[(((phase)*3 + (x))*3 + (y))*3 + (z)])

void __smooth__(double **phi, double **phiNew,
                double *relaxCoeff, double *kappaPhi,
                double *dab, double *Rotation_matrix, double *Inv_rotation_matrix, int FUNCTION_ANISOTROPY,
                long NUMPHASES, long NUMCOMPONENTS, long DIMENSION,
                long sizeX, long sizeY, long sizeZ,
                long xStep, long yStep, long padding,
                double DELTA_X, double DELTA_Y, double DELTA_Z,
                double DELTA_t)
{
	 long x, y, z;

    long index[3][3][3];

    double phiAniso[MAX_NUM_PHASES*27];

    double aniso[MAX_NUM_PHASES] = {0.0};
    double dfdphiSum = 0.0;

    long phase, p;
	// check the data present declaration , declare arrays with sizes , for size look in the main file
    #pragma acc data present(phi, phiNew,relaxCoeff,kappaPhi,dab, Rotation_matrix, Inv_rotation_matrix, FUNCTION_ANISOTROPY,NUMPHASES,NUMCOMPONENTS,DIMENSION, sizeX,sizeY,sizeZ,xStep, yStep,padding,DELTA_X, DELTA_Y,DELTA_Z,DELTA_t) create(index , phiAniso,aniso,dfdphiSum, phase , p)
    {

    		#pragma acc parallel loop collapse(6)
		    for (long i = padding; i < sizeX - padding; i++)
		    {
		        for (long j = (DIMENSION >= 2 ? padding : 0); j < (DIMENSION >= 2 ? sizeY - padding : 1); j++)
		        {
		            for (long k = (DIMENSION == 3 ? padding : 0); k < (DIMENSION == 3 ? sizeZ - padding : 1); k++)
		            {
		                for (long x = 0; x < 3; x++)
		                {
		                    for (long y = 0; y < 3; y++)
		                    {
		                        for (long z = 0; z < 3; z++)
		                        {
		                            index[x][y][z] = (k + z - 1) + (j + y - 1) * yStep + (i + x - 1) * xStep;
		                        }
		                    }
		                }
		            }
		        }
		    }

		    if(DIMENSION ==3)
		    {
			    #pragma acc parallel loop collapse(4)
			    for (phase = 0; phase < NUMPHASES; phase++)
		        {
		            for (x = 0; x < 3; x++)
		            {
		                for (y = 0; y < 3; y++)
		                {
		             
	                        for (z = 0; z < 3; z++)
	                        {
	                            phiAniso(phase, x, y, z) = phi[phase][index[x][y][z]];
	                        }

		                }
		            }
		        }
		    }	

		    else
		    {
			    #pragma acc parallel loop collapse(3)
			    for (phase = 0; phase < NUMPHASES; phase++)
		        {
		            for (x = 0; x < 3; x++)
		            {
		                for (y = 0; y < 3; y++)
		                {
		                    phiAniso(phase, x, y, 1) = phi[phase][index[x][y][1]];  
		                }
		            }
		        }
		    }



		    if (FUNCTION_ANISOTROPY == 0)
        	{	

        		if(DIMENSION == 1)
        		{
        			#pragma acc parallel loop
        			for (phase = 0; phase < NUMPHASES; phase++)
            		{	
            			aniso[phase] = (phiAniso(phase, 0, 1, 1) - 2.0*phiAniso(phase, 1, 1, 1) + phiAniso(phase, 2, 1, 1))/(DELTA_X*DELTA_X);
            		}
        		}
        		else if (DIMENSION == 2)
        		{
        			#pragma acc parallel loop
        			for(phase = 0 ; phase<NUMPHASES ; phase++)
        			{
	    				//Centre
	                    aniso[phase] = -3.0*phiAniso(phase, 1, 1, 1)/(DELTA_X*DELTA_Y);

	                    // Nearest neighbours
	                    aniso[phase] += 0.5*(phiAniso(phase, 0, 1, 1) + phiAniso(phase, 2, 1, 1))/(DELTA_X*DELTA_X);
	                    aniso[phase] += 0.5*(phiAniso(phase, 1, 0, 1) + phiAniso(phase, 1, 2, 1))/(DELTA_Y*DELTA_Y);

	                    // Second-nearest neighbours
	                    aniso[phase] += 0.25*(phiAniso(phase, 0, 0, 1) + phiAniso(phase, 0, 2, 1) + phiAniso(phase, 2, 2, 1) + phiAniso(phase, 2, 0, 1))/(DELTA_X*DELTA_Y);
        			}
        		}

        		else if (DIMENSION == 3)
        		{
        			#pragma acc parallel loop
        			for(phase = 0 ; phase<NUMPHASES ; phase++)
        			{
        				aniso[phase] = -4.0*phiAniso(phase, 1, 1, 1)/(DELTA_X*DELTA_X);

	                    // Nearest neighbours
	                    aniso[phase] += (phiAniso(phase, 0, 1, 1) + phiAniso(phase, 2, 1, 1))/(3.0*DELTA_X*DELTA_X);
	                    aniso[phase] += (phiAniso(phase, 1, 0, 1) + phiAniso(phase, 1, 2, 1))/(3.0*DELTA_Y*DELTA_Y);
	                    aniso[phase] += (phiAniso(phase, 1, 1, 0) + phiAniso(phase, 1, 1, 2))/(3.0*DELTA_Z*DELTA_Z);

	                    // Second-nearest neighbours
	                    aniso[phase] += (phiAniso(phase, 0, 0, 1) + phiAniso(phase, 0, 2, 1) + phiAniso(phase, 2, 2, 1) + phiAniso(phase, 2, 0, 1))/(6.0*DELTA_X*DELTA_Y);
	                    aniso[phase] += (phiAniso(phase, 1, 0, 0) + phiAniso(phase, 1, 0, 2) + phiAniso(phase, 1, 2, 2) + phiAniso(phase, 1, 2, 0))/(6.0*DELTA_Y*DELTA_Z);
	                    aniso[phase] += (phiAniso(phase, 0, 1, 0) + phiAniso(phase, 0, 1, 2) + phiAniso(phase, 2, 1, 2) + phiAniso(phase, 2, 1, 0))/(6.0*DELTA_Z*DELTA_X);
        			}
        		}	
        	}

        	else if (FUNCTION_ANISOTROPY == 1 || FUNCTION_ANISOTROPY == 2)
        	{
	            #pragma acc parallel loop
	            for (phase = 0; phase < NUMPHASES; phase++)
	            {
	                aniso[phase] = calcAnisotropy_01(phiAniso, dab, kappaPhi, Rotation_matrix, Inv_rotation_matrix, phase, NUMPHASES, DIMENSION, DELTA_X, DELTA_Y, DELTA_Z);
	            }
        	}

        	if(FUNCTION_ANISOTROPY == 0)
        	{
        		#pragma acc parallel loop 
        		for(phase = 0 ; phase<NUMPHASES ; phase++)
        		{
        			dfdphiSum = 0.0;
        			#pragma acc parallel loop reduction(+:dfdphiSum)
	           		for(p = 0 ; p<NUMPHASES ;p++)
	           		{
	           		  if (p == phase)
	                    continue;

	           		  dfdphiSum += 2.0*kappaPhi[phase*NUMPHASES + p]*(aniso[p] - aniso[phase]);
	           		}

	           		phiNew[phase][index[1][1][1]] = phi[phase][index[1][1][1]] - DELTA_t*FunctionTau(phi, relaxCoeff, index[1][1][1], NUMPHASES)*dfdphiSum/(double)NUMPHASES;
        		}
        	}
        	else if(FUNCTION_ANISOTROPY == 1 || FUNCTION_ANISOTROPY == 2)
        	{
				#pragma acc parallel loop 
        		for(phase = 0 ; phase<NUMPHASES ; phase++)
				{	
					dfdphiSum = 0.0;
	        		#pragma acc parallel loop reduction(+:dfdphiSum)
	           		for(p = 0 ; p<NUMPHASES ;p++)
	           		{
	           		  if (p == phase)
	                    continue;

	           		  dfdphiSum += 2.0*(aniso[p] - aniso[phase]);
	           		}
	           	 	 phiNew[phase][index[1][1][1]] = phi[phase][index[1][1][1]] - DELTA_t*FunctionTau(phi, relaxCoeff, index[1][1][1], NUMPHASES)*dfdphiSum/(double)NUMPHASES;
	           	}
        	}  
    }
}

void smooth(double **phi, double **phiNew,
            domainInfo* simDomain, controls* simControls,
            simParameters* simParams, subdomainInfo* subdomain,
            dim3 gridSize, dim3 blockSize)
{
    __smooth__(phi, phiNew,
               simParams->relax_coeff_dev, simParams->kappaPhi_dev,
               simParams->dab_dev, simParams->Rotation_matrix_dev, simParams->Inv_Rotation_matrix_dev, 0,
               simDomain->numPhases, simDomain->numComponents, simDomain->DIMENSION,
               subdomain->sizeX, subdomain->sizeY, subdomain->sizeZ,
               subdomain->xStep, subdomain->yStep, subdomain->padding,
               simDomain->DELTA_X, simDomain->DELTA_Y, simDomain->DELTA_Z,
               simControls->DELTA_t);
}

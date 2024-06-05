#include "computeDrivingForce.hpp"

void __computeDrivingForce__helper__(double **phi, double **comp,
                             double **dfdphi,
                             double **phaseComp,
                             double *F0_A, double *F0_B, double *F0_C,
                             double molarVolume,
                             double *theta_i, double *theta_ij, double *theta_ijk,
                             int ELASTICITY,
                             long NUMPHASES, long NUMCOMPONENTS, long DIMENSION,
                             long sizeX, long sizeY, long sizeZ,
                             long xStep, long yStep, long padding,int idx)
{
    double Bpq_hphi[MAX_NUM_PHASES];
   // check the data present declaration , declare arrays with sizes , for size look in the main file 

    #pragma acc data present(phi[:NUMPHASES], comp[:NUMCOMPONENTS-1],dfdphi[:NUMPHASES],phaseComp[:NUMPHASES*NUMCOMPONENTS-1],F0_A[:NUMPHASES*(NUMCOMPONENTS-1)*(NUMCOMPONENTS-1), F0_B[:NUMPHASES*(NUMCOMPONENTS-1)], F0_C[:NUMPHASES],molarVolume,theta_i[:NUMPHASES],theta_ij[:NUMPHASES*NUMPHASES],theta_ijk[:NUMPHASES*NUMPHASES*NUMPHASES],ELASTICITY,NUMPHASES,NUMCOMPONENTS,DIMENSION,sizeX,sizeY,sizeZ,xStep,yStep,padding,idx) create(Bpq_hphi[:MAX_NUM_PHASES])
   {
        #pragma acc parallel loop
        for (long p = 0; p < NUMPHASES; p++)
        {
            Bpq_hphi[p] = dfdphi[p][idx];
        }
        #pragma acc parallel loop reduction(+:dfdphi[:NUMPHASES*2])//replace 2 with range of idx 
        for (long phase = 0; phase < NUMPHASES; phase++)
        {
            if (ELASTICITY)
            {
                dfdphi[phase][idx] = 0.0;
                #pragma acc parallel loop 
                for (long p = 0; p < NUMPHASES; p++)
                    dfdphi[phase][idx] += calcInterp5thDiff(phi, phase, p, idx, NUMPHASES)*Bpq_hphi[p];

            }
             #pragma acc parallel loop // multiple operations on psi , how to apply reduction 
            for (long p = 0; p < NUMPHASES; p++)
            {
                /*
                 * \psi_{p} = f^{p} - \sum_{i=1}^{K-1} (c^{p}_{i}\mu_{i})
                 */
                psi = 0.0;

                psi += calcPhaseEnergy(phaseComp, p, F0_A, F0_B, F0_C, idx, NUMPHASES, NUMCOMPONENTS);
                 #pragma acc parallel loop 
                for (long component = 0; component < NUMCOMPONENTS-1; component++)
                    psi -= calcDiffusionPotential(phaseComp, p, component, F0_A, F0_B, idx, NUMPHASES, NUMCOMPONENTS)*phaseComp[(component*NUMPHASES) + p][idx];

                /*
                 * \frac{\delta F}{\delta\phi_{phase}} += \sum_{p=1}^{N} (\frac{\partial h(\phi_{p})}{\partial \phi_{phase}} \frac{\psi_{p}}{V_{m}})
                 */
                psi *= calcInterp5thDiff(phi, p, phase, idx, NUMPHASES);

                dfdphi[phase][idx] += psi/molarVolume;
            }

            /*
             * Potential function
             * \frac{\delta F}{\delta\phi_{phase}} += \frac{\partial g(\phi_{phase})}{\partial\phi_{phase}}
             */
            dfdphi[phase][idx] += calcDoubleWellDerivative(phi, phase,
                                                       theta_i, theta_ij, theta_ijk,
                                                       idx, NUMPHASES);
        }

   } 

void __computeDrivingForce__(double **phi, double **comp,
                             double **dfdphi,
                             double **phaseComp,
                             double *F0_A, double *F0_B, double *F0_C,
                             double molarVolume,
                             double *theta_i, double *theta_ij, double *theta_ijk,
                             int ELASTICITY,
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
                #pragma acc data copyin(idx)
                __computeDrivingForce__helper__(phi,comp,
                                                dfdphi,
                                                phaseComp,
                                                F0_A, F0_B,F0_C,
                                                molarVolume,
                                                theta_i, theta_ij, theta_ijk,
                                                ELASTICITY,
                                                NUMPHASES, NUMCOMPONENTS, DIMENSION,
                                                sizeX, sizeY, sizeZ,
                                                xStep,yStep,padding);
            }
        }
    }
}

void computeDrivingForce_Chemical(double **phi, double **comp,
                                  double **dfdphi,
                                  double **phaseComp, double **mu,
                                  domainInfo* simDomain, controls* simControls,
                                  simParameters* simParams, subdomainInfo* subdomain,
                                  dim3 gridSize, dim3 blockSize)
{

    if (simControls->FUNCTION_F == 1 || simControls->FUNCTION_F == 3  || simControls->FUNCTION_F == 4)
    {
        __computeDrivingForce__(phi, comp,
                                dfdphi,
                                phaseComp,
                                simParams->F0_A_dev, simParams->F0_B_dev, simParams->F0_C_dev,
                                simParams->molarVolume,
                                simParams->theta_i_dev, simParams->theta_ij_dev, simParams->theta_ijk_dev,
                                simControls->ELASTICITY,
                                simDomain->numPhases, simDomain->numComponents, simDomain->DIMENSION,
                                subdomain->sizeX, subdomain->sizeY, subdomain->sizeZ,
                                subdomain->xStep, subdomain->yStep, subdomain->padding);
    }
    else if (simControls->FUNCTION_F == 2)
    {
        __computeDrivingForce_02__(phi, comp,
                                   dfdphi, phaseComp,
                                   mu,
                                   simParams->molarVolume,
                                   simParams->theta_i_dev, simParams->theta_ij_dev, simParams->theta_ijk_dev,
                                   simControls->ELASTICITY,
                                   simParams->T, simDomain->thermo_phase_dev,
                                   simDomain->numPhases, simDomain->numComponents, simDomain->DIMENSION,
                                   subdomain->sizeX, subdomain->sizeY, subdomain->sizeZ,
                                   subdomain->xStep, subdomain->yStep, subdomain->padding);
    }
}

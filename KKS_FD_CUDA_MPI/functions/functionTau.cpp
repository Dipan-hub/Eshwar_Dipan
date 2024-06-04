#include "functionTau.hpp"
#include <accelmath.h> // OpenACC math library

/*
 *  Weight phase-field mobility using local phase-field values
 *  Replicated from GP solver
 */
double FunctionTau(double **phi, double *relaxCoeff, long idx, long NUMPHASES)
{
    double sum = 0.0, sum1 = 0.0;
    long a, b;

    #pragma acc parallel loop reduction(+:sum,sum1) present(phi, relaxCoeff)
    for (a = 0; a < NUMPHASES; a++)
    {
        for (b = 0; b < NUMPHASES; b++)
        {
            if (a < b)
            {
                sum  += relaxCoeff[a*NUMPHASES + b]*phi[a][idx]*phi[b][idx];
                sum1 += phi[a][idx]*phi[b][idx];
            }
        }
    }

    if (sum1)
    {
        return sum/sum1;
    }
    else
    {
        return relaxCoeff[1];
    }
}

void calculateTau(domainInfo *simDomain, controls *simControls, simParameters *simParams)
{
    double Tol = 1e-9;

    double sum = 0.0;
    double minTau = -1e12;

    if (simControls->FUNCTION_F != 2)
    {
        double ***dmudc = Malloc3M(simDomain->numPhases, simDomain->numComponents-1, simDomain->numComponents-1);

        for (long i = 0; i < simDomain->numPhases; i++)
        {
            for (long j = 0; j < simDomain->numComponents-1; j++)
            {
                for (long k = 0; k < simDomain->numComponents-1; k++)
                {
                    if (j == k)
                        dmudc[i][j][k] = 2.0*simParams->F0_A_host[i][j][k];
                    else
                        dmudc[i][j][k] = simParams->F0_A_host[i][j][k];
                }
            }
        }

        // Mobility matrix
        for (long i = 0; i < simDomain->numPhases; i++)
        {
            double **inverted = MallocM(simDomain->numComponents-1, simDomain->numComponents-1);
            double **dmudc_local = MallocM(simDomain->numComponents-1, simDomain->numComponents-1);

            for (long j = 0; j < simDomain->numComponents-1; j++)
            {
                for (long k = 0; k < simDomain->numComponents-1; k++)
                {
                    dmudc_local[j][k] = dmudc[i][j][k];
                }
            }

            matinvnew(dmudc_local, inverted, simDomain->numComponents-1);

            matrixMultiply(simParams->diffusivity_host[i], inverted, simParams->mobility_host[i], simDomain->numComponents-1);

            FreeM(inverted, simDomain->numComponents-1);
            FreeM(dmudc_local, simDomain->numComponents-1);
        }

        // Relaxation Coefficients
        double **mobilityInv = MallocM(simDomain->numComponents-1, simDomain->numComponents-1);
        double deltac[simDomain->numComponents-1], deltamu[simDomain->numComponents-1];

        double **inverted = MallocM(simDomain->numComponents-1, simDomain->numComponents-1);

        // Reusing inverted to hold mobility of the reference phase
        for (long i = 0; i < simDomain->numComponents-1; i++)
        {
            for (long j = 0; j < simDomain->numComponents-1; j++)
            {
                inverted[i][j] = simParams->mobility_host[simDomain->numPhases-1][i][j];
            }
        }

        matinvnew(inverted, mobilityInv, simDomain->numComponents-1);

        for (long a = 0; a < simDomain->numPhases-1; a++)
        {
            for (long k = 0; k < simDomain->numComponents-1; k++)
                deltac[k] = simParams->ceq_host[a][simDomain->numPhases-1][k] - simParams->ceq_host[a][a][k];

            for (long i = 0; i < simDomain->numComponents-1; i++)
            {
                deltamu[i] = 0.0;

                for (long j = 0; j < simDomain->numComponents-1; j++)
                {
                    deltamu[i] += deltac[j]*mobilityInv[i][j];
                }
            }

            sum = 0.0;
            for (long i = 0; i < simDomain->numComponents-1; i++)
            {
                sum += deltac[i]*deltamu[i];
            }

            simParams->Tau_host[a][simDomain->numPhases-1] = (3.0*0.783333*simParams->kappaPhi_host[a][simDomain->numPhases-1]*sum)/(simParams->theta_ij_host[a][simDomain->numPhases-1]*simParams->molarVolume);
            simParams->Tau_host[simDomain->numPhases-1][a] = simParams->Tau_host[a][simDomain->numPhases-1];

            if (a == 0)
                minTau = simParams->Tau_host[a][simDomain->numPhases-1];

            if (simParams->Tau_host[a][simDomain->numPhases-1] > minTau)
                minTau = simParams->Tau_host[a][simDomain->numPhases-1];
        }

        for (long a = 0; a < simDomain->numPhases; a++)
        {
            for (long b = 0; b < simDomain->numPhases; b++)
            {
                simParams->Tau_host[a][b] = 1.0/minTau;
                simParams->relax_coeff_host[a][b] = 1.0/minTau;
            }
        }

        FreeM(mobilityInv, simDomain->numComponents-1);
        FreeM(inverted, simDomain->numComponents-1);
        Free3M(dmudc, simDomain->numPhases, simDomain->numComponents-1);
    }
    else if (simControls->FUNCTION_F == 2)
    {
        // Relaxation Coefficients
        double **diffInv = MallocM(simDomain->numComponents-1, simDomain->numComponents-1);
        double **diff = MallocM(simDomain->numComponents-1, simDomain->numComponents-1);
        double dmudc[(simDomain->numComponents-1)*(simDomain->numComponents-1)];
        double **mobilityInv = MallocM(simDomain->numComponents-1, simDomain->numComponents-1);
        double deltac[simDomain->numComponents-1], deltamu[simDomain->numComponents-1];
        double y[simDomain->numComponents-1];

        // Get diffusivity of matrix phase
        for (long i = 0; i < simDomain->numComponents-1; i++)
        {
            for (long j = 0; j < simDomain->numComponents-1; j++)
            {
                diff[i][j] = simParams->diffusivity_host[simDomain->numPhases-1][i][j];
            }
        }

        int P[simDomain->numComponents];

        // Get D^{-1}
        LUPDecompose(diff, simDomain->numComponents-1, Tol, P);
        LUPInvert(diff, P, simDomain->numComponents-1, diffInv);

        // Get equilibrium compositions of matrix phase
        for (long i = 0; i < simDomain->numComponents-1; i++)
        {
            y[i] = simParams->ceq_host[simDomain->numPhases-1][simDomain->numPhases-1][i];
            sum += y[i];
        }
        y[simDomain->numComponents-1] = 1.0 - sum;

        // Get dmudc in matrix
        (*dmudc_tdb[simDomain->thermo_phase_host[simDomain->numPhases-1]])(simParams->T, y, dmudc);

        // Get mobility inverse as dmudc*D^{-1}
        for (long i = 0; i < simDomain->numComponents-1; i++)
        {
            for (long j = 0; j < simDomain->numComponents-1; j++)
            {
                mobilityInv[i][j] = 0.0;

                for (long k = 0; k < simDomain->numComponents-1; k++)
                {
                    mobilityInv[i][j] += dmudc[i*(simDomain->numComponents-1) + k]*diffInv[k][j];
                }
            }
        }

        for (long a = 0; a < simDomain->numPhases-1; a++)
        {
            for (long k = 0; k < simDomain->numComponents-1; k++)
                deltac[k] = simParams->ceq_host[a][simDomain->numPhases-1][k] - simParams->ceq_host[a][a][k];

            for (long i = 0; i < simDomain->numComponents-1; i++)
            {
                deltamu[i] = 0.0;

                for (long j = 0; j < simDomain->numComponents-1; j++)
                {
                    deltamu[i] += deltac[j]*mobilityInv[i][j];
                }
            }

            sum = 0.0;
            for (long i = 0; i < simDomain->numComponents-1; i++)
            {
                sum += deltac[i]*deltamu[i];
            }

            simParams->Tau_host[a][simDomain->numPhases-1] = (3.0*0.783333*simParams->kappaPhi_host[a][simDomain->numPhases-1]*sum)/(simParams->theta_ij_host[a][simDomain->numPhases-1]*simParams->molarVolume);
            simParams->Tau_host[simDomain->numPhases-1][a] = simParams->Tau_host[a][simDomain->numPhases-1];

            if (a == 0)
                minTau = simParams->Tau_host[a][simDomain->numPhases-1];

            if (simParams->Tau_host[a][simDomain->numPhases-1] > minTau)
                minTau = simParams->Tau_host[a][simDomain->numPhases-1];
        }

        for (long a = 0; a < simDomain->numPhases; a++)
        {
            for (long b = 0; b < simDomain->numPhases; b++)
            {
                simParams->Tau_host[a][b] = 1.0/minTau;
                simParams->relax_coeff_host[a][b] = 1.0/minTau;
            }
        }

        FreeM(diff, simDomain->numComponents-1);
        FreeM(mobilityInv, simDomain->numComponents-1);
        FreeM(diffInv, simDomain->numComponents-1);
    }
}

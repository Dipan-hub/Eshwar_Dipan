#include <cmath>
#include <openacc.h>
#include <iostream>

// Utility kernel to compute the change in every cell
void computeChange(double *A, double *B, long DIMENSION, long sizeX, long sizeY, long sizeZ) {
    #pragma acc parallel loop collapse(3)
    for (long i = 0; i < sizeX; ++i) {
        for (long j = 0; j < sizeY; ++j) {
            for (long k = 0; k < sizeZ; ++k) {
                long idx = (j + i*sizeY)*sizeZ + k;
                if ((j < sizeY && DIMENSION >= 2) || (DIMENSION == 1 && j == 0)) {
                    if ((k < sizeZ && DIMENSION == 3) || (DIMENSION < 3 && k == 0)) {
                        A[idx] = std::fabs(B[idx] - A[idx]);
                    }
                }
            }
        }
    }
}

// Utility kernel to reset array
void resetArray(double **arr, long numArr, long DIMENSION, long sizeX, long sizeY, long sizeZ) {
    #pragma acc parallel loop collapse(3)
    for (long i = 0; i < sizeX; ++i) {
        for (long j = 0; j < sizeY; ++j) {
            for (long k = 0; k < sizeZ; ++k) {
                long idx = (j + i*sizeY)*sizeZ + k;
                if ((j < sizeY && DIMENSION >= 2) || (DIMENSION == 1 && j == 0)) {
                    if ((k < sizeZ && DIMENSION == 3) || (DIMENSION < 3 && k == 0)) {
                        for (long iter = 0; iter < numArr; iter++) {
                            arr[iter][idx] = 0.0;
                        }
                    }
                }
            }
        }
    }
}

// Aggregate max, min, relative change for each subdomain
void printStats(double **phi, double **comp, double **phiNew, double **compNew, double *maxerr, double *maxVal, double *minVal,
                domainInfo simDomain, subdomainInfo subdomain) {
    long i, j = 0;

    #pragma acc data copyin(compNew[0:simDomain.numComponents-1][0:subdomain.numCompCells], \
                            phiNew[0:simDomain.numPhases][0:subdomain.numCompCells]) \
                    copyout(maxVal[0:simDomain.numComponents-1+simDomain.numPhases], \
                            minVal[0:simDomain.numComponents-1+simDomain.numPhases], \
                            maxerr[0:simDomain.numComponents-1+simDomain.numPhases])
    {
        // Get all stats for compositions
        for (i = 0; i < simDomain.numComponents - 1; i++) {
            double local_max = -DBL_MAX;
            double local_min = DBL_MAX;
            double local_err = -DBL_MAX;

            #pragma acc parallel loop reduction(max:local_max) reduction(min:local_min)
            for (long idx = 0; idx < subdomain.numCompCells; ++idx) {
                if (compNew[i][idx] > local_max) local_max = compNew[i][idx];
                if (compNew[i][idx] < local_min) local_min = compNew[i][idx];
            }

            maxVal[j] = local_max;
            minVal[j] = local_min;

            computeChange(comp[i], compNew[i], simDomain.DIMENSION, subdomain.sizeX, subdomain.sizeY, subdomain.sizeZ);

            #pragma acc parallel loop reduction(max:local_err)
            for (long idx = 0; idx < subdomain.numCompCells; ++idx) {
                if (comp[i][idx] > local_err) local_err = comp[i][idx];
            }

            maxerr[j] = local_err;
            j++;
        }

        // Get all stats for phi
        for (i = 0; i < simDomain.numPhases; i++) {
            double local_max = -DBL_MAX;
            double local_min = DBL_MAX;
            double local_err = -DBL_MAX;

            #pragma acc parallel loop reduction(max:local_max) reduction(min:local_min)
            for (long idx = 0; idx < subdomain.numCompCells; ++idx) {
                if (phiNew[i][idx] > local_max) local_max = phiNew[i][idx];
                if (phiNew[i][idx] < local_min) local_min = phiNew[i][idx];
            }

            maxVal[j] = local_max;
            minVal[j] = local_min;

            computeChange(phi[i], phiNew[i], simDomain.DIMENSION, subdomain.sizeX, subdomain.sizeY, subdomain.sizeZ);

            #pragma acc parallel loop reduction(max:local_err)
            for (long idx = 0; idx < subdomain.numCompCells; ++idx) {
                if (phi[i][idx] > local_err) local_err = phi[i][idx];
            }

            maxerr[j] = local_err;
            j++;
        }
    }
}

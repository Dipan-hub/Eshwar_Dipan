#ifndef UPDATEPHI_HPP_
#define UPDATEPHI_HPP_

#include <stdio.h>

#include "helper_string.h"
//#include "helper_cuda.hpp"

#include "structures.hpp"
#include "Thermo.h"
#include "utilityKernels.hpp"
#include "matrix.hpp"
#include "functionA_01.hpp"
#include "functionA_02.hpp"
#include "functionTau.hpp"
#include "boundary.hpp"

/*
 * Function prototype for the kernel that solves d(\phi_{i})/dt = -L/N \sum_{j=1, i \neq j}^{N} (df/dphi_{i} - df/dphi_{j})
 */
void updatePhi_kernel(double **phi, double **dfdphi, double **phiNew,
                      double *relaxCoeff, double *kappaPhi,
                      double *dab, int FUNCTION_ANISOTROPY,
                      long NUMPHASES, long NUMCOMPONENTS, long DIMENSION, long FUNCTION_F,
                      long sizeX, long sizeY, long sizeZ,
                      long xStep, long yStep, long padding,
                      double DELTA_X, double DELTA_Y, double DELTA_Z,
                      double DELTA_t);

/*
 * Host-side wrapper function for updatePhi_kernel
 */
#ifdef __cplusplus
extern "C"
#endif
void updatePhi(double **phi, double **dfdphi, double **phiNew, double **phaseComp,
               domainInfo* simDomain, controls* simControls,
               simParameters* simParams, subdomainInfo* subdomain,
               dim3 gridSize, dim3 blockSize);

#endif

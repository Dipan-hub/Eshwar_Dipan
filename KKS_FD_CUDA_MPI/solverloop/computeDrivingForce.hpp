#ifndef COMPUTEDRIVINGFORCE_HPP_
#define COMPUTEDRIVINGFORCE_HPP_

#include <stdio.h>
#include "structures.hpp"
#include "utilityKernels.hpp"
#include "Thermo.hpp"
#include "matrix.hpp"
#include "functionF.hpp"
#include "functionW_01.hpp"
#include "functionW_02.hpp"
#include "functionH.hpp"

/*
 * Explicit calculation of the right-hand side of the Allen-Cahn equation.
 * Evaluation of the mobility function in the Cahn-Hilliard equation.
 */
void computeDrivingForce_kernel(double **phi, double **comp,
                                double **dfdphi,
                                double **phaseComp,
                                double *F0_A, double *F0_B, double *F0_C,
                                double molarVolume,
                                double *theta_i, double *theta_ij, double *theta_ijk,
                                int ELASTICITY,
                                long NUMPHASES, long NUMCOMPONENTS, long DIMENSION,
                                long sizeX, long sizeY, long sizeZ,
                                long xStep, long yStep, long padding);

void computeDrivingForce_02_kernel(double **phi, double **comp,
                                   double **dfdphi, double **phaseComp,
                                   double **mu,
                                   double molarVolume,
                                   double *theta_i, double *theta_ij, double *theta_ijk,
                                   int ELASTICITY,
                                   double temperature, long *thermo_phase,
                                   long NUMPHASES, long NUMCOMPONENTS, long DIMENSION,
                                   long sizeX, long sizeY, long sizeZ,
                                   long xStep, long yStep, long padding);

/*
 * Wrapper function for computeDrivingForce_kernel
 */
#ifdef __cplusplus
extern "C"
#endif
void computeDrivingForce_Chemical(double **phi, double **comp,
                                  double **dfdphi,
                                  double **phaseComp, double **mu,
                                  domainInfo* simDomain, controls* simControls,
                                  simParameters* simParams, subdomainInfo* subdomain,
                                  dim3 gridSize, dim3 blockSize);

#endif

#ifndef CALCPHASECOMP_HPP_
#define CALCPHASECOMP_HPP_

#include <stdio.h>
#include "structures.hpp"
#include "utilityKernels.hpp"
#include "Thermo.hpp"
#include "matrix.hpp"
#include "functionF.hpp"
#include "functionH.hpp"

/*
 * Calculate phase compositions for Function_F != 2
 */
void calcPhaseComp_kernel(double **phi, double **comp,
                          double **phaseComp,
                          double *F0_A, double *F0_B, double *F0_C,
                          long NUMPHASES, long NUMCOMPONENTS, long DIMENSION,
                          long sizeX, long sizeY, long sizeZ, long padding);

/*
 *  Initialise diffusion potentials for Function_F == 2
 */
void initMu_kernel(double **phi, double **comp, double **phaseComp, double **mu,
                   long *thermo_phase, double temperature,
                   long NUMPHASES, long NUMCOMPONENTS, long DIMENSION,
                   long sizeX, long sizeY, long sizeZ,
                   long xStep, long yStep, long padding);

/*
 * Calculate phase compositions for Function_F == 2
 */
void calcPhaseComp_02_kernel(double **phi, double **comp,
                             double **phaseComp, double **mu, double *cguess,
                             double temperature, long *thermo_phase,
                             long NUMPHASES, long NUMCOMPONENTS, long DIMENSION,
                             long sizeX, long sizeY, long sizeZ, long padding);

/*
 * Wrapper function for calcPhaseComp_kernel
 */
#ifdef __cplusplus
extern "C"
#endif
void calcPhaseComp(double **phi, double **comp,
                   double **phaseComp, double **mu,
                   domainInfo* simDomain, controls* simControls,
                   simParameters* simParams, subdomainInfo* subdomain,
                   dim3 gridSize, dim3 blockSize);

#endif

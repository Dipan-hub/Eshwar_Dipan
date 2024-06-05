#ifndef UPDATECOMPOSITION_HPP_
#define UPDATECOMPOSITION_HPP_

#include <stdio.h>
#include "structures.hpp"
#include "Thermo.hpp"
#include "utilityKernels.hpp"
#include "matrix.hpp"
#include "functionH.hpp"
#include "functionF.hpp"
#include "boundary.hpp"

/*
 * Function prototypes for kernels that solve dc_{j}/dt = div(M grad(mu)) using a fourth order FD stencil and forward Euler
 */

// For functions F_01, F_03, F_04
void updateComposition_kernel(double **phi, double **phiNew,
                              double **comp, double **compNew,
                              double **phaseComp,
                              double *F0_A, double *F0_B,
                              double *mobility, double *diffusivity, double *kappaPhi, double *theta_ij,
                              long NUMPHASES, long NUMCOMPONENTS, long DIMENSION,
                              long sizeX, long sizeY, long sizeZ,
                              long xStep, long yStep, long padding, int antiTrapping,
                              double DELTA_X, double DELTA_Y, double DELTA_Z,
                              double DELTA_t);

// For function F_02
void updateComposition_02_kernel(double **phi, double **phiNew,
                                 double **comp, double **compNew, double **mu,
                                 double **phaseComp, long *thermo_phase,
                                 double *diffusivity, double *kappaPhi, double *theta_ij,
                                 double temperature, double molarVolume,
                                 long NUMPHASES, long NUMCOMPONENTS, long DIMENSION,
                                 long sizeX, long sizeY, long sizeZ,
                                 long xStep, long yStep, long padding,
                                 double DELTA_X, double DELTA_Y, double DELTA_Z,
                                 double DELTA_t);

void updateMu_02_kernel(double **phi, double **comp,
                        double **phiNew, double **compNew,
                        double **phaseComp, double **mu,
                        long *thermo_phase, double temperature, double molarVolume,
                        long NUMPHASES, long NUMCOMPONENTS, long DIMENSION,
                        long sizeX, long sizeY, long sizeZ,
                        long xStep, long yStep, long padding,
                        double DELTA_X, double DELTA_Y, double DELTA_Z,
                        double DELTA_t);

/*
 * Host-side wrapper function for updateComposition_kernel
 */
#ifdef __cplusplus
extern "C"
#endif
void updateComposition(double **phi, double **comp, double **phiNew, double **compNew,
                       double **phaseComp, double **mu,
                       domainInfo simDomain, controls simControls,
                       simParameters simParams, subdomainInfo subdomain,
                       dim3 gridSize, dim3 blockSize);

#endif

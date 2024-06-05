#ifndef SMOOTH_HPP_
#define SMOOTH_HPP_

#include <stdio.h>
#include "structures.hpp"
#include "Thermo.hpp"
// #include "functionA_02.hpp"
#include "functionF.hpp"
#include "functionTau.hpp"
#include "utilityKernels.hpp"
#include "functionA_01.hpp"

void smooth_kernel(double **phi, double **phiNew,
                   double *relaxCoeff, double *kappaPhi,
                   double *dab, double *Rotation_matrix, double *Inv_rotation_matrix, int FUNCTION_ANISOTROPY,
                   long NUMPHASES, long NUMCOMPONENTS, long DIMENSION,
                   long sizeX, long sizeY, long sizeZ,
                   long xStep, long yStep, long padding,
                   double DELTA_X, double DELTA_Y, double DELTA_Z,
                   double DELTA_t);

#ifdef __cplusplus
extern "C"
#endif
void smooth(double **phi, double **phiNew,
            domainInfo* simDomain, controls* simControls,
            simParameters* simParams, subdomainInfo* subdomain,
            dim3 gridSize, dim3 blockSize);

#endif

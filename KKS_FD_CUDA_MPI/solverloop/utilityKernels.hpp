#ifndef UTILITYKERNELS_HPP_
#define UTILITYKERNELS_HPP_

#include <cmath>
#include "Thermo.h"
#include "structures.hpp"

void computeChange(double *A, double *B, long DIMENSION, long sizeX, long sizeY, long sizeZ);

void resetArray(double **arr, long numArr, long sizeX, long sizeY, long sizeZ);

void printStats(double **phi, double **comp, double **phiNew, double **compNew, 
                double *maxerr, double *maxVal, double *minVal, 
                domainInfo simDomain, subdomainInfo subdomain);

void resetArray(double **arr, long numArr, long DIMENSION, long sizeX, long sizeY, long sizeZ);

#endif // UTILITYKERNELS_H_

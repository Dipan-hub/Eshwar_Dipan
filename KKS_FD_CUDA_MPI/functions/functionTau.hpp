#ifndef FUNCTIONTAU_HPP_
#define FUNCTIONTAU_HPP_

#include "structures.hpp"
#include "matrix.hpp"
#include "utilityFunctions.h"

double FunctionTau(double **phi, double *relaxCoeff, long idx, long NUMPHASES);

void calculateTau(domainInfo *simDomain, controls *simControls, simParameters *simParams);

#endif

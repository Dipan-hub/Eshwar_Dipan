#ifndef ANISOTROPY_01_HPP_
#define ANISOTROPY_01_HPP_

#include <cmath>
#include <cstdio>

void anisotropy_01_dAdq(double *qab, double *dadq, long a, long b, double *dab, long NUMPHASES);

double anisotropy_01_function_ac(double *qab, long a, long b, double *dab, long NUMPHASES);

#endif 

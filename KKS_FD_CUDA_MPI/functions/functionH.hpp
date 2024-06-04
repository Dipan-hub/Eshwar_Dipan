#ifndef FUNCTIONH_HPP_
#define FUNCTIONH_HPP_

#include <stdio.h>

double calcInterp5th(double **phi, long a, long idx, long NUMPHASES);
double calcInterp5thDiff(double **phi, long a, long b, long idx, long NUMPHASES);
double calcInterp3rd(double **phi, long a, long idx, long NUMPHASES);
double calcInterp3rdDiff(double **phi, long a, long b, long idx, long NUMPHASES);

#endif // FUNCTIONH_HPP_

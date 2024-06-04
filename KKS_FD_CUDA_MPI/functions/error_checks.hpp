#ifndef ERROR_CHECKS_HPP_
#define ERROR_CHECKS_HPP_

#ifndef ENABLE_CUFFTMP
#define ENABLE_CUFFTMP 0
#endif

#if ENABLE_CUFFTMP == 1

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>

// Define a generic error check macro
#define CHECK_ERROR(ans) { check_error((ans), __FILE__, __LINE__); }
void check_error(int code, const char *file, int line, bool abort=true);

// Function to assess error
int assess_error(double error);

#endif // ENABLE_CUFFTMP
#endif // ERROR_CHECKS_HPP_

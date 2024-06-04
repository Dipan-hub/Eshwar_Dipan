#include "error_checks.hpp"

#if ENABLE_CUFFTMP == 1

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>

void check_error(int code, const char *file, int line, bool abort)
{
    if (code != 0)  // Assuming non-zero code indicates an error
    {
        fprintf(stderr, "ERROR: %d %s %d\n", code, file, line);
        if (abort) exit(code);
    }
}

int assess_error(double error)
{
    if (error > 1e-6)
    {
        printf("FAILED with error %e\n", error);
        return 1;
    }
    else
    {
        printf("PASSED with error %e\n", error);
        return 0;
    }
}

#endif

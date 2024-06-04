#include "functionW_02.hpp"

/*
 * Calculate g'(\phi)
 *
 * Arguments:
 *              1. double **phi -> all the phase volume-fraction values
 *              2. long phase -> differentiate wrt to this phase
 *              3. double *theta_i   -> coefficients for theta_i, one per phase
 *              4. double *theta_ij  -> coefficients for theta_ij, one per pair of phases
 *              5. double *theta_ijk -> coefficients for theta_ijk, one per triplet of phases
 *              6. long idx -> position of cell in 1D
 *              7. long NUMPHASES -> number of phases
 * Return:
 *              numerical evaluation of derivative of interpolation polynomial, as a double datatype
 */
double calcDoubleWellDerivative(double **phi, long phase,
                                double *theta_i, double *theta_ij, double *theta_ijk,
                                long idx, long NUMPHASES)
{
    if (NUMPHASES < 2)
        return 0.0;

    double ans = 0.0;

    #pragma acc parallel loop reduction(+:ans) present(phi[0:NUMPHASES][0:idx+1], theta_i[0:NUMPHASES], theta_ij[0:NUMPHASES*NUMPHASES], theta_ijk[0:NUMPHASES*NUMPHASES*NUMPHASES])
    for (long i = 0; i < NUMPHASES; i++)
    {
        double temp1 = phi[i][idx] * phi[i][idx];

        // Derivative of \phi^{2}(1 - \phi)^{2}
        if (i == phase)
            ans += theta_i[i] * 2.0 * phi[i][idx] * (1.0 - phi[i][idx]) * (1.0 - 2.0 * phi[i][idx]);

        #pragma acc loop reduction(+:ans)
        for (long j = 0; j < NUMPHASES; j++)
        {
            if (j == i)
                continue;

            double temp2 = phi[j][idx] * phi[j][idx];

            // Derivative of \sum_{i=1}^{N} \sum_{j=1, j!= i}^{N} \phi^{2}_{i}\phi^{2}_{j}
            if (i == phase)
                ans += 2.0 * theta_ij[j + i * NUMPHASES] * phi[i][idx] * temp2;
            else if (j == phase)
                ans += 2.0 * theta_ij[j + i * NUMPHASES] * temp1 * phi[j][idx];

            #pragma acc loop reduction(+:ans)
            for (long k = 0; k < NUMPHASES; k++)
            {
                if (k == i || k == j)
                    continue;

                // Derivative of (ijk)^2
                if (i == phase)
                    ans += 2.0 * theta_ijk[(j + i * NUMPHASES) * NUMPHASES + k] * phi[i][idx] * temp2 * phi[k][idx] * phi[k][idx];
                else if (j == phase)
                    ans += 2.0 * theta_ijk[(j + i * NUMPHASES) * NUMPHASES + k] * temp1 * phi[j][idx] * phi[k][idx] * phi[k][idx];
                else if (k == phase)
                    ans += 2.0 * theta_ijk[(j + i * NUMPHASES) * NUMPHASES + k] * temp1 * temp2 * phi[k][idx];
            }
        }
    }

    return ans;
}

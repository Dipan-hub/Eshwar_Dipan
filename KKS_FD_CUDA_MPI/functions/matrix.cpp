#include "matrix.hpp"
#include <accelmath.h>
#include <openacc.h>
#include <cmath>
#include <cstdlib>

// Sum of two vectors
void vectorsum(double *y1, double *y2, double *sum, long size) {
    #pragma acc parallel loop
    for (long j = 0; j < size; j++) {
        sum[j] = y1[j] + y2[j];
    }
}

// Multiplication of matrix and vector
void multiply(double **inv, double *y, double *prod, long size) {
    #pragma acc parallel loop
    for (long i = 0; i < size; i++) {
        double sum = 0;
        for (long j = 0; j < size; j++) {
            sum += inv[i][j] * y[j];
        }
        prod[i] = sum;
    }
}

// Multiplication of two matrices
void multiply2d(double **m1, double **m2, double **prod, long size) {
    #pragma acc parallel loop
    for (long k = 0; k < size; k++) {
        for (long i = 0; i < size; i++) {
            double sum = 0;
            for (long j = 0; j < size; j++) {
                sum += m1[k][j] * m2[j][i];
            }
            prod[k][i] = sum;
        }
    }
}

// Matrix inversion using LU decomposition
void matinvnew(double **coeffmatrix, double **inv, long size) {
    int *tag = new int[size];
    double **factor = MallocM(size, size);
    double **inv1 = MallocM(size, size);
    double **iden = MallocM(size, size);
    double **prod = MallocM(size, size);
    double *vec1 = MallocV(size);
    double *vec = MallocV(size);

    // Making the Upper Triangular Matrix.
    for (long k = 0; k < size; k++) {
        tag[k] = k;
    }
    for (long k = 0; k < size; k++) {
        pivot(coeffmatrix, factor, k, tag, size);
        #pragma acc parallel loop
        for (long i = k + 1; i < size; i++) {
            double fact = -coeffmatrix[i][k] / coeffmatrix[k][k];
            factor[i][k] = -fact;
            #pragma acc loop
            for (long j = k; j < size; j++) {
                coeffmatrix[i][j] = fact * coeffmatrix[k][j] + coeffmatrix[i][j];
            }
        }
    }
    #pragma acc parallel loop
    for (long i = 0; i < size; i++) {
        for (long j = 0; j < size; j++) {
            if (i == j) factor[i][j] = 1;
            if (j > i) factor[i][j] = 0;
        }
    }

    // The Identity Matrix.
    #pragma acc parallel loop
    for (long i = 0; i < size; i++) {
        for (long j = 0; j < size; j++) {
            if (i == j) iden[i][j] = 1;
            else iden[i][j] = 0;
        }
    }

    // Forward and backward substitution to get the final identity matrix.
    for (long i = 0; i < size; i++) {
        substitutef(factor, iden, i, vec1, size);
        substituteb(coeffmatrix, vec1, vec, size);
        #pragma acc parallel loop
        for (long j = 0; j < size; j++) {
            inv1[j][i] = vec[j];
        }
    }

    colswap(inv1, inv, tag, size);
    multiply2d(factor, coeffmatrix, prod, size);
    rowswap(prod, coeffmatrix, tag, size);

    FreeM(factor, size);
    FreeM(iden, size);
    FreeM(inv1, size);
    FreeM(prod, size);
    free(vec1);
    free(vec);

    delete[] tag;
}

// Back Substitution
void substituteb(double **fac, double *y, double *vec, long size) {
    vec[size - 1] = y[size - 1] * pow(fac[size - 1][size - 1], -1);
    for (long i = size - 2; i >= 0; i--) {
        double sum = 0;
        #pragma acc loop
        for (long j = i + 1; j < size; j++) {
            sum -= fac[i][j] * vec[j];
        }
        vec[i] = (y[i] + sum) * pow(fac[i][i], -1);
    }
}

// Forward Substitution
void substitutef(double **fac, double **y1, int index, double *vec, long size) {
    double *d = new double[size];
    #pragma acc parallel loop
    for (long i = 0; i < size; i++) {
        d[i] = y1[i][index];
    }
    vec[0] = d[0];
    for (long i = 1; i < size; i++) {
        double sum = 0;
        #pragma acc loop
        for (long j = 0; j < i; j++) {
            sum -= fac[i][j] * vec[j];
        }
        vec[i] = d[i] + sum;
    }
    delete[] d;
}

// Modulus operator
double mod(double k) {
    return (k < 0) ? -k : k;
}

//2nd

// Pivoting
void pivot(double **coeffmatrix, double **factor, int k, int *tag, long size) {
    double swap, big;
    int tagswap, tag1;
    big = mod(coeffmatrix[k][k]);
    tag1 = k;
    #pragma acc parallel loop reduction(max:big)
    for (int i = k + 1; i < size; i++) {
        if (mod(coeffmatrix[i][k]) > big) {
            tag1 = i;
            big = coeffmatrix[i][k];
        }
    }
    tagswap = tag[k];
    tag[k] = tag[tag1];
    tag[tag1] = tagswap;

    #pragma acc parallel loop
    for (int i = 0; i < size; i++) {
        swap = coeffmatrix[k][i];
        coeffmatrix[k][i] = coeffmatrix[tag1][i];
        coeffmatrix[tag1][i] = swap;
    }

    #pragma acc parallel loop
    for (int i = 0; i < k; i++) {
        swap = factor[k][i];
        factor[k][i] = factor[tag1][i];
        factor[tag1][i] = swap;
    }
}

// Swapping Columns To get the final identity matrix because of the initial swapping for pivoting
void colswap(double **m1, double **m2, int *tag, long size) {
    #pragma acc parallel loop collapse(3)
    for (int k = 0; k < size; k++) {
        for (int j = 0; j < size; j++) {
            for (int p = 0; p < size; p++) {
                m2[p][tag[j]] = m1[p][j];
            }
        }
    }
}

// Switching rows
void rowswap(double **m1, double **m2, int *tag, long size) {
    #pragma acc parallel loop collapse(3)
    for (int k = 0; k < size; k++) {
        for (int j = 0; j < size; j++) {
            for (int p = 0; p < size; p++) {
                m2[tag[j]][p] = m1[j][p];
            }
        }
    }
}

// LU Decomposition
int LUPDecompose(double **A, int N, double Tol, int *P) {
    int imax;
    double maxA, *ptr, absA;

    #pragma acc parallel loop
    for (int i = 0; i <= N; i++) {
        P[i] = i; // Unit permutation matrix, P[N] initialized with N
    }

    for (int i = 0; i < N; i++) {
        maxA = 0.0;
        imax = i;

        #pragma acc parallel loop reduction(max:maxA)
        for (int k = i; k < N; k++) {
            absA = fabs(A[k][i]);
            if (absA > maxA) {
                maxA = absA;
                imax = k;
            }
        }

        if (maxA < Tol) return 0; // failure, matrix is degenerate

        if (imax != i) {
            // pivoting P
            int j = P[i];
            P[i] = P[imax];
            P[imax] = j;

            // pivoting rows of A
            ptr = A[i];
            A[i] = A[imax];
            A[imax] = ptr;

            // counting pivots starting from N (for determinant)
            P[N]++;
        }

        #pragma acc parallel loop
        for (int j = i + 1; j < N; j++) {
            A[j][i] /= A[i][i];

            #pragma acc loop
            for (int k = i + 1; k < N; k++) {
                A[j][k] -= A[j][i] * A[i][k];
            }
        }
    }

    return 1; // decomposition done
}

// Inversion after LU-decomposition
void LUPInvert(double **A, int *P, int N, double **IA) {
    #pragma acc parallel loop collapse(2)
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
            IA[i][j] = (P[i] == j) ? 1.0 : 0.0;

            #pragma acc loop
            for (int k = 0; k < i; k++) {
                IA[i][j] -= A[i][k] * IA[k][j];
            }
        }

        for (int i = N - 1; i >= 0; i--) {
            #pragma acc loop
            for (int k = i + 1; k < N; k++) {
                IA[i][j] -= A[i][k] * IA[k][j];
            }

            IA[i][j] /= A[i][i];
        }
    }
}

// AB = C, N is the dimension of all 3 matrices
void matrixMultiply(double **A, double **B, double **C, int N) {
    #pragma acc parallel loop collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0.0;

            #pragma acc loop
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// A is an N*N*3*3 matrix, B is an N-element vector, C is of length N
void multiply(double *A, double *B, double *C, long ip1, long ip2, long NUMPHASES, int N) {
    #pragma acc parallel loop
    for (int i = 0; i < N; i++) {
        double sum = 0.0;
        for (int j = 0; j < N; j++) {
            sum += A[((ip1 * NUMPHASES + ip2) * 3 + i) * 3 + j] * B[j];
        }
        C[i] = sum;
    }
}

//3rd

//extern void LUPDecomposeC1(double A[][MAX_NUM_COMP], long N, double Tol, int *P) 
extern int LUPDecomposeC1(double A[][MAX_NUM_COMP], long N, double Tol, int *P)
{
    long i, j, k, imax;
    double maxA, ptr, absA;

    #pragma acc parallel loop
    for (i = 0; i <= N; i++) {
        P[i] = i; // Unit permutation matrix, P[N] initialized with N
    }

    for (i = 0; i < N; i++) {
        maxA = 0.0;
        imax = i;

        #pragma acc parallel loop reduction(max:maxA)
        for (k = i; k < N; k++) {
            absA = fabs(A[k][i]);
            if (absA > maxA) {
                maxA = absA;
                imax = k;
            }
        }

        if (maxA < Tol) return 0; // failure, matrix is degenerate

        if (imax != i) {
            // pivoting P
            j = P[i];
            P[i] = P[imax];
            P[imax] = j;

            // pivoting rows of A
            #pragma acc parallel loop
            for (j = 0; j < N; j++) {
                ptr = A[i][j];
                A[i][j] = A[imax][j];
                A[imax][j] = ptr;
            }

            // counting pivots starting from N (for determinant)
            P[N]++;
        }

        #pragma acc parallel loop
        for (j = i + 1; j < N; j++) {
            A[j][i] /= A[i][i];

            #pragma acc loop
            for (k = i + 1; k < N; k++) {
                A[j][k] -= A[j][i] * A[i][k];
            }
        }
    }

    return 1; // decomposition done
}

//extern void LUPDecomposeC2(double A[(MAX_NUM_COMP) * (MAX_NUM_COMP)], long N, double Tol, int *P) 
extern int LUPDecomposeC2(double A[(MAX_NUM_COMP) * (MAX_NUM_COMP)], long N, double Tol, int *P)
{
    long i, j, k, imax;
    double maxA, ptr, absA;

    #pragma acc parallel loop
    for (i = 0; i <= N; i++) {
        P[i] = i; // Unit permutation matrix, P[N] initialized with N
    }

    for (i = 0; i < N; i++) {
        maxA = 0.0;
        imax = i;

        #pragma acc parallel loop reduction(max:maxA)
        for (k = i; k < N; k++) {
            absA = fabs(A[k * N + i]);
            if (absA > maxA) {
                maxA = absA;
                imax = k;
            }
        }

        if (maxA < Tol) return 0; // failure, matrix is degenerate

        if (imax != i) {
            // pivoting P
            j = P[i];
            P[i] = P[imax];
            P[imax] = j;

            // pivoting rows of A
            #pragma acc parallel loop
            for (j = 0; j < N; j++) {
                ptr = A[i * N + j];
                A[i * N + j] = A[imax * N + j];
                A[imax * N + j] = ptr;
            }

            // counting pivots starting from N (for determinant)
            P[N]++;
        }

        #pragma acc parallel loop
        for (j = i + 1; j < N; j++) {
            A[j * N + i] /= A[i * N + i];

            #pragma acc loop
            for (k = i + 1; k < N; k++) {
                A[j * N + k] -= A[j * N + i] * A[i * N + k];
            }
        }
    }

    return 1; // decomposition done
}

extern void LUPSolveC1(double A[][MAX_NUM_COMP], int *P, double *b, long N, double *x) {
    #pragma acc parallel loop
    for (long i = 0; i < N; i++) {
        x[i] = b[P[i]];

        #pragma acc loop
        for (long k = 0; k < i; k++) {
            x[i] -= A[i][k] * x[k];
        }
    }

    #pragma acc parallel loop
    for (long i = N - 1; i >= 0; i--) {
        #pragma acc loop
        for (long k = i + 1; k < N; k++) {
            x[i] -= A[i][k] * x[k];
        }

        x[i] /= A[i][i];
    }
}


//4th


extern void LUPSolveC2(double A[(MAX_NUM_COMP)*(MAX_NUM_COMP)], int *P, double *b, long N, double *x) {
    #pragma acc parallel loop
    for (long i = 0; i < N; i++) {
        x[i] = b[P[i]];

        #pragma acc loop
        for (long k = 0; k < i; k++) {
            x[i] -= A[i * N + k] * x[k];
        }
    }

    #pragma acc parallel loop
    for (long i = N - 1; i >= 0; i--) {
        #pragma acc loop
        for (long k = i + 1; k < N; k++) {
            x[i] -= A[i * N + k] * x[k];
        }

        x[i] /= A[i * N + i];
    }
}

extern void LUPInvertC1(double A[][MAX_NUM_COMP], int *P, long N, double IA[][MAX_NUM_COMP]) {
    #pragma acc parallel loop collapse(2)
    for (long j = 0; j < N; j++) {
        for (long i = 0; i < N; i++) {
            IA[i][j] = P[i] == j ? 1.0 : 0.0;

            #pragma acc loop
            for (long k = 0; k < i; k++) {
                IA[i][j] -= A[i][k] * IA[k][j];
            }
        }

        #pragma acc parallel loop
        for (long i = N - 1; i >= 0; i--) {
            #pragma acc loop
            for (long k = i + 1; k < N; k++) {
                IA[i][j] -= A[i][k] * IA[k][j];
            }

            IA[i][j] /= A[i][i];
        }
    }
}

extern void LUPInvertC2(double A[(MAX_NUM_COMP)*(MAX_NUM_COMP)], int *P, long N, double IA[][MAX_NUM_COMP]) {
    #pragma acc parallel loop collapse(2)
    for (long j = 0; j < N; j++) {
        for (long i = 0; i < N; i++) {
            IA[i][j] = P[i] == j ? 1.0 : 0.0;

            #pragma acc loop
            for (long k = 0; k < i; k++) {
                IA[i][j] -= A[i * N + k] * IA[k][j];
            }
        }

        #pragma acc parallel loop
        for (long i = N - 1; i >= 0; i--) {
            #pragma acc loop
            for (long k = i + 1; k < N; k++) {
                IA[i][j] -= A[i * N + k] * IA[k][j];
            }

            IA[i][j] /= A[i * N + i];
        }
    }
}

//extern void LUPDecomposePC1(double A[][MAX_NUM_PHASE_COMP], long N, double Tol, int *P)
extern int LUPDecomposePC1(double A[][MAX_NUM_PHASE_COMP], long N, double Tol, int *P)
 {
    long i, j, k, imax;
    double maxA, ptr, absA;

    #pragma acc parallel loop
    for (i = 0; i <= N; i++) {
        P[i] = i; // Unit permutation matrix, P[N] initialized with N
    }

    for (i = 0; i < N; i++) {
        maxA = 0.0;
        imax = i;

        #pragma acc parallel loop reduction(max:maxA)
        for (k = i; k < N; k++) {
            absA = fabs(A[k][i]);
            if (absA > maxA) {
                maxA = absA;
                imax = k;
            }
        }

        if (maxA < Tol) return 0; // failure, matrix is degenerate

        if (imax != i) {
            // pivoting P
            j = P[i];
            P[i] = P[imax];
            P[imax] = j;

            // pivoting rows of A
            #pragma acc parallel loop
            for (j = 0; j < N; j++) {
                ptr = A[i][j];
                A[i][j] = A[imax][j];
                A[imax][j] = ptr;
            }

            // counting pivots starting from N (for determinant)
            P[N]++;
        }

        #pragma acc parallel loop
        for (j = i + 1; j < N; j++) {
            A[j][i] /= A[i][i];

            #pragma acc loop
            for (k = i + 1; k < N; k++) {
                A[j][k] -= A[j][i] * A[i][k];
            }
        }
    }

    return 1; // decomposition done
}

//extern void LUPDecomposePC2(double A[(MAX_NUM_PHASE_COMP)*(MAX_NUM_PHASE_COMP)], long N, double Tol, int *P)
extern int LUPDecomposePC2(double A[(MAX_NUM_PHASE_COMP)*(MAX_NUM_PHASE_COMP)], long N, double Tol, int *P)
 {
    long i, j, k, imax;
    double maxA, ptr, absA;

    #pragma acc parallel loop
    for (i = 0; i <= N; i++) {
        P[i] = i; // Unit permutation matrix, P[N] initialized with N
    }

    for (i = 0; i < N; i++) {
        maxA = 0.0;
        imax = i;

        #pragma acc parallel loop reduction(max:maxA)
        for (k = i; k < N; k++) {
            absA = fabs(A[k * N + i]);
            if (absA > maxA) {
                maxA = absA;
                imax = k;
            }
        }

        if (maxA < Tol) return 0; // failure, matrix is degenerate

        if (imax != i) {
            // pivoting P
            j = P[i];
            P[i] = P[imax];
            P[imax] = j;

            // pivoting rows of A
            #pragma acc parallel loop
            for (j = 0; j < N; j++) {
                ptr = A[i * N + j];
                A[i * N + j] = A[imax * N + j];
                A[imax * N + j] = ptr;
            }

            // counting pivots starting from N (for determinant)
            P[N]++;
        }

        #pragma acc parallel loop
        for (j = i + 1; j < N; j++) {
            A[j * N + i] /= A[i * N + i];

            #pragma acc loop
            for (k = i + 1; k < N; k++) {
                A[j * N + k] -= A[j * N + i] * A[i * N + k];
            }
        }
    }

    return 1; // decomposition done
}

extern void LUPSolvePC1(double A[][MAX_NUM_PHASE_COMP], int *P, double *b, long N, double *x) {
    #pragma acc parallel loop
    for (long i = 0; i < N; i++) {
        x[i] = b[P[i]];

        #pragma acc loop
        for (long k = 0; k < i; k++) {
            x[i] -= A[i][k] * x[k];
        }
    }

    #pragma acc parallel loop
    for (long i = N - 1; i >= 0; i--) {
        #pragma acc loop
        for (long k = i + 1; k < N; k++) {
            x[i] -= A[i][k] * x[k];
        }

        x[i] /= A[i][i];
    }
}

extern void LUPSolvePC2(double A[(MAX_NUM_PHASE_COMP)*(MAX_NUM_PHASE_COMP)], int *P, double *b, long N, double *x) {
    #pragma acc parallel loop
    for (long i = 0; i < N; i++) {
        x[i] = b[P[i]];

        #pragma acc loop
        for (long k = 0; k < i; k++) {
            x[i] -= A[i * N + k] * x[k];
        }
    }

    #pragma acc parallel loop
    for (long i = N - 1; i >= 0; i--) {
        #pragma acc loop
        for (long k = i + 1; k < N; k++) {
            x[i] -= A[i * N + k] * x[k];
        }

        x[i] /= A[i * N + i];
    }
}

extern void LUPInvertPC1(double A[][MAX_NUM_PHASE_COMP], int *P, long N, double IA[][MAX_NUM_PHASE_COMP]) {
    #pragma acc parallel loop collapse(2)
    for (long j = 0; j < N; j++) {
        for (long i = 0; i < N; i++) {
            IA[i][j] = P[i] == j ? 1.0 : 0.0;

            #pragma acc loop
            for (long k = 0; k < i; k++) {
                IA[i][j] -= A[i][k] * IA[k][j];
            }
        }

        #pragma acc parallel loop
        for (long i = N - 1; i >= 0; i--) {
            #pragma acc loop
            for (long k = i + 1; k < N; k++) {
                IA[i][j] -= A[i][k] * IA[k][j];
            }

            IA[i][j] /= A[i][i];
        }
    }
}

extern void LUPInvertPC2(double A[(MAX_NUM_PHASE_COMP)*(MAX_NUM_PHASE_COMP)], int *P, long N, double IA[][MAX_NUM_PHASE_COMP]) {
    #pragma acc parallel loop collapse(2)
    for (long j = 0; j < N; j++) {
        for (long i = 0; i < N; i++) {
            IA[i][j] = P[i] == j ? 1.0 : 0.0;

            #pragma acc loop
            for (long k = 0; k < i; k++) {
                IA[i][j] -= A[i * N + k] * IA[k][j];
            }
        }

        #pragma acc parallel loop
        for (long i = N - 1; i >= 0; i--) {
            #pragma acc loop
            for (long k = i + 1; k < N; k++) {
                IA[i][j] -= A[i * N + k] * IA[k][j];
            }

            IA[i][j] /= A[i * N + i];
        }
    }
}


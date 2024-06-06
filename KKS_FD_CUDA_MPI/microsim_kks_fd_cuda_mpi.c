// Standard libraries
#include <numeric>
#include <vector>
#include <complex>
#include <random>
#include <cstdlib>
#include <cstdio>

#include <mpi.h>
#include <mpi-ext.h>
#include <sys/stat.h>
#include <unistd.h>

// Flags and constants
#define MASTER 0

#ifndef ENABLE_CUFFTMP
#define ENABLE_CUFFTMP 0
#endif

#ifndef ENABLE_HDF5
#define ENABLE_HDF5 0
#endif

// C header files
// From ./functions/
#include "structures.hpp"
#include "inputReader.h"
#include "initialize_variables.h"
#include "filling.h"
#include "utilityFunctions.h"
#include "helper_string.h"
// From ./solverloop/
#include "fileWriter.h"

// OpenACC header files
// From ./functions
#include "functionF.hpp"
// From ./solverloop/
#include "boundary.hpp"
#include "smooth.hpp"
#include "calcPhaseComp.hpp"
#include "computeDrivingForce.hpp"
#include "updateComposition.hpp"
#include "updatePhi.hpp"
#include "computeElastic.cuh"
#include "utilityKernels.hpp"

int main(int argc, char *argv[])
{
    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;

    int MPI_Enabled = 1;
    int rank, size;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);

    if (size == 1)
        MPI_Enabled = 0;

    if (MPI_Enabled && rank == MASTER)
        printf("MPI enabled\n");

    domainInfo simDomain;
    controls simControls;
    simParameters simParams;
    fillParameters simFill;
    subdomainInfo subdomain;

    int numDevices = 1;  // Assuming one device per process for OpenACC

    // Phase-field variables on host-side
    double *phi;
    double *comp;
    double *mu;

    // for-loop iterators
    long i, j;

    // Variables for finding and storing statistics
    double *maxerr;
    double *maxerr_h;
    double *maxVal, *minVal;
    double *maxVal_h, *minVal_h;

    // Set up simulation using only one process
    if (rank == MASTER)
    {
        // Create directory for output storage
        mkdir("DATA", 0777);
    }

    if (MPI_Enabled)
    {
        MPI_Barrier(comm);
    }

    // Read input from specified input file
    if (readInput_MPI(&simDomain, &simControls, &simParams, rank, argv))
    {
        if (rank == MASTER)
        {
            printf("\n----------------------------------------------------------------------------ERROR---------------------------------------------------------------------------\n");
            printf("\nSolver not compiled to handle required number of phases and/or components.\n");
            printf("Please select the appropriate number for both, using the MicroSim GUI or by running GEdata_writer.py with the desired input file through the command line\n");
            if (MAX_NUM_PHASES < simDomain.numPhases)
                printf("Currently, the maximum number of phases supported is %ld, but attempted to run with %d phases\n", MAX_NUM_PHASES, simDomain.numPhases);
            if (MAX_NUM_COMP < simDomain.numComponents)
                printf("Currently, the maximum number of components supported is %ld, but you have attempted to run with %d components\n", MAX_NUM_COMP, simDomain.numComponents);
            printf("\n------------------------------------------------------------------------------------------------------------------------------------------------------------\n");
        }

        MPI_Abort(comm, 1);
    }

    read_boundary_conditions(&simDomain, &simControls, &simParams, rank, argv);

    if (MPI_Enabled)
    {
        MPI_Barrier(comm);
    }

    // Fix padding size
    simControls.padding = 1;
    simControls.numworkers = size;

    // Initialize free energy, simulation parameters
    calcFreeEnergyCoeffs(&simDomain, &simControls, &simParams);
    moveParamsToGPU(&simDomain, &simControls, &simParams);

    // Create subdirectories for every processor's share of the output
    char directory[1000];
    sprintf(directory, "DATA/Processor_%d", rank);
    mkdir(directory, 0777);

    if (MPI_Enabled)
    {
        MPI_Barrier(comm);
    }

    // Distribute data to all the processors
    decomposeDomain(simDomain, &simControls, &subdomain, rank, size);

    // Decide appropriate kernel grid size and block size
    //Finad an alternative
    //calcKernelParams(&gridSize, &blockSize, simDomain, simControls, &subdomain);
    // Allocate memory on CPUs
    phi = (double*)malloc(sizeof(double) * simDomain.numPhases * subdomain.numCompCells);
    comp = (double*)malloc(sizeof(double) * (simDomain.numComponents-1) * subdomain.numCompCells);
    if (simControls.FUNCTION_F == 2)
        mu = (double*)malloc(sizeof(double) * (simDomain.numComponents-1) * subdomain.numCompCells);

    for (i = 0; i < simDomain.numPhases*subdomain.numCompCells; i++)
        phi[i] = 0.0;

    for (i = 0; i < (simDomain.numComponents-1)*subdomain.numCompCells; i++)
        comp[i] = 0.0;

    if (simControls.FUNCTION_F == 2)
        for (i = 0; i < (simDomain.numComponents-1)*subdomain.numCompCells; i++)
            mu[i] = 0.0;

#if ENABLE_CUFFTMP == 0
    simControls.ELASTICITY = 0;
#elif ENABLE_CUFFTMP == 1

    int FFT_Alternate= 0;

    double **BpqHost, *B_calc, *kx, *ky, *kz;

    const size_t my_nx = subdomain.numX;
    const size_t my_ny = subdomain.numY;
    const size_t my_nz = subdomain.numZ;

    printf("cuFFTMp is enabled\n");
    printf("For processor %d: %d, %d, %d\n", rank, (int)my_nx, (int)my_ny, (int)my_nz);

    // Create a plan
    // Plan creation for OpenACC can be different. This is a placeholder.
    acc_plan plan = 0;

    if (simControls.ELASTICITY)
    {
        // Plan creation logic here
    }

    // OpenACC variables
    double **phiElDev[simDomain.numPhases], **dfEldphi[simDomain.numPhases];

    if (simControls.ELASTICITY)
    {
        for (i = 0; i < simDomain.numPhases; i++)
        {
            // Memory allocation for OpenACC
        }
    }

    if (simControls.ELASTICITY)
    {
        BpqHost = (double**)malloc(sizeof(double*)*simDomain.numPhases*simDomain.numPhases);

        for (i = 0; i < simDomain.numPhases*simDomain.numPhases; i++)
        {
            BpqHost[i] = (double*)malloc(sizeof(double)*my_nx*my_ny*my_nz);
        }

        B_calc = (double*)malloc(sizeof(double)*my_nx*my_ny*my_nz);
        kx     = (double*)malloc(sizeof(double)*my_nx*my_ny*my_nz);
        ky     = (double*)malloc(sizeof(double)*my_nx*my_ny*my_nz);
        kz     = (double*)malloc(sizeof(double)*my_nx*my_ny*my_nz);

        calc_k(kx, ky, kz,
               my_nx, my_ny, my_nz,
               simDomain, simControls, simParams, subdomain);

        for (i = 0; i < simDomain.numPhases; i++)
        {
            for (j = 0; j < simDomain.numPhases; j++)
            {
                calculate_Bn(B_calc, kx, ky, kz, simDomain, simParams, subdomain, i, j);
                memcpy(BpqHost[i*simDomain.numPhases + j], B_calc, sizeof(double)*my_nx*my_ny*my_nz);
            }
        }

        free(B_calc);
        free(kx);
        free(ky);
        free(kz);
    }
#endif

//2nd

if (MPI_Enabled)
{
    MPI_Barrier(comm);
    #pragma acc wait
}

// Not restarting
if (simControls.restart == 0 && simControls.startTime == 0)
{
    // Read geometry from specified filling file
    readFill(&simFill, argv, rank);

    // Initialise the domain on the CPU using the read geometry
    fillDomain(simDomain, subdomain, simParams, phi, comp, &simFill);

    if (rank == MASTER)
        printf("Finished filling\n");
}

// If restarting
if (!(simControls.restart == 0) || !(simControls.startTime == 0))
{
    if (rank == MASTER)
        printf("Reading from disk\n");

    #if ENABLE_HDF5 == 1
    if (simControls.writeHDF5)
    {
        readHDF5(phi, comp, mu,
                 simDomain, subdomain,
                 simControls, rank, comm, argv);
    }
    else
        #endif
    {
        read_domain(phi, comp, mu, simDomain, subdomain, simControls, rank, comm, argv);
    }
}

if (MPI_Enabled)
{
    MPI_Barrier(comm);
    #pragma acc wait
}

if (rank == MASTER)
    printf("\nAllocating memory on GPUs\n");

// Allocate memory for values required to be calculated during solution,
// like residuals and explicit calculations of derivatives

// These pointers can only be dereferenced by device-side code
// Allows dereferencing in kernels using [phase/component][location]
#pragma acc enter data create(compDev[:simDomain.numComponents-1], compNewDev[:simDomain.numComponents-1], phiDev[:simDomain.numPhases], dfdphiDev[:simDomain.numPhases], phiNewDev[:simDomain.numPhases], phaseCompDev[:simDomain.numPhases*(simDomain.numComponents-1)])

if (simControls.FUNCTION_F == 2)
    #pragma acc enter data create(muDev[:simDomain.numComponents-1])

// These pointers can be dereferenced at the phase or component level on the host-side
// Useful for data transfer and CUB compatibility
compHost    = (double**)malloc((simDomain.numComponents-1)*sizeof(double*));
compNewHost = (double**)malloc((simDomain.numComponents-1)*sizeof(double*));
phiHost     = (double**)malloc(simDomain.numPhases*sizeof(double*));
dfdphiHost  = (double**)malloc(simDomain.numPhases*sizeof(double*));
phiNewHost  = (double**)malloc(simDomain.numPhases*sizeof(double*));
phaseCompHost = (double**)malloc(sizeof(double*)*simDomain.numPhases*(simDomain.numComponents-1));

if (simControls.FUNCTION_F == 2)
    muHost = (double**)malloc((simDomain.numComponents-1)*sizeof(double*));

// Memory on the device is allocated using the host-side pointers
// The pointers are then copied to the device-side pointers so that they point to the same data
for (j = 0; j < simDomain.numComponents-1; j++)
{
    #pragma acc enter data create(compHost[j][:subdomain.numCompCells], compNewHost[j][:subdomain.numCompCells])
    #pragma acc update device(compDev[j:1], compNewDev[j:1])

    if (simControls.FUNCTION_F == 2)
    {
        #pragma acc enter data create(muHost[j][:subdomain.numCompCells])
        #pragma acc update device(muDev[j:1])
    }
}

for (j = 0; j < simDomain.numPhases; j++)
{
    #pragma acc enter data create(phiHost[j][:subdomain.numCompCells], dfdphiHost[j][:subdomain.numCompCells], phiNewHost[j][:subdomain.numCompCells])
    #pragma acc update device(phiDev[j:1], dfdphiDev[j:1], phiNewDev[j:1])
}

for (j = 0; j < simDomain.numPhases*(simDomain.numComponents-1); j++)
{
    #pragma acc enter data create(phaseCompHost[j][:subdomain.numCompCells])
    #pragma acc update device(phaseCompDev[j:1])
}

// Required for computing and printing statistics
#pragma acc enter data create(maxerr[:simDomain.numPhases+simDomain.numComponents-1], maxVal[:simDomain.numPhases+simDomain.numComponents-1], minVal[:simDomain.numPhases+simDomain.numComponents-1])

maxerr_h = (double*)malloc(sizeof(double)*(simDomain.numPhases+simDomain.numComponents-1));
maxVal_h = (double*)malloc(sizeof(double)*(simDomain.numPhases+simDomain.numComponents-1));
minVal_h = (double*)malloc(sizeof(double)*(simDomain.numPhases+simDomain.numComponents-1));

if (rank == MASTER)
    printf("Allocated memory on GPUs\n");

if (MPI_Enabled)
{
    MPI_Barrier(comm);
    #pragma acc wait
}

// Move fields from host to device
#pragma acc parallel loop present(phiHost[:simDomain.numPhases][:subdomain.numCompCells])
for (i = 0; i < simDomain.numPhases; i++)
{
    #pragma acc update device(phiHost[i][:subdomain.numCompCells])
}

#pragma acc parallel loop present(compHost[:simDomain.numComponents-1][:subdomain.numCompCells])
for (i = 0; i < simDomain.numComponents-1; i++)
{
    #pragma acc update device(compHost[i][:subdomain.numCompCells])
}

if (simControls.FUNCTION_F == 2)
{
    #pragma acc parallel loop present(muHost[:simDomain.numComponents-1][:subdomain.numCompCells])
    for (i = 0; i < simDomain.numComponents-1; i++)
    {
        #pragma acc update device(muHost[i][:subdomain.numCompCells])
    }
}

//Moving buffers to neighbours
if (MPI_Enabled)
{
    for (i = 0; i < simDomain.numPhases; i++)
    {
        MPI_Sendrecv(phiHost[i]+subdomain.shiftPointer, subdomain.shiftPointer, MPI_DOUBLE, subdomain.nbBack, 0,
                     phiHost[i]+subdomain.numCompCells-subdomain.shiftPointer, subdomain.shiftPointer, MPI_DOUBLE, subdomain.nbFront, 0,
                     comm, MPI_STATUS_IGNORE);

        MPI_Sendrecv(phiHost[i]+subdomain.numCompCells-2*subdomain.shiftPointer, subdomain.shiftPointer, MPI_DOUBLE, subdomain.nbFront, 1,
                     phiHost[i], subdomain.shiftPointer, MPI_DOUBLE, subdomain.nbBack, 1,
                     comm, MPI_STATUS_IGNORE);
    }

    for (i = 0; i < simDomain.numComponents-1; i++)
    {
        MPI_Sendrecv(compHost[i]+subdomain.shiftPointer, subdomain.shiftPointer, MPI_DOUBLE, subdomain.nbBack, 2,
                     compHost[i]+subdomain.numCompCells-subdomain.shiftPointer, subdomain.shiftPointer, MPI_DOUBLE, subdomain.nbFront, 2,
                     comm, MPI_STATUS_IGNORE);

        MPI_Sendrecv(compHost[i]+subdomain.numCompCells-2*subdomain.shiftPointer, subdomain.shiftPointer, MPI_DOUBLE, subdomain.nbFront, 3,
                     compHost[i], subdomain.shiftPointer, MPI_DOUBLE, subdomain.nbBack, 3,
                     comm, MPI_STATUS_IGNORE);
    }

    MPI_Barrier(comm);
    #pragma acc wait
}

applyBoundaryCondition(phiDev, 0, simDomain.numPhases,
                       simDomain, simControls,
                       simParams, subdomain,
                       gridSize, blockSize);

applyBoundaryCondition(compDev, 2, simDomain.numComponents-1,
                       simDomain, simControls,
                       simParams, subdomain,
                       gridSize, blockSize);

//3rd

// Copy old field to new field
for (i = 0; i < simDomain.numPhases; i++)
    #pragma acc parallel loop present(phiNewHost[i][:subdomain.numCompCells], phiHost[i][:subdomain.numCompCells])
    for (int k = 0; k < subdomain.numCompCells; k++)
        phiNewHost[i][k] = phiHost[i][k];

for (i = 0; i < simDomain.numComponents-1; i++)
    #pragma acc parallel loop present(compNewHost[i][:subdomain.numCompCells], compHost[i][:subdomain.numCompCells])
    for (int k = 0; k < subdomain.numCompCells; k++)
        compNewHost[i][k] = compHost[i][k];

if (MPI_Enabled)
{
    MPI_Barrier(comm);
    #pragma acc wait
}

// Smooth
for (j = 1; j <= simControls.nsmooth; j++)
{
    smooth(phiDev, phiNewDev,
           &simDomain, &simControls,
           &simParams, &subdomain,
           gridSize, blockSize);

    for (i = 0; i < simDomain.numPhases; i++)
        #pragma acc parallel loop present(phiHost[i][:subdomain.numCompCells], phiNewHost[i][:subdomain.numCompCells])
        for (int k = 0; k < subdomain.numCompCells; k++)
            phiHost[i][k] = phiNewHost[i][k];

    if (MPI_Enabled)
    {
        for (i = 0; i < simDomain.numPhases; i++)
        {
            MPI_Sendrecv(phiHost[i]+subdomain.shiftPointer, subdomain.shiftPointer, MPI_DOUBLE, subdomain.nbBack, 0,
                         phiHost[i]-subdomain.shiftPointer+subdomain.numCompCells, subdomain.shiftPointer, MPI_DOUBLE, subdomain.nbFront, 0,
                         comm, MPI_STATUS_IGNORE);

            MPI_Sendrecv(phiHost[i]+subdomain.numCompCells-2*subdomain.shiftPointer, subdomain.shiftPointer, MPI_DOUBLE, subdomain.nbFront, 1,
                         phiHost[i], subdomain.shiftPointer, MPI_DOUBLE, subdomain.nbBack, 1,
                         comm, MPI_STATUS_IGNORE);
        }
    }

    applyBoundaryCondition(phiDev, 0, simDomain.numPhases,
                           simDomain, simControls,
                           simParams, subdomain,
                           gridSize, blockSize);

    calcPhaseComp(phiDev, compDev,
                  phaseCompDev, muDev,
                  &simDomain, &simControls,
                  &simParams, &subdomain,
                  gridSize, blockSize);

    if (j == simControls.nsmooth)
    {
        for (i = 0; i < simDomain.numPhases; i++)
            #pragma acc parallel loop present(phiHost[i][:subdomain.numCompCells], phiNewHost[i][:subdomain.numCompCells])
            for (int k = 0; k < subdomain.numCompCells; k++)
                phiNewHost[i][k] = phiHost[i][k];

        if (rank == MASTER)
            printf("\nFinished smoothing\n");
    }
}

if (rank == MASTER)
    printf("\nStarting solution procedure\n");

// Solution timeloop
for (simControls.count = simControls.startTime; simControls.count <= simControls.startTime + simControls.numSteps; simControls.count++)
{
    if (simControls.count % simControls.trackProgress == 0)
    {
        if (rank == MASTER)
            printf("\nTime: %le\n", (double)(simControls.count)*simControls.DELTA_t);

        printStats(phiHost, compHost,
                   phiNewHost, compNewHost,
                   maxerr, maxVal, minVal,
                   simDomain, subdomain,
                   gridSize, blockSize);

        #pragma acc update host(maxerr[:simDomain.numPhases+simDomain.numComponents-1], maxVal[:simDomain.numPhases+simDomain.numComponents-1], minVal[:simDomain.numPhases+simDomain.numComponents-1])
        double ans1, ans2, ans3;

        for (i = 0; i < simDomain.numPhases; i++)
        {
            MPI_Reduce(maxerr_h+i+simDomain.numComponents-1, &ans1, 1, MPI_DOUBLE, MPI_MAX, MASTER, comm);
            MPI_Reduce(maxVal_h+i+simDomain.numComponents-1, &ans2, 1, MPI_DOUBLE, MPI_MAX, MASTER, comm);
            MPI_Reduce(minVal_h+i+simDomain.numComponents-1, &ans3, 1, MPI_DOUBLE, MPI_MIN, MASTER, comm);

            if (rank == MASTER)
                printf("%*s, Max = %le, Min = %le, Relative_Change = %le\n", 5, simDomain.phaseNames[i], ans2, ans3, ans1);

            if (fabs(ans1) > 2 && rank == MASTER)
            {
                MPI_Abort(comm, 1);
            }
        }

        for (i = 0; i < simDomain.numComponents-1; i++)
        {
            MPI_Reduce(maxerr_h+i, &ans1, 1, MPI_DOUBLE, MPI_MAX, MASTER, comm);
            MPI_Reduce(maxVal_h+i, &ans2, 1, MPI_DOUBLE, MPI_MAX, MASTER, comm);
            MPI_Reduce(minVal_h+i, &ans3, 1, MPI_DOUBLE, MPI_MIN, MASTER, comm);

            if (rank == MASTER)
                printf("%*s, Max = %le, Min = %le, Relative_Change = %le\n", 5, simDomain.componentNames[i], ans2, ans3, ans1);

            if (fabs(ans1) > 2 && rank == MASTER)
            {
                MPI_Abort(comm, 1);
            }
        }
    }

    // Copy new field to old field
    for (i = 0; i < simDomain.numPhases; i++)
        #pragma acc parallel loop present(phiHost[i][:subdomain.numCompCells], phiNewHost[i][:subdomain.numCompCells])
        for (int k = 0; k < subdomain.numCompCells; k++)
            phiHost[i][k] = phiNewHost[i][k];

    for (i = 0; i < simDomain.numComponents-1; i++)
        #pragma acc parallel loop present(compHost[i][:subdomain.numCompCells], compNewHost[i][:subdomain.numCompCells])
        for (int k = 0; k < subdomain.numCompCells; k++)
            compHost[i][k] = compNewHost[i][k];

    // Writing to file
    if (simControls.count % simControls.saveInterval == 0 || simControls.count == simControls.startTime + simControls.numSteps)
    {
        for (i = 0; i < simDomain.numPhases; i++)
            #pragma acc update host(phi[:simDomain.numPhases*subdomain.numCompCells], phiHost[i][:subdomain.numCompCells])

        for (i = 0; i < simDomain.numComponents-1; i++)
            #pragma acc update host(comp[:simDomain.numComponents*subdomain.numCompCells], compHost[i][:subdomain.numCompCells])

        if (simControls.FUNCTION_F == 2)
            for (i = 0; i < simDomain.numComponents-1; i++)
                #pragma acc update host(mu[:simDomain.numComponents*subdomain.numCompCells], muHost[i][:subdomain.numCompCells])

        if (simControls.writeFormat == 0)
            writeVTK_BINARY(phi, comp, mu, simDomain, subdomain, simControls, rank, comm, argv);
        else if (simControls.writeFormat == 1)
            writeVTK_ASCII(phi, comp, mu, simDomain, subdomain, simControls, rank, comm, argv);

        #if ENABLE_HDF5 == 1
        if (simControls.writeHDF5)
            writeHDF5(phi, comp, mu,
                      simDomain, subdomain,
                      simControls, rank, comm, argv);
            if (rank == MASTER && simControls.writeHDF5)
                printf("Wrote to file\n");
        #endif

        if (rank == MASTER && simControls.writeHDF5 == 0)
            printf("Wrote to file\n");

        if (simControls.count == simControls.startTime + simControls.numSteps)
            break;
    }

    // Moving buffers to neighbours
    if (MPI_Enabled)
    {
        for (i = 0; i < simDomain.numPhases; i++)
        {
            MPI_Sendrecv(phiHost[i]+subdomain.shiftPointer, subdomain.shiftPointer, MPI_DOUBLE, subdomain.nbBack, 0,
                         phiHost[i]-subdomain.shiftPointer+subdomain.numCompCells, subdomain.shiftPointer, MPI_DOUBLE, subdomain.nbFront, 0,
                         comm, MPI_STATUS_IGNORE);

            MPI_Sendrecv(phiHost[i]+subdomain.numCompCells-2*subdomain.shiftPointer, subdomain.shiftPointer, MPI_DOUBLE, subdomain.nbFront, 1,
                         phiHost[i], subdomain.shiftPointer, MPI_DOUBLE, subdomain.nbBack, 1,
                         comm, MPI_STATUS_IGNORE);
        }

        for (i = 0; i < simDomain.numComponents-1; i++)
        {

            MPI_Sendrecv(compHost[i]+subdomain.shiftPointer, subdomain.shiftPointer, MPI_DOUBLE, subdomain.nbBack, 2,
                         compHost[i]-subdomain.shiftPointer+subdomain.numCompCells, subdomain.shiftPointer, MPI_DOUBLE, subdomain.nbFront, 2,
                         comm, MPI_STATUS_IGNORE);

            MPI_Sendrecv(compHost[i]+subdomain.numCompCells-2*subdomain.shiftPointer, subdomain.shiftPointer, MPI_DOUBLE, subdomain.nbFront, 3,
                         compHost[i], subdomain.shiftPointer, MPI_DOUBLE, subdomain.nbBack, 3,
                         comm, MPI_STATUS_IGNORE);

        }

        if (simControls.FUNCTION_F == 2)
        {
            for (i = 0; i < simDomain.numComponents-1; i++)
            {
                MPI_Sendrecv(muHost[i]+subdomain.shiftPointer, subdomain.shiftPointer, MPI_DOUBLE, subdomain.nbBack, 2,
                             muHost[i]-subdomain.shiftPointer+subdomain.numCompCells, subdomain.shiftPointer, MPI_DOUBLE, subdomain.nbFront, 2,
                             comm, MPI_STATUS_IGNORE);

                MPI_Sendrecv(muHost[i]+subdomain.numCompCells-2*subdomain.shiftPointer, subdomain.shiftPointer, MPI_DOUBLE, subdomain.nbFront, 3,
                             muHost[i], subdomain.shiftPointer, MPI_DOUBLE, subdomain.nbBack, 3,
                             comm, MPI_STATUS_IGNORE);
            }
        }
    }

    applyBoundaryCondition(phiDev, 0, simDomain.numPhases,
                           simDomain, simControls,
                           simParams, subdomain,
                           gridSize, blockSize);

    applyBoundaryCondition(compDev, 2, simDomain.numComponents-1,
                           simDomain, simControls,
                           simParams, subdomain,
                           gridSize, blockSize);

    if (simControls.FUNCTION_F == 2)
    {
        applyBoundaryCondition(muDev, 1, simDomain.numComponents-1,
                               simDomain, simControls,
                               simParams, subdomain,
                               gridSize, blockSize);
    }

    // Solution procedure
    calcPhaseComp(phiDev, compDev,
                  phaseCompDev, muDev,
                  &simDomain, &simControls,
                  &simParams, &subdomain,
                  gridSize, blockSize);

    resetArray(dfdphiDev, simDomain.numPhases, simDomain.DIMENSION, subdomain.sizeX, subdomain.sizeY, subdomain.sizeZ, gridSize, blockSize);

    #if ENABLE_CUFFTMP == 1
    if (simControls.ELASTICITY)
    {
        if (FFT_Alternate == 1)
        {
            moveTocudaLibXtDesc(phiDev, phiElDev,
                                simDomain, simControls, simParams, subdomain,
                                stream, gridSize, blockSize, comm);

            for (i = 0; i < simDomain.numPhases; i++)
            {
                if (simControls.eigenSwitch[i] == 1)
                {
                    CUFFT_CHECK(cufftXtExecDescriptor(plan, phiElDev[i], phiElDev[i], CUFFT_FORWARD));
                }
            }

            computeDrivingForce_Elastic(phiElDev, dfEldphi, BpqHost,
                                        simDomain, simControls, simParams, subdomain,
                                        stream, comm);

            for (i = 0; i < simDomain.numPhases; i++)
            {
                if (simControls.eigenSwitch[i] == 1)
                {
                    CUFFT_CHECK(cufftXtExecDescriptor(plan, dfEldphi[i], dfEldphi[i], CUFFT_INVERSE));
                }
            }

            moveFromcudaLibXtDesc(dfdphiDev, dfEldphi,
                                  simDomain, simControls, simParams, subdomain,
                                  stream, gridSize, blockSize);
            FFT_Alternate = 0;
        }
        else if (FFT_Alternate == 0)
        {
            moveTocudaLibXtDesc(phiDev, dfEldphi,
                                simDomain, simControls, simParams, subdomain,
                                stream, gridSize, blockSize, comm);

            for (i = 0; i < simDomain.numPhases; i++)
            {
                if (simControls.eigenSwitch[i] == 1)
                {
                    CUFFT_CHECK(cufftXtExecDescriptor(plan, dfEldphi[i], dfEldphi[i], CUFFT_FORWARD));
                }
            }

            computeDrivingForce_Elastic(dfEldphi, phiElDev, BpqHost,
                                        simDomain, simControls, simParams, subdomain,
                                        stream, comm);

            for (i = 0; i < simDomain.numPhases; i++)
            {
                if (simControls.eigenSwitch[i] == 1)
                {
                    CUFFT_CHECK(cufftXtExecDescriptor(plan, phiElDev[i], phiElDev[i], CUFFT_INVERSE));
                }
            }

            moveFromcudaLibXtDesc(dfdphiDev, phiElDev,
                                  simDomain, simControls, simParams, subdomain,
                                  stream, gridSize, blockSize);

            FFT_Alternate = 1;
        }
    }
    #endif

    computeDrivingForce_Chemical(phiDev, compDev,
                                 dfdphiDev, phaseCompDev, muDev,
                                 &simDomain, &simControls,
                                 &simParams, &subdomain,
                                 gridSize, blockSize);

    updatePhi(phiDev, dfdphiDev, phiNewDev, phaseCompDev,
              &simDomain, &simControls,
              &simParams, &subdomain,
              gridSize, blockSize);

    if (MPI_Enabled)
    {
        for (i = 0; i < simDomain.numPhases; i++)
        {
            MPI_Sendrecv(phiNewHost[i]+subdomain.shiftPointer, subdomain.shiftPointer, MPI_DOUBLE, subdomain.nbBack, 0,
                         phiNewHost[i]-subdomain.shiftPointer+subdomain.numCompCells, subdomain.shiftPointer, MPI_DOUBLE, subdomain.nbFront, 0,
                         comm, MPI_STATUS_IGNORE);

            MPI_Sendrecv(phiNewHost[i]+subdomain.numCompCells-2*subdomain.shiftPointer, subdomain.shiftPointer, MPI_DOUBLE, subdomain.nbFront, 1,
                         phiNewHost[i], subdomain.shiftPointer, MPI_DOUBLE, subdomain.nbBack, 1,
                         comm, MPI_STATUS_IGNORE);
        }
    }

    applyBoundaryCondition(phiNewDev, 0, simDomain.numPhases,
                           simDomain, simControls,
                           simParams, subdomain,
                           gridSize, blockSize);

    updateComposition(phiDev, compDev, phiNewDev, compNewDev,
                      phaseCompDev, muDev,
                      simDomain, simControls,
                      simParams, subdomain,
                      gridSize, blockSize);

    if (MPI_Enabled)
    {
        for (i = 0; i < simDomain.numComponents-1; i++)
        {
            MPI_Sendrecv(compNewHost[i]+subdomain.shiftPointer, subdomain.shiftPointer, MPI_DOUBLE, subdomain.nbBack, 2,
                         compNewHost[i]-subdomain.shiftPointer+subdomain.numCompCells, subdomain.shiftPointer, MPI_DOUBLE, subdomain.nbFront, 2,
                         comm, MPI_STATUS_IGNORE);

            MPI_Sendrecv(compNewHost[i]+subdomain.numCompCells-2*subdomain.shiftPointer, subdomain.shiftPointer, MPI_DOUBLE, subdomain.nbFront, 3,
                         compNewHost[i], subdomain.shiftPointer, MPI_DOUBLE, subdomain.nbBack, 3,
                         comm, MPI_STATUS_IGNORE);
        }

        if (simControls.FUNCTION_F == 2)
        {
            for (i = 0; i < simDomain.numComponents-1; i++)
            {
                MPI_Sendrecv(muHost[i]+subdomain.shiftPointer, subdomain.shiftPointer, MPI_DOUBLE, subdomain.nbBack, 2,
                             muHost[i]-subdomain.shiftPointer+subdomain.numCompCells, subdomain.shiftPointer, MPI_DOUBLE, subdomain.nbFront, 2,
                             comm, MPI_STATUS_IGNORE);

                MPI_Sendrecv(muHost[i]+subdomain.numCompCells-2*subdomain.shiftPointer, subdomain.shiftPointer, MPI_DOUBLE, subdomain.nbFront, 3,
                             muHost[i], subdomain.shiftPointer, MPI_DOUBLE, subdomain.nbBack, 3,
                             comm, MPI_STATUS_IGNORE);
            }
        }
    }

    applyBoundaryCondition(compNewDev, 2, simDomain.numComponents-1,
                           simDomain, simControls,
                           simParams, subdomain,
                           gridSize, blockSize);

    if (simControls.FUNCTION_F == 2)
    {
        applyBoundaryCondition(muDev, 1, simDomain.numComponents-1,
                               simDomain, simControls,
                               simParams, subdomain,
                               gridSize, blockSize);
    }

} // End solution timeloop


//4th

free(minVal_h);
free(maxVal_h);
free(maxerr_h);

#pragma acc exit data delete(minVal[:simDomain.numPhases+simDomain.numComponents-1], maxVal[:simDomain.numPhases+simDomain.numComponents-1], maxerr[:simDomain.numPhases+simDomain.numComponents-1])

for (j = 0; j < simDomain.numComponents-1; j++)
{
    #pragma acc exit data delete(compHost[j][:subdomain.numCompCells], compNewHost[j][:subdomain.numCompCells])
    if (simControls.FUNCTION_F == 2)
        #pragma acc exit data delete(muHost[j][:subdomain.numCompCells])
}

#pragma acc exit data delete(compDev[:simDomain.numComponents-1], compNewDev[:simDomain.numComponents-1])

if (simControls.FUNCTION_F == 2)
    #pragma acc exit data delete(muDev[:simDomain.numComponents-1])

for (j = 0; j < simDomain.numPhases; j++)
{
    #pragma acc exit data delete(phiHost[j][:subdomain.numCompCells], phiNewHost[j][:subdomain.numCompCells], dfdphiHost[j][:subdomain.numCompCells])
}

#if ENABLE_CUFFTMP == 1
if (simControls.ELASTICITY)
{
    for (j = 0; j < simDomain.numPhases; j++)
    {
        // Equivalent OpenACC calls for cufftXtFree
    }
}
#endif

#pragma acc exit data delete(phiDev[:simDomain.numPhases], dfdphiDev[:simDomain.numPhases], phiNewDev[:simDomain.numPhases])

for (j = 0; j < simDomain.numPhases*(simDomain.numComponents-1); j++)
{
    #pragma acc exit data delete(phaseCompHost[j][:subdomain.numCompCells])
}

#pragma acc exit data delete(phaseCompDev[:simDomain.numPhases*(simDomain.numComponents-1)])

free(compHost);
free(compNewHost);

if (simControls.FUNCTION_F == 2)
{
    free(muHost);
    free(mu);
}

free(phiHost);
free(dfdphiHost);
free(phiNewHost);

free(phaseCompHost);

freeVars(&simDomain, &simControls, &simParams);

free(phi);
free(comp);

#if ENABLE_CUFFTMP == 1
if (simControls.ELASTICITY)
{
    for (i = 0; i < simDomain.numPhases*simDomain.numPhases; i++)
    {
        // Equivalent OpenACC calls for cudaFree
    }

    // Equivalent OpenACC calls for cufftDestroy and cudaStreamDestroy
}
#endif

MPI_Finalize();

return 0;
}

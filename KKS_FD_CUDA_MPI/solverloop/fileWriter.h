#ifndef FILEWRITER_H_
#define FILEWRITER_H_

#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <string.h>
#include <arpa/inet.h>
#include <endian.h>

#ifndef ENABLE_HDF5
#define ENABLE_HDF5 1
#endif

#include "structures.hpp"

#define  IS_BIG_ENDIAN     (1 == htons(1))
#define  IS_LITTLE_ENDIAN  (!IS_BIG_ENDIAN)

double swap_bytes(double value);

/*
 * Files from each MPI process can not be merged and read directly
 * Reconstruct vtk files into a single vtk file if needed (resource-intensive)
 * Writes only .vtk files in each DATA/Processor_%d directory
 * N phases and C-1 components are written
 */

#ifdef __cplusplus
extern "C"
#endif
void writeVTK_ASCII(double *phi, double *comp, double *mu,
                    domainInfo simDomain, subdomainInfo subdomain,
                    controls simControls, int rank, MPI_Comm comm,
                    char *argv[]);

#ifdef __cplusplus
extern "C"
#endif
void writeVTK_BINARY(double *phi, double *comp, double *mu,
                     domainInfo simDomain, subdomainInfo subdomain,
                     controls simControls, int rank, MPI_Comm comm,
                     char *argv[]);

#ifdef __cplusplus
extern "C"
#endif
int readVTK_ASCII(FILE *fp, double *phi, double *comp, double *mu,
                  domainInfo simDomain, subdomainInfo subdomain,
                  controls simControls);

#ifdef __cplusplus
extern "C"
#endif
int readVTK_BINARY(FILE *fp, double *phi, double *comp, double *mu,
                   domainInfo simDomain, subdomainInfo subdomain,
                   controls simControls);

#ifdef __cplusplus
extern "C"
#endif
void read_domain(double *phi, double *comp, double *mu,
                 domainInfo simDomain, subdomainInfo subdomain,
                 controls simControls, int rank, MPI_Comm comm,
                 char *argv[]);

/*
 * Parallel file format that can merge chunks from each MPI process
 * Writes .h5 files
 */
#ifdef __cplusplus
extern "C"
#endif
void writeHDF5(double *phi, double *comp, double *mu,
               domainInfo simDomain, subdomainInfo subdomain,
               controls simControls, int rank, MPI_Comm comm,
               char *argv[]);

#ifdef __cplusplus
extern "C"
#endif
void readHDF5(double *phi, double *comp, double *mu,
              domainInfo simDomain, subdomainInfo subdomain,
              controls simControls, int rank, MPI_Comm comm,
              char *argv[]);

void writeDoubleArrayToVTK(double *arr, char *dirName, char *fileName, long nx, long ny, long nz);

#endif

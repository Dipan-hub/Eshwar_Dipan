/******************************************************************************
 *                       Code generated with sympy 1.8                        *
 *                                                                            *
 *              See http://www.sympy.org/ for more information.               *
 *                                                                            *
 *                       This file is part of 'project'                       *
 ******************************************************************************/

#ifndef PROJECT__TDBS_THERMO__H
#define PROJECT__TDBS_THERMO__H

void GE_0(double T, double *y, double *Ge);
void GE_1(double T, double *y, double *Ge);
void Mu_0(double T, double *y, double *Mu);
void Mu_1(double T, double *y, double *Mu);
void dmudc_0(double T, double *y, double *Dmudc);
void dmudc_1(double T, double *y, double *Dmudc);

static void (*free_energy_tdb[])(double T, double *y, double *Ge) = {GE_0, GE_1}; 
static void (*Mu_tdb[])(double T, double *y, double *Mu) = {Mu_0, Mu_1}; 
static void (*dmudc_tdb[])(double T, double *y, double *Dmudc) = {dmudc_0, dmudc_1}; 

#endif

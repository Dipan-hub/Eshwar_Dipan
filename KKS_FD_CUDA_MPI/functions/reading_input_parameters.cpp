  #include "inputReader.h"
  #include <fstream>
  #include <iostream>
  #include <vector>
  #include <sstream>

  int readInput_MPI(domainInfo *simDomain, controls *simControls,
                    simParameters *simParams, int rank, char *argv[])
  {
      std::ifstream fr(argv[1]);
      std::ofstream fp;
      if (fr)
      {
          if (!(rank))
              std::cout << "\nReading input parameters from " << argv[1] << "\n";
      }
      else
      {
          if (!(rank))
              std::cout << "\nFile " << argv[1] << " not found\n";
          return -1;
      }

      std::string tempbuff;
      std::string outfile = std::string(argv[1]) + ".out";
      fp.open(outfile);

      // Setting defaults
      simParams->alpha = 2.94;
      simParams->ISOTHERMAL = 1;
      simControls->multiphase = 1;
      simControls->restart = 0;
      simControls->writeHDF5 = 0;
      simControls->FUNCTION_ANISOTROPY = 0;
      simControls->ANISOTROPY = 0;
      simControls->ANISOTROPY_GRADIENT = 0;
      simControls->ELASTICITY = 0;

      while (std::getline(fr, tempbuff))
      {
          std::istringstream iss(tempbuff);
          std::string tmpstr1, tmpstr2;
          if (iss >> tmpstr1 && tmpstr1[0] != '#')
          {
              if (iss >> std::ws && std::getline(iss, tmpstr2, ';'))
              {
                  tmpstr2 = tmpstr2.substr(tmpstr2.find('=') + 1);
                  if (tmpstr2[0] == ' ') tmpstr2.erase(0, 1);

                  if (tmpstr1 == "MESH_X")
                  {
                      simDomain->MESH_X = std::stol(tmpstr2);
                      fp << tmpstr1 << " = " << simDomain->MESH_X << "\n";
                  }
                  else if (tmpstr1 == "MESH_Y")
                  {
                      simDomain->MESH_Y = std::stol(tmpstr2);
                      fp << tmpstr1 << " = " << simDomain->MESH_Y << "\n";
                  }
                  else if (tmpstr1 == "MESH_Z")
                  {
                      simDomain->MESH_Z = std::stol(tmpstr2);
                      fp << tmpstr1 << " = " << simDomain->MESH_Z << "\n";
                  }
                  else if (tmpstr1 == "DELTA_X")
                  {
                      simDomain->DELTA_X = std::stod(tmpstr2);
                      fp << tmpstr1 << " = " << simDomain->DELTA_X << "\n";
                  }
                  else if (tmpstr1 == "DELTA_Y")
                  {
                      simDomain->DELTA_Y = std::stod(tmpstr2);
                      fp << tmpstr1 << " = " << simDomain->DELTA_Y << "\n";
                  }
                  else if (tmpstr1 == "DELTA_Z")
                  {
                      simDomain->DELTA_Z = std::stod(tmpstr2);
                      fp << tmpstr1 << " = " << simDomain->DELTA_Z << "\n";
                  }
                  else if (tmpstr1 == "DIMENSION")
                  {
                      simDomain->DIMENSION = std::stoi(tmpstr2);
                      fp << tmpstr1 << " = " << simDomain->DIMENSION << "\n";
                  }
                  else if (tmpstr1 == "NUMPHASES")
                  {
                      simDomain->numPhases = std::stoi(tmpstr2);
                      fp << tmpstr1 << " = " << simDomain->numPhases << "\n";
                  }
                  else if (tmpstr1 == "NUMCOMPONENTS")
                  {
                      simDomain->numComponents = std::stoi(tmpstr2);
                      fp << tmpstr1 << " = " << simDomain->numComponents << "\n";
                  }
                  else if (tmpstr1 == "PHASES")
                  {
                      simDomain->phaseNames.resize(simDomain->numPhases);
                      simDomain->phase_map.resize(simDomain->numPhases);

                      simDomain->thermo_phase_host.resize(simDomain->numPhases);
                      simDomain->thermo_phase_dev.resize(simDomain->numPhases);

                      for (long i = 0; i < simDomain->numPhases; i++)
                      {
                          simDomain->phaseNames[i].resize(30);
                          simDomain->phase_map[i].resize(30);
                      }

                      populate_string_array(simDomain->phaseNames, tmpstr2, simDomain->numPhases);

                      for (long i = 0; i < simDomain->numPhases; i++)
                      {
                          fp << tmpstr1 << " = " << simDomain->phaseNames[i] << "\n";
                      }
                  }
                  else if (tmpstr1 == "COMPONENTS")
                  {
                      simDomain->componentNames.resize(simDomain->numComponents);
                      for (long i = 0; i < simDomain->numComponents; i++)
                          simDomain->componentNames[i].resize(30);

                      populate_string_array(simDomain->componentNames, tmpstr2, simDomain->numComponents);
                      for (long i = 0; i < simDomain->numComponents; i++)
                      {
                          fp << tmpstr1 << " = " << simDomain->componentNames[i] << "\n";
                      }

                      if (simDomain->numComponents > 1 && simDomain->numPhases > 0)
  {
      simParams->F0_Beq_host = MallocM(simDomain->numPhases, simDomain->numComponents - 1);

      simParams->DELTA_T = MallocM(simDomain->numPhases, simDomain->numPhases);
      simParams->DELTA_C = MallocM(simDomain->numPhases, simDomain->numComponents - 1);

      simParams->dcbdT = Malloc3M(simDomain->numPhases, simDomain->numPhases, simDomain->numComponents - 1);
      simParams->dBbdT = MallocM(simDomain->numPhases, simDomain->numComponents - 1);

      simParams->slopes = Malloc3M(simDomain->numPhases, simDomain->numPhases, simDomain->numPhases);

      simParams->gamma_host = MallocM(simDomain->numPhases, simDomain->numPhases);
      simParams->gamma_dev.resize(simDomain->numPhases * simDomain->numPhases);

      simParams->Tau_host = MallocM(simDomain->numPhases, simDomain->numPhases);

      simParams->relax_coeff_host = MallocM(simDomain->numPhases, simDomain->numPhases);
      simParams->relax_coeff_dev.resize(simDomain->numPhases * simDomain->numPhases);

      simParams->kappaPhi_host = MallocM(simDomain->numPhases, simDomain->numPhases);
      simParams->kappaPhi_dev.resize(simDomain->numPhases * simDomain->numPhases);

      simParams->diffusivity_host = Malloc3M(simDomain->numPhases, simDomain->numComponents - 1, simDomain->numComponents - 1);
      simParams->diffusivity_dev.resize(simDomain->numPhases * (simDomain->numComponents - 1) * (simDomain->numComponents - 1));

      simParams->mobility_host = Malloc3M(simDomain->numPhases, simDomain->numComponents - 1, simDomain->numComponents - 1);
      simParams->mobility_dev.resize(simDomain->numPhases * (simDomain->numComponents - 1) * (simDomain->numComponents - 1));

      simParams->F0_A_host = Malloc3M(simDomain->numPhases, simDomain->numComponents - 1, simDomain->numComponents - 1);
      simParams->F0_A_dev.resize(simDomain->numPhases * (simDomain->numComponents - 1) * (simDomain->numComponents - 1));

      simParams->F0_B_host = MallocM(simDomain->numPhases, simDomain->numComponents - 1);
      simParams->F0_B_dev.resize(simDomain->numPhases * (simDomain->numComponents - 1));

      simParams->F0_C_host.resize(simDomain->numPhases);
      simParams->F0_C_dev.resize(simDomain->numPhases);

      simParams->ceq_host = Malloc3M(simDomain->numPhases, simDomain->numPhases, simDomain->numComponents - 1);
      simParams->ceq_dev.resize(simDomain->numPhases * simDomain->numPhases * (simDomain->numComponents - 1));

      simParams->cfill_host = Malloc3M(simDomain->numPhases, simDomain->numPhases, simDomain->numComponents - 1);
      simParams->cfill_dev.resize(simDomain->numPhases * simDomain->numPhases * (simDomain->numComponents - 1));

      simParams->cguess_host = Malloc3M(simDomain->numPhases, simDomain->numPhases, simDomain->numComponents - 1);
      simParams->cguess_dev.resize(simDomain->numPhases * simDomain->numPhases * (simDomain->numComponents - 1));

      simParams->theta_ijk_host = Malloc3M(simDomain->numPhases, simDomain->numPhases, simDomain->numPhases);
      for (long i = 0; i < simDomain->numPhases; i++)
          for (long j = 0; j < simDomain->numPhases; j++)
              for (long k = 0; k < simDomain->numPhases; k++)
                  simParams->theta_ijk_host[i][j][k] = 0.0;
      simParams->theta_ijk_dev.resize(simDomain->numPhases * simDomain->numPhases * simDomain->numPhases);

      simParams->theta_ij_host = MallocM(simDomain->numPhases, simDomain->numPhases);
      simParams->theta_ij_dev.resize(simDomain->numPhases * simDomain->numPhases);

      simParams->theta_i_host.resize(simDomain->numPhases);
      simParams->theta_i_dev.resize(simDomain->numPhases);

      simParams->Rotation_matrix_host = Malloc4M(simDomain->numPhases, simDomain->numPhases, 3, 3);
      simParams->Rotation_matrix_dev.resize(simDomain->numPhases * simDomain->numPhases * 3 * 3);

      simParams->Inv_Rotation_matrix_host = Malloc4M(simDomain->numPhases, simDomain->numPhases, 3, 3);
      simParams->Inv_Rotation_matrix_dev.resize(simDomain->numPhases * simDomain->numPhases * 3 * 3);

      simParams->dab_host = MallocM(simDomain->numPhases, simDomain->numPhases);
      simParams->dab_dev.resize(simDomain->numPhases * simDomain->numPhases);

      simControls->eigenSwitch.resize(simDomain->numPhases);
      simParams->eigen_strain.resize(simDomain->numPhases);
      simParams->Stiffness_c.resize(simDomain->numPhases);

      for (long i = 0; i < 6; i++)
      {
          simControls->boundary[i].resize(4); // 4 = Number of scalar fields
      }
  }
  else
  {
      std::cout << "Invalid number of components and/or phases\n";
  }
                  }

  else if (tmpstr1 == "DELTA_t")
  {
      simControls->DELTA_t = std::stod(tmpstr2);
      fp << tmpstr1 << " = " << simControls->DELTA_t << "\n";
  }
  else if (tmpstr1 == "RESTART")
  {
      simControls->restart = std::stol(tmpstr2);
      fp << tmpstr1 << " = " << simControls->restart << "\n";
  }
  else if (tmpstr1 == "STARTTIME")
  {
      simControls->startTime = std::stol(tmpstr2);
      fp << tmpstr1 << " = " << simControls->startTime << "\n";
      simControls->count = simControls->startTime;
  }
  else if (tmpstr1 == "NTIMESTEPS")
  {
      simControls->numSteps = std::stol(tmpstr2);
      fp << tmpstr1 << " = " << simControls->numSteps << "\n";
  }
  else if (tmpstr1 == "NSMOOTH")
  {
      simControls->nsmooth = std::stol(tmpstr2);
      fp << tmpstr1 << " = " << simControls->nsmooth << "\n";
  }
  else if (tmpstr1 == "SAVET")
  {
      simControls->saveInterval = std::stol(tmpstr2);
      fp << tmpstr1 << " = " << simControls->saveInterval << "\n";
  }
  else if (tmpstr1 == "TRACK_PROGRESS")
  {
      simControls->trackProgress = std::stol(tmpstr2);
      fp << tmpstr1 << " = " << simControls->trackProgress << "\n";
  }
  else if (tmpstr1 == "WRITEFORMAT")
  {
      if (tmpstr2 == "ASCII")
          simControls->writeFormat = 1;
      else if (tmpstr2 == "BINARY")
          simControls->writeFormat = 0;
      else
          simControls->writeFormat = 2;
      fp << tmpstr1 << " = " << simControls->writeFormat << "\n";
  }
  else if (tmpstr1 == "WRITEHDF5")
  {
      simControls->writeHDF5 = std::stoi(tmpstr2);
      fp << tmpstr1 << " = " << simControls->writeHDF5 << "\n";
  }
  else if (tmpstr1 == "GAMMA")
  {
      if (simDomain->numPhases > 0)
          populate_matrix(simParams->gamma_host, tmpstr2, simDomain->numPhases);

      for (long i = 0; i < simDomain->numPhases; i++)
      {
          for (long j = 0; j < simDomain->numPhases; j++)
          {
              fp << tmpstr1 << "[" << i << "][" << j << "] = " << simParams->gamma_host[i][j] << "\n";
          }
      }
  }
  else if (tmpstr1 == "DIFFUSIVITY")
  {
      populate_diffusivity_matrix(simParams->diffusivity_host, tmpstr2, simDomain->numComponents);
      for (long i = 0; i < simDomain->numPhases; i++)
      {
          for (long j = 0; j < simDomain->numComponents - 1; j++)
          {
              for (long k = 0; k < simDomain->numComponents - 1; k++)
              {
                  fp << tmpstr1 << "[" << i << "][" << j << "][" << k << "] = " << simParams->diffusivity_host[i][j][k] << "\n";
              }
          }
      }
  }
  else if (tmpstr1 == "Tau")
  {
      if (simDomain->numPhases > 0)
          populate_matrix(simParams->Tau_host, tmpstr2, simDomain->numPhases);

      for (long i = 0; i < simDomain->numPhases; i++)
      {
          for (long j = 0; j < simDomain->numPhases; j++)
          {
              fp << tmpstr1 << "[" << i << "][" << j << "] = " << simParams->Tau_host[i][j] << "\n";
          }
      }
  }
  else if (tmpstr1 == "alpha")
  {
      simParams->alpha = std::stod(tmpstr2);
      fp << tmpstr1 << " = " << simParams->alpha << "\n";
  }
  else if (tmpstr1 == "epsilon")
  {
      simParams->epsilon = std::stod(tmpstr2);
      fp << tmpstr1 << " = " << simParams->epsilon << "\n";
  }
  else if (tmpstr1 == "ceq")
  {
      populate_thermodynamic_matrix(simParams->ceq_host, tmpstr2, simDomain->numComponents);
      for (long i = 0; i < simDomain->numPhases; i++)
      {
          for (long j = 0; j < simDomain->numPhases; j++)
          {
              for (long k = 0; k < simDomain->numComponents - 1; k++)
              {
                  fp << tmpstr1 << "[" << i << "][" << j << "][" << k << "] = " << simParams->ceq_host[i][j][k] << "\n";
              }
          }
      }
  }
  else if (tmpstr1 == "cfill")
  {
      populate_thermodynamic_matrix(simParams->cfill_host, tmpstr2, simDomain->numComponents);
      for (long i = 0; i < simDomain->numPhases; i++)
      {
          for (long j = 0; j < simDomain->numPhases; j++)
          {
              for (long k = 0; k < simDomain->numComponents - 1; k++)
              {
                  fp << tmpstr1 << "[" << i << "][" << j << "][" << k << "] = " << simParams->cfill_host[i][j][k] << "\n";
              }
          }
      }
  }
  else if (tmpstr1 == "c_guess")
  {
      populate_thermodynamic_matrix(simParams->cguess_host, tmpstr2, simDomain->numComponents);
      for (long i = 0; i < simDomain->numPhases; i++)
      {
          for (long j = 0; j < simDomain->numPhases; j++)
          {
              for (long k = 0; k < simDomain->numComponents - 1; k++)
              {
                  fp << tmpstr1 << "[" << i << "][" << j << "][" << k << "] = " << simParams->cguess_host[i][j][k] << "\n";
              }
          }
      }
  }
  else if (tmpstr1 == "R")
  {
      simParams->R = std::stod(tmpstr2);
      fp << tmpstr1 << " = " << simParams->R << "\n";
  }
  else if (tmpstr1 == "V")
  {
      simParams->molarVolume = std::stod(tmpstr2);
      fp << tmpstr1 << " = " << simParams->molarVolume << "\n";
  }
  else if (tmpstr1 == "T")
  {
      simParams->T = std::stod(tmpstr2);
      fp << tmpstr1 << " = " << simParams->T << "\n";
  }
  else if (tmpstr1 == "Equilibrium_temperature")
  {
      simParams->Teq = std::stod(tmpstr2);
      fp << tmpstr1 << " = " << simParams->Teq << "\n";
  }
  else if (tmpstr1 == "Filling_temperature")
  {
      simParams->Tfill = std::stod(tmpstr2);
      fp << tmpstr1 << " = " << simParams->Tfill << "\n";
  }
  else if (tmpstr1 == "Function_F" && simDomain->numPhases > 1 && simDomain->numComponents - 1 > 0)
  {
      simControls->FUNCTION_F = std::stoi(tmpstr2);
      fp << tmpstr1 << " = " << simControls->FUNCTION_F << "\n";
  }
  else if (tmpstr1 == "A")
  {
      populate_A_matrix(simParams->F0_A_host, tmpstr2, simDomain->numComponents);
  }
  else if (tmpstr1 == "tdbfname" && simDomain->numPhases > 1);
  else if (tmpstr1 == "num_thermo_phases")
  {
      simDomain->numThermoPhases = std::stoi(tmpstr2);
      simDomain->phases_tdb.resize(simDomain->numThermoPhases);

      for (long i = 0; i < simDomain->numThermoPhases; i++)
          simDomain->phases_tdb[i].resize(51);
  }
  else if (tmpstr1 == "tdb_phases" && simDomain->numThermoPhases > 0)
  {
      populate_string_array(simDomain->phases_tdb, tmpstr2, simDomain->numThermoPhases);
  }
  else if (tmpstr1 == "phase_map" && simDomain->numPhases > 1)
  {
      populate_string_array(simDomain->phase_map, tmpstr2, simDomain->numPhases);
  }
  else if (tmpstr1 == "theta_i" && simDomain->numPhases > 1)
  {
      populate_thetai_matrix(simParams->theta_i_host, tmpstr2, simDomain->numPhases);
  }
  else if (tmpstr1 == "theta_ij" && simDomain->numPhases > 1)
  {
      populate_thetaij_matrix(simParams->theta_ij_host, tmpstr2, simDomain->numPhases);
  }
  else if (tmpstr1 == "Gamma_abc" && simDomain->numPhases > 1)
  {
      populate_matrix3M(simParams->theta_ijk_host, tmpstr2, simDomain->numPhases);
  }
  else if (tmpstr1 == "ISOTHERMAL")
  {
      simControls->ISOTHERMAL = std::stoi(tmpstr2);
  }
  else if (tmpstr1 == "dTdt" && simDomain->numPhases > 1)
  {
      simControls->dTdt = std::stod(tmpstr2);
      fp << tmpstr1 << " = " << simControls->dTdt << "\n";
  }
  else if (tmpstr1 == "T_update" && simDomain->numPhases > 1)
  {
      simControls->T_update = std::stoi(tmpstr2);
      fp << tmpstr1 << " = " << simControls->T_update << "\n";
  }
  else if (tmpstr1 == "SEED")
  {
      simParams->SEED = std::stol(tmpstr2);
      fp << tmpstr1 << " = " << simParams->SEED << "\n";
  }
  else if (tmpstr1 == "Function_anisotropy")
  {
      simControls->FUNCTION_ANISOTROPY = std::stoi(tmpstr2);
      fp << tmpstr1 << " = " << simControls->FUNCTION_ANISOTROPY << "\n";
  }
  else if (tmpstr1 == "Anisotropy_type" && simControls->FUNCTION_ANISOTROPY != 0)
  {
      simControls->FOLD = std::stoi(tmpstr2);
      fp << tmpstr1 << " = " << simControls->FOLD << "\n";
  }
  else if (tmpstr1 == "Rotation_matrix" && simControls->FUNCTION_ANISOTROPY != 0)
  {
      populate_rotation_matrix(simParams->Rotation_matrix_host, simParams->Inv_Rotation_matrix_host, tmpstr2);

      for (long i = 0; i < simDomain->numPhases; i++)
      {
          for (long j = 0; j < simDomain->numPhases; j++)
          {
              for (int ii = 0; ii < 3; ii++)
              {
                  for (int jj = 0; jj < 3; jj++)
                  {
                      fp << tmpstr1 << "[" << i << "][" << j << "][" << ii << "][" << jj << "] = " << simParams->Rotation_matrix_host[i][j][ii][jj] << "\n";
                  }
              }
          }
      }

      for (long i = 0; i < simDomain->numPhases; i++)
      {
          for (long j = 0; j < simDomain->numPhases; j++)
          {
              for (int ii = 0; ii < 3; ii++)
              {
                  for (int jj = 0; jj < 3; jj++)
                  {
                      fp << "Inv_" << tmpstr1 << "[" << i << "][" << j << "][" << ii << "][" << jj << "] = " << simParams->Inv_Rotation_matrix_host[i][j][ii][jj] << "\n";
                  }
              }
          }
      }
  }
  else if (tmpstr1 == "dab")
  {
      populate_matrix(simParams->dab_host, tmpstr2, simDomain->numPhases);
      for (long i = 0; i < simDomain->numPhases; i++)
      {
          for (long j = 0; j < simDomain->numPhases; j++)
          {
              fp << tmpstr1 << "[" << i << "][" << j << "] = " << simParams->dab_host[i][j] << "\n";
          }
      }
  }
  else if (tmpstr1 == "ELASTICITY")
  {
      simControls->ELASTICITY = std::stoi(tmpstr2);
      fp << "ELASTICITY = " << simControls->ELASTICITY << "\n";
  }
  else if (tmpstr1 == "EIGEN_STRAIN" && simDomain->numPhases > 0)
  {
      populate_symmetric_tensor(simParams->eigen_strain, tmpstr2, simDomain->numPhases);
  }
  else if (tmpstr1 == "VOIGT_ISOTROPIC" && simDomain->numPhases > 0)
  {
      populate_cubic_stiffness(simParams->Stiffness_c, tmpstr2);
  }

                  }

              }
      }


      for (long i = 0; i < simDomain->numPhases; i++)
  {
      fp << "eps[" << i << "].xx = " << simParams->eigen_strain[i].xx << "\n";
      fp << "eps[" << i << "].yy = " << simParams->eigen_strain[i].yy << "\n";
      fp << "eps[" << i << "].zz = " << simParams->eigen_strain[i].zz << "\n";
      fp << "eps[" << i << "].yz = " << simParams->eigen_strain[i].yz << "\n";
      fp << "eps[" << i << "].xz = " << simParams->eigen_strain[i].xz << "\n";
      fp << "eps[" << i << "].xy = " << simParams->eigen_strain[i].xy << "\n";
  }

  for (long i = 0; i < simDomain->numPhases; i++)
  {
      fp << "Stiffness[" << i << "].C11 = " << simParams->Stiffness_c[i].C11 << "\n";
      fp << "Stiffness[" << i << "].C12 = " << simParams->Stiffness_c[i].C12 << "\n";
      fp << "Stiffness[" << i << "].C44 = " << simParams->Stiffness_c[i].C44 << "\n";
  }

  if (simControls->writeHDF5 == 1)
      simControls->writeFormat = 2;

  if (simControls->restart != 0)
      simControls->nsmooth = 0;

  if (simDomain->DIMENSION == 2)
      simDomain->MESH_Z = 1;

  fr.close();
  fp.close();

  if (simDomain->numPhases > MAX_NUM_PHASES || simDomain->numComponents > MAX_NUM_COMP)
      return 1;
  else
      return 0;
  }


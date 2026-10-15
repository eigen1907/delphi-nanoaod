#ifndef SKELANA_PSCEMF_HPC_HAC_STIC_HPP
#define SKELANA_PSCEMF_HPC_HAC_STIC_HPP

#include "skelana/mtrack.hpp"

namespace skelana
{
  /* +KEEP,PSCEMF
   *                            ( PA extra-module EMNC (22) )
   */
  inline const int LENEMF = 8;
  extern "C" struct
  {
    int nemf;
    int kemf[MTRACK][LENEMF];
  } pscemf_;

  /* +KEEP,PSCHPC
   *                            ( PA extra-module EMNC (22) )
   */
  inline const int LENHPC = 8;
  extern "C" struct
  {
    int nhpc;
    int khpc[MTRACK][LENHPC];
  } pschpc_;

  /* +KEEP,PSCHAC
   *                            ( PA extra-module HCNC (23) )
   */
  inline const int LENHAC = 8;
  extern "C" struct
  {
    int nhac;
    int khac[MTRACK][LENHAC];
  } pschac_;

  /* +KEEP,PSCSTC
   *                            ( PA extra-module SSTC (33) )
   */
  inline const int LENSTC = 9;
  extern "C" struct
  {
    int nstic;
    int kstic[MTRACK][LENSTC];
  } pscstc_;

  inline int &NEMF = pscemf_.nemf;
  inline int &NHPC = pschpc_.nhpc;
  inline int &NHAC = pschac_.nhac;
  inline int &NSTIC = pscstc_.nstic;

  inline int &KEMF(int i, int j) { return pscemf_.kemf[j - 1][i - 1]; }
  inline float &QEMF(int i, int j) { return *reinterpret_cast<float *>(&pscemf_.kemf[j - 1][i - 1]); }

  inline int &KHPC(int i, int j) { return pschpc_.khpc[j - 1][i - 1]; }
  inline float &QHPC(int i, int j) { return *reinterpret_cast<float *>(&pschpc_.khpc[j - 1][i - 1]); }

  inline int &KHAC(int i, int j) { return pschac_.khac[j - 1][i - 1]; }
  inline float &QHAC(int i, int j) { return *reinterpret_cast<float *>(&pschac_.khac[j - 1][i - 1]); }

  inline int &KSTIC(int i, int j) { return pscstc_.kstic[j - 1][i - 1]; }
  inline float &QSTIC(int i, int j) { return *reinterpret_cast<float *>(&pscstc_.kstic[j - 1][i - 1]); }

}   // namespace skelana

#endif // SKELANA_PSCEMF_HPC_HAC_STIC_HPP

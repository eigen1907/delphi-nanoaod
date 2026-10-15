#ifndef SKELETON_PSCGRC_HPP
#define SKELETON_PSCGRC_HPP

#include "skelana/mtrack.hpp"

namespace skelana
{
/*+KEEP,PSCGRC.                     RICH Gas information
*                            ( PA extra-module HAID (26) )
*
*     NGRIC                - Number of tracks with RICH gas info
*     QGRIC(LENGRC,MTRACK) - Real    array of RICH gas information
*     KGRIC(LENGRC,MTRACK) - Integer array of RICH gas information
*
*     QGRIC(1,I) - THEG , Cerenkov angle in gas
*     QGRIC(2,I) - SIGG , Cerenkov angle resolution
*     KGRIC(3,I) - NPHG , Number of photoelectrons in ring
*     QGRIC(4,I) - NEPG , Expected number of photoel. in gas
*     KGRIC(5,I) - FLAGG, RICH sector flag in gas
*
*/
    extern "C" struct
    {
        int ngric;
        int kgric[MTRACK][5];
        float theg[MTRACK];
        float sigg[MTRACK];
        int nphg[MTRACK];
        int nepg[MTRACK];
        float flagg[MTRACK];
    } pscgrc_;

    inline int &NGRIC = pscgrc_.ngric;
    inline int &KGRIC(int i, int j) { return pscgrc_.kgric[j - 1][i - 1]; }
    inline float &QGRIC(int i, int j) { return *reinterpret_cast<float*>(&pscgrc_.kgric[j - 1][i - 1]); }
    inline float &THEG(int i) { return pscgrc_.theg[i - 1]; }
    inline float &SIGG(int i) { return pscgrc_.sigg[i - 1]; }
    inline int &NPHG(int i) { return pscgrc_.nphg[i - 1]; }
    inline int &NEPG(int i) { return pscgrc_.nepg[i - 1]; }
    inline float &FLAGG(int i) { return pscgrc_.flagg[i - 1]; }
} // namespace skelana

#endif // SKELETON_PSCGRC_HPP

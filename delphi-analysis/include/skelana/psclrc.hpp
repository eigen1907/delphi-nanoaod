#ifndef SKELANA_PSCLRC_HPP
#define SKELANA_PSCLRC_HPP

#include "skelana/mtrack.hpp"

namespace skelana
{
/*+KEEP,PSCLRC.                Barrel RICH liquid information
*                            ( PA extra-module HAID (26) )
*
*     NLRIC                - Number of tracks with RICH liq
*     LENLRC               - Length of RICH liq information
*     QLRIC(LENLRC,MTRACK) - Real    array of RICH liq information
*     KLRIC(LENLRC,MTRACK) - Integer array of RICH liq information
*
*     QLRIC(1,I) - THEL , Cerenkov angle in liquid
*     QLRIC(2,I) - SIGL , Cerenkov angle resolution
*     KLRIC(3,I) - NPHL , Number of photoelectrons in ring
*     QLRIC(4,I) - NEPL , Expected number of photoel. in liquid
*     KLRIC(5,I) - FLAGL, RICH sector flag in liquid
*
*
*/
    inline const int LENLRC = 5;
    extern "C" struct
    {
        int nlric;
        int klric[MTRACK][LENLRC];
        float thel[MTRACK];
        float sigl[MTRACK];
        int nphl[MTRACK];
        int nepl[MTRACK];
        float flagl[MTRACK];
    } psclrc_;

    inline int &NLRIC = psclrc_.nlric;
    inline int &KLRIC(int i, int j) { return psclrc_.klric[j - 1][i - 1]; }
    inline float &QLRIC(int i, int j) { return *reinterpret_cast<float*>(&psclrc_.klric[j - 1][i - 1]); }
    inline float &THEL(int i) { return psclrc_.thel[i - 1]; }
    inline float &SIGL(int i) { return psclrc_.sigl[i - 1]; }
    inline int &NPHL(int i) { return psclrc_.nphl[i - 1]; }
    inline int &NEPL(int i) { return psclrc_.nepl[i - 1]; }
    inline float &FLAGL(int i) { return psclrc_.flagl[i - 1]; }
} // namespace skelana

#endif // SKELANA_PSCLRC_HPP

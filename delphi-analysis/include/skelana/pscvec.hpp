#ifndef SKELANA_PSCVEC_HPP
#define SKELANA_PSCVEC_HPP

#include "skelana/mtrack.hpp"

namespace skelana
{
    /* +KEEP,PSCVEC.         VECP and other information for VECSUB
    *
    *      NVECP               - Number of particles in VECP
    *     NCVECP               - Number of charged particles
    *     NNVECP               - Number of neutral particles
    *     NVECMC               - Number of simulated tracks
    *     NJET                 - Number of jets
    *       VECP(10, 3*MTRACK) - Real    vector of particle info
    *      IVECP(10, 3*MTRACK) - Integer vector of particle info
    *      LVECP    (3*MTRACK) - LPA/LST links    of the tracks
    *     INVECP    (3*MTRACK) - LPA/LST indecies of the tracks
    *     LVLOCK    (3*MTRACK) - Bitted word per track
    *
    *      VECP( 1,I) - P(x)
    *      VECP( 2,I) - P(y)
    *      VECP( 3,I) - P(z)
    *      VECP( 4,I) - E
    *      VECP( 5,I) - Mass
    *      VECP( 6,I) - |P|
    *      VECP( 7,I) - Charge
    *      VECP( 8,I) - Mass code
    *     IVECP( 9,I) - Mass identification
    *     IVECP(10,I) - Jet number
    */
        
        inline const int MJET = 30;
    
        inline const int LVPART = 1;
        inline const int LVJET = 260;
        inline const int LVTHRU = 250;
        inline const int LVSPHE = 255;
    
        extern "C" struct
        {
            float vecp[3 * MTRACK][10];
        } pucppp_;

        // First array in VECSUB's PUCLLL common: PA/Sim bank links.
        extern "C" struct
        {
            int lvecp[3 * MTRACK];
        } puclll_;

        extern "C" struct
        {
            int nvecp;
            int ncvecp;
            int nnvecp;
            int nvecmc;
            int njet;
            int lvlock[3 * MTRACK];
            int invecp[3 * MTRACK];
        } pscvec_;
    
        inline float &VECP(int i, int j) { return pucppp_.vecp[j - 1][i - 1]; }
        inline int &IVECP(int i, int j) { return *reinterpret_cast<int *>(&pucppp_.vecp[j - 1][i - 1]); }

        inline int &NVECP = pscvec_.nvecp;
        inline int &NCVECP = pscvec_.ncvecp;
        inline int &NNVECP = pscvec_.nnvecp;
        inline int &NVECMC = pscvec_.nvecmc;
        inline int &NJET = pscvec_.njet;
        inline int &LVLOCK(int i) { return pscvec_.lvlock[i - 1]; }
        inline int &LVECP(int i) { return puclll_.lvecp[i - 1]; }
        inline int &INVEC(int i) { return pscvec_.invecp[i - 1]; }
    
} // namespace skelana

#endif // SKELANA_PSCVEC_HPP

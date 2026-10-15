#ifndef SKELANA_PSCLUJ_HPP
#define SKELANA_PSCLUJ_HPP

namespace skelana
{
/* +KEEP,PSCLUJ.                LUJETS (LUND) like event record
*
*     NP      - number of particles in current event
*
*     KP(I,1) - status code KS of the current particle
*     KP(I,2) - particle code KF (Particle Data Group)
*     KP(I,3) - index of parent particle, or jet, or 0
*     KP(I,4) - index of the first daughter, if none 0
*     KP(I,5) - index of the last  daughter, if none 0
*
*     PP(I,1) - Px, momentum in the x direction, in GeV/c
*     PP(I,2) - Py, momentum in the y direction, in GeV/c
*     PP(I,3) - Pz, momentum in the z direction, in GeV/c
*     PP(I,4) - E , energy, in GeV
*     PP(I,5) - m , mass, in GeV/c**2
*
*     VP(I,1) - x position of production vertex (DST unpackers use cm)
*     VP(I,2) - y position of production vertex (DST unpackers use cm)
*     VP(I,3) - z position of production vertex (DST unpackers use cm)
*     VP(I,4) - time of production
*     VP(I,5) - proper lifetime of the particle, or 0.
*
*/
    inline const int NPMAX = 4000;
    extern "C" struct
    {
        int np;
        int kp[5][NPMAX];
        float pp[5][NPMAX];
        float vp[5][NPMAX];
    } pscluj_;

    inline int &NP = pscluj_.np;
    inline int &KP(int i, int j) { return pscluj_.kp[j - 1][i - 1]; }
    inline float &PP(int i, int j) { return pscluj_.pp[j - 1][i - 1]; }
    inline float &VP(int i, int j) { return pscluj_.vp[j - 1][i - 1]; }   
    
} // namespace skelana

#endif // SKELANA_PSCLUJ_HPP

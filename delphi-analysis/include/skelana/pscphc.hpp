#ifndef SKELANA_PSCPHC_HPP
#define SKELANA_PSCPHC_HPP

namespace skelana
{
    /* +KEEP,PSCPHC.                 Photon conversion information
     *                                       ( PXPC bank (24) )
     *
     *     NPHOC                  - Number of converted photons
     *     QPHOC(LENPHC, NPCMAX)  - Real    array of conv. info
     *     KPHOC(LENPHC, NPCMAX)  - Integer array of conv. info
     *
     * Fields (SKELANA v1.05 A.1.8):
     *     KPHOC( 1,I) - index of first decay product
     *     KPHOC( 2,I) - index of second decay product
     *     KPHOC( 3,I) - legacy photon lookup; current PSHPHC searches Reco LVECP,
     *                    so this is not a reliable Sim index
     *     KPHOC( 4,I) - PXPHOT code
     *                    (photon converted in front of TPC, mass code 21)
     *                    21 conversion from 2 TPC tracks, type I pair
     *                    22 conversion from 2 TPC tracks, type II pair
     *                    23 conversion from 2 TPC tracks, type III pair
     *                    24 conversion from 1 TPC track,  type I pair
     *                    Legacy banks may use negative codes.
     *     QPHOC( 5,I) - Px of the photon
     *     QPHOC( 6,I) - Py of the photon
     *     QPHOC( 7,I) - Pz of the photon
     *     QPHOC( 8,I) - Photon energy
     *     QPHOC( 9,I) - Photon track length (beam spot -> conversion)
     *     QPHOC(10,I) - X position of the conversion
     *     QPHOC(11,I) - Y position of the conversion
     *     QPHOC(12,I) - Z position of the conversion
     */

    inline const int NPCMAX = 50;
    inline const int LENPHC = 12;

    extern "C" struct
    {
        int nphoc;
        int kphoc[NPCMAX][LENPHC];
    } pscphc_;

    inline int   &NPHOC = pscphc_.nphoc;
    inline int   &KPHOC(int i, int j) { return pscphc_.kphoc[j - 1][i - 1]; }
    inline float &QPHOC(int i, int j) { return *reinterpret_cast<float *>(&pscphc_.kphoc[j - 1][i - 1]); }

} // namespace skelana

#endif // SKELANA_PSCPHC_HPP

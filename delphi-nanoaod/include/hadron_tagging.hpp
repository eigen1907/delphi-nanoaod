#ifndef DELPHI_HADRON_TAGGING_HPP
#define DELPHI_HADRON_TAGGING_HPP

#include <cstdint>
#include <cmath>
#include <vector>

#include <Math/Vector4D.h> // ROOT::Math::XYZTVectorF

// FCC-style gen B hadron tagging: thrust hemispheres (w.r.t. ThrustWithMissP axis),
// per-Part flags via gen ancestry (Part_simIdx -> SimPart_genIdx -> walk parents).

namespace delphi_hadron_tagging {

constexpr int kAbsBc = 541;
constexpr int kAbsBs = 531;
constexpr int kAbsBu = 521;
constexpr int kAbsBd = 511;
constexpr int kAbsLb = 5122;

inline bool hasHeavyAncestor(int genIdx, int absPdgTarget, int nGen,
                             const std::vector<int16_t> &parentIdx,
                             const std::vector<int16_t> &pdgId)
{
    int g = genIdx;
    for (int step = 0; step < nGen + 2 && g >= 0 && g < nGen; ++step)
    {
        if (std::abs(static_cast<int>(pdgId[static_cast<size_t>(g)])) == absPdgTarget)
            return true;
        int p = static_cast<int>(parentIdx[static_cast<size_t>(g)]);
        if (p < 0 || p >= nGen)
            break;
        g = p;
    }
    return false;
}

inline double cosThetaThrust(const ROOT::Math::XYZTVectorF &p4, double thrustTheta, double thrustPhi)
{
    const double px = p4.X();
    const double py = p4.Y();
    const double pz = p4.Z();
    const double pm = std::sqrt(px * px + py * py + pz * pz);
    if (pm <= 0.0)
        return 0.0;
    const double st = std::sin(thrustTheta);
    const double tx = st * std::cos(thrustPhi);
    const double ty = st * std::sin(thrustPhi);
    const double tz = std::cos(thrustTheta);
    return (px * tx + py * ty + pz * tz) / pm;
}

void fillEventHemisphereCounts(
    int nGen,
    const std::vector<int16_t> &genPdg,
    const std::vector<ROOT::Math::XYZTVectorF> &genP4,
    double thrustTheta,
    double thrustPhi,
    int &n_Bc_Emin,
    int &n_Bs_Emin,
    int &n_Bu_Emin,
    int &n_Bd_Emin,
    int &n_Lb_Emin,
    int &n_Bc_Emax,
    int &n_Bs_Emax,
    int &n_Bu_Emax,
    int &n_Bd_Emax,
    int &n_Lb_Emax);

void fillExclusiveLabels(
    int n_Bc_Emin, int n_Bs_Emin, int n_Bu_Emin, int n_Bd_Emin, int n_Lb_Emin,
    int &label_Bc_Emin, int &label_Bs_Emin, int &label_Bu_Emin, int &label_Bd_Emin, int &label_Lb_Emin,
    int &label_light_Emin, int &label_hasBc_Emin, int &label_has1Bc_Emin,
    int n_Bc_Emax, int n_Bs_Emax, int n_Bu_Emax, int n_Bd_Emax, int n_Lb_Emax,
    int &label_Bc_Emax, int &label_Bs_Emax, int &label_Bu_Emax, int &label_Bd_Emax, int &label_Lb_Emax,
    int &label_light_Emax, int &label_hasBc_Emax, int &label_has1Bc_Emax);

void fillPartFlavourFlags(
    int nPart,
    int nSim,
    int nGen,
    const std::vector<int16_t> &partSimIdx,
    const std::vector<int16_t> &simGenIdx,
    const std::vector<int16_t> &genParentIdx,
    const std::vector<int16_t> &genPdgId,
    std::vector<int8_t> &partFromBc,
    std::vector<int8_t> &partFromBs,
    std::vector<int8_t> &partFromBu,
    std::vector<int8_t> &partFromBd,
    std::vector<int8_t> &partFromLb);

} // namespace delphi_hadron_tagging

#endif

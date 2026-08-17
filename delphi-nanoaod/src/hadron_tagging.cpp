#include "hadron_tagging.hpp"

#include <cmath>

namespace delphi_hadron_tagging {

static void bumpHemisphere(double cth, int &nEmin, int &nEmax)
{
    if (std::abs(cth) < 1e-15)
        return;
    if (cth > 0.0)
        ++nEmin;
    else if (cth < 0.0)
        ++nEmax;
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
    int &n_Lb_Emax)
{
    n_Bc_Emin = n_Bs_Emin = n_Bu_Emin = n_Bd_Emin = n_Lb_Emin = 0;
    n_Bc_Emax = n_Bs_Emax = n_Bu_Emax = n_Bd_Emax = n_Lb_Emax = 0;

    if (nGen <= 0)
        return;

    const size_t n = static_cast<size_t>(nGen);
    for (size_t g = 0; g < n; ++g)
    {
        const int ap = std::abs(static_cast<int>(genPdg[g]));
        if (ap != kAbsBc && ap != kAbsBs && ap != kAbsBu && ap != kAbsBd && ap != kAbsLb)
            continue;
        const double cth = cosThetaThrust(genP4[g], thrustTheta, thrustPhi);
        if (ap == kAbsBc)
            bumpHemisphere(cth, n_Bc_Emin, n_Bc_Emax);
        else if (ap == kAbsBs)
            bumpHemisphere(cth, n_Bs_Emin, n_Bs_Emax);
        else if (ap == kAbsBu)
            bumpHemisphere(cth, n_Bu_Emin, n_Bu_Emax);
        else if (ap == kAbsBd)
            bumpHemisphere(cth, n_Bd_Emin, n_Bd_Emax);
        else if (ap == kAbsLb)
            bumpHemisphere(cth, n_Lb_Emin, n_Lb_Emax);
    }
}

void fillExclusiveLabels(
    int n_Bc_Emin, int n_Bs_Emin, int n_Bu_Emin, int n_Bd_Emin, int n_Lb_Emin,
    int &label_Bc_Emin, int &label_Bs_Emin, int &label_Bu_Emin, int &label_Bd_Emin, int &label_Lb_Emin,
    int &label_light_Emin, int &label_hasBc_Emin, int &label_has1Bc_Emin,
    int n_Bc_Emax, int n_Bs_Emax, int n_Bu_Emax, int n_Bd_Emax, int n_Lb_Emax,
    int &label_Bc_Emax, int &label_Bs_Emax, int &label_Bu_Emax, int &label_Bd_Emax, int &label_Lb_Emax,
    int &label_light_Emax, int &label_hasBc_Emax, int &label_has1Bc_Emax)
{
    label_Bc_Emin = (n_Bc_Emin == 1 && n_Bs_Emin == 0 && n_Bu_Emin == 0 && n_Bd_Emin == 0 && n_Lb_Emin == 0) ? 1 : 0;
    label_Bs_Emin = (n_Bc_Emin == 0 && n_Bs_Emin == 1 && n_Bu_Emin == 0 && n_Bd_Emin == 0 && n_Lb_Emin == 0) ? 1 : 0;
    label_Bu_Emin = (n_Bc_Emin == 0 && n_Bs_Emin == 0 && n_Bu_Emin == 1 && n_Bd_Emin == 0 && n_Lb_Emin == 0) ? 1 : 0;
    label_Bd_Emin = (n_Bc_Emin == 0 && n_Bs_Emin == 0 && n_Bu_Emin == 0 && n_Bd_Emin == 1 && n_Lb_Emin == 0) ? 1 : 0;
    label_Lb_Emin = (n_Bc_Emin == 0 && n_Bs_Emin == 0 && n_Bu_Emin == 0 && n_Bd_Emin == 0 && n_Lb_Emin == 1) ? 1 : 0;
    label_light_Emin = (n_Bc_Emin == 0 && n_Bs_Emin == 0 && (n_Bu_Emin > 0 || n_Bd_Emin > 0) && n_Lb_Emin == 0) ? 1 : 0;
    label_hasBc_Emin = (n_Bc_Emin > 0) ? 1 : 0;
    label_has1Bc_Emin = (n_Bc_Emin == 1) ? 1 : 0;

    label_Bc_Emax = (n_Bc_Emax == 1 && n_Bs_Emax == 0 && n_Bu_Emax == 0 && n_Bd_Emax == 0 && n_Lb_Emax == 0) ? 1 : 0;
    label_Bs_Emax = (n_Bc_Emax == 0 && n_Bs_Emax == 1 && n_Bu_Emax == 0 && n_Bd_Emax == 0 && n_Lb_Emax == 0) ? 1 : 0;
    label_Bu_Emax = (n_Bc_Emax == 0 && n_Bs_Emax == 0 && n_Bu_Emax == 1 && n_Bd_Emax == 0 && n_Lb_Emax == 0) ? 1 : 0;
    label_Bd_Emax = (n_Bc_Emax == 0 && n_Bs_Emax == 0 && n_Bu_Emax == 0 && n_Bd_Emax == 1 && n_Lb_Emax == 0) ? 1 : 0;
    label_Lb_Emax = (n_Bc_Emax == 0 && n_Bs_Emax == 0 && n_Bu_Emax == 0 && n_Bd_Emax == 0 && n_Lb_Emax == 1) ? 1 : 0;
    label_light_Emax = (n_Bc_Emax == 0 && n_Bs_Emax == 0 && (n_Bu_Emax > 0 || n_Bd_Emax > 0) && n_Lb_Emax == 0) ? 1 : 0;
    label_hasBc_Emax = (n_Bc_Emax > 0) ? 1 : 0;
    label_has1Bc_Emax = (n_Bc_Emax == 1) ? 1 : 0;
}

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
    std::vector<int8_t> &partFromLb)
{
    partFromBc.assign(static_cast<size_t>(nPart), 0);
    partFromBs.assign(static_cast<size_t>(nPart), 0);
    partFromBu.assign(static_cast<size_t>(nPart), 0);
    partFromBd.assign(static_cast<size_t>(nPart), 0);
    partFromLb.assign(static_cast<size_t>(nPart), 0);

    if (nPart <= 0 || nGen <= 0)
        return;

    for (int ip = 0; ip < nPart; ++ip)
    {
        const size_t i = static_cast<size_t>(ip);
        const int simIdx = static_cast<int>(partSimIdx[i]);
        if (simIdx < 0 || simIdx >= nSim)
            continue;
        const int genIdx = static_cast<int>(simGenIdx[static_cast<size_t>(simIdx)]);
        if (genIdx < 0 || genIdx >= nGen)
            continue;

        if (hasHeavyAncestor(genIdx, kAbsBc, nGen, genParentIdx, genPdgId))
            partFromBc[i] = 1;
        if (hasHeavyAncestor(genIdx, kAbsBs, nGen, genParentIdx, genPdgId))
            partFromBs[i] = 1;
        if (hasHeavyAncestor(genIdx, kAbsBu, nGen, genParentIdx, genPdgId))
            partFromBu[i] = 1;
        if (hasHeavyAncestor(genIdx, kAbsBd, nGen, genParentIdx, genPdgId))
            partFromBd[i] = 1;
        if (hasHeavyAncestor(genIdx, kAbsLb, nGen, genParentIdx, genPdgId))
            partFromLb[i] = 1;
    }
}

} // namespace delphi_hadron_tagging

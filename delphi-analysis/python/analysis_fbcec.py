#!/usr/bin/env python3
"""
FBCEC + FBCC Analysis
-------------------------------
One-point correlators for charged particles, binned in angle between
track momentum and beam axis.

Observables (filled with weight q_i * E_i for FBCEC, q_i for FBCC; no
1/sin(theta) Jacobian applied at fill time):

  FBCEC_forward          — 180 bins in theta [0,180] deg
                           → dN(qE)/dtheta after /bin_width
  FBCEC_forward_cos       — 140 bins in cos(theta) [-0.9397,+0.9397]
                           (|cosθ|<cos20°; out-of-range tracks go to
                           under/overflow)
                           → dN(qE)/dcos(theta) after /bin_width
  FBCC_forward{_cos}      — analogous, weight = q
  <obs>_forward{_cos}_pos / _neg — per-charge components (FBCEC weight = E,
                           FBCC weight = 1). Combined = pos − neg. These feed
                           the downstream per-charge efficiency correction and
                           the route-A covariance blocks (cov_*_{pp,mm,pm}).

Forward only — the backward (180−angle) folding has been dropped.
The theta -> cos(theta) change of variable carries the 1/sin(theta)
Jacobian implicitly via the differing bin widths, so the cos-binned
variants give dN/dcos(theta) directly without any per-track factor.

Track selection (nominal): |d0| < 0.6 cm, |z0*sin(theta)| < 1 cm,
baseline pT > 0.4 GeV.
Momentum cut sets: baseline pT > 0.4; pT > 1, 2, 3, 4, 5, 6, 8 GeV;
and |p| > 2, 3, 4, 5, 6, 8 GeV (each on top of the pT > 0.4 baseline).

Cut variants (additive, per base systematic, not crossed with each other):
  `_dpp0p5`    — extra dpp < 0.5 tracking-quality cut
  `_d0z0loose` — loose IP acceptance |d0| < 4 cm, |z0*sin(theta)| < 10 cm
Both get the full nominal + systematic + covariance treatment.

For MC:
  - Fills gen-level histograms (no selection, no systematics)
  - Fills reco-level histograms (after selection AND event selection, all systematics)

For Data:
  - Fills reco-level histograms (after selection AND event selection, all systematics)

Covariance matrix and 2D histograms are computed by default for data, skipped for MC.
Use --compute-covariance to force on (e.g. for MC), --no-covariance to force off.

Usage:
  MC:   python analysis_fbcec.py input.root output.root --is-mc
  Data: python analysis_fbcec.py input.root output.root
  MC with covariance: python analysis_fbcec.py input.root output.root --is-mc --compute-covariance
"""

import ROOT
import numpy as np
import argparse
import math
import sys
from array import array

from common_functions import (
    thrust_axis_fast,
    thrust_theta,
    missing_p,
    apply_fake_drop_charged,
    randomly_drop_particles,
    P4Block,
    match_angular,
)
from binning_and_selections import (
    apply_track_selection_delphi,
    apply_event_selection_delphi,
)
from vertex_reweight import (
    VertexReweighter3D,
    detect_period as rw_detect_period,
    find_3d_table,
)

# ============================================================================
# BINNING
# ============================================================================

# Theta: 0-180 degrees, 1 degree bins
theta_bins = np.linspace(0, 180, 181)

# cos(theta): 140 uniform bins inside |cosθ| < cos(20°) = 0.9397, for
# dN/dcos(theta) variants. Tracks with |cosθ| ≥ 0.9397 (i.e. θ outside
# [20°, 160°]) land in under/overflow rather than a fiducial bin. Decoupled
# from theta_bins on purpose — the cov accumulator and finalize_errors
# handle the per-key matrix size.
cos_theta_bins = np.linspace(-0.9397, 0.9397, 141)

# Impact-parameter binning (post track-selection cuts so distributions stay
# inside |d0|<0.6 cm and |z0·sinθ|<1 cm; ranges have a small margin). Width
# ≪ data resolution (~280 μm) so peak shapes resolve cleanly.
#   d0_vtx    : 1000 bins / 2 cm  → 20 μm/bin
#   z0_vtx    : 1000 bins / 6 cm  → 60 μm/bin (z0 can grow as z0sinθ/sin θ)
#   z0sin_vtx : 1000 bins / 2.4 cm → 24 μm/bin
D0_VTX_NBINS,    D0_VTX_RANGE    = 1000, (-1.0, 1.0)
Z0_VTX_NBINS,    Z0_VTX_RANGE    = 1000, (-3.0, 3.0)
Z0SIN_VTX_NBINS, Z0SIN_VTX_RANGE = 1000, (-1.2, 1.2)

# Single track energy bins
energy_bins = np.array([
    0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
    0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0,
    1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5,
    1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 1.85, 1.9, 1.95, 2.0,
    2.5, 3.0, 3.5, 4.0, 4.5, 5.0,
    6, 7, 8, 9, 10, 12, 14, 16, 18, 20,
    25, 30, 35, 40, 45, 50, 100,
], dtype=np.float64)

# Momentum cut labels, thresholds, and types ("pt" = transverse, "p" = total)
# Baseline is pT > 0.4 (transverse momentum), enforced in apply_track_selection_delphi
# via charged_pt_min; the "" entry below means "no extra cut beyond the pT>0.4
# baseline". The _pmag* entries are additional |p| cuts on top of that baseline.
P_CUTS = [
    ("", 0.4, "pt"),          # baseline pT > 0.4
    ("_p1", 1.0, "pt"),       # pT > 1
    ("_p2", 2.0, "pt"),       # pT > 2
    ("_p3", 3.0, "pt"),       # pT > 3
    ("_p4", 4.0, "pt"),       # pT > 4
    ("_p5", 5.0, "pt"),       # pT > 5
    ("_p6", 6.0, "pt"),       # pT > 6
    ("_p8", 8.0, "pt"),       # pT > 8
    ("_pmag2", 2.0, "p"),     # |p| > 2 (with baseline pT > 0.4)
    ("_pmag3", 3.0, "p"),     # |p| > 3 (with baseline pT > 0.4)
    ("_pmag4", 4.0, "p"),     # |p| > 4 (with baseline pT > 0.4)
    ("_pmag5", 5.0, "p"),     # |p| > 5 (with baseline pT > 0.4)
    ("_pmag6", 6.0, "p"),     # |p| > 6 (with baseline pT > 0.4)
    ("_pmag8", 8.0, "p"),     # |p| > 8 (with baseline pT > 0.4)
]

# Angular matching cutoff for gen-reco matching (radians)
MATCH_DR = 0.05

# Tighter dpp (delta-p/p) cuts for tracking-quality study. Filled in
# parallel with the existing distributions (no dpp cut ≡ dpp < 1) for
# nominal systematic only, on FBCEC/FBCC observables only.
DPP_CUTS = [
    (0.25, "_dpp0p25"),
    (0.5,  "_dpp0p5"),
    (0.75, "_dpp0p75"),
]

# Species classification of gen-matched tracks for the per-species
# matched/chargeOK/chargeMisID histograms (MC nominal only).
# 0=other, 1=pi, 2=K, 3=p.
SPECIES_NAMES = ('other', 'pi', 'K', 'p')

# PDG codes (TPCNtuple tgenBefore.pid convention).
_PDG_TO_SPECIES = {
    211: 1, -211: 1,
    321: 2, -321: 2,
    2212: 3, -2212: 3,
}
# DELPHI internal codes (NanoAOD SimPart_pdgId convention, verified
# against GenPart cross-reference: 41=pi+, 42=K+, 65=p; sign carries
# the charge).
_DELPHI_TO_SPECIES = {
    41: 1, -41: 1,
    42: 2, -42: 2,
    65: 3, -65: 3,
}


def species_codes(pdg_array, convention='pdg'):
    """Classify each gen particle by species code.

    convention='pdg'    → expect PDG IDs (±211, ±321, ±2212)
    convention='delphi' → expect DELPHI internal IDs (±41, ±42, ±65)

    Returns int8 array same length as input; 0 for "other".
    """
    table = _PDG_TO_SPECIES if convention == 'pdg' else _DELPHI_TO_SPECIES
    out = np.zeros(len(pdg_array), dtype=np.int8)
    for i, p in enumerate(pdg_array):
        out[i] = table.get(int(p), 0)
    return out


# ============================================================================
# SYSTEMATIC CONFIGURATION (charged-only)
# ============================================================================

SYSTEMATICS = {
    'nominal': {
        'type': 'selection',
        'charged_pt_min': 0.4,
    },

    # Charged pT-scale systematics (±1%): scales transverse momentum (px,py),
    # pz left unchanged — see apply_systematic_variation.
    'charged_scale_up': {
        'type': 'momentum',
        'momentum_scale_charged': 1.01,
    },
    'charged_scale_down': {
        'type': 'momentum',
        'momentum_scale_charged': 0.99,
    },

    # Efficiency systematics
    'fake_drop': {
        'type': 'efficiency',
        'apply_fake_drop': True,
    },
    'charged_eff': {
        'type': 'efficiency',
        'drop_charged_fraction': 0.02,
    },

    # MC d0/z0sin resolution smearing (additive Gaussian per (pT, θ) bin).
    # Two independent variants — d0 only and z0sin only — produce parallel
    # histogram sets. No-op when smearer is None (data, or sigma file missing).
    'smear_d0': {
        'type': 'resolution_smear',
        'smear_kind': 'd0',
    },
    'smear_z0sin': {
        'type': 'resolution_smear',
        'smear_kind': 'z0sin',
    },
}


# Auxiliary cut VARIANT (not a systematic uncertainty): tighter dpp cut on
# charged tracks. For every systematic in SYSTEMATICS we additionally
# produce a `_dpp0p5` partner — same kinematic variation plus the extra
# dpp < 0.5 cut on charged tracks. So every histogram family (FBCEC/FBCC,
# Track*, IP, per-species matching, NChargedSele, ...) gets parallel
# `*_<syst>` and `*_<syst>_dpp0p5` variants.
DPP_VARIANT_TAG = '_dpp0p5'        # suffix appended to syst name
DPP_VARIANT_MAX = 0.5              # threshold

# Auxiliary cut VARIANT (not a systematic uncertainty): looser impact-parameter
# acceptance. For every BASE systematic we additionally produce a `_d0z0loose`
# partner using |d0| < 4 cm and |z0·sinθ| < 10 cm in place of the nominal
# 0.6 / 1.0 cm track-quality cut. Like the dpp variant this is a cut variant,
# not an uncertainty, and it is ADDITIVE — it is NOT crossed with `_dpp0p5`
# (no `_dpp0p5_d0z0loose` combination is produced).
D0Z0_VARIANT_TAG = '_d0z0loose'    # suffix appended to syst name
D0Z0_VARIANT_CFG = {'charged_d0_max': 4.0, 'charged_z0sin_max': 10.0}


def runs_with_dpp_variant(systematics):
    """Return a dict pairing each entry in `systematics` with a `_dpp0p5`
    partner that has the same config plus `charged_dpp_max = 0.5`.

    Used to drive both the booking loop and the event loop so that every
    syst variant gets a dpp-cut companion. The dpp0p5 entries live outside
    SYSTEMATICS (they are a cut variant, not an uncertainty)."""
    out = dict(systematics)
    for syst_name, cfg in list(systematics.items()):
        new_cfg = dict(cfg)
        new_cfg['charged_dpp_max'] = DPP_VARIANT_MAX
        out[f"{syst_name}{DPP_VARIANT_TAG}"] = new_cfg
    return out


def runs_with_d0z0_variant(systematics):
    """Return a dict pairing each entry in `systematics` with a `_d0z0loose`
    partner that has the same config plus the loose IP acceptance
    (|d0| < 4 cm, |z0·sinθ| < 10 cm). Mirrors runs_with_dpp_variant; the
    partners live outside SYSTEMATICS (cut variant, not an uncertainty)."""
    out = dict(systematics)
    for syst_name, cfg in list(systematics.items()):
        new_cfg = dict(cfg)
        new_cfg.update(D0Z0_VARIANT_CFG)
        out[f"{syst_name}{D0Z0_VARIANT_TAG}"] = new_cfg
    return out


def build_systematics_to_run(systematics):
    """Full systematic expansion shared by both entry points (analysis_fbcec
    main() and analysis_fbcec_nanoaod). Each base systematic yields:
      - itself                 (tight IP 0.6/1.0, no dpp cut)
      - a `_dpp0p5` partner     (dpp < 0.5 tracking-quality cut)
      - a `_d0z0loose` partner  (loose IP acceptance |d0|<4, |z0sinθ|<10)
    The dpp and d0z0loose variants are ADDITIVE: they are NOT crossed with
    each other. For the 9 base systematics this gives 9+9+9 = 27 entries."""
    out = runs_with_dpp_variant(systematics)            # base + `_dpp0p5`
    for syst_name, cfg in runs_with_d0z0_variant(systematics).items():
        out.setdefault(syst_name, cfg)                 # adds the `_d0z0loose` set
    return out


def cov_target_systs():
    """Syst names whose covariance is accumulated and finalized into per-bin
    statistical errors + cov TH2Ds: nominal and its dpp / d0z0loose partners.
    Shared by main() and analysis_fbcec_nanoaod so the finalize/rewrite step
    stays in sync."""
    return ('nominal', f'nominal{DPP_VARIANT_TAG}', f'nominal{D0Z0_VARIANT_TAG}')


def _strip_dpp_variant(syst_name):
    """Return the base syst name for `*_dpp0p5` partners; else identity."""
    if syst_name.endswith(DPP_VARIANT_TAG):
        return syst_name[: -len(DPP_VARIANT_TAG)]
    return syst_name


def _is_diagnostic_eligible(syst_name):
    """True if the syst should get the IP-diagnostic + per-species hists.
    Covers nominal + smear systs + their `_dpp0p5` partners."""
    return _strip_dpp_variant(syst_name) in ('nominal', 'smear_d0', 'smear_z0sin')


def _diag_suffix(syst_name):
    """Suffix used by IP-diagnostic hist names (`d0_vtx_*`, `z0_vtx_*`,
    `z0sin_vtx_*`, TProfile RMS variants).

    Nominal hists are bare (`d0_vtx_all`); nominal's dpp0p5 partner takes
    just `_dpp0p5`; smear variants and their dpp partners use the full
    `_<syst>` form. So the four flavours produced by an MC run are:

        d0_vtx_all                          (nominal)
        d0_vtx_all_dpp0p5                   (nominal + dpp<0.5)
        d0_vtx_all_smear_d0                 (smear_d0)
        d0_vtx_all_smear_d0_dpp0p5          (smear_d0 + dpp<0.5)
    """
    base = _strip_dpp_variant(syst_name)
    dpp_part = DPP_VARIANT_TAG if syst_name != base else ''
    base_part = '' if base == 'nominal' else f"_{base}"
    return f"{base_part}{dpp_part}"


# ============================================================================
# HISTOGRAM CREATION
# ============================================================================

def create_counter_histogram():
    """
    Bin 1: Total events processed (E ~ 91.25 GeV)
    Bin 2: Pass event selection (baseline pT>0.4, d0<0.6, z0sin<1)
    Bin 3: Pass event selection + has tracks with pT>1
    Bin 4: Pass event selection + has tracks with pT>2
    Bin 5: Pass event selection + has tracks with pT>4
    Bin 6: Pass event selection + has tracks with |p|>2
    Bin 7: Pass event selection + has tracks with |p|>4
    """
    counter = ROOT.TH1D('counter', 'Event Counter', 7, 0, 7)
    counter.GetXaxis().SetBinLabel(1, 'Total Events')
    counter.GetXaxis().SetBinLabel(2, 'Pass EvtSel')
    counter.GetXaxis().SetBinLabel(3, 'Pass EvtSel + pT>1')
    counter.GetXaxis().SetBinLabel(4, 'Pass EvtSel + pT>2')
    counter.GetXaxis().SetBinLabel(5, 'Pass EvtSel + pT>4')
    counter.GetXaxis().SetBinLabel(6, 'Pass EvtSel + |p|>2')
    counter.GetXaxis().SetBinLabel(7, 'Pass EvtSel + |p|>4')
    return counter


def create_gen_histograms(do_2d=False):
    """Gen-level histograms (no event selection): no-cut, pT>0.4 baseline,
    pT>{1,2,3,4,5,6,8}, and |p|>{2,3,4,5,6,8} sets."""
    hists = {}

    for suffix in ("", "_cut", "_cut1", "_cut2", "_cut3", "_cut4", "_cut5",
                   "_cut6", "_cut8",
                   "_pmag2", "_pmag3", "_pmag4", "_pmag5", "_pmag6", "_pmag8"):
        for obs in ("FBCEC", "FBCC"):
            # forward only; combined (= pos − neg) + per-charge components
            for comp in ("", "_pos", "_neg"):
                hists[f"{obs}_forward{comp}{suffix}"] = ROOT.TH1D(
                    f"{obs}_forward{comp}{suffix}", "",
                    len(theta_bins) - 1, array('d', theta_bins))
                # cos(theta) variant → dN/dcos(theta) after /bin_width
                hists[f"{obs}_forward_cos{comp}{suffix}"] = ROOT.TH1D(
                    f"{obs}_forward_cos{comp}{suffix}", "",
                    len(cos_theta_bins) - 1, array('d', cos_theta_bins))

            if do_2d:
                hists[f"{obs}2d_forward{suffix}"] = ROOT.TH2D(f"{obs}2d_forward{suffix}", "",
                    len(theta_bins) - 1, array('d', theta_bins),
                    len(energy_bins) - 1, array('d', energy_bins))
                # 2D cos(theta) variant
                hists[f"{obs}2d_forward_cos{suffix}"] = ROOT.TH2D(
                    f"{obs}2d_forward_cos{suffix}", "",
                    len(cos_theta_bins) - 1, array('d', cos_theta_bins),
                    len(energy_bins) - 1, array('d', energy_bins))

        hists[f"TrackE{suffix}"] = ROOT.TH1D(f"TrackE{suffix}", "",
            len(energy_bins) - 1, array('d', energy_bins))

        for sign in ("pos", "neg", "all"):
            hists[f"TrackP_{sign}{suffix}"] = ROOT.TH1D(f"TrackP_{sign}{suffix}", "", 1000, 0, 100)
            hists[f"TrackPt_{sign}{suffix}"] = ROOT.TH1D(f"TrackPt_{sign}{suffix}", "", 1000, 0, 100)
            hists[f"TrackTheta_{sign}{suffix}"] = ROOT.TH1D(f"TrackTheta_{sign}{suffix}", "",
                len(theta_bins) - 1, array('d', theta_bins))
            hists[f"TrackPhi_{sign}{suffix}"] = ROOT.TH1D(f"TrackPhi_{sign}{suffix}", "", 360, -180, 180)

        # Per-species (pi/K/p) Track* histograms — filled from
        # SimPart_pdgId classification via species_codes(). MC only.
        for sp in ("pi", "K", "p"):
            hists[f"TrackP_{sp}{suffix}"]  = ROOT.TH1D(f"TrackP_{sp}{suffix}",  "", 1000, 0, 100)
            hists[f"TrackPt_{sp}{suffix}"] = ROOT.TH1D(f"TrackPt_{sp}{suffix}", "", 1000, 0, 100)
            hists[f"TrackTheta_{sp}{suffix}"] = ROOT.TH1D(f"TrackTheta_{sp}{suffix}", "",
                len(theta_bins) - 1, array('d', theta_bins))
            hists[f"TrackPhi_{sp}{suffix}"] = ROOT.TH1D(f"TrackPhi_{sp}{suffix}", "", 360, -180, 180)

        # Per-charge × per-species TrackTheta only (pos/neg × pi/K/p) — used
        # to mirror the fake-by-species decomposition from correction_fbcc.
        for sign in ("pos", "neg"):
            for sp in ("pi", "K", "p"):
                hists[f"TrackTheta_{sign}_{sp}{suffix}"] = ROOT.TH1D(
                    f"TrackTheta_{sign}_{sp}{suffix}", "",
                    len(theta_bins) - 1, array('d', theta_bins))

        hists[f"NCharged{suffix}"] = ROOT.TH1D(f"NCharged{suffix}", "", 100, 0, 100)

    # --- Primary Z->qqbar flavor split (bb / cc / light) of the baseline
    # FBCEC/FBCC (combined + cos). Event-level truth tag from the GenPart JETSET
    # record; the SAME per-event tag is applied to the reco observable. Filled
    # via fill_observables(..., fbcec_only=True), so only the 4 combined histos
    # exist per flavor. SetDirectory(0): gen and reco share these names (written
    # to separate gen/ and reco/ dirs), so detach to avoid a gDirectory clash.
    for flav in ('bb', 'cc', 'light'):
        for obs in ('FBCEC', 'FBCC'):
            for nm, edges in ((f'{obs}_forward_{flav}', theta_bins),
                              (f'{obs}_forward_cos_{flav}', cos_theta_bins)):
                h = ROOT.TH1D(nm, "", len(edges) - 1, array('d', edges))
                h.SetDirectory(0)
                hists[nm] = h

    # --- Gen FBCEC/FBCC gated by the RECO event selection, at the pT>2
    # nominal-downstream cut (inclusive + bb/cc/light flavor). Filled by
    # fill_gen_evtsel ONLY when the reco event passed apply_event_selection_delphi
    # — unlike the standard gen histos, these carry the reco-level event cuts, so
    # gen vs reco is a matched (acceptance x efficiency) comparison. gen-only
    # names, no reco collision. ---
    for comp in ("", "_bb", "_cc", "_light"):
        for obs in ("FBCEC", "FBCC"):
            hists[f"{obs}_forward_recosel{comp}"] = ROOT.TH1D(
                f"{obs}_forward_recosel{comp}", "",
                len(theta_bins) - 1, array('d', theta_bins))
            hists[f"{obs}_forward_cos_recosel{comp}"] = ROOT.TH1D(
                f"{obs}_forward_cos_recosel{comp}", "",
                len(cos_theta_bins) - 1, array('d', cos_theta_bins))

    # ...same reco-event gate at ALL the pT and |p| tiers (inclusive only; the
    # bb/cc/light flavor split stays at the pT>2 nominal `_recosel` below, exactly
    # mirroring the ungated set where flavor exists only at `_cut2`). Tier -> track
    # cut is defined in fill_gen_evtsel and MUST match process_gen_arrays.
    # `_recosel_cut2` is included so the `_cutN` series is uniform for downstream
    # loops; it is identical by construction to the bare `_recosel` (== ungated
    # `_cut2`, reco-gated), which is kept as the nominal shorthand.
    for tier in ("_cut", "_cut1", "_cut2", "_cut3", "_cut4", "_cut5", "_cut6", "_cut8",
                 "_pmag2", "_pmag3", "_pmag4", "_pmag5", "_pmag6", "_pmag8"):
        for obs in ("FBCEC", "FBCC"):
            hists[f"{obs}_forward_recosel{tier}"] = ROOT.TH1D(
                f"{obs}_forward_recosel{tier}", "",
                len(theta_bins) - 1, array('d', theta_bins))
            hists[f"{obs}_forward_cos_recosel{tier}"] = ROOT.TH1D(
                f"{obs}_forward_cos_recosel{tier}", "",
                len(cos_theta_bins) - 1, array('d', cos_theta_bins))

    # Per-charge N+/N- components for the INCLUSIVE recosel tiers (bare `_recosel`
    # + every _cutN/_pmag), filled via charge_decomp=True. These are the route-A
    # per-charge efficiency-correction inputs for a downstream correction that
    # uses the reco-event-selection-gated gen. Flavor splits stay combined-only
    # (as on the reco side). gen-only names, no reco collision.
    for pfx in ("_recosel",
                "_recosel_cut", "_recosel_cut1", "_recosel_cut2", "_recosel_cut3",
                "_recosel_cut4", "_recosel_cut5", "_recosel_cut6", "_recosel_cut8",
                "_recosel_pmag2", "_recosel_pmag3", "_recosel_pmag4",
                "_recosel_pmag5", "_recosel_pmag6", "_recosel_pmag8"):
        for obs in ("FBCEC", "FBCC"):
            for sgn in ("pos", "neg"):
                hists[f"{obs}_forward_{sgn}{pfx}"] = ROOT.TH1D(
                    f"{obs}_forward_{sgn}{pfx}", "",
                    len(theta_bins) - 1, array('d', theta_bins))
                hists[f"{obs}_forward_cos_{sgn}{pfx}"] = ROOT.TH1D(
                    f"{obs}_forward_cos_{sgn}{pfx}", "",
                    len(cos_theta_bins) - 1, array('d', cos_theta_bins))

    # Gen total energy sums (binned 0-110 GeV like the reco `evis`): E_vis-style
    # (charged pT>0.4 + neutral 0.5<E<50) and correlator-style (charged pT>2),
    # each un-gated and reco-event-selection gated (`_recosel`). Filled in
    # process_gen_arrays (un-gated) and fill_gen_evtsel (gated); all 20-160 deg.
    for nm in ("SumE", "SumE_recosel", "SumE_pt2", "SumE_pt2_recosel"):
        hists[nm] = ROOT.TH1D(nm, ";#SigmaE [GeV];events", 220, 0.0, 110.0)

    # sqrt(s') = m_Z, the effective collision energy after ISR. Companion to
    # SumE (= E_Z); `_recosel` shows how the event selection sculpts the
    # hard-ISR tail. Filled from the Z 4-momentum -- see primary_z_4mom.
    for nm in ("sqrt_sprime", "sqrt_sprime_recosel"):
        hists[nm] = ROOT.TH1D(nm, ";#sqrt{s'} [GeV];events", 220, 0.0, 110.0)

    return hists


def create_reco_histograms(syst_name, do_2d=False, is_mc=False):
    """Reco-level histograms for one systematic, over all P_CUTS momentum-cut
    sets (pT>0.4 baseline, pT>{1,2,3,4,5,6,8}, |p|>{2,3,4,5,6,8}). The IP acceptance
    (tight 0.6/1.0 vs loose `_d0z0loose` 4/10) is carried by syst_config."""
    hists = {}
    s = syst_name

    for pcut_label, pcut_val, pcut_type in P_CUTS:
        tag = f"{pcut_label}_{s}"

        for obs in ("FBCEC", "FBCC"):
            # forward only; combined (= pos − neg) + per-charge components.
            # Per-charge pos/neg feed the downstream per-charge efficiency
            # correction and the route-A covariance blocks (see finalize_errors).
            for comp in ("", "_pos", "_neg"):
                hists[f"{obs}_forward{comp}{tag}"] = ROOT.TH1D(
                    f"{obs}_forward{comp}{tag}", "",
                    len(theta_bins) - 1, array('d', theta_bins))
                # cos(theta) variant → dN/dcos(theta) after /bin_width
                hists[f"{obs}_forward_cos{comp}{tag}"] = ROOT.TH1D(
                    f"{obs}_forward_cos{comp}{tag}", "",
                    len(cos_theta_bins) - 1, array('d', cos_theta_bins))

            if do_2d:
                hists[f"{obs}2d_forward{tag}"] = ROOT.TH2D(f"{obs}2d_forward{tag}", "",
                    len(theta_bins) - 1, array('d', theta_bins),
                    len(energy_bins) - 1, array('d', energy_bins))
                # 2D cos(theta) variant
                hists[f"{obs}2d_forward_cos{tag}"] = ROOT.TH2D(
                    f"{obs}2d_forward_cos{tag}", "",
                    len(cos_theta_bins) - 1, array('d', cos_theta_bins),
                    len(energy_bins) - 1, array('d', energy_bins))

        hists[f"TrackE{tag}"] = ROOT.TH1D(f"TrackE{tag}", "",
            len(energy_bins) - 1, array('d', energy_bins))

        for sign in ("pos", "neg", "all"):
            hists[f"TrackP_{sign}{tag}"] = ROOT.TH1D(f"TrackP_{sign}{tag}", "", 1000, 0, 100)
            hists[f"TrackPt_{sign}{tag}"] = ROOT.TH1D(f"TrackPt_{sign}{tag}", "", 1000, 0, 100)
            hists[f"TrackTheta_{sign}{tag}"] = ROOT.TH1D(f"TrackTheta_{sign}{tag}", "",
                len(theta_bins) - 1, array('d', theta_bins))
            hists[f"TrackPhi_{sign}{tag}"] = ROOT.TH1D(f"TrackPhi_{sign}{tag}", "", 360, -180, 180)

        hists[f"NChargedSele{tag}"] = ROOT.TH1D(f"NChargedSele{tag}", "", 100, 0, 100)
        hists[f"NPartSele{tag}"] = ROOT.TH1D(f"NPartSele{tag}", "", 200, 0, 200)
        hists[f"SumESele{tag}"] = ROOT.TH1D(f"SumESele{tag}", "", 200, 0, 200)

        # --- Vertex-reweighted FBCEC/FBCC (MC nominal only) ---
        # Single mode `_rw_z0sin_charge`: per-track w(pT, θ, z0sinθ, q) from
        # the v73 3D weight tables. Forward only.
        if is_mc and syst_name == 'nominal':
            for obs in ("FBCEC", "FBCC"):
                nm_f = f"{obs}_forward_rw_z0sin_charge{tag}"
                hists[nm_f] = ROOT.TH1D(nm_f, "",
                    len(theta_bins) - 1, array('d', theta_bins))
                # cos(theta) variant
                nm_f_c = f"{obs}_forward_cos_rw_z0sin_charge{tag}"
                hists[nm_f_c] = ROOT.TH1D(nm_f_c, "",
                    len(cos_theta_bins) - 1, array('d', cos_theta_bins))

        # --- z0sin-charge-reweighted single-track kinematics (MC nominal only) ---
        # Per-track w(pT, z0sinθ, q) applied at fill time. Suffix `_rw_z0sin_charge`.
        # Companions to the unweighted TrackE / TrackP_<sign> / etc. histos
        # for studying per-track-quantity data/MC agreement after reweighting.
        if is_mc and syst_name == 'nominal':
            hists[f"TrackE_rw_z0sin_charge{tag}"] = ROOT.TH1D(
                f"TrackE_rw_z0sin_charge{tag}", "",
                len(energy_bins) - 1, array('d', energy_bins))
            for sign in ("pos", "neg", "all"):
                hists[f"TrackP_{sign}_rw_z0sin_charge{tag}"]     = ROOT.TH1D(
                    f"TrackP_{sign}_rw_z0sin_charge{tag}",     "", 1000, 0, 100)
                hists[f"TrackPt_{sign}_rw_z0sin_charge{tag}"]    = ROOT.TH1D(
                    f"TrackPt_{sign}_rw_z0sin_charge{tag}",    "", 1000, 0, 100)
                hists[f"TrackTheta_{sign}_rw_z0sin_charge{tag}"] = ROOT.TH1D(
                    f"TrackTheta_{sign}_rw_z0sin_charge{tag}", "",
                    len(theta_bins) - 1, array('d', theta_bins))
                hists[f"TrackPhi_{sign}_rw_z0sin_charge{tag}"]   = ROOT.TH1D(
                    f"TrackPhi_{sign}_rw_z0sin_charge{tag}",   "", 360, -180, 180)
            if do_2d:
                for direction in ("forward",):
                    for obs in ("FBCEC", "FBCC"):
                        nm = f"{obs}2d_{direction}_rw_z0sin_charge{tag}"
                        hists[nm] = ROOT.TH2D(nm, "",
                            len(theta_bins) - 1, array('d', theta_bins),
                            len(energy_bins) - 1, array('d', energy_bins))
                        # cos(theta) variant
                        nm_c = f"{obs}2d_{direction}_cos_rw_z0sin_charge{tag}"
                        hists[nm_c] = ROOT.TH2D(nm_c, "",
                            len(cos_theta_bins) - 1, array('d', cos_theta_bins),
                            len(energy_bins) - 1, array('d', energy_bins))

        # --- MC matching histograms: matched, fake, charge mis-ID ---
        if is_mc:
            for sign in ("pos", "neg"):
                hists[f"TrackTheta_{sign}_matched{tag}"] = ROOT.TH1D(
                    f"TrackTheta_{sign}_matched{tag}", "",
                    len(theta_bins) - 1, array('d', theta_bins))
                hists[f"TrackTheta_{sign}_fake{tag}"] = ROOT.TH1D(
                    f"TrackTheta_{sign}_fake{tag}", "",
                    len(theta_bins) - 1, array('d', theta_bins))
                hists[f"TrackTheta_{sign}_chargeOK{tag}"] = ROOT.TH1D(
                    f"TrackTheta_{sign}_chargeOK{tag}", "",
                    len(theta_bins) - 1, array('d', theta_bins))
                hists[f"TrackTheta_{sign}_chargeMisID{tag}"] = ROOT.TH1D(
                    f"TrackTheta_{sign}_chargeMisID{tag}", "",
                    len(theta_bins) - 1, array('d', theta_bins))
                # z0sin-charge-reweighted variants of MC-matching histos
                if syst_name == 'nominal':
                    for kind in ('matched', 'fake', 'chargeOK', 'chargeMisID'):
                        nm = f"TrackTheta_{sign}_{kind}_rw_z0sin_charge{tag}"
                        hists[nm] = ROOT.TH1D(nm, "",
                            len(theta_bins) - 1, array('d', theta_bins))

            # --- Per-species (pi/K/p) split of matched, chargeOK, chargeMisID ---
            # Gen-level species from matched gen partner. Filled for nominal
            # + smear systs and their `_dpp0p5` partners (so we get the
            # composition shift under each track-quality variant). `fake`
            # has no gen partner so is not species-split. The
            # `_rw_z0sin_charge` companion is nominal-only.
            if _is_diagnostic_eligible(syst_name):
                for sign in ("pos", "neg"):
                    for kind in ('matched', 'chargeOK', 'chargeMisID'):
                        for sp in ('pi', 'K', 'p'):
                            nm = f"TrackTheta_{sign}_{kind}_{sp}{tag}"
                            hists[nm] = ROOT.TH1D(nm, "",
                                len(theta_bins) - 1, array('d', theta_bins))
                            if syst_name == 'nominal':
                                nm_rw = f"TrackTheta_{sign}_{kind}_{sp}_rw_z0sin_charge{tag}"
                                hists[nm_rw] = ROOT.TH1D(nm_rw, "",
                                    len(theta_bins) - 1, array('d', theta_bins))

        # --- Tighter dpp cut variants for FBCEC/FBCC (nominal syst only) ---
        # Parallel histograms with dpp < {0.25, 0.5, 0.75}, on top of the
        # existing (no-dpp-cut ≡ dpp < 1) distributions. theta + cos, 1D
        # (and 2D when do_2d). No Track*/NCharged variants — fbcec_only.
        if syst_name == 'nominal':
            for _dpp_max, dpp_tag in DPP_CUTS:
                dpp_pcut_tag = f"{pcut_label}{dpp_tag}_{s}"
                for obs in ("FBCEC", "FBCC"):
                    hists[f"{obs}_forward{dpp_pcut_tag}"] = ROOT.TH1D(
                        f"{obs}_forward{dpp_pcut_tag}", "",
                        len(theta_bins) - 1, array('d', theta_bins))
                    hists[f"{obs}_forward_cos{dpp_pcut_tag}"] = ROOT.TH1D(
                        f"{obs}_forward_cos{dpp_pcut_tag}", "",
                        len(cos_theta_bins) - 1, array('d', cos_theta_bins))
                    if do_2d:
                        hists[f"{obs}2d_forward{dpp_pcut_tag}"] = ROOT.TH2D(
                            f"{obs}2d_forward{dpp_pcut_tag}", "",
                            len(theta_bins) - 1, array('d', theta_bins),
                            len(energy_bins) - 1, array('d', energy_bins))
                        hists[f"{obs}2d_forward_cos{dpp_pcut_tag}"] = ROOT.TH2D(
                            f"{obs}2d_forward_cos{dpp_pcut_tag}", "",
                            len(cos_theta_bins) - 1, array('d', cos_theta_bins),
                            len(energy_bins) - 1, array('d', energy_bins))

    # --- Primary-vertex diagnostics (nominal only, once per event) ---
    # Mirrors analysis_v0: PV = first Vtx with status != 4 (status=4 = decay).
    # Only the NanoAOD analyzer can fill these (the Vtx_* collection is absent
    # in TPCNtuple); TPCNtuple jobs leave the histograms empty.
    if syst_name == 'nominal':
        hists['pv_x']           = ROOT.TH1D('pv_x',           ';PV x [cm];events',   200, -1.0, 1.0)
        hists['pv_y']           = ROOT.TH1D('pv_y',           ';PV y [cm];events',   200, -1.0, 1.0)
        hists['pv_z']           = ROOT.TH1D('pv_z',           ';PV z [cm];events',   200, -5.0, 5.0)
        hists['pv_rho']         = ROOT.TH1D('pv_rho',         ';PV #rho [cm];events',100,  0.0, 1.0)
        hists['pv_xy']          = ROOT.TH2D('pv_xy',          ';PV x [cm];PV y [cm]',     200, -1.0, 1.0, 200, -1.0, 1.0)
        hists['pv_z_vs_rho']    = ROOT.TH2D('pv_z_vs_rho',    ';PV #rho [cm];PV z [cm]',  100,  0.0, 1.0, 200, -5.0, 5.0)
        hists['pv_n_outgoing']  = ROOT.TH1D('pv_n_outgoing',  ';PV N_{outgoing};events',  100, 0, 100)
        hists['n_pv_candidates']= ROOT.TH1D('n_pv_candidates',';N PV candidates (Vtx_status != 4);events', 20, 0, 20)

        # --- QED-rejection / on-Z-pole event diagnostics (nominal only) ---
        # Two stages per observable: `_presel` = every on-pole event reaching
        # track selection (BEFORE the hadronic event cut), `_postsel` = events
        # passing apply_event_selection_delphi. Overlaying them proves the cut
        # removes the low-E_vis (gamma-gamma / two-photon, beam-gas) and boosted
        # (hard single-photon ISR) populations, leaving a sample peaked at
        # sqrt(s) = M_Z. Built from the same selected charged+neutral set
        # (20-160 deg) as the event cut. Filled by _fill_qed_event_diag.
        for stage in ('presel', 'postsel'):
            hists[f'evis_{stage}'] = ROOT.TH1D(
                f'evis_{stage}', ';E_{vis} [GeV];events', 220, 0.0, 110.0)
            hists[f'evis_over_roots_{stage}'] = ROOT.TH1D(
                f'evis_over_roots_{stage}', ';E_{vis}/#sqrt{s};events', 120, 0.0, 1.2)
            hists[f'sprime_{stage}'] = ROOT.TH1D(
                f'sprime_{stage}', ";#sqrt{s'} (visible mass) [GeV];events", 220, 0.0, 110.0)
            hists[f'sprime_over_roots_{stage}'] = ROOT.TH1D(
                f'sprime_over_roots_{stage}', ";#sqrt{s'}/#sqrt{s};events", 120, 0.0, 1.2)
            hists[f'echarged_over_roots_{stage}'] = ROOT.TH1D(
                f'echarged_over_roots_{stage}', ';E_{ch}/#sqrt{s};events', 120, 0.0, 1.2)
            hists[f'boost_{stage}'] = ROOT.TH1D(
                f'boost_{stage}', ';#Sigma p_{z}/E_{vis};events', 200, -1.0, 1.0)
            hists[f'absboost_{stage}'] = ROOT.TH1D(
                f'absboost_{stage}', ';|#Sigma p_{z}|/E_{vis};events', 100, 0.0, 1.0)
            hists[f'nch_{stage}'] = ROOT.TH1D(
                f'nch_{stage}', ';N_{ch};events', 100, 0, 100)
            hists[f'costhrust_{stage}'] = ROOT.TH1D(
                f'costhrust_{stage}', ';cos#theta_{thrust};events', 100, -1.0, 1.0)
            hists[f'nch_vs_evis_{stage}'] = ROOT.TH2D(
                f'nch_vs_evis_{stage}', ';N_{ch};E_{vis} [GeV]',
                60, 0, 60, 110, 0.0, 110.0)

        # --- Reverse-selection (rejected-event) control region (nominal only) ---
        # FBCC/FBCEC computed on events FAILING the hadronic event selection.
        # Contrasted with the selected-sample FBCC (the `_forward` baseline),
        # this exposes the charge asymmetry of the rejected, QED-contaminated
        # population (gamma-gamma, radiative, low-multiplicity). Baseline charged
        # set only; filled via fill_observables(..., fbcec_only=True).
        for obs in ("FBCEC", "FBCC"):
            hists[f'{obs}_forward_revsel'] = ROOT.TH1D(
                f'{obs}_forward_revsel', "",
                len(theta_bins) - 1, array('d', theta_bins))
            hists[f'{obs}_forward_cos_revsel'] = ROOT.TH1D(
                f'{obs}_forward_cos_revsel', "",
                len(cos_theta_bins) - 1, array('d', cos_theta_bins))
        hists['n_revsel'] = ROOT.TH1D('n_revsel', ';;rejected events', 1, 0, 1)

        # --- Primary Z->qqbar flavor split (bb / cc / light) of the baseline
        # reco FBCEC/FBCC (combined + cos). MC only; SAME event-level truth tag
        # as the gen split. Nominal syst only. SetDirectory(0) so the shared
        # gen/reco names don't clash in gDirectory. ---
        if is_mc:
            for flav in ('bb', 'cc', 'light'):
                for obs in ('FBCEC', 'FBCC'):
                    for nm, edges in ((f'{obs}_forward_{flav}', theta_bins),
                                      (f'{obs}_forward_cos_{flav}', cos_theta_bins)):
                        hh = ROOT.TH1D(nm, "", len(edges) - 1, array('d', edges))
                        hh.SetDirectory(0)
                        hists[nm] = hh

    # --- Impact parameter diagnostics (nominal + smear systs) ---
    # Suffix: empty for nominal, `_smear_d0` / `_smear_z0sin` for the smear
    # systematics so we get parallel sets of d0/z0/z0sin distributions
    # showing the un-smeared (nominal) vs smeared shapes for closure plots.
    if _is_diagnostic_eligible(syst_name):
        sm_sfx = _diag_suffix(syst_name)
        for sign in ("pos", "neg", "all"):
            hists[f"d0_vtx_{sign}{sm_sfx}"]    = ROOT.TH1D(f"d0_vtx_{sign}{sm_sfx}",    "",
                D0_VTX_NBINS,    D0_VTX_RANGE[0],    D0_VTX_RANGE[1])
            hists[f"z0_vtx_{sign}{sm_sfx}"]    = ROOT.TH1D(f"z0_vtx_{sign}{sm_sfx}",    "",
                Z0_VTX_NBINS,    Z0_VTX_RANGE[0],    Z0_VTX_RANGE[1])
            hists[f"z0sin_vtx_{sign}{sm_sfx}"] = ROOT.TH1D(f"z0sin_vtx_{sign}{sm_sfx}", "",
                Z0SIN_VTX_NBINS, Z0SIN_VTX_RANGE[0], Z0SIN_VTX_RANGE[1])
        # TProfile RMS vs theta/pT — same axes as the nominal block for direct comparison.
        _theta_diag = np.linspace(0, 180, 37)
        _pt_diag = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 45.0])
        for var in ("d0_vtx", "z0sin_vtx"):
            hists[f"{var}_rms_vs_theta{sm_sfx}"] = ROOT.TProfile(f"{var}_rms_vs_theta{sm_sfx}", "",
                len(_theta_diag) - 1, array('d', _theta_diag), 's')
            hists[f"{var}_rms_vs_pt{sm_sfx}"] = ROOT.TProfile(f"{var}_rms_vs_pt{sm_sfx}", "",
                len(_pt_diag) - 1, array('d', _pt_diag), 's')

    # Remainder of the diagnostic block (rw companions, TProfiles, 4D THn)
    # stays nominal-only.
    if syst_name == 'nominal':
        # Reweighted d0_vtx / z0sin_vtx (MC only) — closure / sanity checks
        # for the per-track w(pT, θ, z0sinθ, q) lookup. Single mode
        # `_rw_z0sin_charge`.
        if is_mc:
            for sign in ("pos", "neg", "all"):
                nm_d0    = f"d0_vtx_{sign}_rw_z0sin_charge"
                nm_z0sin = f"z0sin_vtx_{sign}_rw_z0sin_charge"
                hists[nm_d0]    = ROOT.TH1D(nm_d0,    "",
                    D0_VTX_NBINS,    D0_VTX_RANGE[0],    D0_VTX_RANGE[1])
                hists[nm_z0sin] = ROOT.TH1D(nm_z0sin, "",
                    Z0SIN_VTX_NBINS, Z0SIN_VTX_RANGE[0], Z0SIN_VTX_RANGE[1])

        # (TProfiles d0_vtx_rms_vs_{theta,pt} and z0sin_vtx_rms_vs_{theta,pt}
        # are booked above in the nominal+smear block; same names so no
        # duplication.)

        # 4D track-quality histogram for IP-reweighting study (charge-asymmetric).
        # Axes:
        #   0: pT     [GeV]  6 bins, edges [0, 1, 2, 4, 6, 8, 50]
        #   1: theta  [deg]  18 bins, 10° wide, [0, 180]
        #   2: d0     [cm]   200 bins, 0.01 cm, [-1, 1]
        #   3: z0sinθ [cm]   200 bins, 0.01 cm, [-1, 1]
        # Total per charge: 6 × 18 × 200 × 200 = 4,320,000 bins (~17 MB at THnF).
        # Replaces the previous separate 3D (pT × IP × θ) histos so the
        # reweighting can be derived per (pT, θ, d0, z0sinθ) bin jointly.
        ip4d_pt_edges = array('d', [0.0, 1.0, 2.0, 4.0, 6.0, 8.0, 50.0])
        ip4d_nbins = array('i', [6, 18, 200, 200])
        ip4d_xmin  = array('d', [0.0,    0.0, -1.0, -1.0])
        ip4d_xmax  = array('d', [50.0, 180.0,  1.0,  1.0])
        for sign in ("pos", "neg"):
            nm = f"vtx_pt_theta_d0_z0sin_{sign}_4d"
            h4d = ROOT.THnF(nm,
                ";p_{T} [GeV];#theta [deg];d_{0} [cm];z_{0}sin#theta [cm]",
                4, ip4d_nbins, ip4d_xmin, ip4d_xmax)
            # pT axis is variable-binned; theta, d0, z0sinθ stay uniform.
            h4d.GetAxis(0).Set(len(ip4d_pt_edges) - 1, ip4d_pt_edges)
            hists[nm] = h4d

    return hists


# ============================================================================
# SYSTEMATIC VARIATION
# ============================================================================

def apply_systematic_variation(px, py, pz, m, q, pt, th, d0, z0, config,
                               dpp=None, smearer=None, rng=None):
    """Apply systematic variations. Charged-only.

    `dpp` (per-track delta-p/p) is passed through unchanged by the
    systematics but filtered by `keep_mask` together with the other
    arrays, so callers can use it for downstream cuts.

    `smearer` (VertexResolutionSmearer) + `rng` (np.random.Generator) are
    consulted only for `type=='resolution_smear'` configs. If either is
    None the smear is silently skipped (acts like nominal)."""
    px_var = px.copy()
    py_var = py.copy()
    pz_var = pz.copy()
    m_var = m.copy()
    q_var = q.copy()
    pt_var = pt.copy()
    th_var = th.copy()
    d0_var = d0.copy()
    z0_var = z0.copy()
    dpp_var = dpp.copy() if dpp is not None else None

    is_charged = np.abs(q_var) > 0.1

    # Resolution smear (charged only) — per-track Gaussian noise on d0 or
    # z0·sinθ, width Δ(pT, θ) from VertexResolutionSmearer.
    if config.get('type') == 'resolution_smear' and smearer is not None and rng is not None:
        kind = config.get('smear_kind', 'd0')
        idx = np.where(is_charged)[0]
        if len(idx) > 0:
            pt_c        = pt_var[idx]
            theta_deg_c = np.degrees(th_var[idx])
            noise = smearer.sample(kind, pt_c, theta_deg_c, rng)
            if kind == 'd0':
                d0_var[idx] += noise
            else:  # 'z0sin' — noise is added to z0·sinθ; map back to z0
                sin_th = np.sin(th_var[idx])
                sin_th_safe = np.where(sin_th > 1e-10, sin_th, 1e-10)
                z0_var[idx] += noise / sin_th_safe

    # Transverse-momentum (pT) scale (charged only): scale px, py only and
    # leave pz unchanged, so the systematic varies pT directly (|p| and θ shift
    # accordingly) rather than scaling the full 3-momentum.
    if config.get('momentum_scale_charged'):
        scale = config['momentum_scale_charged']
        px_var[is_charged] *= scale
        py_var[is_charged] *= scale
        pt_var[is_charged] = np.sqrt(px_var[is_charged]**2 + py_var[is_charged]**2)
        th_var[is_charged] = np.arctan2(pt_var[is_charged], pz_var[is_charged])

    # Efficiency
    keep_mask = np.ones(len(px_var), dtype=bool)

    if config.get('apply_fake_drop'):
        charged_indices = np.where(is_charged)[0]
        if len(charged_indices) > 0:
            fake_dropped = apply_fake_drop_charged(
                px_var[is_charged], py_var[is_charged],
                pz_var[is_charged], m_var[is_charged],
                seed=config.get('fake_drop_seed')
            )
            keep_mask[charged_indices] = fake_dropped['mask']

    if config.get('drop_charged_fraction', 0.0) > 0:
        charged_indices = np.where(is_charged)[0]
        if len(charged_indices) > 0:
            dropped = randomly_drop_particles(
                px_var[is_charged], py_var[is_charged],
                pz_var[is_charged], m_var[is_charged],
                drop_fraction=config['drop_charged_fraction'],
                seed=config.get('drop_charged_seed')
            )
            keep_mask[charged_indices] = dropped['mask']

    # Tighter dpp cut: drop charged tracks with dpp >= dpp_max or dpp < 0
    # (invalid fit). Operates via keep_mask so all downstream histograms
    # see only the surviving subset.
    dpp_max_cfg = config.get('charged_dpp_max')
    if dpp_max_cfg is not None and dpp_var is not None:
        charged_indices = np.where(is_charged)[0]
        if len(charged_indices) > 0:
            dpp_c_arr = dpp_var[charged_indices]
            dpp_pass = (dpp_c_arr >= 0) & (dpp_c_arr < dpp_max_cfg)
            keep_mask[charged_indices] = keep_mask[charged_indices] & dpp_pass

    return {
        'px': px_var[keep_mask], 'py': py_var[keep_mask],
        'pz': pz_var[keep_mask], 'm': m_var[keep_mask],
        'q': q_var[keep_mask], 'pt': pt_var[keep_mask],
        'th': th_var[keep_mask], 'd0': d0_var[keep_mask],
        'z0': z0_var[keep_mask],
        'dpp': dpp_var[keep_mask] if dpp_var is not None else None,
    }


# ============================================================================
# FBCEC COMPUTATION
# ============================================================================

def fill_reweighted_observables(angle_deg, e_c, q_c, w_zc,
                                hists_dict, prefix=""):
    """
    Fill vertex-reweighted FBCEC/FBCC variants for one event.

    `w_zc` is a per-track weight array (z0sin_charge mode, single 3D table)
    aligned to the same length as the input track arrays. Each track's
    (qE, q) contribution is multiplied by its own w.
    """
    angle_rad = np.radians(angle_deg)
    sin_angle = np.sin(angle_rad)
    cos_angle = np.cos(angle_rad)
    n = len(e_c)
    for i in range(n):
        if sin_angle[i] < 1e-10:
            continue
        a_fwd = angle_deg[i]
        c_fwd = cos_angle[i]
        ei = e_c[i]
        qi = q_c[i]
        wi = float(w_zc[i])
        w_cec = qi * ei * wi
        w_cc = qi * wi
        hists_dict[f"FBCEC_forward_rw_z0sin_charge{prefix}"].Fill(a_fwd, w_cec)
        hists_dict[f"FBCC_forward_rw_z0sin_charge{prefix}"].Fill(a_fwd, w_cc)
        # cos(theta) variant
        hists_dict[f"FBCEC_forward_cos_rw_z0sin_charge{prefix}"].Fill(c_fwd, w_cec)
        hists_dict[f"FBCC_forward_cos_rw_z0sin_charge{prefix}"].Fill(c_fwd, w_cc)


def fill_observables(angle_deg, e_c, q_c, hists_dict, prefix="", p_c=None,
                     pt_c=None, phi_c=None, do_2d=False,
                     w_z0sin_charge=None, fbcec_only=False,
                     species=None, charge_decomp=False):
    """
    Fill FBCEC, FBCC, and track kinematics histograms for one event.

    If `w_z0sin_charge` (per-track weight array, same length as e_c) is
    given, also fill `_rw_z0sin_charge` companion histograms for every
    per-track quantity. The caller is responsible for ensuring the
    matching `_rw_z0sin_charge` histograms have been booked.

    If `fbcec_only=True`, only FBCEC/FBCC histograms (theta + cos, 1D
    + 2D when do_2d) are filled; Track* and NCharged-style hists are
    skipped. Used by the dpp-variant fills to keep the booked-hist
    inventory tight.

    `charge_decomp=True` forces the per-charge `_pos`/`_neg` FBCEC/FBCC
    components to be filled EVEN when `fbcec_only=True` — i.e. per-charge
    without the Track* block. Used by the reco-gated `_recosel` fills so
    they carry the route-A per-charge efficiency-correction inputs while
    staying free of the (unwanted) track-kinematics histograms.
    """
    angle_rad = np.radians(angle_deg)
    sin_angle = np.sin(angle_rad)
    cos_angle = np.cos(angle_rad)

    for i in range(len(e_c)):
        if sin_angle[i] < 1e-10:
            continue

        jacobian = 1.0
        a_fwd = angle_deg[i]
        c_fwd = cos_angle[i]            # cos(forward angle)
        ei = e_c[i]
        qi = q_c[i]

        w_cec = qi * ei * jacobian
        w_cc = qi * jacobian

        # 1D forward observables (combined = N⁺−N⁻ / E⁺−E⁻); backward dropped
        hists_dict[f"FBCEC_forward{prefix}"].Fill(a_fwd, w_cec)
        hists_dict[f"FBCC_forward{prefix}"].Fill(a_fwd, w_cc)
        # cos(theta)-binned → dN/dcos(theta)
        hists_dict[f"FBCEC_forward_cos{prefix}"].Fill(c_fwd, w_cec)
        hists_dict[f"FBCC_forward_cos{prefix}"].Fill(c_fwd, w_cc)

        # Per-charge components (skipped for the fbcec_only dpp fine-scan,
        # unless charge_decomp forces them on — see the recosel fills).
        # These are the inputs to the downstream per-charge efficiency
        # correction and the route-A covariance blocks: FBCEC weight = E,
        # FBCC weight = 1, with the charge sign picking the pos/neg histogram.
        if (not fbcec_only) or charge_decomp:
            sgn = "pos" if qi > 0 else "neg"
            hists_dict[f"FBCEC_forward_{sgn}{prefix}"].Fill(a_fwd, ei)
            hists_dict[f"FBCC_forward_{sgn}{prefix}"].Fill(a_fwd, 1.0)
            hists_dict[f"FBCEC_forward_cos_{sgn}{prefix}"].Fill(c_fwd, ei)
            hists_dict[f"FBCC_forward_cos_{sgn}{prefix}"].Fill(c_fwd, 1.0)

        # 2D forward observables (only if requested)
        if do_2d:
            hists_dict[f"FBCEC2d_forward{prefix}"].Fill(a_fwd, ei, w_cec)
            hists_dict[f"FBCC2d_forward{prefix}"].Fill(a_fwd, ei, w_cc)
            # 2D cos(theta) variants
            hists_dict[f"FBCEC2d_forward_cos{prefix}"].Fill(c_fwd, ei, w_cec)
            hists_dict[f"FBCC2d_forward_cos{prefix}"].Fill(c_fwd, ei, w_cc)

        if fbcec_only:
            continue

        # Track kinematics
        hists_dict[f"TrackE{prefix}"].Fill(ei)

        sign = "pos" if qi > 0 else "neg"
        hists_dict[f"TrackTheta_{sign}{prefix}"].Fill(a_fwd)
        hists_dict[f"TrackTheta_all{prefix}"].Fill(a_fwd)
        if p_c is not None:
            hists_dict[f"TrackP_{sign}{prefix}"].Fill(p_c[i])
            hists_dict[f"TrackP_all{prefix}"].Fill(p_c[i])
        if pt_c is not None:
            hists_dict[f"TrackPt_{sign}{prefix}"].Fill(pt_c[i])
            hists_dict[f"TrackPt_all{prefix}"].Fill(pt_c[i])
        if phi_c is not None:
            hists_dict[f"TrackPhi_{sign}{prefix}"].Fill(phi_c[i])
            hists_dict[f"TrackPhi_all{prefix}"].Fill(phi_c[i])

        # Per-species fills (pi/K/p) — gen-level only; species is None on reco.
        if species is not None:
            sp_code = int(species[i])
            sp_label = {1: "pi", 2: "K", 3: "p"}.get(sp_code)
            if sp_label is not None:
                hists_dict[f"TrackTheta_{sp_label}{prefix}"].Fill(a_fwd)
                if p_c is not None:
                    hists_dict[f"TrackP_{sp_label}{prefix}"].Fill(p_c[i])
                if pt_c is not None:
                    hists_dict[f"TrackPt_{sp_label}{prefix}"].Fill(pt_c[i])
                if phi_c is not None:
                    hists_dict[f"TrackPhi_{sp_label}{prefix}"].Fill(phi_c[i])
                # Per-charge × per-species TrackTheta (parallels the
                # correction_fbcc fake-by-species decomposition).
                hists_dict[f"TrackTheta_{sign}_{sp_label}{prefix}"].Fill(a_fwd)

        # z0sin-charge-reweighted variants of every per-track quantity
        if w_z0sin_charge is not None:
            wi = float(w_z0sin_charge[i])
            hists_dict[f"TrackE_rw_z0sin_charge{prefix}"].Fill(ei, wi)
            hists_dict[f"TrackTheta_{sign}_rw_z0sin_charge{prefix}"].Fill(a_fwd, wi)
            hists_dict[f"TrackTheta_all_rw_z0sin_charge{prefix}"].Fill(a_fwd, wi)
            if p_c is not None:
                hists_dict[f"TrackP_{sign}_rw_z0sin_charge{prefix}"].Fill(p_c[i], wi)
                hists_dict[f"TrackP_all_rw_z0sin_charge{prefix}"].Fill(p_c[i], wi)
            if pt_c is not None:
                hists_dict[f"TrackPt_{sign}_rw_z0sin_charge{prefix}"].Fill(pt_c[i], wi)
                hists_dict[f"TrackPt_all_rw_z0sin_charge{prefix}"].Fill(pt_c[i], wi)
            if phi_c is not None:
                hists_dict[f"TrackPhi_{sign}_rw_z0sin_charge{prefix}"].Fill(phi_c[i], wi)
                hists_dict[f"TrackPhi_all_rw_z0sin_charge{prefix}"].Fill(phi_c[i], wi)
            if do_2d:
                hists_dict[f"FBCEC2d_forward_rw_z0sin_charge{prefix}"].Fill(a_fwd, ei, w_cec * wi)
                hists_dict[f"FBCC2d_forward_rw_z0sin_charge{prefix}"].Fill(a_fwd, ei, w_cc * wi)
                # cos(theta) variants
                hists_dict[f"FBCEC2d_forward_cos_rw_z0sin_charge{prefix}"].Fill(c_fwd, ei, w_cec * wi)
                hists_dict[f"FBCC2d_forward_cos_rw_z0sin_charge{prefix}"].Fill(c_fwd, ei, w_cc * wi)


# ============================================================================
# GEN-LEVEL PROCESSING
# ============================================================================

def load_gen_particles(tree_gen_before, tree_gen, event_idx):
    """Load gen-level particles with energy conservation fallback."""
    tree_gen_before.GetEntry(event_idx)
    E = tree_gen_before.Energy
    if abs(E - 91.25) > 1:
        return None

    px = np.array(tree_gen_before.px)
    py = np.array(tree_gen_before.py)
    pz = np.array(tree_gen_before.pz)
    m = np.array(tree_gen_before.mass)
    q = np.array(tree_gen_before.charge)
    pid = np.array(tree_gen_before.pid) if hasattr(tree_gen_before, 'pid') else None

    if len(px) == 0:
        return None

    e = np.sqrt(px**2 + py**2 + pz**2 + m**2)

    # Energy conservation check — fallback to tgen
    if abs(np.sum(e) - E) > 0.1:
        tree_gen.GetEntry(event_idx)
        px = np.array(tree_gen.px)
        py = np.array(tree_gen.py)
        pz = np.array(tree_gen.pz)
        m = np.array(tree_gen.mass)
        q = np.array(tree_gen.charge)
        pid = np.array(tree_gen.pid) if hasattr(tree_gen, 'pid') else None
        e = np.sqrt(px**2 + py**2 + pz**2 + m**2)

    species = species_codes(pid, 'pdg') if pid is not None else np.zeros(len(px), dtype=np.int8)
    return {'px': px, 'py': py, 'pz': pz, 'm': m, 'q': q, 'e': e, 'E': E, 'species': species}


def process_gen_level(tree_gen_before, tree_gen, event_idx, gen_hists, do_2d=False):
    """Process gen-level: fill FBCEC histograms. 5 pT-cut sets + 2 |p|-cut sets."""
    data = load_gen_particles(tree_gen_before, tree_gen, event_idx)
    if data is None:
        return
    process_gen_arrays(data['px'], data['py'], data['pz'], data['q'], data['e'],
                       gen_hists, do_2d=do_2d, species=data.get('species'))


def process_gen_arrays(px, py, pz, q, e, gen_hists, do_2d=False, species=None,
                       flavor=None, z_4mom=None):
    """Pure-numpy gen-level processor. Same body as process_gen_level after
    particle loading; callable from either the TPCNtuple tree wrapper or
    the NanoAOD reader once it has extracted SimPart leaves into arrays.

    `species` (optional): per-particle species code (1=pi, 2=K, 3=p, 0=other);
    when provided, per-species Track* hists are also filled by fill_observables.
    """
    # Select charged particles
    is_charged = np.abs(q) > 0.5
    px_c = px[is_charged]
    py_c = py[is_charged]
    pz_c = pz[is_charged]
    q_c = q[is_charged]
    e_c = e[is_charged]
    sp_c = species[is_charged] if species is not None else None

    # Gen total energy sums (un-gated): E_vis-style (charged pT>0.4 + neutral
    # 0.5<E<50) and correlator-style (charged pT>2), both within 20-160 deg.
    _pt = np.sqrt(px**2 + py**2)
    _th = np.arctan2(_pt, pz)
    _acc = (_th > np.deg2rad(20)) & (_th < np.deg2rad(160))
    # SumE = sqrt(s) - E_ISR: neutrinos INCLUDED, ISR photons EXCLUDED. Taken
    # from the Z 4-momentum (primary_z_4mom) rather than summing leaves: the
    # leaves have neutrinos stripped in build_gen_data, and ISR photons can NOT
    # be tagged in kk2f's pruned GenPart. Everything the Z decays to sums to
    # its 4-momentum, so E_Z is exactly that sum. NOTE this redefines SumE --
    # it was the 4pi visible sum (~89.5 = sqrt(s) - E_nu) before 2026-07-15.
    if z_4mom is not None:
        gen_hists["SumE"].Fill(float(z_4mom[0]))
        gen_hists["sqrt_sprime"].Fill(float(z_4mom[1]))
    else:
        gen_hists["SumE"].Fill(float(e.sum()))   # no Z in record (data/pruned)
    gen_hists["SumE_pt2"].Fill(float(e[_acc & is_charged & (_pt > 2.0)].sum()))

    if len(q_c) == 0:
        return

    p_mag = np.sqrt(px_c**2 + py_c**2 + pz_c**2)
    pt_c = np.sqrt(px_c**2 + py_c**2)
    cos_angle = np.divide(pz_c, p_mag, out=np.zeros_like(pz_c), where=p_mag > 0)
    angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    th_c = np.arctan2(pt_c, pz_c)
    phi_c = np.degrees(np.arctan2(py_c, px_c))

    # --- Set 1: no cuts (all charged) ---
    gen_hists["NCharged"].Fill(len(q_c))
    fill_observables(angle_deg, e_c, q_c, gen_hists, prefix="",
                     p_c=p_mag, pt_c=pt_c, phi_c=phi_c, do_2d=do_2d,
                     species=sp_c)

    # --- Set 2: theta 20-160 deg + pT > 0.4 GeV (baseline) ---
    theta_accept = (th_c > np.deg2rad(20)) & (th_c < np.deg2rad(160))
    sel_cut = theta_accept & (pt_c > 0.4)
    if np.any(sel_cut):
        gen_hists["NCharged_cut"].Fill(np.sum(sel_cut))
        fill_observables(angle_deg[sel_cut], e_c[sel_cut], q_c[sel_cut],
                         gen_hists, prefix="_cut", p_c=p_mag[sel_cut],
                         pt_c=pt_c[sel_cut], phi_c=phi_c[sel_cut], do_2d=do_2d,
                         species=(sp_c[sel_cut] if sp_c is not None else None))

    # --- Set 3: theta 20-160 deg + pT > 1 GeV ---
    sel_cut1 = theta_accept & (pt_c > 1.0)
    if np.any(sel_cut1):
        gen_hists["NCharged_cut1"].Fill(np.sum(sel_cut1))
        fill_observables(angle_deg[sel_cut1], e_c[sel_cut1], q_c[sel_cut1],
                         gen_hists, prefix="_cut1", p_c=p_mag[sel_cut1],
                         pt_c=pt_c[sel_cut1], phi_c=phi_c[sel_cut1], do_2d=do_2d,
                         species=(sp_c[sel_cut1] if sp_c is not None else None))

    # --- Set 4: theta 20-160 deg + pT > 2 GeV (the nominal-downstream cut) ---
    sel_cut2 = theta_accept & (pt_c > 2.0)
    if np.any(sel_cut2):
        gen_hists["NCharged_cut2"].Fill(np.sum(sel_cut2))
        fill_observables(angle_deg[sel_cut2], e_c[sel_cut2], q_c[sel_cut2],
                         gen_hists, prefix="_cut2", p_c=p_mag[sel_cut2],
                         pt_c=pt_c[sel_cut2], phi_c=phi_c[sel_cut2], do_2d=do_2d,
                         species=(sp_c[sel_cut2] if sp_c is not None else None))
        # Primary Z->qqbar flavor split at the pT>2 nominal-downstream cut
        # (combined FBCEC/FBCC). Sums to the pT>2 inclusive FBCC_forward_cut2.
        if flavor is not None:
            fill_observables(angle_deg[sel_cut2], e_c[sel_cut2], q_c[sel_cut2],
                             gen_hists, prefix=f"_{flavor}", do_2d=False,
                             fbcec_only=True)

    # --- theta 20-160 deg + pT > 3 GeV ---
    sel_cut3 = theta_accept & (pt_c > 3.0)
    if np.any(sel_cut3):
        gen_hists["NCharged_cut3"].Fill(np.sum(sel_cut3))
        fill_observables(angle_deg[sel_cut3], e_c[sel_cut3], q_c[sel_cut3],
                         gen_hists, prefix="_cut3", p_c=p_mag[sel_cut3],
                         pt_c=pt_c[sel_cut3], phi_c=phi_c[sel_cut3], do_2d=do_2d,
                         species=(sp_c[sel_cut3] if sp_c is not None else None))

    # --- Set 5: theta 20-160 deg + pT > 4 GeV ---
    sel_cut4 = theta_accept & (pt_c > 4.0)
    if np.any(sel_cut4):
        gen_hists["NCharged_cut4"].Fill(np.sum(sel_cut4))
        fill_observables(angle_deg[sel_cut4], e_c[sel_cut4], q_c[sel_cut4],
                         gen_hists, prefix="_cut4", p_c=p_mag[sel_cut4],
                         pt_c=pt_c[sel_cut4], phi_c=phi_c[sel_cut4], do_2d=do_2d,
                         species=(sp_c[sel_cut4] if sp_c is not None else None))

    # --- theta 20-160 deg + pT > 5 GeV ---
    sel_cut5 = theta_accept & (pt_c > 5.0)
    if np.any(sel_cut5):
        gen_hists["NCharged_cut5"].Fill(np.sum(sel_cut5))
        fill_observables(angle_deg[sel_cut5], e_c[sel_cut5], q_c[sel_cut5],
                         gen_hists, prefix="_cut5", p_c=p_mag[sel_cut5],
                         pt_c=pt_c[sel_cut5], phi_c=phi_c[sel_cut5], do_2d=do_2d,
                         species=(sp_c[sel_cut5] if sp_c is not None else None))

    # --- Set 6: theta 20-160 deg + pT > 6 GeV ---
    sel_cut6 = theta_accept & (pt_c > 6.0)
    if np.any(sel_cut6):
        gen_hists["NCharged_cut6"].Fill(np.sum(sel_cut6))
        fill_observables(angle_deg[sel_cut6], e_c[sel_cut6], q_c[sel_cut6],
                         gen_hists, prefix="_cut6", p_c=p_mag[sel_cut6],
                         pt_c=pt_c[sel_cut6], phi_c=phi_c[sel_cut6], do_2d=do_2d,
                         species=(sp_c[sel_cut6] if sp_c is not None else None))

    # --- Set 7: theta 20-160 deg + pT > 8 GeV ---
    sel_cut8 = theta_accept & (pt_c > 8.0)
    if np.any(sel_cut8):
        gen_hists["NCharged_cut8"].Fill(np.sum(sel_cut8))
        fill_observables(angle_deg[sel_cut8], e_c[sel_cut8], q_c[sel_cut8],
                         gen_hists, prefix="_cut8", p_c=p_mag[sel_cut8],
                         pt_c=pt_c[sel_cut8], phi_c=phi_c[sel_cut8], do_2d=do_2d,
                         species=(sp_c[sel_cut8] if sp_c is not None else None))

    # --- Set 8: theta 20-160 deg + pT > 0.4 (baseline) + |p| > 2 GeV ---
    sel_pmag2 = theta_accept & (pt_c > 0.4) & (p_mag > 2.0)
    if np.any(sel_pmag2):
        gen_hists["NCharged_pmag2"].Fill(np.sum(sel_pmag2))
        fill_observables(angle_deg[sel_pmag2], e_c[sel_pmag2], q_c[sel_pmag2],
                         gen_hists, prefix="_pmag2", p_c=p_mag[sel_pmag2],
                         pt_c=pt_c[sel_pmag2], phi_c=phi_c[sel_pmag2], do_2d=do_2d,
                         species=(sp_c[sel_pmag2] if sp_c is not None else None))

    # --- theta 20-160 deg + pT > 0.4 (baseline) + |p| > 3 GeV ---
    sel_pmag3 = theta_accept & (pt_c > 0.4) & (p_mag > 3.0)
    if np.any(sel_pmag3):
        gen_hists["NCharged_pmag3"].Fill(np.sum(sel_pmag3))
        fill_observables(angle_deg[sel_pmag3], e_c[sel_pmag3], q_c[sel_pmag3],
                         gen_hists, prefix="_pmag3", p_c=p_mag[sel_pmag3],
                         pt_c=pt_c[sel_pmag3], phi_c=phi_c[sel_pmag3], do_2d=do_2d,
                         species=(sp_c[sel_pmag3] if sp_c is not None else None))

    # --- Set 9: theta 20-160 deg + pT > 0.4 (baseline) + |p| > 4 GeV ---
    sel_pmag4 = theta_accept & (pt_c > 0.4) & (p_mag > 4.0)
    if np.any(sel_pmag4):
        gen_hists["NCharged_pmag4"].Fill(np.sum(sel_pmag4))
        fill_observables(angle_deg[sel_pmag4], e_c[sel_pmag4], q_c[sel_pmag4],
                         gen_hists, prefix="_pmag4", p_c=p_mag[sel_pmag4],
                         pt_c=pt_c[sel_pmag4], phi_c=phi_c[sel_pmag4], do_2d=do_2d,
                         species=(sp_c[sel_pmag4] if sp_c is not None else None))

    # --- theta 20-160 deg + pT > 0.4 (baseline) + |p| > 5 GeV ---
    sel_pmag5 = theta_accept & (pt_c > 0.4) & (p_mag > 5.0)
    if np.any(sel_pmag5):
        gen_hists["NCharged_pmag5"].Fill(np.sum(sel_pmag5))
        fill_observables(angle_deg[sel_pmag5], e_c[sel_pmag5], q_c[sel_pmag5],
                         gen_hists, prefix="_pmag5", p_c=p_mag[sel_pmag5],
                         pt_c=pt_c[sel_pmag5], phi_c=phi_c[sel_pmag5], do_2d=do_2d,
                         species=(sp_c[sel_pmag5] if sp_c is not None else None))

    # --- Set 10: theta 20-160 deg + pT > 0.4 (baseline) + |p| > 6 GeV ---
    sel_pmag6 = theta_accept & (pt_c > 0.4) & (p_mag > 6.0)
    if np.any(sel_pmag6):
        gen_hists["NCharged_pmag6"].Fill(np.sum(sel_pmag6))
        fill_observables(angle_deg[sel_pmag6], e_c[sel_pmag6], q_c[sel_pmag6],
                         gen_hists, prefix="_pmag6", p_c=p_mag[sel_pmag6],
                         pt_c=pt_c[sel_pmag6], phi_c=phi_c[sel_pmag6], do_2d=do_2d,
                         species=(sp_c[sel_pmag6] if sp_c is not None else None))

    # --- Set 11: theta 20-160 deg + pT > 0.4 (baseline) + |p| > 8 GeV ---
    sel_pmag8 = theta_accept & (pt_c > 0.4) & (p_mag > 8.0)
    if np.any(sel_pmag8):
        gen_hists["NCharged_pmag8"].Fill(np.sum(sel_pmag8))
        fill_observables(angle_deg[sel_pmag8], e_c[sel_pmag8], q_c[sel_pmag8],
                         gen_hists, prefix="_pmag8", p_c=p_mag[sel_pmag8],
                         pt_c=pt_c[sel_pmag8], phi_c=phi_c[sel_pmag8], do_2d=do_2d,
                         species=(sp_c[sel_pmag8] if sp_c is not None else None))


# ============================================================================
# RECO-LEVEL PROCESSING
# ============================================================================

def process_reco_level(tree_reco, event_idx, systematics,
                       reco_hists, counter, cov_accum=None, do_2d=False,
                       gen_data=None, vtx_rw=None, period=None,
                       smearer=None, rng=None):
    """
    Process reco-level for all systematics.
    Baseline pT>0.4 + IP selection (config-driven d0/z0sin), then the P_CUTS
    momentum sets: pT>0.4 baseline, pT>{1,2,3,4,5,6,8}, |p|>{2,3,4,5,6,8}.

    Thin wrapper around process_reco_arrays for the TPCNtuple format.
    """
    tree_reco.GetEntry(event_idx)

    E = tree_reco.Energy
    if abs(E - 91.25) > 1:
        return

    px = np.array(tree_reco.px)
    py = np.array(tree_reco.py)
    pz = np.array(tree_reco.pz)
    m = np.array(tree_reco.mass)
    q = np.array(tree_reco.charge)
    pt = np.array(tree_reco.pt)
    th = np.array(tree_reco.theta)
    d0 = np.array(tree_reco.d0)
    z0 = np.array(tree_reco.z0)
    dpp = np.array(tree_reco.dpp) if hasattr(tree_reco, 'dpp') else np.full(len(px), -1.0)

    process_reco_arrays(E, px, py, pz, m, q, pt, th, d0, z0,
                        systematics, reco_hists, counter,
                        cov_accum=cov_accum, do_2d=do_2d,
                        gen_data=gen_data, vtx_rw=vtx_rw, period=period,
                        dpp=dpp, smearer=smearer, rng=rng)


def fill_gen_evtsel(px, py, pz, q, e, gen_hists, flavor=None, z_4mom=None):
    """Fill gen FBCEC/FBCC gated by the RECO event selection (theta 20-160,
    charged). The CALLER invokes this only when the reco event passed
    apply_event_selection_delphi, so these gen histograms carry the reco-level
    event cuts (unlike the standard gen histos, which have per-track cuts only)
    — i.e. gen vs reco is a matched (acceptance x efficiency) comparison.
    Fills, at the SAME per-track tiers as the ungated process_gen_arrays:
      - `_recosel` (pT>2, nominal) + bb/cc/light flavor split (`_recosel_{flav}`)
      - `_recosel_{cut,cut1,cut3,cut4,cut5,cut6,cut8}` (pT>0.4/1/3/4/5/6/8, incl)
      - `_recosel_{pmag2..8}` (pT>0.4 AND |p|>2/3/4/5/6/8, inclusive)
    all with `_cos` twins. Flavor is only at the pT>2 nominal (as in the ungated
    set). `_recosel` == the ungated `_cut2`, reco-gated. Every INCLUSIVE tier
    also carries the per-charge `_pos`/`_neg` FBCEC/FBCC components (via
    charge_decomp=True) — the route-A per-charge efficiency-correction inputs for
    a downstream correction that uses this reco-gated gen. Flavor stays combined."""
    is_charged = np.abs(q) > 0.5
    # Gen total energy sums for reco-selected events: E_vis-style (charged
    # pT>0.4 + neutral 0.5<E<50) and correlator-style (charged pT>2), 20-160 deg.
    _pt = np.sqrt(px**2 + py**2)
    _th = np.arctan2(_pt, pz)
    _acc = (_th > np.deg2rad(20)) & (_th < np.deg2rad(160))
    _vis = _acc & ((is_charged & (_pt > 0.4)) | ((~is_charged) & (e > 0.5) & (e < 50.0)))
    gen_hists["SumE_recosel"].Fill(float(e[_vis].sum()))
    gen_hists["SumE_pt2_recosel"].Fill(float(e[_acc & is_charged & (_pt > 2.0)].sum()))
    # sqrt(s') in reco-selected events: shows the selection cutting the hard-ISR
    # tail. Deliberately NOT applied to SumE_recosel above, which stays a
    # fiducial VISIBLE sum (neutrinos out) so it stays comparable to reco evis.
    if z_4mom is not None:
        gen_hists["sqrt_sprime_recosel"].Fill(float(z_4mom[1]))
    if not np.any(is_charged):
        return
    px_c = px[is_charged]; py_c = py[is_charged]; pz_c = pz[is_charged]
    q_c = q[is_charged];   e_c = e[is_charged]
    pt_c = np.sqrt(px_c**2 + py_c**2)
    p_mag = np.sqrt(px_c**2 + py_c**2 + pz_c**2)
    th_c = np.arctan2(pt_c, pz_c)
    cos_angle = np.divide(pz_c, p_mag, out=np.zeros_like(pz_c), where=p_mag > 0)
    angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    theta_accept = (th_c > np.deg2rad(20)) & (th_c < np.deg2rad(160))

    # Nominal pT>2 recosel (+ bb/cc/light flavor split) — the matched partner
    # of the reco `_p2` sample. `_recosel` == the ungated `_cut2`, reco-gated;
    # `_recosel_cut2` is the identical duplicate that keeps the `_cutN` tier
    # series uniform. Flavor split is on the bare `_recosel` only.
    sel = theta_accept & (pt_c > 2.0)
    if np.any(sel):
        for _p in ("_recosel", "_recosel_cut2"):
            fill_observables(angle_deg[sel], e_c[sel], q_c[sel], gen_hists,
                             prefix=_p, do_2d=False, fbcec_only=True,
                             charge_decomp=True)
        if flavor is not None:
            fill_observables(angle_deg[sel], e_c[sel], q_c[sel], gen_hists,
                             prefix=f"_recosel_{flavor}", do_2d=False, fbcec_only=True)

    # Same reco-event gate at the OTHER pT and |p| tiers (inclusive only). Masks
    # MUST match process_gen_arrays exactly (theta 20-160, then the pT or |p|
    # threshold). Each filled independently so a tier isn't skipped when a
    # higher-pT tier is empty.
    for pfx, m in (
            ("_recosel_cut",   theta_accept & (pt_c > 0.4)),
            ("_recosel_cut1",  theta_accept & (pt_c > 1.0)),
            ("_recosel_cut3",  theta_accept & (pt_c > 3.0)),
            ("_recosel_cut4",  theta_accept & (pt_c > 4.0)),
            ("_recosel_cut5",  theta_accept & (pt_c > 5.0)),
            ("_recosel_cut6",  theta_accept & (pt_c > 6.0)),
            ("_recosel_cut8",  theta_accept & (pt_c > 8.0)),
            ("_recosel_pmag2", theta_accept & (pt_c > 0.4) & (p_mag > 2.0)),
            ("_recosel_pmag3", theta_accept & (pt_c > 0.4) & (p_mag > 3.0)),
            ("_recosel_pmag4", theta_accept & (pt_c > 0.4) & (p_mag > 4.0)),
            ("_recosel_pmag5", theta_accept & (pt_c > 0.4) & (p_mag > 5.0)),
            ("_recosel_pmag6", theta_accept & (pt_c > 0.4) & (p_mag > 6.0)),
            ("_recosel_pmag8", theta_accept & (pt_c > 0.4) & (p_mag > 8.0)),
    ):
        if np.any(m):
            fill_observables(angle_deg[m], e_c[m], q_c[m], gen_hists,
                             prefix=pfx, do_2d=False, fbcec_only=True,
                             charge_decomp=True)


def _fill_qed_event_diag(h, stage, e_all, e_c, sum_px, sum_py, sum_pz, theta_thrust, roots):
    """Fill the per-event QED-rejection / on-Z-pole diagnostics for one stage.

    stage='presel' is filled for every on-pole event reaching track selection
    (BEFORE the hadronic event cut); stage='postsel' only for events passing
    apply_event_selection_delphi. Overlaying the two shows the cut stripping
    the low-E_vis (gamma-gamma / two-photon, beam-gas) and boosted (hard
    single-photon ISR) populations, leaving a sample peaked at sqrt(s) = M_Z.

    E_vis and the longitudinal boost are built from the SAME selected
    charged+neutral set (20-160 deg) that apply_event_selection_delphi cuts on,
    so `evis_presel` is exactly the quantity compared to 0.5*sqrt(s). A genuine
    Z->hadrons event is produced at rest (sum_pz ~ 0); a hard ISR photon
    collinear with the beam escapes down the beampipe and boosts the visible
    system, so boost = sum_pz/E_vis fingerprints radiative events. `h` is the
    nominal reco-hist dict; `roots` is the per-event sqrt(s) (Event_cmEnergy).
    """
    E_vis = float(np.sum(e_all))
    E_ch  = float(np.sum(e_c))
    n_ch  = len(e_c)
    # Effective sqrt(s') = invariant mass of the visible system. A hard ISR
    # photon collinear with the beam escapes down the beampipe, so the visible
    # (hadronic) system is left with M_vis < M_Z -> sqrt(s') sits below the pole.
    p2 = sum_px * sum_px + sum_py * sum_py + sum_pz * sum_pz
    sprime = float(np.sqrt(max(E_vis * E_vis - p2, 0.0)))
    h[f'evis_{stage}'].Fill(E_vis)
    h[f'sprime_{stage}'].Fill(sprime)
    h[f'nch_{stage}'].Fill(n_ch)
    h[f'costhrust_{stage}'].Fill(float(np.cos(theta_thrust)))
    h[f'nch_vs_evis_{stage}'].Fill(n_ch, E_vis)
    if roots and roots > 0:
        h[f'evis_over_roots_{stage}'].Fill(E_vis / roots)
        h[f'echarged_over_roots_{stage}'].Fill(E_ch / roots)
        h[f'sprime_over_roots_{stage}'].Fill(sprime / roots)
    if E_vis > 0:
        b = float(sum_pz) / E_vis
        h[f'boost_{stage}'].Fill(b)
        h[f'absboost_{stage}'].Fill(abs(b))


def process_reco_arrays(E, px, py, pz, m, q, pt, th, d0, z0,
                        systematics, reco_hists, counter,
                        cov_accum=None, do_2d=False,
                        gen_data=None, vtx_rw=None, period=None,
                        dpp=None, smearer=None, rng=None, flavor=None):
    """Pure-numpy reco-level processor. Callable from either the TPCNtuple
    tree wrapper above or the NanoAOD reader once it has built per-particle
    arrays from Part_* + Trac_* collections."""
    if len(px) == 0:
        return False

    # Nominal reco event-selection decision, returned so the caller can gate the
    # gen-event-selected histograms (gen FBCC carrying the reco-level event cut).
    nominal_pass = False

    for syst_name, syst_config in systematics.items():

        # Apply systematic variation
        varied = apply_systematic_variation(
            px, py, pz, m, q, pt, th, d0, z0, syst_config, dpp=dpp,
            smearer=smearer, rng=rng,
        )
        px_v, py_v, pz_v, m_v, q_v = (
            varied['px'], varied['py'], varied['pz'], varied['m'], varied['q']
        )
        pt_v, th_v, d0_v, z0_v = (
            varied['pt'], varied['th'], varied['d0'], varied['z0']
        )
        dpp_v = varied['dpp']  # may be None when caller has no dpp info

        if len(px_v) == 0:
            continue

        # --- Track selection: baseline pT > 0.4 + IP acceptance ---
        # Baseline momentum cut is transverse pT (charged_pt_min). d0/z0
        # acceptance defaults to the tight 0.6/1.0 cm; `_d0z0loose` partners
        # carry charged_d0_max=4 / charged_z0sin_max=10 in their config.
        selection = apply_track_selection_delphi(
            px=px_v, py=py_v, pz=pz_v, m=m_v, q=q_v,
            th=th_v, pt=pt_v, d0=d0_v, z0=z0_v,
            charged_pt_min=syst_config.get('charged_pt_min', 0.4),
            charged_d0_max=syst_config.get('charged_d0_max', 0.6),
            charged_z0sin_max=syst_config.get('charged_z0sin_max', 1.0),
        )

        sel_all = selection['sel']
        if not np.any(sel_all):
            continue

        # Extract selected particles
        px_sel = px_v[sel_all]
        py_sel = py_v[sel_all]
        pz_sel = pz_v[sel_all]
        m_sel = m_v[sel_all]
        q_sel = q_v[sel_all]

        sel_c = np.abs(q_sel) > 0.1

        # Compute thrust for event selection
        e_sel = np.sqrt(px_sel**2 + py_sel**2 + pz_sel**2 + m_sel**2)
        p3_all = np.stack([px_sel, py_sel, pz_sel], axis=1)
        axis, T = thrust_axis_fast(p3_all, include_met=True)

        e_c = e_sel[sel_c]
        e_all = e_sel

        # Event selection
        theta_thrust = thrust_theta(axis, T, fold=False)
        pass_sel = apply_event_selection_delphi(
            e_c=e_c, e_n=e_all,
            theta_Tu=theta_thrust, E_reco=E
        )['pass_reco']

        # --- QED-rejection / on-Z-pole diagnostics: pre-selection stage ---
        # Filled for every on-pole event reaching here, BEFORE the event cut,
        # so the overlay with `_postsel` shows what the cut throws away.
        # Nominal only, once per event.
        _qed_spx = _qed_spy = _qed_spz = None
        if syst_name == 'nominal':
            nominal_pass = bool(pass_sel)
            _qed_spx = float(np.sum(px_sel))
            _qed_spy = float(np.sum(py_sel))
            _qed_spz = float(np.sum(pz_sel))
            _fill_qed_event_diag(reco_hists[syst_name], 'presel',
                                 e_all, e_c, _qed_spx, _qed_spy, _qed_spz,
                                 theta_thrust, E)

        if not pass_sel:
            # --- Reverse-selection control region (nominal only) ---
            # FBCC/FBCEC in the REJECTED events (QED-contaminated: gamma-gamma,
            # radiative, low-multiplicity). Compared to the selected-sample FBCC,
            # this shows the contamination is concentrated in what the cut
            # removes. Baseline charged set (pT>0.4 already in track selection);
            # fbcec_only keeps the booked inventory to the 4 combined histos.
            if syst_name == 'nominal' and np.any(sel_c):
                q_c_rev  = q_sel[sel_c]
                px_c_rev = px_sel[sel_c]
                py_c_rev = py_sel[sel_c]
                pz_c_rev = pz_sel[sel_c]
                p_mag_rev = np.sqrt(px_c_rev**2 + py_c_rev**2 + pz_c_rev**2)
                cos_rev = np.divide(pz_c_rev, p_mag_rev,
                                    out=np.zeros_like(pz_c_rev), where=p_mag_rev > 0)
                angle_deg_rev = np.degrees(np.arccos(np.clip(cos_rev, -1.0, 1.0)))
                fill_observables(angle_deg_rev, e_c, q_c_rev,
                                 reco_hists[syst_name], prefix='_revsel',
                                 do_2d=False, fbcec_only=True)
                reco_hists[syst_name]['n_revsel'].Fill(0.5)
            continue

        # --- Counter: event passes event selection (nominal only, once per event) ---
        if syst_name == 'nominal':
            counter.Fill(1.5)  # Pass EvtSel
            # QED-rejection diagnostics: post-selection stage (event passed).
            _fill_qed_event_diag(reco_hists[syst_name], 'postsel',
                                 e_all, e_c, _qed_spx, _qed_spy, _qed_spz,
                                 theta_thrust, E)

        # --- Impact parameter diagnostics (nominal + smear + dpp_lt_0p5) ---
        # Smear systs: d0_v / z0_v carry the smeared values, so the same fill
        # code produces smeared closure histograms (suffix `_smear_*`).
        # dpp_lt_0p5: the keep_mask drops bad-fit tracks before this block, so
        # what gets filled is the surviving-track d0/z0 (suffix `_dpp_lt_0p5`).
        # The v73 rw companion + 4D THnF stay nominal-only.
        if _is_diagnostic_eligible(syst_name):
            h = reco_hists[syst_name]
            sm_sfx = _diag_suffix(syst_name)
            is_charged_v = np.abs(q_v) > 0.1
            keep = sel_all & is_charged_v
            keep_idx = np.flatnonzero(keep)

            # Pre-compute per-track z0sin_charge weights once (nominal only,
            # MC). The v73 reweighter assumes nominal kinematics.
            w_zc_keep = None
            if syst_name == 'nominal' and vtx_rw is not None and len(keep_idx) > 0:
                pt_keep        = pt_v[keep]
                theta_deg_keep = np.degrees(th_v[keep])
                z0sin_keep     = z0_v[keep] * np.sin(th_v[keep])
                q_keep         = q_v[keep]
                w_zc_keep = vtx_rw.weights(pt_keep, theta_deg_keep, z0sin_keep, q_keep)

            for jj, i in enumerate(keep_idx):
                sin_th_i = np.sin(th_v[i])
                z0sin_i = z0_v[i] * sin_th_i
                d0_i = d0_v[i]
                theta_deg_i = np.degrees(th_v[i])
                pt_i = pt_v[i]

                sign = "pos" if q_v[i] > 0 else "neg"

                h[f'd0_vtx_all{sm_sfx}'].Fill(d0_i)
                h[f'd0_vtx_{sign}{sm_sfx}'].Fill(d0_i)
                h[f'z0_vtx_all{sm_sfx}'].Fill(z0_v[i])
                h[f'z0_vtx_{sign}{sm_sfx}'].Fill(z0_v[i])
                h[f'z0sin_vtx_all{sm_sfx}'].Fill(z0sin_i)
                h[f'z0sin_vtx_{sign}{sm_sfx}'].Fill(z0sin_i)

                h[f'd0_vtx_rms_vs_theta{sm_sfx}'].Fill(theta_deg_i, d0_i)
                h[f'd0_vtx_rms_vs_pt{sm_sfx}'].Fill(pt_i, d0_i)
                h[f'z0sin_vtx_rms_vs_theta{sm_sfx}'].Fill(theta_deg_i, z0sin_i)
                h[f'z0sin_vtx_rms_vs_pt{sm_sfx}'].Fill(pt_i, z0sin_i)

                if syst_name == 'nominal':
                    # 4D for IP-reweighting study (pT x θ x d0 x z0sinθ).
                    # pT clipped to <50 so high-pT tracks fold into the last
                    # visible bin [8,50] rather than going to THn overflow.
                    pt_clip = min(pt_i, 49.999)
                    vtx4d_vals = array('d', [pt_clip, theta_deg_i, d0_i, z0sin_i])
                    h[f'vtx_pt_theta_d0_z0sin_{sign}_4d'].Fill(vtx4d_vals)

                    # Reweighted d0/z0sin (closure / sanity check, MC only)
                    if w_zc_keep is not None:
                        wi = float(w_zc_keep[jj])
                        h['d0_vtx_all_rw_z0sin_charge'].Fill(d0_i, wi)
                        h[f'd0_vtx_{sign}_rw_z0sin_charge'].Fill(d0_i, wi)
                        h['z0sin_vtx_all_rw_z0sin_charge'].Fill(z0sin_i, wi)
                        h[f'z0sin_vtx_{sign}_rw_z0sin_charge'].Fill(z0sin_i, wi)

        # Extract charged tracks
        q_c = q_sel[sel_c]
        px_c = px_sel[sel_c]
        py_c = py_sel[sel_c]
        pz_c = pz_sel[sel_c]

        # Compute angle, momentum, phi
        p_mag = np.sqrt(px_c**2 + py_c**2 + pz_c**2)
        pt_c = np.sqrt(px_c**2 + py_c**2)
        cos_angle = np.divide(pz_c, p_mag, out=np.zeros_like(pz_c), where=p_mag > 0)
        angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
        phi_deg = np.degrees(np.arctan2(py_c, px_c))

        # Per-charged-track impact parameters for vertex reweighting
        # (only used when the reweighter is wired in)
        d0_c = d0_v[sel_all][sel_c]
        z0sin_c = z0_v[sel_all][sel_c] * np.sin(np.radians(angle_deg))
        # Per-charged-track dpp for the tighter-dpp study (nominal only).
        # None when caller has no dpp info; -1 entries are invalid fits.
        dpp_c = dpp_v[sel_all][sel_c] if dpp_v is not None else None

        # --- Counter: pT threshold availability (nominal only, once per event) ---
        if syst_name == 'nominal':
            if np.any(pt_c > 1.0):
                counter.Fill(2.5)  # Pass EvtSel + pT>1
            if np.any(pt_c > 2.0):
                counter.Fill(3.5)  # Pass EvtSel + pT>2
            if np.any(pt_c > 4.0):
                counter.Fill(4.5)  # Pass EvtSel + pT>4
            if np.any(p_mag > 2.0):
                counter.Fill(5.5)  # Pass EvtSel + |p|>2
            if np.any(p_mag > 4.0):
                counter.Fill(6.5)  # Pass EvtSel + |p|>4

        s = syst_name
        h = reco_hists[s]

        # --- Loop over momentum cuts ---
        for pcut_label, pcut_val, pcut_type in P_CUTS:
            if pcut_type == "p":
                sel_p = p_mag > pcut_val
            elif pcut_val > 0.4:
                sel_p = pt_c > pcut_val
            else:
                sel_p = np.ones(len(pt_c), dtype=bool)

            if not np.any(sel_p):
                continue

            tag = f"{pcut_label}_{s}"

            # Per-track z0sin_charge weight (MC nominal only) — single 3D
            # mode w(pT, θ, z0sinθ, q). Feeds fill_observables for the
            # per-track-quantity `_rw_z0sin_charge` companions and is also
            # used directly for the FBCEC/FBCC `_rw_z0sin_charge` fills below.
            w_zc = None
            if vtx_rw is not None and syst_name == 'nominal':
                w_zc = vtx_rw.weights(
                    pt_c[sel_p],
                    angle_deg[sel_p],
                    z0sin_c[sel_p],
                    q_c[sel_p],
                )

            fill_observables(
                angle_deg[sel_p], e_c[sel_p], q_c[sel_p], h,
                prefix=tag, p_c=p_mag[sel_p], pt_c=pt_c[sel_p],
                phi_c=phi_deg[sel_p], do_2d=do_2d,
                w_z0sin_charge=w_zc,
            )
            h[f"NChargedSele{tag}"].Fill(np.sum(sel_p))
            h[f"NPartSele{tag}"].Fill(len(px_sel))
            h[f"SumESele{tag}"].Fill(np.sum(e_all))

            # Primary Z->qqbar flavor split at the pT>2 nominal-downstream cut
            # (nominal, MC): combined FBCEC/FBCC tagged by truth flavor. Sums to
            # the pT>2 inclusive FBCC_forward_p2_nominal.
            if flavor is not None and syst_name == 'nominal' and pcut_label == "_p2":
                fill_observables(angle_deg[sel_p], e_c[sel_p], q_c[sel_p], h,
                                 prefix=f"_{flavor}", do_2d=False, fbcec_only=True)

            # Vertex-reweighted FBCEC/FBCC variants (MC, nominal only).
            if w_zc is not None:
                fill_reweighted_observables(
                    angle_deg[sel_p], e_c[sel_p], q_c[sel_p],
                    w_zc, h, prefix=tag,
                )

            # --- Tighter-dpp variant FBCEC/FBCC fills (nominal only) ---
            # Tracks with dpp < 0 are invalid fits and are excluded by the
            # `dpp_c >= 0` clause (equivalently they fail every cut here).
            if syst_name == 'nominal' and dpp_c is not None:
                for dpp_max, dpp_tag in DPP_CUTS:
                    sel_dpp = sel_p & (dpp_c >= 0) & (dpp_c < dpp_max)
                    if not np.any(sel_dpp):
                        continue
                    dpp_pcut_tag = f"{pcut_label}{dpp_tag}_{s}"
                    fill_observables(
                        angle_deg[sel_dpp], e_c[sel_dpp], q_c[sel_dpp], h,
                        prefix=dpp_pcut_tag,
                        p_c=p_mag[sel_dpp], pt_c=pt_c[sel_dpp],
                        phi_c=phi_deg[sel_dpp], do_2d=do_2d,
                        fbcec_only=True,
                    )

            # --- Gen-reco matching for matched/fake decomposition (MC) ---
            # Run the matching for nominal + smear systs and their _dpp0p5
            # partners; the per-syst varied/filtered track set drives the
            # per-species matched/chargeOK/chargeMisID histograms.
            if gen_data is not None and _is_diagnostic_eligible(syst_name):
                # Build reco P4Block for tracks passing this pT cut
                px_sel_p = px_c[sel_p]
                py_sel_p = py_c[sel_p]
                pz_sel_p = pz_c[sel_p]
                q_sel_p = q_c[sel_p]
                pt_sel_p = pt_c[sel_p]
                th_sel_p = np.arctan2(pt_sel_p, pz_sel_p)
                phi_sel_p = np.arctan2(py_sel_p, px_sel_p)
                e_sel_p = e_c[sel_p]
                angle_sel_p = angle_deg[sel_p]

                rec_block = P4Block.build(
                    px_sel_p, py_sel_p, pz_sel_p, q_sel_p,
                    pt_sel_p, th_sel_p, phi_sel_p, e_sel_p
                )

                # Build gen P4Block: charged, with pT and theta cuts matching this pT threshold
                gpx = gen_data['px']
                gpy = gen_data['py']
                gpz = gen_data['pz']
                gq  = gen_data['q']
                ge  = gen_data['e']
                # Species code per gen particle (0=other, 1=pi, 2=K, 3=p);
                # loader fills zeros when convention unknown.
                gspecies = gen_data.get('species')
                if gspecies is None:
                    gspecies = np.zeros(len(gq), dtype=np.int8)

                g_is_charged = np.abs(gq) > 0.5
                gpx_c = gpx[g_is_charged]
                gpy_c = gpy[g_is_charged]
                gpz_c = gpz[g_is_charged]
                gq_c = gq[g_is_charged]
                ge_c = ge[g_is_charged]
                gsp_c = gspecies[g_is_charged]
                gpt_c = np.sqrt(gpx_c**2 + gpy_c**2)
                gth_c = np.arctan2(gpt_c, gpz_c)
                gphi_c = np.arctan2(gpy_c, gpx_c)
                gp_mag_c = np.sqrt(gpx_c**2 + gpy_c**2 + gpz_c**2)
                gcos_c = np.divide(gpz_c, gp_mag_c, out=np.zeros_like(gpz_c), where=gp_mag_c > 0)
                gangle_deg_c = np.degrees(np.arccos(np.clip(gcos_c, -1.0, 1.0)))

                # Apply gen-level track cuts: theta [20,160] deg + pT>0.4
                # baseline + momentum threshold (matches the reco pT baseline).
                g_theta_accept = (gth_c > np.deg2rad(20)) & (gth_c < np.deg2rad(160))
                if pcut_type == "p":
                    g_sel = g_theta_accept & (gpt_c > 0.4) & (gp_mag_c > pcut_val)
                else:
                    g_sel = g_theta_accept & (gpt_c > pcut_val)

                if np.any(g_sel):
                    gen_block = P4Block.build(
                        gpx_c[g_sel], gpy_c[g_sel], gpz_c[g_sel], gq_c[g_sel],
                        gpt_c[g_sel], gth_c[g_sel], gphi_c[g_sel], ge_c[g_sel]
                    )
                    gq_g_sel = gq_c[g_sel]
                    gsp_g_sel = gsp_c[g_sel]  # species code aligned to gen_block

                    # Match: same charge enforced by match_angular
                    ireco, igen, imiss, ifake = match_angular(rec_block, gen_block, MATCH_DR)

                    # Fill matched reco tracks
                    for k, ri in enumerate(ireco):
                        sign = "pos" if q_sel_p[ri] > 0 else "neg"
                        ang = angle_sel_p[ri]
                        h[f"TrackTheta_{sign}_matched{tag}"].Fill(ang)
                        if w_zc is not None:
                            h[f"TrackTheta_{sign}_matched_rw_z0sin_charge{tag}"].Fill(
                                ang, float(w_zc[ri]))
                        # per-species split of the gen partner (0=other → skip)
                        sp_code = int(gsp_g_sel[igen[k]])
                        if sp_code > 0:
                            sp = SPECIES_NAMES[sp_code]
                            h[f"TrackTheta_{sign}_matched_{sp}{tag}"].Fill(ang)
                            if w_zc is not None:
                                h[f"TrackTheta_{sign}_matched_{sp}_rw_z0sin_charge{tag}"].Fill(
                                    ang, float(w_zc[ri]))

                    # Fill fake reco tracks (no gen match)
                    for fi in ifake:
                        sign = "pos" if q_sel_p[fi] > 0 else "neg"
                        h[f"TrackTheta_{sign}_fake{tag}"].Fill(angle_sel_p[fi])
                        if w_zc is not None:
                            h[f"TrackTheta_{sign}_fake_rw_z0sin_charge{tag}"].Fill(
                                angle_sel_p[fi], float(w_zc[fi]))

                    # Angular-only matching (no charge requirement) for charge mis-ID
                    ireco_nc, igen_nc, _, _ = match_angular(
                        rec_block, gen_block, MATCH_DR, require_same_charge=False)

                    for ri, gi in zip(ireco_nc, igen_nc):
                        sign = "pos" if q_sel_p[ri] > 0 else "neg"
                        kind = 'chargeOK' if q_sel_p[ri] * gq_g_sel[gi] > 0 else 'chargeMisID'
                        ang = angle_sel_p[ri]
                        h[f"TrackTheta_{sign}_{kind}{tag}"].Fill(ang)
                        if w_zc is not None:
                            h[f"TrackTheta_{sign}_{kind}_rw_z0sin_charge{tag}"].Fill(
                                ang, float(w_zc[ri]))
                        sp_code = int(gsp_g_sel[gi])
                        if sp_code > 0:
                            sp = SPECIES_NAMES[sp_code]
                            h[f"TrackTheta_{sign}_{kind}_{sp}{tag}"].Fill(ang)
                            if w_zc is not None:
                                h[f"TrackTheta_{sign}_{kind}_{sp}_rw_z0sin_charge{tag}"].Fill(
                                    ang, float(w_zc[ri]))
                else:
                    # No gen tracks pass cuts — all reco are fakes
                    for i in range(len(q_sel_p)):
                        sign = "pos" if q_sel_p[i] > 0 else "neg"
                        h[f"TrackTheta_{sign}_fake{tag}"].Fill(angle_sel_p[i])
                        if w_zc is not None:
                            h[f"TrackTheta_{sign}_fake_rw_z0sin_charge{tag}"].Fill(
                                angle_sel_p[i], float(w_zc[i]))

        # Covariance accumulation (all momentum cuts).
        #  - nominal: pass dpp_c so the function also runs the dpp<0.5 partner
        #    pass into `*_dpp0p5` keys.
        #  - nominal_d0z0loose: these tracks are already loose-IP-selected;
        #    accumulate into `*_d0z0loose` keys (dpp_c=None → no dpp cross).
        if cov_accum is not None and syst_name == 'nominal':
            _accumulate_covariance(angle_deg, e_c, q_c, p_mag, pt_c, cov_accum,
                                   dpp_c=dpp_c)
        elif cov_accum is not None and syst_name == f'nominal{D0Z0_VARIANT_TAG}':
            _accumulate_covariance(angle_deg, e_c, q_c, p_mag, pt_c, cov_accum,
                                   dpp_c=None, key_base_suffix=D0Z0_VARIANT_TAG)

    return nominal_pass


# ============================================================================
# COVARIANCE
# ============================================================================

COV_OBS_NAMES = ("FBCEC", "FBCC")
COV_DIRECTIONS = ("forward",)  # backward dropped — forward measurement only
COV_BINNINGS = ("", "_cos")  # parallel theta-binned and cos(theta)-binned
COV_BLOCKS = ("pp", "mm", "pm")  # per-charge covariance blocks (route A)


def cov_1d_keys():
    """Base keys for the per-charge FORWARD covariance. For each key,
    create_cov_accumulators allocates three blocks (pp/mm/pm) built from the
    per-event per-charge bin vectors n^+(b), n^-(b). Variants: bare
    (-> *_nominal), `_dpp0p5`, `_d0z0loose`. Backward is no longer measured."""
    keys = []
    for pcut_label, _, _ in P_CUTS:
        for obs in COV_OBS_NAMES:
            for binning in COV_BINNINGS:
                base = f"{obs}_forward{binning}{pcut_label}"
                keys.append(base)
                keys.append(f"{base}{DPP_VARIANT_TAG}")
                keys.append(f"{base}{D0Z0_VARIANT_TAG}")
    return keys


def _cov_nbins_for_key(key):
    """Per-key bin count: cos-binned keys use len(cos_theta_bins)-1, others
    len(theta_bins)-1. (`_cos` survives any variant suffix, so a plain
    membership test is enough.)"""
    return (len(cos_theta_bins) - 1) if "_cos" in key else (len(theta_bins) - 1)


def create_cov_accumulators():
    """Per base key -> {'pp','mm','pm'} blocks (route A, per-charge covariance).
    Each block is NxN (theta) or MxM (cos). pp/mm are symmetric outer-product
    sums of n^+/n^-; pm is the (asymmetric) n^+ x n^- cross sum."""
    out = {}
    for k in cov_1d_keys():
        n = _cov_nbins_for_key(k)
        out[k] = {b: np.zeros((n, n)) for b in COV_BLOCKS}
    return out


def _accumulate_covariance(angle_deg, e_c, q_c, p_mag, pt_c, cov_accum, dpp_c=None,
                           key_base_suffix=""):
    """Accumulate per-event FBCEC and FBCC into covariance running sums for
    both theta- and cos(theta)-binned histograms.

    When `dpp_c` is provided, also runs a second accumulation pass with the
    `dpp >= 0 & dpp < DPP_VARIANT_MAX` tracking-quality cut, writing into the
    `<base>_dpp0p5` keys whose cov is finalized into the `*_nominal_dpp0p5`
    histograms.

    `key_base_suffix` is appended to every base key before the dpp suffix —
    used by the loose-IP variant (`_d0z0loose`), which calls this with the
    already-loose-selected tracks and dpp_c=None (no dpp cross), so it writes
    into `<base>_d0z0loose` keys → `*_nominal_d0z0loose` histograms.

    Per-charge (route A): instead of collapsing + and - into one signed vector,
    we accumulate the three blocks pp = Σ n⁺⊗n⁺, mm = Σ n⁻⊗n⁻, pm = Σ n⁺⊗n⁻,
    where n^±(b) is the per-event per-charge weighted bin content (weight = E for
    FBCEC, 1 for FBCC). finalize_errors propagates these through the per-charge
    efficiencies downstream. Forward only."""
    nBins_th = len(theta_bins) - 1
    nBins_co = len(cos_theta_bins) - 1
    # forward theta bin indices (uniform 1° bins, edges 0..180)
    bin_th = np.clip(np.floor(angle_deg).astype(int), 0, nBins_th - 1)
    # forward cos(theta) bin indices: uniform inside [cos_lo, cos_hi); out-of-range
    # tracks get bin = -1 and are dropped (matches TH1D under/overflow at fill).
    cos_lo = float(cos_theta_bins[0])
    cos_hi = float(cos_theta_bins[-1])
    inv_w  = nBins_co / (cos_hi - cos_lo)
    cos_fwd = np.cos(np.radians(angle_deg))
    in_co   = (cos_fwd >= cos_lo) & (cos_fwd < cos_hi)
    bin_co  = np.where(in_co,
                       np.floor((cos_fwd - cos_lo) * inv_w).astype(int), -1)

    is_pos = q_c > 0
    is_neg = q_c < 0
    # per-charge weight magnitude (charge sign selects the +/- bucket)
    wmag = {"FBCEC": e_c, "FBCC": np.ones(len(q_c))}

    # Per-pcut base masks (boolean so they compose with the optional dpp filter).
    pcut_base_masks = {}
    for pcut_label, pcut_val, pcut_type in P_CUTS:
        if pcut_label == "":
            pcut_base_masks[pcut_label] = np.ones(len(q_c), dtype=bool)
        elif pcut_type == "p":
            pcut_base_masks[pcut_label] = p_mag > pcut_val
        else:
            pcut_base_masks[pcut_label] = pt_c > pcut_val

    # Two passes: bare nominal cov, and dpp<DPP_VARIANT_MAX partner.
    # dpp = -1 marks an invalid fit and is excluded from the dpp pass too.
    passes = [("", None)]
    if dpp_c is not None:
        passes.append((DPP_VARIANT_TAG, (dpp_c >= 0) & (dpp_c < DPP_VARIANT_MAX)))

    for key_suffix, dpp_mask in passes:
        for pcut_label, _, _ in P_CUTS:
            mask = pcut_base_masks[pcut_label]
            if dpp_mask is not None:
                mask = mask & dpp_mask
            mpos = mask & is_pos
            mneg = mask & is_neg
            if not (mpos.any() or mneg.any()):
                continue
            for obs_name, wm in wmag.items():
                for binning, bins, nB in [("",     bin_th, nBins_th),
                                          ("_cos", bin_co, nBins_co)]:
                    # per-charge weighted bin vectors (cos: drop out-of-range)
                    vp = np.zeros(nB)
                    vn = np.zeros(nB)
                    bp, wp = bins[mpos], wm[mpos]
                    bn, wn = bins[mneg], wm[mneg]
                    if binning == "_cos":
                        kp = bp >= 0; bp, wp = bp[kp], wp[kp]
                        kn = bn >= 0; bn, wn = bn[kn], wn[kn]
                    if bp.size:
                        np.add.at(vp, bp, wp)
                    if bn.size:
                        np.add.at(vn, bn, wn)
                    key = f"{obs_name}_forward{binning}{pcut_label}{key_base_suffix}{key_suffix}"
                    blocks = cov_accum[key]
                    nzp = np.nonzero(vp)[0]
                    nzn = np.nonzero(vn)[0]
                    if nzp.size:
                        blocks['pp'][np.ix_(nzp, nzp)] += np.outer(vp[nzp], vp[nzp])
                    if nzn.size:
                        blocks['mm'][np.ix_(nzn, nzn)] += np.outer(vn[nzn], vn[nzn])
                    if nzp.size and nzn.size:
                        blocks['pm'][np.ix_(nzp, nzn)] += np.outer(vp[nzp], vn[nzn])


def finalize_errors(reco_hists, cov_accum, n_events, save_cov_matrices=True):
    """Per-charge covariance finalization for the FORWARD FBCEC/FBCC observables
    (route A). For each cov-target syst (nominal and its `_dpp0p5` / `_d0z0loose`
    partners) and each (obs, binning, pcut):

      - read the running block sums pp/mm/pm and the per-charge means
        n^+ , n^- (the `*_pos` / `*_neg` central-histogram contents),
      - form the per-charge covariance blocks (sum convention, matching the
        old Var(Q)=Σq² − (Σq)²/N estimator for the TOTAL over events):
            C_pp = Σ n⁺⊗n⁺ − (Σn⁺)(Σn⁺)ᵀ/N        (and mm, pm likewise),
      - set per-charge central-hist errors  √C_pp[b,b] / √C_mm[b,b],
      - set the COMBINED central-hist (N⁺−N⁻) error from the ε=1 combination
        Var = C_pp[b,b] + C_mm[b,b] − 2·C_pm[b,b]  (the correct stat error of
        the *uncorrected* difference),
      - when save_cov_matrices, store the three blocks as
        cov_<combined>_{pp,mm,pm} TH2Ds. Downstream forms the efficiency-
        corrected covariance as
            D₊C_ppD₊ + D₋C_mmD₋ − D₊C_pmD₋ − D₋C_pmᵀD₊,  D_± = diag(1/ε^±).
    """
    if n_events < 2:
        return

    # (accumulator-key variant suffix, syst dict name)
    variants = (("", 'nominal'),
                (DPP_VARIANT_TAG, f'nominal{DPP_VARIANT_TAG}'),
                (D0Z0_VARIANT_TAG, f'nominal{D0Z0_VARIANT_TAG}'))

    for variant, syst in variants:
        target = reco_hists.get(syst)
        if not target:
            continue
        for pcut_label, _, _ in P_CUTS:
            for obs in COV_OBS_NAMES:
                for binning in COV_BINNINGS:
                    acc_key = f"{obs}_forward{binning}{pcut_label}{variant}"
                    blocks = cov_accum.get(acc_key)
                    if blocks is None:
                        continue
                    tag = f"{pcut_label}_{syst}"
                    k_comb = f"{obs}_forward{binning}{tag}"
                    k_pos  = f"{obs}_forward{binning}_pos{tag}"
                    k_neg  = f"{obs}_forward{binning}_neg{tag}"
                    if k_pos not in target or k_neg not in target:
                        continue
                    hp, hn = target[k_pos], target[k_neg]
                    nB = blocks['pp'].shape[0]
                    mp = np.array([hp.GetBinContent(b + 1) for b in range(nB)])
                    mn = np.array([hn.GetBinContent(b + 1) for b in range(nB)])
                    C_pp = blocks['pp'] - np.outer(mp, mp) / n_events
                    C_mm = blocks['mm'] - np.outer(mn, mn) / n_events
                    C_pm = blocks['pm'] - np.outer(mp, mn) / n_events

                    # per-charge central-hist statistical errors
                    for b in range(nB):
                        hp.SetBinError(b + 1, np.sqrt(max(0.0, C_pp[b, b])))
                        hn.SetBinError(b + 1, np.sqrt(max(0.0, C_mm[b, b])))
                    # combined (uncorrected N⁺−N⁻) error, ε=1 case
                    if k_comb in target:
                        hc = target[k_comb]
                        for b in range(nB):
                            var_b = C_pp[b, b] + C_mm[b, b] - 2.0 * C_pm[b, b]
                            hc.SetBinError(b + 1, np.sqrt(max(0.0, var_b)))

                    if save_cov_matrices:
                        axis_bins = cos_theta_bins if binning == "_cos" else theta_bins
                        for blabel, mat in (('pp', C_pp), ('mm', C_mm), ('pm', C_pm)):
                            cov_name = f"cov_{k_comb}_{blabel}"
                            h2 = ROOT.TH2D(cov_name,
                                           f"Cov[{blabel}] for {k_comb}",
                                           nB, array('d', axis_bins),
                                           nB, array('d', axis_bins))
                            for a in range(nB):
                                row = mat[a]
                                for c in range(nB):
                                    h2.SetBinContent(a + 1, c + 1, row[c])
                            target[cov_name] = h2


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='FBCEC analysis (forward charge-weighted energy correlator) with systematics'
    )
    parser.add_argument('infile', help='Input ROOT file')
    parser.add_argument('outfile', help='Output ROOT file')
    parser.add_argument('--is-mc', action='store_true',
                        help='Process as MC (fills gen-level histograms)')
    parser.add_argument('--systematic', default='all',
                        help='Run specific systematic or "all" (default: all)')
    parser.add_argument('--compute-covariance', action='store_true', default=None,
                        help='Compute covariance matrix and 2D histograms (nominal only). '
                             'Default: enabled for data, disabled for MC.')
    parser.add_argument('--no-covariance', action='store_true',
                        help='Disable covariance computation even for data.')
    parser.add_argument('--period', choices=['auto', '94c', '95d'], default='auto',
                        help='Data-taking period for vertex reweighting; '
                             'auto-detects from infile path if "auto" (default).')
    parser.add_argument('--reweight-dir', default='.',
                        help='Directory containing vertex_reweight_<period>_v73_3d.root '
                             '(default: cwd).')

    args = parser.parse_args()

    # Determine systematics. Each base entry gets a `_dpp0p5` partner (dpp<0.5
    # cut variant) and a `_d0z0loose` partner (loose IP acceptance) — both cut
    # variants, not uncertainties, and additive (not crossed). See
    # build_systematics_to_run.
    if args.systematic == 'all':
        systematics_to_run = build_systematics_to_run(SYSTEMATICS)
    elif args.systematic in SYSTEMATICS:
        systematics_to_run = build_systematics_to_run(
            {args.systematic: SYSTEMATICS[args.systematic]})
    else:
        print(f"ERROR: Unknown systematic '{args.systematic}'")
        print(f"Available: {', '.join(SYSTEMATICS.keys())}")
        return 1

    # Default: covariance on for data, off for MC
    if args.no_covariance:
        do_2d = False
    elif args.compute_covariance:
        do_2d = True
    else:
        do_2d = not args.is_mc

    print(f"Running FBCEC analysis on: {args.infile}")
    print(f"MC mode: {args.is_mc}")
    print(f"Systematics: {', '.join(systematics_to_run.keys())}")
    print(f"Covariance + 2D: {do_2d}")

    # Vertex reweighter (MC only) — loads the v73 3D weight TH3F
    # (z0sin_vtx_pos/neg) from `vertex_reweight_<period>_v73_3d.root`.
    period = args.period if args.period != 'auto' else rw_detect_period(args.infile)
    vtx_rw = None
    if args.is_mc:
        if period is None:
            print("WARNING: cannot determine period from infile; "
                  "vertex reweighting disabled (no _rw_ histograms will be filled)")
        else:
            table_path = find_3d_table(period, search_dir=args.reweight_dir)
            if table_path is None:
                print(f"WARNING: vertex_reweight_{period}_v73_3d.root not found "
                      f"in {args.reweight_dir}; reweighting disabled")
            else:
                try:
                    vtx_rw = VertexReweighter3D(table_path)
                    print(f"Vertex reweighter loaded from {table_path}; period={period}")
                except Exception as exc:
                    print(f"WARNING: vertex reweighter disabled — {exc}")
                    vtx_rw = None

    # Open input
    fin = ROOT.TFile.Open(args.infile, 'READ')
    if not fin or fin.IsZombie():
        print(f"ERROR: Cannot open {args.infile}")
        return 1

    tree_reco = fin.Get('t')
    if not tree_reco:
        print("ERROR: Cannot find tree 't'")
        return 1

    tree_gen_before = None
    tree_gen = None
    if args.is_mc:
        tree_gen_before = fin.Get('tgenBefore')
        tree_gen = fin.Get('tgen')
        if not tree_gen_before or not tree_gen:
            print("ERROR: MC mode but cannot find 'tgenBefore' or 'tgen'")
            return 1

    n_events = tree_reco.GetEntries()
    print(f"Processing {n_events} events")

    # Create histograms
    counter = create_counter_histogram()

    gen_hists = None
    if args.is_mc:
        gen_hists = create_gen_histograms(do_2d=do_2d)

    reco_hists = {}
    for syst_name in systematics_to_run:
        reco_hists[syst_name] = create_reco_histograms(syst_name, do_2d=do_2d, is_mc=args.is_mc)

    # Covariance accumulators
    cov_accum = None
    if do_2d:
        cov_accum = create_cov_accumulators()

    # Event loop
    for iEvt in range(n_events):
        if iEvt % 1000 == 0:
            print(f"  Processing event {iEvt}/{n_events}")

        # Gen level (MC only)
        gen_data = None
        if args.is_mc and gen_hists is not None:
            process_gen_level(tree_gen_before, tree_gen, iEvt, gen_hists, do_2d=do_2d)
            # Also load gen data for matching in reco-level
            gen_data = load_gen_particles(tree_gen_before, tree_gen, iEvt)

        # Reco level
        counter.Fill(0.5)
        process_reco_level(
            tree_reco, iEvt, systematics_to_run,
            reco_hists, counter,
            cov_accum=cov_accum,
            do_2d=do_2d,
            gen_data=gen_data,
            vtx_rw=vtx_rw,
            period=period,
        )

    print(f"Processed {n_events} events")

    # Write output
    fout = ROOT.TFile(args.outfile, 'RECREATE')

    # Gen
    if gen_hists is not None:
        fout.mkdir("gen")
        fout.cd("gen")
        for h in gen_hists.values():
            h.Write()

    # Reco
    fout.mkdir("reco")
    fout.cd("reco")
    for syst_name, hists in reco_hists.items():
        for h in hists.values():
            h.Write()

    # Counter
    fout.cd()
    counter.Write()

    # Finalize covariance: set bin errors and save covariance matrices for
    # nominal AND its `_dpp0p5` / `_d0z0loose` partners (all populated by
    # _accumulate_covariance).
    if cov_accum is not None:
        finalize_errors(reco_hists, cov_accum, n_events, save_cov_matrices=True)
        # Re-write the touched hist dicts (now with correct errors + new cov TH2Ds)
        fout.cd("reco")
        for syst in cov_target_systs():
            for h in reco_hists.get(syst, {}).values():
                h.Write(h.GetName(), ROOT.TObject.kOverwrite)

    fout.Close()
    fin.Close()

    print(f"Output written to {args.outfile}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
"""V0 (K0s) Tag-and-Probe: Charge-Dependent Tracking Performance via V0 Decays
============================================================================
Uses uproot. Runs on treefy'd nanoaod (TTree 't').

For a K0s candidate, charge conservation forces the daughters to be (+, -).
If we pick the higher-quality daughter as the "tag" (with reliable charge),
the other daughter (probe) MUST have the opposite charge:

  tag=- => probe should be + : denominator tag_neg(theta), numerator probe_pos_given_tag_neg(theta)
  tag=+ => probe should be - : ... etc.

Charge mis-ID failure modes:
  same-sign     : tag=+,probe=+ or tag=-,probe=-  (sign was flipped)
  zero-charge   : probe Part_charge==0            (charge measurement lost)

Everything is binned vs probe theta, split by pT/pmag thresholds on the probe.
Momentum binning matches analysis_tau.py so the two measurements are directly
comparable.

Usage:
    python analysis_v0.py input_ttree.root output.root [--is-mc] [--max-events N]
"""

import ROOT
import uproot
import awkward as ak
import numpy as np
import argparse
import sys
from array import array


# ============================================================================
# CONSTANTS
# ============================================================================

M_PI  = 0.13957
M_P   = 0.93827
M_K0S = 0.49761
M_LAM = 1.11568

# K0s mass window for the tag-and-probe selection (chosen from the smoke
# spectrum we already have; widen later if needed).
K0S_WINDOW = (0.485, 0.510)

# "cleank0s" V0-quality cuts -- applied on top of the baseline track-pT cut.
# These suppress combinatorial background under the K0s peak. Default values
# came from a flight-significance scan on the Y13709.1 smoke sample; tighten
# if you need higher purity at the cost of statistics.
CLEANK0S_FLIGHT_MIN  = 4.0       # V0_xyFlightDistanceOverError > this
CLEANK0S_VTXPROB_MIN = 0.01      # V0_vertexChi2Prob          > this
LAM_VETO_LO          = 1.108     # reject V0s where either p-pi mass
LAM_VETO_HI          = 1.124     # falls inside the Lambda window

# Tau-parity track-quality cuts -- applied to the TAG only (the probe is
# unrestricted by design, so charge-loss / mis-ID failure modes of the probe
# are observable). Same numerical values as analysis_tau's "good charged
# track" definition. V0 daughters may have large d0 by construction (they
# come from a displaced vertex), but selecting tags within these cuts keeps
# the tag side directly comparable to the tau measurement.
TAG_D0_MAX     = 0.6   # |d0|             < 0.6 cm
TAG_Z0SIN_MAX  = 1.0   # |z0 * sin theta| < 1.0 cm


# ============================================================================
# BINNING (matched to analysis_tau.py)
# ============================================================================

THETA_BINS    = np.linspace(20, 160, 29)
P_BINS        = np.array([0.5, 1, 2, 3, 5, 8, 12, 20, 30, 45, 60], dtype=np.float64)
PT_CUTS       = [0.4, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 20.0]
PMAG_CUTS     = [1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 20.0]
PT_HIST_BINS  = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 5, 7, 10, 15, 20, 30, 45, 60], dtype=np.float64)


# ============================================================================
# LOAD
# ============================================================================

def load_all_events(infile, max_events=-1):
    f = uproot.open(infile)
    tree = f["t"]
    keys = tree.keys()

    # Resolve the renamed Part_fourMomentum branch (same logic as analysis_tau).
    px_key = None
    for c in ["Part_fourMomentum_fCoordinates_fX",
              "Part_fourMomentum.fCoordinates.fX",
              "Part_fourMomentum/Part_fourMomentum.fCoordinates.fX"]:
        if c in keys:
            px_key = c
            py_key = c.replace("fX", "fY")
            pz_key = c.replace("fX", "fZ")
            en_key = c.replace("fX", "fT")
            break
    if px_key is None:
        sys.exit(f"ERROR: cannot find Part_fourMomentum. Sample keys: {keys[:20]}")

    branches = [px_key, py_key, pz_key, en_key, "Part_charge", "Part_lock",
                "nV0", "V0_firstPartIdx", "V0_secondPartIdx",
                "V0_tagFlag", "V0_suggestedMass",
                "V0_xyFlightDistanceOverError", "V0_vertexChi2Prob",
                "nPhotonConv", "PhotonConv_firstDaughterIdx", "PhotonConv_secondDaughterIdx"]

    # Trac_* fields needed for dpp = sigma(p)/p (matches analysis_tau's tag election),
    # plus impact parameters for the tau-parity tag d0/z0 cuts. Vtx_* fields
    # for the V0 secondary-vertex position diagnostics (data/MC comparison).
    optional = ["GenPart_pdgId", "Trac_partIdx",
                "Trac_perigee_fArray", "Trac_weightMatrix_fRep_fArray",
                "Trac_impParToVertexRPhi", "Trac_impParToVertexZ",
                "Part_originVtxIdx",
                "Vtx_position_fCoordinates_fX",
                "Vtx_position_fCoordinates_fY",
                "Vtx_position_fCoordinates_fZ",
                "Vtx_status", "Vtx_nOutgoing"]
    for k in optional:
        if k in keys:
            branches.append(k)

    data = tree.arrays(branches, entry_stop=max_events if max_events > 0 else None)

    out = {
        "px": data[px_key], "py": data[py_key], "pz": data[pz_key], "en": data[en_key],
        "charge": data["Part_charge"],
        "lock":   data["Part_lock"],
        "nV0":     data["nV0"],
        "i1":      data["V0_firstPartIdx"],
        "i2":      data["V0_secondPartIdx"],
        "tagFlag": data["V0_tagFlag"],
        "sugMass": data["V0_suggestedMass"],
        "flight":  data["V0_xyFlightDistanceOverError"],
        "vtxProb": data["V0_vertexChi2Prob"],
        "nPhotonConv": data["nPhotonConv"],
        "convI1":  data["PhotonConv_firstDaughterIdx"],
        "convI2":  data["PhotonConv_secondDaughterIdx"],
    }
    for k in optional:
        if k in data.fields:
            out[k] = data[k]
    return out


def compute_d0z0_per_part(trac_partIdx, trac_d0, trac_z0, n_part):
    """Map Trac_impParToVertexRPhi / Trac_impParToVertexZ from Trac index to
    Part index. Part entries with no Trac counterpart get NaN.
    """
    d0 = np.full(n_part, np.nan, dtype=np.float64)
    z0 = np.full(n_part, np.nan, dtype=np.float64)
    ti = np.asarray(trac_partIdx, dtype=np.int32)
    if len(ti) == 0:
        return d0, z0
    valid = (ti >= 0) & (ti < n_part)
    d0[ti[valid]] = np.asarray(trac_d0, dtype=np.float64)[valid]
    z0[ti[valid]] = np.asarray(trac_z0, dtype=np.float64)[valid]
    return d0, z0


def compute_dpp_per_part(trac_partIdx, perigee_flat, weight_flat, n_part):
    """Compute dpp = sigma(p)/p for every Part_* index in an event.

    Mirrors analysis_tau.py:
      dpp = (1/sqrt(W11)) / |perigee[1]|     # curvature parameter and its weight
    Returns array of size n_part. Indices without a Trac entry, or with
    pathological fits, get -1.
    """
    out = np.full(n_part, -1.0, dtype=np.float64)
    n_trac = len(trac_partIdx)
    if n_trac == 0:
        return out
    per = np.asarray(perigee_flat, dtype=np.float64)
    wm  = np.asarray(weight_flat,  dtype=np.float64)
    if per.size != 5 * n_trac or wm.size != 15 * n_trac:
        return out
    per = per.reshape(n_trac, 5)
    wm  = wm.reshape(n_trac, 15)
    curv = per[:, 1]
    w_cc = wm[:, 1]
    ok = (np.abs(curv) > 1e-10) & (w_cc > 0)
    dpp_trac = np.full(n_trac, -1.0)
    dpp_trac[ok] = (1.0 / np.sqrt(w_cc[ok])) / np.abs(curv[ok])
    ti = np.asarray(trac_partIdx, dtype=np.int32)
    valid_ti = (ti >= 0) & (ti < n_part)
    out[ti[valid_ti]] = dpp_trac[valid_ti]
    return out


# ============================================================================
# KINEMATIC HELPERS
# ============================================================================

def inv_mass(px1, py1, pz1, m1, px2, py2, pz2, m2):
    e1 = np.sqrt(px1*px1 + py1*py1 + pz1*pz1 + m1*m1)
    e2 = np.sqrt(px2*px2 + py2*py2 + pz2*pz2 + m2*m2)
    px, py, pz = px1+px2, py1+py2, pz1+pz2
    s = (e1+e2)*(e1+e2) - (px*px + py*py + pz*pz)
    return np.sqrt(s) if s > 0 else 0.0


def kin(px, py, pz):
    p  = float(np.sqrt(px*px + py*py + pz*pz))
    pt = float(np.sqrt(px*px + py*py))
    theta = float(np.degrees(np.arctan2(pt, pz)))
    return p, pt, theta


# ============================================================================
# HISTOGRAMS
# ============================================================================

CUT_LEVELS = ['nocut', 'baseline', 'angle', 'highp', 'cleank0s']


def book_histograms():
    h = {}

    # ---- Cutflow ----
    h['cutflow'] = ROOT.TH1D('cutflow', '', 9, 0, 9)
    for i, l in enumerate(['Events', 'V0_total', 'V0_baseline', 'V0_angle',
                           'V0_highp', 'V0_cleank0s', 'V0_inK0Swindow',
                           'V0_tag_elected', 'TP_filled']):
        h['cutflow'].GetXaxis().SetBinLabel(i + 1, l)

    # ---- Mass spectra at multiple cut levels ----
    for tag in CUT_LEVELS:
        h[f'mpipi_{tag}_full'] = ROOT.TH1D(f'mpipi_{tag}_full',
            ';M(#pi^{+}#pi^{-}) [GeV];V0 candidates', 200, 0.0, 2.0)
        h[f'mpipi_{tag}_zoom'] = ROOT.TH1D(f'mpipi_{tag}_zoom',
            ';M(#pi^{+}#pi^{-}) [GeV];V0 candidates', 100, 0.40, 0.60)
        h[f'mppi_{tag}_full']  = ROOT.TH1D(f'mppi_{tag}_full',
            ';M(p#pi^{-}) [GeV];V0 candidates', 200, 0.0, 2.0)
        h[f'mppi_{tag}_zoom']  = ROOT.TH1D(f'mppi_{tag}_zoom',
            ';M(p#pi^{-}) [GeV];V0 candidates',  80, 1.05, 1.20)
        h[f'mpip_{tag}_full']  = ROOT.TH1D(f'mpip_{tag}_full',
            ';M(#pi^{+}#bar{p}) [GeV];V0 candidates', 200, 0.0, 2.0)
        h[f'mpip_{tag}_zoom']  = ROOT.TH1D(f'mpip_{tag}_zoom',
            ';M(#pi^{+}#bar{p}) [GeV];V0 candidates',  80, 1.05, 1.20)

    # 2D QA (no cut)
    h['mpipi_vs_flight'] = ROOT.TH2D('mpipi_vs_flight',
        ';xy flight / #sigma;M(#pi#pi) [GeV]', 50, 0, 50, 100, 0.40, 0.60)
    h['mpipi_vs_vtxprob'] = ROOT.TH2D('mpipi_vs_vtxprob',
        ';V0 vertex chi^{2} prob;M(#pi#pi) [GeV]', 50, 0, 1, 100, 0.40, 0.60)
    h['mpipi_vs_pK0s'] = ROOT.TH2D('mpipi_vs_pK0s',
        ';p(K0_{s}) [GeV];M(#pi#pi) [GeV]', 50, 0, 50, 100, 0.40, 0.60)
    h['mpipi_vs_thetaK0s'] = ROOT.TH2D('mpipi_vs_thetaK0s',
        ';#theta(K0_{s}) [deg];M(#pi#pi) [GeV]', len(THETA_BINS)-1, array('d', THETA_BINS),
        100, 0.40, 0.60)

    # ---- V0 secondary-vertex position (data/MC comparison) ----
    # Looked up via Part_originVtxIdx of either daughter (the V0 decay
    # vertex). Filled at two cut levels:
    #   _cleank0s : cleank0s cut (no mass window) — checks bulk V0 vertex shape
    #   _inwindow : cleank0s + K0s mass window — clean signal sample
    for level in ('cleank0s', 'inwindow'):
        h[f'v0_vtx_x_{level}']   = ROOT.TH1D(f'v0_vtx_x_{level}',
            ';V0 vertex x [cm];V0 candidates', 100, -20.0, 20.0)
        h[f'v0_vtx_y_{level}']   = ROOT.TH1D(f'v0_vtx_y_{level}',
            ';V0 vertex y [cm];V0 candidates', 100, -20.0, 20.0)
        h[f'v0_vtx_z_{level}']   = ROOT.TH1D(f'v0_vtx_z_{level}',
            ';V0 vertex z [cm];V0 candidates', 100, -50.0, 50.0)
        h[f'v0_vtx_rho_{level}'] = ROOT.TH1D(f'v0_vtx_rho_{level}',
            ';V0 vertex #rho = #sqrt{x^{2}+y^{2}} [cm];V0 candidates',
            100, 0.0, 50.0)
        h[f'v0_vtx_xy_{level}']  = ROOT.TH2D(f'v0_vtx_xy_{level}',
            ';V0 vertex x [cm];V0 vertex y [cm]',
            100, -20.0, 20.0, 100, -20.0, 20.0)
        h[f'v0_vtx_z_vs_rho_{level}'] = ROOT.TH2D(f'v0_vtx_z_vs_rho_{level}',
            ';V0 vertex #rho [cm];V0 vertex z [cm]',
            50, 0.0, 50.0, 100, -50.0, 50.0)

    # ---- Primary-vertex position (per-event, data/MC comparison) ----
    # Selected per event as the Vtx_* with Vtx_incomingIdx == -1 (no
    # parent particle = production vertex). If multiple match, take the
    # one with most outgoing tracks. Filled once per processed event
    # regardless of V0 selection — gives a beam-spot-drift diagnostic.
    h['pv_x']        = ROOT.TH1D('pv_x',
        ';primary vertex x [cm];events',  200, -1.0, 1.0)
    h['pv_y']        = ROOT.TH1D('pv_y',
        ';primary vertex y [cm];events',  200, -1.0, 1.0)
    h['pv_z']        = ROOT.TH1D('pv_z',
        ';primary vertex z [cm];events',  200, -5.0, 5.0)
    h['pv_rho']      = ROOT.TH1D('pv_rho',
        ';primary vertex #rho [cm];events', 100, 0.0, 1.0)
    h['pv_xy']       = ROOT.TH2D('pv_xy',
        ';primary vertex x [cm];primary vertex y [cm]',
        100, -1.0, 1.0, 100, -1.0, 1.0)
    h['pv_z_vs_rho'] = ROOT.TH2D('pv_z_vs_rho',
        ';primary vertex #rho [cm];primary vertex z [cm]',
         50,  0.0, 1.0, 100, -5.0, 5.0)
    h['pv_n_outgoing'] = ROOT.TH1D('pv_n_outgoing',
        ';primary vertex N_{outgoing};events', 100, 0, 100)
    h['n_pv_candidates'] = ROOT.TH1D('n_pv_candidates',
        ';# Vtx with incomingIdx == -1;events', 10, 0, 10)

    # ---- Tag-Probe conditional histograms ----
    # Tag denominators are vs probe theta (same convention as analysis_tau).
    # Probe charge has 3 categories (pos / neg / zero), tag charge has 2.
    for pcut in PT_CUTS:
        t = f"pt{pcut:.1f}".replace('.', 'p')
        h[f'tag_pos_{t}'] = ROOT.TH1D(f'tag_pos_{t}', '', len(THETA_BINS)-1, array('d', THETA_BINS))
        h[f'tag_neg_{t}'] = ROOT.TH1D(f'tag_neg_{t}', '', len(THETA_BINS)-1, array('d', THETA_BINS))
        for pc in ('pos', 'neg', 'zero'):
            for tc in ('pos', 'neg'):
                h[f'probe_{pc}_tag_{tc}_{t}'] = ROOT.TH1D(
                    f'probe_{pc}_tag_{tc}_{t}', '', len(THETA_BINS)-1, array('d', THETA_BINS))

    for pcut in PMAG_CUTS:
        t = f"pmag{pcut:.1f}".replace('.', 'p')
        h[f'tag_pos_{t}'] = ROOT.TH1D(f'tag_pos_{t}', '', len(THETA_BINS)-1, array('d', THETA_BINS))
        h[f'tag_neg_{t}'] = ROOT.TH1D(f'tag_neg_{t}', '', len(THETA_BINS)-1, array('d', THETA_BINS))
        for pc in ('pos', 'neg', 'zero'):
            for tc in ('pos', 'neg'):
                h[f'probe_{pc}_tag_{tc}_{t}'] = ROOT.TH1D(
                    f'probe_{pc}_tag_{tc}_{t}', '', len(THETA_BINS)-1, array('d', THETA_BINS))

    # ---- Inclusive kinematics (no pT/p threshold) ----
    for sign in ('pos', 'neg'):
        h[f'tag_pt_{sign}']    = ROOT.TH1D(f'tag_pt_{sign}',    f';p_{{T}} [GeV];tag ({sign})',
            len(PT_HIST_BINS)-1, array('d', PT_HIST_BINS))
        h[f'tag_p_{sign}']     = ROOT.TH1D(f'tag_p_{sign}',     f';p [GeV];tag ({sign})',
            len(PT_HIST_BINS)-1, array('d', PT_HIST_BINS))
        h[f'tag_theta_{sign}'] = ROOT.TH1D(f'tag_theta_{sign}', f';#theta [deg];tag ({sign})',
            len(THETA_BINS)-1, array('d', THETA_BINS))
    for sign in ('pos', 'neg', 'zero'):
        h[f'probe_pt_{sign}']    = ROOT.TH1D(f'probe_pt_{sign}',    f';p_{{T}} [GeV];probe ({sign})',
            len(PT_HIST_BINS)-1, array('d', PT_HIST_BINS))
        h[f'probe_p_{sign}']     = ROOT.TH1D(f'probe_p_{sign}',     f';p [GeV];probe ({sign})',
            len(PT_HIST_BINS)-1, array('d', PT_HIST_BINS))
        h[f'probe_theta_{sign}'] = ROOT.TH1D(f'probe_theta_{sign}', f';#theta [deg];probe ({sign})',
            len(THETA_BINS)-1, array('d', THETA_BINS))

    # ---- dpp QA (mirrors analysis_tau dpp histograms; K0s window only) ----
    h['dpp_all_v0']     = ROOT.TH1D('dpp_all_v0',
        ';#sigma(p)/p;V0 daughter tracks', 100, 0.0, 0.2)
    h['dpp_tag']        = ROOT.TH1D('dpp_tag',
        ';#sigma(p)/p;tag tracks',         100, 0.0, 0.2)
    h['dpp_probe']      = ROOT.TH1D('dpp_probe',
        ';#sigma(p)/p;probe tracks',       100, 0.0, 0.2)
    h['dpp_vs_theta_all_v0'] = ROOT.TH2D('dpp_vs_theta_all_v0',
        ';#theta [deg];#sigma(p)/p',
        len(THETA_BINS)-1, array('d', THETA_BINS), 100, 0.0, 0.2)
    h['dpp_vs_theta_tag']    = ROOT.TH2D('dpp_vs_theta_tag',
        ';#theta [deg];#sigma(p)/p (tag)',
        len(THETA_BINS)-1, array('d', THETA_BINS), 100, 0.0, 0.2)
    h['dpp_vs_theta_probe']  = ROOT.TH2D('dpp_vs_theta_probe',
        ';#theta [deg];#sigma(p)/p (probe)',
        len(THETA_BINS)-1, array('d', THETA_BINS), 100, 0.0, 0.2)
    h['dpp_status'] = ROOT.TH1D('dpp_status', ';status;K0s candidates', 4, 0, 4)
    for i, l in enumerate(['total_K0s', 'both_valid', 'one_invalid', 'both_invalid']):
        h['dpp_status'].GetXaxis().SetBinLabel(i + 1, l)

    return h


def fill_mass_spectra(h, level, m_pipi, m_ppi, m_pip):
    h[f'mpipi_{level}_full'].Fill(m_pipi)
    h[f'mpipi_{level}_zoom'].Fill(m_pipi)
    h[f'mppi_{level}_full' ].Fill(m_ppi)
    h[f'mppi_{level}_zoom' ].Fill(m_ppi)
    h[f'mpip_{level}_full' ].Fill(m_pip)
    h[f'mpip_{level}_zoom' ].Fill(m_pip)


def fill_tag_probe(h, tag_q, tag_pt, tag_p, tag_theta,
                      probe_q, probe_pt, probe_p, probe_theta):
    """Tag must have charge != 0; probe charge can be -1, 0, or +1."""
    ts = 'pos' if tag_q > 0 else 'neg'
    if probe_q > 0:
        ps = 'pos'
    elif probe_q < 0:
        ps = 'neg'
    else:
        ps = 'zero'

    # Inclusive kinematics
    h[f'tag_pt_{ts}'].Fill(tag_pt)
    h[f'tag_p_{ts}'].Fill(tag_p)
    h[f'tag_theta_{ts}'].Fill(tag_theta)
    h[f'probe_pt_{ps}'].Fill(probe_pt)
    h[f'probe_p_{ps}'].Fill(probe_p)
    h[f'probe_theta_{ps}'].Fill(probe_theta)

    for pcut in PT_CUTS:
        if probe_pt > pcut:
            t = f"pt{pcut:.1f}".replace('.', 'p')
            h[f'tag_{ts}_{t}'].Fill(probe_theta)
            h[f'probe_{ps}_tag_{ts}_{t}'].Fill(probe_theta)

    for pcut in PMAG_CUTS:
        if probe_pt > 0.4 and probe_p > pcut:
            t = f"pmag{pcut:.1f}".replace('.', 'p')
            h[f'tag_{ts}_{t}'].Fill(probe_theta)
            h[f'probe_{ps}_tag_{ts}_{t}'].Fill(probe_theta)


# ============================================================================
# TAG ELECTION
# ============================================================================
# Tag-eligibility per daughter -- track-quality cuts copied from
# analysis_tau.fill_tag_and_probe + tau "good charged track" definition:
#   - Part_lock == 0                         (tau "good track")
#   - dpp = sigma(p)/p valid                 (curvature fit converged)
#   - |d0|             < TAG_D0_MAX          (tau d0 cut)
#   - |z0 * sin theta| < TAG_Z0SIN_MAX       (tau z0 cut)
#
# Election: among the eligible daughters, smaller-dpp wins. If neither is
# eligible the V0 is skipped.
#
# NOT applied (intentional difference from tau):
#   - The tau "at-least-one-in-barrel" rule is a topology guard derived from
#     back-to-back tau pairs (acolinearity < 20 deg => other track in
#     [20, 160] when one is in [40, 140]). K0s daughters are boosted along
#     the K0s flight direction, not back-to-back, so no theta cut is applied
#     to the tag, the probe, or to the V0 as a whole.
#
# The probe also carries no extra cut: it can be charge==0 (neutral Part
# entry with no Trac), lock-flagged, or with any d0/z0 -- that's exactly the
# point of the efficiency / charge-mis-ID measurement.


def elect_tag(lock1, dpp1, d0_1, z0sin_1,
              lock2, dpp2, d0_2, z0sin_2):
    """Return 0 if daughter A is the tag, 1 if B, or -1 if no tag eligible.

    NaN d0/z0sin (no Trac entry, e.g. neutral Part) make the daughter
    tag-ineligible automatically.
    """
    def eligible(lock, dpp, d0, z0sin):
        if lock != 0 or dpp < 0:
            return False
        # NaN comparisons return False, so the np.isnan guards are explicit
        if np.isnan(d0) or np.isnan(z0sin):
            return False
        return abs(d0) < TAG_D0_MAX and z0sin < TAG_Z0SIN_MAX

    elig1 = eligible(lock1, dpp1, d0_1, z0sin_1)
    elig2 = eligible(lock2, dpp2, d0_2, z0sin_2)
    if elig1 and elig2:
        return 0 if dpp1 < dpp2 else 1
    if elig1:
        return 0
    if elig2:
        return 1
    return -1


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("infile")
    parser.add_argument("outfile")
    parser.add_argument("--is-mc", action="store_true")
    parser.add_argument("--max-events", type=int, default=-1)
    args = parser.parse_args()

    print(f"Loading {args.infile}...")
    data = load_all_events(args.infile, args.max_events)
    h = book_histograms()

    n_events = len(data["nV0"])
    print(f"Processing {n_events} events...")

    has_dpp = ("Trac_partIdx" in data and "Trac_perigee_fArray" in data
               and "Trac_weightMatrix_fRep_fArray" in data)
    has_d0z0 = ("Trac_partIdx" in data and "Trac_impParToVertexRPhi" in data
                and "Trac_impParToVertexZ" in data)
    has_vtx_pos = ("Part_originVtxIdx" in data
                   and "Vtx_position_fCoordinates_fX" in data
                   and "Vtx_position_fCoordinates_fY" in data
                   and "Vtx_position_fCoordinates_fZ" in data)
    if not has_vtx_pos:
        print("WARNING: Part_originVtxIdx / Vtx_position_* not in input — "
              "V0 vertex-position histograms (v0_vtx_*) will stay empty.")
    has_pv = (has_vtx_pos
              and "Vtx_status" in data
              and "Vtx_nOutgoing" in data)
    if not has_pv:
        print("WARNING: Vtx_status / Vtx_nOutgoing not in input — "
              "primary-vertex histograms (pv_*) will stay empty.")
    if not has_dpp:
        print("WARNING: Trac_partIdx / Trac_perigee_fArray / Trac_weightMatrix_fRep_fArray "
              "not in input — every dpp will be invalid and every V0 will be skipped at "
              "the tag election. Make sure your treefy'd file has Trac_* fields.")
    if not has_d0z0:
        print("WARNING: Trac_impParToVertexRPhi / Trac_impParToVertexZ not in input -- "
              "the tau-parity d0/z0 tag cuts will reject every tag. Either re-treefy with "
              "the impact-parameter branches, or set TAG_D0_MAX / TAG_Z0SIN_MAX to inf.")

    n_v0_total = 0
    n_v0_window = 0
    n_tp_filled = 0

    for ev in range(n_events):
        h['cutflow'].Fill(0)
        nV0 = int(data["nV0"][ev])
        if nV0 == 0:
            continue

        px   = np.asarray(data["px"][ev])
        py   = np.asarray(data["py"][ev])
        pz   = np.asarray(data["pz"][ev])
        q    = np.asarray(data["charge"][ev]).astype(int)
        lock = np.asarray(data["lock"][ev]).astype(np.int64)
        i1   = np.asarray(data["i1"][ev]).astype(int)
        i2   = np.asarray(data["i2"][ev]).astype(int)
        flt  = np.asarray(data["flight"][ev]).astype(float)
        vtx  = np.asarray(data["vtxProb"][ev]).astype(float)

        # Per-event Part-indexed dpp / d0 / z0 lookups.
        n_part = len(px)
        if has_dpp:
            dpp_part = compute_dpp_per_part(
                np.asarray(data["Trac_partIdx"][ev]),
                np.asarray(data["Trac_perigee_fArray"][ev]),
                np.asarray(data["Trac_weightMatrix_fRep_fArray"][ev]),
                n_part)
        else:
            dpp_part = np.full(n_part, -1.0)
        if has_d0z0:
            d0_part, z0_part = compute_d0z0_per_part(
                np.asarray(data["Trac_partIdx"][ev]),
                np.asarray(data["Trac_impParToVertexRPhi"][ev]),
                np.asarray(data["Trac_impParToVertexZ"][ev]),
                n_part)
        else:
            d0_part = np.full(n_part, np.nan)
            z0_part = np.full(n_part, np.nan)

        # ---- Hadronic event selection (≥7 charged tracks with quality cuts) ----
        # Same baseline as fbcec / tau: |q|>0.5, lock==0, pT>0.4, |d0|<0.6,
        # |z0 sinθ|<1. Gates the per-event PV histos and the per-V0 V0-vertex
        # histos so they all share the same hadronic-event sample (no
        # background contamination from beam-gas / 2-photon / etc.).
        pt_all   = np.sqrt(px*px + py*py)
        sin_th   = np.sqrt(px*px + py*py) / np.sqrt(px*px + py*py + pz*pz + 1e-30)
        z0sin_all = np.where(np.isfinite(z0_part),
                             np.abs(z0_part * sin_th), 999.0)
        d0_abs    = np.where(np.isfinite(d0_part), np.abs(d0_part), 999.0)
        good_charged = ((np.abs(q) > 0) & (lock == 0)
                        & (pt_all > 0.4) & (d0_abs < 0.6) & (z0sin_all < 1.0))
        n_charged = int(np.sum(good_charged))
        if n_charged < 7:
            continue

        # Per-event vertex position arrays for the V0 secondary-vertex
        # data/MC comparison. Daughter j1 inherits its origin vertex from
        # the V0 decay (Part_originVtxIdx). vtx_x/y/z hold ONE entry per
        # reconstructed Vtx in the event.
        if has_vtx_pos:
            origin_vtx_part = np.asarray(data["Part_originVtxIdx"][ev]).astype(int)
            vtx_x = np.asarray(data["Vtx_position_fCoordinates_fX"][ev], dtype=float)
            vtx_y = np.asarray(data["Vtx_position_fCoordinates_fY"][ev], dtype=float)
            vtx_z = np.asarray(data["Vtx_position_fCoordinates_fZ"][ev], dtype=float)
            n_vtx_ev = len(vtx_x)
        else:
            origin_vtx_part = None
            vtx_x = vtx_y = vtx_z = None
            n_vtx_ev = 0

        # ---- Primary-vertex selection (per event) ----
        # DELPHI convention: Vtx[0] is the event PV (status=0 = track-fitted,
        # status=1 = default IP). Status=4 vertices are decay (secondary).
        # We take the first Vtx with status != 4 — almost always index 0.
        if has_pv and n_vtx_ev > 0:
            vtx_status = np.asarray(data["Vtx_status"][ev], dtype=np.int64)
            n_out      = np.asarray(data["Vtx_nOutgoing"][ev], dtype=np.int64)
            non_decay  = np.flatnonzero(vtx_status != 4)
            h['n_pv_candidates'].Fill(len(non_decay))
            if len(non_decay) > 0:
                pv_idx = int(non_decay[0])  # first non-decay vertex = PV
                pvx = float(vtx_x[pv_idx])
                pvy = float(vtx_y[pv_idx])
                pvz = float(vtx_z[pv_idx])
                pvrho = float(np.sqrt(pvx*pvx + pvy*pvy))
                h['pv_x'].Fill(pvx)
                h['pv_y'].Fill(pvy)
                h['pv_z'].Fill(pvz)
                h['pv_rho'].Fill(pvrho)
                h['pv_xy'].Fill(pvx, pvy)
                h['pv_z_vs_rho'].Fill(pvrho, pvz)
                h['pv_n_outgoing'].Fill(int(n_out[pv_idx]))

        # Per-event set of (sorted) Part-index pairs flagged as photon conversions.
        # A V0 whose two daughters match a PhotonConv pair is a gamma->e+e-.
        nConv = int(data["nPhotonConv"][ev])
        conv_pairs = set()
        if nConv > 0:
            ci1 = np.asarray(data["convI1"][ev]).astype(int)
            ci2 = np.asarray(data["convI2"][ev]).astype(int)
            for k in range(nConv):
                a, b = int(ci1[k]), int(ci2[k])
                if a >= 0 and b >= 0:
                    conv_pairs.add((min(a, b), max(a, b)))

        for iv in range(nV0):
            j1, j2 = int(i1[iv]), int(i2[iv])

            px1, py1, pz1 = float(px[j1]), float(py[j1]), float(pz[j1])
            px2, py2, pz2 = float(px[j2]), float(py[j2]), float(pz[j2])
            q1, q2 = int(q[j1]), int(q[j2])

            p1, pt1, th1 = kin(px1, py1, pz1)
            p2, pt2, th2 = kin(px2, py2, pz2)

            m_pipi = inv_mass(px1, py1, pz1, M_PI, px2, py2, pz2, M_PI)
            m_ppi  = inv_mass(px1, py1, pz1, M_P,  px2, py2, pz2, M_PI)
            m_pip  = inv_mass(px1, py1, pz1, M_PI, px2, py2, pz2, M_P)

            n_v0_total += 1
            h['cutflow'].Fill(1)

            # No-cut spectra
            fill_mass_spectra(h, 'nocut', m_pipi, m_ppi, m_pip)

            # 2D QA on no-cut
            pK0s = float(np.sqrt((px1+px2)**2 + (py1+py2)**2 + (pz1+pz2)**2))
            thK0s = float(np.degrees(np.arctan2(np.sqrt((px1+px2)**2 + (py1+py2)**2), pz1+pz2)))
            h['mpipi_vs_flight'].Fill(flt[iv], m_pipi)
            h['mpipi_vs_vtxprob'].Fill(vtx[iv], m_pipi)
            h['mpipi_vs_pK0s'].Fill(pK0s, m_pipi)
            h['mpipi_vs_thetaK0s'].Fill(thK0s, m_pipi)

            # Cut-progression mass spectra
            cut_baseline = (pt1 > 0.4) and (pt2 > 0.4)
            cut_angle    = cut_baseline and (20 < th1 < 160) and (20 < th2 < 160)
            cut_highp    = cut_angle    and (p1 > 1.0) and (p2 > 1.0)

            # cleank0s: baseline + V0 displacement + vertex-fit quality + Lambda veto + gamma-conversion veto
            in_lam_window = (LAM_VETO_LO < m_ppi < LAM_VETO_HI) or \
                            (LAM_VETO_LO < m_pip < LAM_VETO_HI)
            is_conv = (min(j1, j2), max(j1, j2)) in conv_pairs
            cut_cleank0s = (cut_baseline
                            and (flt[iv] > CLEANK0S_FLIGHT_MIN)
                            and (vtx[iv] > CLEANK0S_VTXPROB_MIN)
                            and not in_lam_window
                            and not is_conv)

            if cut_baseline:
                h['cutflow'].Fill(2)
                fill_mass_spectra(h, 'baseline', m_pipi, m_ppi, m_pip)
            if cut_angle:
                h['cutflow'].Fill(3)
                fill_mass_spectra(h, 'angle', m_pipi, m_ppi, m_pip)
            if cut_highp:
                h['cutflow'].Fill(4)
                fill_mass_spectra(h, 'highp', m_pipi, m_ppi, m_pip)
            if cut_cleank0s:
                h['cutflow'].Fill(5)
                fill_mass_spectra(h, 'cleank0s', m_pipi, m_ppi, m_pip)

            # ---- V0 secondary-vertex position diagnostics ----
            # The V0 decay vertex is the origin vertex of either daughter
            # (j1 and j2 must agree if reconstruction is consistent — we
            # use j1 and trust the consistency).
            v0_vx = v0_vy = v0_vz = None
            if has_vtx_pos and cut_cleank0s and origin_vtx_part is not None:
                if 0 <= j1 < len(origin_vtx_part):
                    iv0 = int(origin_vtx_part[j1])
                    if 0 <= iv0 < n_vtx_ev:
                        v0_vx = float(vtx_x[iv0])
                        v0_vy = float(vtx_y[iv0])
                        v0_vz = float(vtx_z[iv0])
                        v0_vrho = float(np.sqrt(v0_vx*v0_vx + v0_vy*v0_vy))
                        h['v0_vtx_x_cleank0s'].Fill(v0_vx)
                        h['v0_vtx_y_cleank0s'].Fill(v0_vy)
                        h['v0_vtx_z_cleank0s'].Fill(v0_vz)
                        h['v0_vtx_rho_cleank0s'].Fill(v0_vrho)
                        h['v0_vtx_xy_cleank0s'].Fill(v0_vx, v0_vy)
                        h['v0_vtx_z_vs_rho_cleank0s'].Fill(v0_vrho, v0_vz)

            # ---- K0s tag-and-probe (gated on cleank0s + mass window) ----
            if not cut_cleank0s:
                continue
            if not (K0S_WINDOW[0] <= m_pipi <= K0S_WINDOW[1]):
                continue
            n_v0_window += 1
            h['cutflow'].Fill(6)

            # V0 vertex position in the clean K0s sample (signal region)
            if v0_vx is not None:
                v0_vrho = float(np.sqrt(v0_vx*v0_vx + v0_vy*v0_vy))
                h['v0_vtx_x_inwindow'].Fill(v0_vx)
                h['v0_vtx_y_inwindow'].Fill(v0_vy)
                h['v0_vtx_z_inwindow'].Fill(v0_vz)
                h['v0_vtx_rho_inwindow'].Fill(v0_vrho)
                h['v0_vtx_xy_inwindow'].Fill(v0_vx, v0_vy)
                h['v0_vtx_z_vs_rho_inwindow'].Fill(v0_vrho, v0_vz)

            dpp1 = float(dpp_part[j1])
            dpp2 = float(dpp_part[j2])
            lock1 = int(lock[j1])
            lock2 = int(lock[j2])

            # dpp QA (K0s window, before tag election)
            h['dpp_status'].Fill(0.5)  # total_K0s
            if dpp1 >= 0:
                h['dpp_all_v0'].Fill(dpp1)
                h['dpp_vs_theta_all_v0'].Fill(th1, dpp1)
            if dpp2 >= 0:
                h['dpp_all_v0'].Fill(dpp2)
                h['dpp_vs_theta_all_v0'].Fill(th2, dpp2)
            nvalid = int(dpp1 >= 0) + int(dpp2 >= 0)
            if nvalid == 2:
                h['dpp_status'].Fill(1.5)
            elif nvalid == 1:
                h['dpp_status'].Fill(2.5)
            else:
                h['dpp_status'].Fill(3.5)

            d0_1 = float(d0_part[j1]); d0_2 = float(d0_part[j2])
            z0_1 = float(z0_part[j1]); z0_2 = float(z0_part[j2])
            z0sin_1 = abs(z0_1 * np.sin(np.radians(th1))) if not np.isnan(z0_1) else float('nan')
            z0sin_2 = abs(z0_2 * np.sin(np.radians(th2))) if not np.isnan(z0_2) else float('nan')

            tag_idx = elect_tag(lock1, dpp1, d0_1, z0sin_1,
                                lock2, dpp2, d0_2, z0sin_2)
            if tag_idx < 0:
                continue
            h['cutflow'].Fill(7)

            if tag_idx == 0:
                tag_q, tag_pt, tag_p, tag_theta, tag_dpp = q1, pt1, p1, th1, dpp1
                pr_q,  pr_pt,  pr_p,  pr_theta,  pr_dpp  = q2, pt2, p2, th2, dpp2
            else:
                tag_q, tag_pt, tag_p, tag_theta, tag_dpp = q2, pt2, p2, th2, dpp2
                pr_q,  pr_pt,  pr_p,  pr_theta,  pr_dpp  = q1, pt1, p1, th1, dpp1

            h['dpp_tag'].Fill(tag_dpp)
            h['dpp_vs_theta_tag'].Fill(tag_theta, tag_dpp)
            if pr_dpp >= 0:
                h['dpp_probe'].Fill(pr_dpp)
                h['dpp_vs_theta_probe'].Fill(pr_theta, pr_dpp)

            fill_tag_probe(h, tag_q, tag_pt, tag_p, tag_theta,
                              pr_q,  pr_pt,  pr_p,  pr_theta)
            h['cutflow'].Fill(8)
            n_tp_filled += 1

    print(f"  V0 candidates total            : {n_v0_total}")
    print(f"  V0 in K0s mass window {K0S_WINDOW}: {n_v0_window}")
    print(f"  T&P fills (per assignment)      : {n_tp_filled}")

    out = ROOT.TFile(args.outfile, "RECREATE")
    for v in h.values():
        v.Write()
    out.Close()
    print(f"Wrote {args.outfile}")


if __name__ == "__main__":
    main()

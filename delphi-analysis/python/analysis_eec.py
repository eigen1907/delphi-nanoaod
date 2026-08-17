#!/usr/bin/env python3
"""
EEC Analysis (Charged-only)
---------------------------
Single-pass EEC analysis with systematic variations for charged particles.

For MC:
  - Fills gen-level histograms (before selection, no systematics)
  - Fills reco-level histograms (after selection AND event selection, all systematics)

For Data:
  - Fills reco-level histograms (after selection AND event selection, all systematics)

Optionally computes covariance matrix (1D or 2D) for nominal systematic only.

Usage:
  MC:   python analysis_eec.py input.root output.root --is-mc
  Data: python analysis_eec.py input.root output.root
  Single systematic: python analysis_eec.py input.root output.root --systematic nominal
  With covariance:   python analysis_eec.py input.root output.root --compute-covariance
  With 2D covariance: python analysis_eec.py input.root output.root --compute-covariance --covariance-2d
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
    heavy_jet_mass,
    calcAngle,
    apply_fake_drop_charged,
    randomly_drop_particles,
    calculate_event_eec_histogram,
    get_flat_bin_index,
    P4Block,
    match_angular,
)
from binning_and_selections import (
    rbins, zbins, eijbins, eijbins_alt, ebins,
    tbins, tbins2, tbinsDelphi, logtbins, logtbins2,
    apply_track_selection_delphi,
    apply_event_selection_delphi,
)

# ============================================================================
# SYSTEMATIC CONFIGURATION (charged-only, no neutral, no multiplicity reweight)
# ============================================================================

SYSTEMATICS = {
    'nominal': {
        'type': 'selection',
        # Baseline charged-track momentum cut is now |p| > 0.4 GeV
        # (total momentum), NOT pT > 0.4. This matches the thrust analyzer
        # (analysis_thrust.SYSTEMATICS) so EEC and thrust share one fiducial
        # definition. apply_track_selection_delphi switches to the |p| cut
        # whenever charged_p_min is set.
        'charged_p_min': 0.4,
    },

    # Momentum scale systematics (charged only)
    'charged_scale_up': {
        'type': 'momentum',
        'momentum_scale_charged': 1.01,
    },
    'charged_scale_down': {
        'type': 'momentum',
        'momentum_scale_charged': 0.99,
    },

    # Efficiency systematics (charged only)
    'fake_drop': {
        'type': 'efficiency',
        'apply_fake_drop': True,
    },
    'charged_eff': {
        'type': 'efficiency',
        'drop_charged_fraction': 0.02,
    },

    # Selection systematics (charged |p| threshold; varied around the 0.4 GeV
    # baseline). Names kept for downstream/bookkeeping compatibility.
    'charged_pt_up': {
        'type': 'selection',
        'charged_p_min': 0.5,
    },
    'charged_pt_down': {
        'type': 'selection',
        'charged_p_min': 0.3,
    },
}


# ============================================================================
# HISTOGRAM CREATION
# ============================================================================

def create_counter_histogram():
    """
    Event counter histogram.
    Bin 1: Total events processed
    Bin 2: Events passing event selection (P-scheme)
    Bin 3: Events passing event selection (E-scheme)
    """
    counter = ROOT.TH1D('counter', 'Event Counter', 3, 0, 3)
    counter.GetXaxis().SetBinLabel(1, 'Total Events')
    counter.GetXaxis().SetBinLabel(2, 'P-scheme')
    counter.GetXaxis().SetBinLabel(3, 'E-scheme')
    return counter


def create_gen_histograms():
    """
    Gen-level histograms: thrust + EEC, no systematics, no selection.
    """
    hists = {}

    # Thrust (before selection)
    for name, bins in [
        ("Thrust_before", tbins), ("Thrust_before2", tbins2),
        ("Thrust_beforeDelphi", tbinsDelphi),
        ("Thrust_before_log", logtbins), ("Thrust_before_log2", logtbins2),
        ("Thrust_before_Escheme", tbins), ("Thrust_before2_Escheme", tbins2),
        ("Thrust_beforeDelphi_Escheme", tbinsDelphi),
        ("Thrust_before_log_Escheme", logtbins), ("Thrust_before_log2_Escheme", logtbins2),
        ("ThrustC_beforeDelphi", tbins2),
        ("ThrustC_beforeDelphi_Escheme", tbins2),
    ]:
        hists[name] = ROOT.TH1D(name, "", len(bins) - 1, array('d', bins))

    # EEC (all charged, no charge categories)
    hists["EEC_r"] = ROOT.TH1D("EEC_r", "", len(rbins) - 1, array('d', rbins))
    hists["EEC_z"] = ROOT.TH1D("EEC_z", "", len(zbins) - 1, array('d', zbins))

    # 2D EEC
    hists["EEC2d_r"] = ROOT.TH2D("EEC2d_r", "",
                                   len(rbins) - 1, array('d', rbins),
                                   len(eijbins) - 1, array('d', eijbins))
    hists["EEC2d_z"] = ROOT.TH2D("EEC2d_z", "",
                                   len(zbins) - 1, array('d', zbins),
                                   len(eijbins) - 1, array('d', eijbins))

    # Kinematic distributions
    hists["NCharged"] = ROOT.TH1D("NCharged", "", 100, 0, 100)
    hists["SumE"] = ROOT.TH1D("SumE", "", 200, 0, 200)

    return hists


def create_reco_histograms(syst_name):
    """
    Reco-level histograms for one systematic.
    """
    hists = {}
    s = syst_name  # shorthand for naming

    # --- Thrust histograms (P-scheme) ---
    for base, bins in [
        ("ThrustMissPNC2", tbins2),
        ("ThrustMissPNCDelphi", tbinsDelphi),
        ("ThrustMissPNCLog2", logtbins2),
        ("ThrustCDelphi", tbins2),
    ]:
        name = f"{base}_{s}"
        hists[name] = ROOT.TH1D(name, "", len(bins) - 1, array('d', bins))

    # --- Thrust histograms (E-scheme) ---
    for base, bins in [
        ("ThrustMissPNC2_Escheme", tbins2),
        ("ThrustMissPNCDelphi_Escheme", tbinsDelphi),
        ("ThrustMissPNCLog2_Escheme", logtbins2),
        ("ThrustCDelphi_Escheme", tbins2),
    ]:
        name = f"{base}_{s}"
        hists[name] = ROOT.TH1D(name, "", len(bins) - 1, array('d', bins))

    # --- EEC histograms (all charged, no charge categories) ---
    hists[f"EEC_r_{s}"] = ROOT.TH1D(f"EEC_r_{s}", "",
                                      len(rbins) - 1, array('d', rbins))
    hists[f"EEC_z_{s}"] = ROOT.TH1D(f"EEC_z_{s}", "",
                                      len(zbins) - 1, array('d', zbins))

    # 2D EEC (r x Eij, z x Eij)
    hists[f"EEC2d_r_{s}"] = ROOT.TH2D(f"EEC2d_r_{s}", "",
                                        len(rbins) - 1, array('d', rbins),
                                        len(eijbins) - 1, array('d', eijbins))
    hists[f"EEC2d_z_{s}"] = ROOT.TH2D(f"EEC2d_z_{s}", "",
                                        len(zbins) - 1, array('d', zbins),
                                        len(eijbins) - 1, array('d', eijbins))

    # --- Kinematic control plots ---
    hists[f"NChargedSele_{s}"] = ROOT.TH1D(f"NChargedSele_{s}", "", 100, 0, 100)
    hists[f"NPartSele_{s}"] = ROOT.TH1D(f"NPartSele_{s}", "", 200, 0, 200)
    hists[f"SumESele_{s}"] = ROOT.TH1D(f"SumESele_{s}", "", 200, 0, 200)
    hists[f"TrackPTSele_{s}"] = ROOT.TH1D(f"TrackPTSele_{s}", "", 1000, 0, 100)
    hists[f"TrackESele_{s}"] = ROOT.TH1D(f"TrackESele_{s}", "",
                                           len(ebins) - 1, array('d', ebins))
    hists[f"MissPNC_{s}"] = ROOT.TH1D(f"MissPNC_{s}", "", 1000, 0, 100)
    hists[f"HJM_{s}"] = ROOT.TH1D(f"HJM_{s}", "", 1000, 0, 100)

    return hists


# ============================================================================
# SYSTEMATIC VARIATION
# ============================================================================

def apply_systematic_variation(px, py, pz, m, q, pt, th, d0, z0, config):
    """
    Apply systematic variations to particle momenta and efficiency.
    Charged-only systematics: momentum scale, fake drop, charged efficiency.
    """
    px_var = px.copy()
    py_var = py.copy()
    pz_var = pz.copy()
    m_var = m.copy()
    q_var = q.copy()
    pt_var = pt.copy()
    th_var = th.copy()
    d0_var = d0.copy()
    z0_var = z0.copy()

    is_charged = np.abs(q_var) > 0.1

    # === Momentum scale (charged only) ===
    if config.get('momentum_scale_charged'):
        scale = config['momentum_scale_charged']
        px_var[is_charged] *= scale
        py_var[is_charged] *= scale
        pz_var[is_charged] *= scale
        pt_var[is_charged] = np.sqrt(px_var[is_charged]**2 + py_var[is_charged]**2)
        th_var[is_charged] = np.arctan2(pt_var[is_charged], pz_var[is_charged])

    # === Efficiency systematics ===
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

    return {
        'px': px_var[keep_mask], 'py': py_var[keep_mask],
        'pz': pz_var[keep_mask], 'm': m_var[keep_mask],
        'q': q_var[keep_mask], 'pt': pt_var[keep_mask],
        'th': th_var[keep_mask], 'd0': d0_var[keep_mask],
        'z0': z0_var[keep_mask],
    }


# ============================================================================
# THRUST CALCULATION
# ============================================================================

def calculate_thrust_variants(px, py, pz, m, q, sel_c, sel_n, include_met=True):
    """
    Calculate P-scheme and E-scheme thrust for all particles and charged-only.
    Returns dict with all thrust values.
    """
    e = np.sqrt(px**2 + py**2 + pz**2 + m**2)
    p3_all = np.stack([px, py, pz], axis=1)

    # P-scheme (all particles)
    axis_all, T_all = thrust_axis_fast(p3_all, include_met=include_met)
    tau = 1 - T_all
    log_tau = np.log(tau) if tau > 0 else -10.0

    # E-scheme (all particles)
    p3_norm = np.linalg.norm(p3_all, axis=1, keepdims=True)
    n_hat = p3_all / np.where(p3_norm > 0, p3_norm, 1.0)
    p3_escheme = e[:, np.newaxis] * n_hat
    axis_escheme, T_escheme = thrust_axis_fast(p3_escheme, include_met=include_met)
    tau_e = 1 - T_escheme
    log_tau_e = np.log(tau_e) if tau_e > 0 else -10.0

    # Charged-only P-scheme
    if np.sum(sel_c) > 0:
        p3_c = p3_all[sel_c]
        axis_c, T_c = thrust_axis_fast(p3_c, include_met=False)
        tau_c = 1 - T_c
    else:
        axis_c, T_c = np.zeros(3), 0.0
        tau_c = 1.0

    # Charged-only E-scheme
    if np.sum(sel_c) > 0:
        p3_c_e = p3_escheme[sel_c]
        axis_c_e, T_c_e = thrust_axis_fast(p3_c_e, include_met=False)
        tau_c_e = 1 - T_c_e
    else:
        axis_c_e, T_c_e = np.zeros(3), 0.0
        tau_c_e = 1.0

    return {
        # P-scheme all
        'axis': axis_all, 'T': T_all,
        'thrust': tau, 'thrust_log': log_tau,
        # E-scheme all
        'axis_escheme': axis_escheme, 'T_escheme': T_escheme,
        'thrust_escheme': tau_e, 'thrust_log_escheme': log_tau_e,
        # Charged P-scheme
        'thrust_charged': tau_c,
        # Charged E-scheme
        'thrust_charged_escheme': tau_c_e,
    }


# ============================================================================
# EEC COMPUTATION
# ============================================================================

def compute_charged_eec(px_c, py_c, pz_c, m_c, E_beam):
    """
    Compute EEC for all charged particle pairs.

    Returns:
        r     : array of angles between pairs
        z     : array of z = (1 - cos(r)) / 2
        eij   : array of E_i * E_j / E_beam^2
        iu, ju: pair indices
    """
    e_c = np.sqrt(px_c**2 + py_c**2 + pz_c**2 + m_c**2)
    p3_c = np.stack((px_c, py_c, pz_c), axis=1)

    norms = np.linalg.norm(p3_c, axis=1)
    denom = np.outer(norms, norms)
    dot = p3_c @ p3_c.T
    cos = np.divide(dot, denom, out=np.ones_like(dot), where=denom > 0)
    r = np.arccos(np.clip(cos, -1.0, 1.0))
    z = 0.5 * (1 - cos)
    ee = np.outer(e_c, e_c) / (E_beam * E_beam)

    iu, ju = np.triu_indices(len(p3_c), k=1)

    return r, z, ee, iu, ju


# ============================================================================
# GEN-LEVEL PROCESSING
# ============================================================================

def load_gen_particles(tree_gen_before, tree_gen, event_idx):
    """
    Load gen-level particles with energy conservation fallback.
    Returns None if event is invalid (energy cut fails).
    """
    tree_gen_before.GetEntry(event_idx)
    E = tree_gen_before.Energy
    if abs(E - 91.25) > 1:
        return None

    px = np.array(tree_gen_before.px)
    py = np.array(tree_gen_before.py)
    pz = np.array(tree_gen_before.pz)
    m = np.array(tree_gen_before.mass)
    q = np.array(tree_gen_before.charge)
    pt = np.array(tree_gen_before.pt)
    th = np.array(tree_gen_before.theta)

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
        pt = np.array(tree_gen.pt)
        th = np.array(tree_gen.theta)
        e = np.sqrt(px**2 + py**2 + pz**2 + m**2)

    return {
        'px': px, 'py': py, 'pz': pz, 'm': m,
        'q': q, 'pt': pt, 'th': th, 'e': e, 'E': E,
    }


def process_gen_level(tree_gen_before, tree_gen, event_idx, gen_hists):
    """Thin flat-skim wrapper: load gen particles for one event (with the
    tgenBefore->tgen energy fallback) and delegate to process_gen_arrays.

    Non-flat readers (raw nanoaod) build the gen particle arrays themselves
    via the V0-aware SimPart leaf walk and call process_gen_arrays directly,
    so the EEC physics is identical bit-for-bit across readers.
    """
    data = load_gen_particles(tree_gen_before, tree_gen, event_idx)
    if data is None:
        return
    process_gen_arrays(
        data['px'], data['py'], data['pz'], data['m'],
        data['q'], data['E'], gen_hists,
    )


def process_gen_arrays(px, py, pz, m, q, E, gen_hists):
    """
    Pure-array gen core (reader-agnostic). Fills gen-level thrust + EEC
    histograms in full phase space — no track selection, no systematics.
    `E` is the event cm energy (used for the EEC E_i*E_j/E^2 normalisation
    and the E-scheme thrust). Charged particles are |q| > 0.5.
    """
    if len(px) == 0:
        return

    e = np.sqrt(px**2 + py**2 + pz**2 + m**2)

    p3 = np.stack((px, py, pz), axis=1)

    # --- P-scheme thrust (all particles) ---
    axis, T = thrust_axis_fast(p3, include_met=False)
    gen_hists["Thrust_before"].Fill(1 - T)
    gen_hists["Thrust_before2"].Fill(1 - T)
    gen_hists["Thrust_beforeDelphi"].Fill(1 - T)
    gen_hists["Thrust_before_log"].Fill(np.log(1 - T))
    gen_hists["Thrust_before_log2"].Fill(np.log(1 - T))

    # --- E-scheme thrust (all particles) ---
    p3_norm = np.linalg.norm(p3, axis=1, keepdims=True)
    n_hat = p3 / np.where(p3_norm > 0, p3_norm, 1.0)
    p3_escheme = e[:, np.newaxis] * n_hat
    axis_e, T_e = thrust_axis_fast(p3_escheme, include_met=False)
    gen_hists["Thrust_before_Escheme"].Fill(1 - T_e)
    gen_hists["Thrust_before2_Escheme"].Fill(1 - T_e)
    gen_hists["Thrust_beforeDelphi_Escheme"].Fill(1 - T_e)
    gen_hists["Thrust_before_log_Escheme"].Fill(np.log(1 - T_e))
    gen_hists["Thrust_before_log2_Escheme"].Fill(np.log(1 - T_e))

    # --- Charged-only thrust ---
    sel_c = np.abs(q) > 0.5
    if np.sum(sel_c) > 0:
        p3_c = p3[sel_c]
        _, T_c = thrust_axis_fast(p3_c, include_met=False)
        gen_hists["ThrustC_beforeDelphi"].Fill(1 - T_c)

        p3_c_e = p3_escheme[sel_c]
        _, T_c_e = thrust_axis_fast(p3_c_e, include_met=False)
        gen_hists["ThrustC_beforeDelphi_Escheme"].Fill(1 - T_c_e)

    # --- Gen-level EEC (all charged) ---
    is_charged = np.abs(q) > 0.5
    px_c, py_c, pz_c, m_c = px[is_charged], py[is_charged], pz[is_charged], m[is_charged]
    if len(px_c) >= 2:
        r, z, ee, iu, ju = compute_charged_eec(px_c, py_c, pz_c, m_c, E)
        for i, j in zip(iu, ju):
            r_val = r[i, j]
            z_val = z[i, j]
            eij_val = ee[i, j]
            gen_hists["EEC_r"].Fill(r_val, eij_val)
            gen_hists["EEC_z"].Fill(z_val, eij_val)
            gen_hists["EEC2d_r"].Fill(r_val, eij_val)
            gen_hists["EEC2d_z"].Fill(z_val, eij_val)

    gen_hists["NCharged"].Fill(np.sum(is_charged))
    gen_hists["SumE"].Fill(np.sum(e))


# ============================================================================
# RECO-LEVEL PROCESSING
# ============================================================================

def process_reco_level(tree_reco, event_idx, systematics,
                       reco_hists, counter,
                       covariance_state=None):
    """Thin flat-skim wrapper: read the flat reco tree 't' for one event and
    delegate to process_reco_arrays (the shared, reader-agnostic core).
    Non-flat readers (raw nanoaod) call the array core directly.
    """
    tree_reco.GetEntry(event_idx)
    process_reco_arrays(
        np.array(tree_reco.px), np.array(tree_reco.py), np.array(tree_reco.pz),
        np.array(tree_reco.mass), np.array(tree_reco.charge),
        np.array(tree_reco.pt), np.array(tree_reco.theta),
        np.array(tree_reco.d0), np.array(tree_reco.z0),
        tree_reco.Energy,
        systematics, reco_hists, counter,
        covariance_state=covariance_state,
    )


def process_reco_arrays(px, py, pz, m, q, pt, th, d0, z0, E,
                        systematics, reco_hists, counter,
                        covariance_state=None):
    """
    Process reco-level particles for all systematics from pre-built per-event
    arrays. Identical physics to the former process_reco_level body; factored
    out so non-flat readers (raw nanoaod) reuse the exact same systematics /
    selection / EEC / covariance path. Optionally accumulates covariance sums
    for nominal (P-scheme).
    """
    if abs(E - 91.25) > 1:
        return

    if len(px) == 0:
        return

    for syst_name, syst_config in systematics.items():

        # --- Apply systematic variation ---
        varied = apply_systematic_variation(
            px, py, pz, m, q, pt, th, d0, z0, syst_config
        )
        px_v, py_v, pz_v, m_v, q_v = (
            varied['px'], varied['py'], varied['pz'], varied['m'], varied['q']
        )
        pt_v, th_v, d0_v, z0_v = (
            varied['pt'], varied['th'], varied['d0'], varied['z0']
        )

        if len(px_v) == 0:
            continue

        # --- Track selection ---
        # Standard DELPHI charged-track IP acceptance: |d0|<4 cm, |z0*sinθ|<10 cm
        # (longitudinal cut looser than transverse). d0/z0 are in cm (SKELANA
        # QTRAC). Overridable per-systematic.
        selection = apply_track_selection_delphi(
            px=px_v, py=py_v, pz=pz_v, m=m_v, q=q_v,
            th=th_v, pt=pt_v, d0=d0_v, z0=z0_v,
            charged_p_min=syst_config.get('charged_p_min', 0.4),
            charged_d0_max=syst_config.get('charged_d0_max', 4.0),
            charged_z0sin_max=syst_config.get('charged_z0sin_max', 10.0),
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
        pt_sel = pt_v[sel_all]

        sel_c = np.abs(q_sel) > 0.1   # charged
        sel_n = ~sel_c                  # neutral

        # --- Thrust ---
        thrust_vals = calculate_thrust_variants(
            px_sel, py_sel, pz_sel, m_sel, q_sel,
            sel_c, sel_n, include_met=True
        )

        # Energies for event selection
        e_sel = np.sqrt(px_sel**2 + py_sel**2 + pz_sel**2 + m_sel**2)
        e_c = e_sel[sel_c]
        e_all = e_sel

        s = syst_name
        h = reco_hists[s]

        # ======== P-scheme event selection ========
        theta_thrust_p = thrust_theta(thrust_vals['axis'], thrust_vals['T'], fold=False)
        pass_p = apply_event_selection_delphi(
            e_c=e_c, e_n=e_all,
            theta_Tu=theta_thrust_p, E_reco=E
        )['pass_reco']

        if pass_p:
            if syst_name == 'nominal':
                counter.Fill(1.5)

            # Thrust histograms
            h[f"ThrustMissPNC2_{s}"].Fill(thrust_vals['thrust'])
            h[f"ThrustMissPNCDelphi_{s}"].Fill(thrust_vals['thrust'])
            h[f"ThrustMissPNCLog2_{s}"].Fill(thrust_vals['thrust_log'])
            h[f"ThrustCDelphi_{s}"].Fill(thrust_vals['thrust_charged'])

            # --- EEC (charged-only) ---
            px_c = px_sel[sel_c]
            py_c = py_sel[sel_c]
            pz_c = pz_sel[sel_c]
            m_c = m_sel[sel_c]

            if len(px_c) >= 2:
                r, z, ee, iu, ju = compute_charged_eec(px_c, py_c, pz_c, m_c, E)

                for idx_i, idx_j in zip(iu, ju):
                    r_val = r[idx_i, idx_j]
                    z_val = z[idx_i, idx_j]
                    eij_val = ee[idx_i, idx_j]

                    h[f"EEC_r_{s}"].Fill(r_val, eij_val)
                    h[f"EEC_z_{s}"].Fill(z_val, eij_val)
                    h[f"EEC2d_r_{s}"].Fill(r_val, eij_val)
                    h[f"EEC2d_z_{s}"].Fill(z_val, eij_val)

                # --- Covariance accumulation (nominal only, P-scheme only) ---
                if covariance_state is not None and syst_name == 'nominal':
                    _accumulate_covariance(r, z, ee, iu, ju, E, covariance_state)

            # Kinematic control plots
            p3_all = np.stack((px_sel, py_sel, pz_sel), axis=1)
            p_miss = missing_p(p3_all)
            p4_all = np.column_stack((e_all, px_sel, py_sel, pz_sel))

            h[f"NChargedSele_{s}"].Fill(np.sum(sel_c))
            h[f"NPartSele_{s}"].Fill(len(px_sel))
            h[f"SumESele_{s}"].Fill(np.sum(e_all))
            h[f"MissPNC_{s}"].Fill(np.linalg.norm(p_miss))
            h[f"HJM_{s}"].Fill(heavy_jet_mass(p4_all, thrust_vals['axis']))
            for k in range(len(e_c)):
                h[f"TrackPTSele_{s}"].Fill(pt_sel[sel_c][k])
                h[f"TrackESele_{s}"].Fill(e_c[k])

        # ======== E-scheme event selection ========
        theta_thrust_e = thrust_theta(thrust_vals['axis_escheme'], thrust_vals['T_escheme'], fold=False)
        pass_e = apply_event_selection_delphi(
            e_c=e_c, e_n=e_all,
            theta_Tu=theta_thrust_e, E_reco=E
        )['pass_reco']

        if pass_e:
            if syst_name == 'nominal':
                counter.Fill(2.5)

            h[f"ThrustMissPNC2_Escheme_{s}"].Fill(thrust_vals['thrust_escheme'])
            h[f"ThrustMissPNCDelphi_Escheme_{s}"].Fill(thrust_vals['thrust_escheme'])
            h[f"ThrustMissPNCLog2_Escheme_{s}"].Fill(thrust_vals['thrust_log_escheme'])
            h[f"ThrustCDelphi_Escheme_{s}"].Fill(thrust_vals['thrust_charged_escheme'])


# ============================================================================
# RESPONSE MATRICES (flattened, no RooUnfold — consumed by unfold_2dflatten*)
# ============================================================================
# Single-pass gen<->reco migration for charged-EEC unfolding, built per matched
# charged-particle pair. Mirrors create_response_matrices.py's
# bookResponseMatricesNoRooUnfold + matched/miss/fake fill, but here it runs in
# the analyzer so the response shares ONE selection / gen definition with the
# spectra. Object names + the 4D THnD layout are exactly what
# unfold_2dflatten*.py reads: reco2d_eij_{r,z}[_match], gen2d_eij_{r,z}[_match],
# response2d_eij_{r,z} (axes: reco_var, reco_Eij, gen_var, gen_Eij).
#
# Method = 2D (var, Eij) migration: pairs are filled with UNIT weight into 2D
# (var x Eij) histograms (Eij is an axis, not the weight), so both the angular
# and the energy-weight migration are unfolded; the EEC is the Eij-weighted
# projection downstream. Decomposition: reco2d = matched + fake (full reco),
# gen2d = matched + miss (full truth), *_match / response2d = matched only.
# Nominal selection only; gated on the reco P-scheme event selection.

def _book_thnd(name, edges):
    """4D THnD with the given per-axis variable bin edges."""
    ndim = len(edges)
    nbins = array('i', [len(e) - 1 for e in edges])
    h = ROOT.THnD(name, name, ndim, nbins,
                  array('d', [0.0] * ndim), array('d', [1.0] * ndim))
    for ax in range(ndim):
        h.GetAxis(ax).Set(len(edges[ax]) - 1, array('d', list(edges[ax])))
    return h


def create_response_objects():
    """Book the flattened response objects consumed by unfold_2dflatten*.py."""
    resp = {}

    def th2(name, xb):
        resp[name] = ROOT.TH2D(name, "", len(xb) - 1, array('d', xb),
                               len(eijbins) - 1, array('d', eijbins))

    def th2_alt(name, xb):
        # alternative eij binning (2D-binning systematic)
        resp[name] = ROOT.TH2D(name, "", len(xb) - 1, array('d', xb),
                               len(eijbins_alt) - 1, array('d', eijbins_alt))

    for var, vb in (("r", rbins), ("z", zbins)):
        th2(f"reco2d_eij_{var}", vb)
        th2(f"gen2d_eij_{var}", vb)
        th2(f"reco2d_eij_{var}_match", vb)
        th2(f"gen2d_eij_{var}_match", vb)
        # alt-eij-binning reco/gen SELECTED 2D (no _match/response/cov) for the
        # 2D-binning systematic.
        th2_alt(f"reco2d_eij_{var}_alt", vb)
        th2_alt(f"gen2d_eij_{var}_alt", vb)

    resp["response2d_eij_r"] = _book_thnd("response2d_eij_r",
                                          [rbins, eijbins, rbins, eijbins])
    resp["response2d_eij_z"] = _book_thnd("response2d_eij_z",
                                          [zbins, eijbins, zbins, eijbins])

    # 1D gen EEC filled with the EXACT eij weight (not the eij axis), with the
    # SAME selections as gen2d_eij_{var} (gen track cut + reco event selection).
    # Compare against the bin-center eij-weighted projection of gen2d_eij_{var}
    # to get the 2D-binning (bin-center) correction on the unfolded result.
    resp["gen1d_r"] = ROOT.TH1D("gen1d_r", "", len(rbins) - 1, array('d', rbins))
    resp["gen1d_z"] = ROOT.TH1D("gen1d_z", "", len(zbins) - 1, array('d', zbins))
    # reco counterpart of gen1d (exact-eij 1D, reco selected) -- the eij binning
    # is irrelevant for the 1D (exact weight), so there is no separate _alt 1D.
    resp["reco1d_r"] = ROOT.TH1D("reco1d_r", "", len(rbins) - 1, array('d', rbins))
    resp["reco1d_z"] = ROOT.TH1D("reco1d_z", "", len(zbins) - 1, array('d', zbins))
    return resp


def _charged_block(px, py, pz, m, q, pt, th, mask):
    """P4Block over the masked (selected-charged) particles + their energies."""
    cpx, cpy, cpz, cm, cq = px[mask], py[mask], pz[mask], m[mask], q[mask]
    cpt, cth = pt[mask], th[mask]
    cphi = np.arctan2(cpy, cpx)
    ce = np.sqrt(cpx * cpx + cpy * cpy + cpz * cpz + cm * cm)
    block = P4Block.build(cpx, cpy, cpz, cq, cpt, cth, cphi, ce, cm)
    return block, ce


def process_response_arrays(px, py, pz, m, q, pt, th, d0, z0, E_reco,
                            gpx, gpy, gpz, gm, gq, E_gen, resp,
                            charged_p_min=0.4, neutral_e_min=0.5,
                            match_r=0.05, require_same_charge=False):
    """
    Build the charged-EEC migration for ONE event from pre-built reco + gen
    arrays (reader-agnostic). Applies the nominal track selection to both
    levels, gates on the reco P-scheme event selection, matches selected
    charged reco<->gen (angular ΔR<match_r; same-charge only if
    require_same_charge, default OFF since the EEC is charge-blind), then fills
    the matched/miss/fake pairs into `resp` (from create_response_objects()).

    gpx/gpy/gpz/gm/gq are the gen final-state arrays (V0-aware leaves when the
    caller is the nanoaod reader). pt/th for gen are derived here.
    """
    if len(px) == 0 or len(gpx) == 0:
        return

    # --- nominal reco track selection (same IP acceptance as the spectra:
    #     |d0|<4 cm, |z0*sinθ|<10 cm; d0/z0 in cm) ---
    rsel = apply_track_selection_delphi(
        px=px, py=py, pz=pz, m=m, q=q, th=th, pt=pt, d0=d0, z0=z0,
        charged_p_min=charged_p_min, neutral_e_min=neutral_e_min,
        charged_d0_max=4.0, charged_z0sin_max=10.0,
    )
    sel_all = rsel['sel']
    sel_c = rsel['sel_c']
    if not np.any(sel_all):
        return

    # --- reco P-scheme event-selection gate (same as the spectra) ---
    px_s, py_s, pz_s, m_s, q_s = (px[sel_all], py[sel_all], pz[sel_all],
                                  m[sel_all], q[sel_all])
    sel_c_final = np.abs(q_s) > 0.1
    thrust_vals = calculate_thrust_variants(
        px_s, py_s, pz_s, m_s, q_s, sel_c_final, ~sel_c_final, include_met=True)
    e_s = np.sqrt(px_s**2 + py_s**2 + pz_s**2 + m_s**2)
    theta_thrust_p = thrust_theta(thrust_vals['axis'], thrust_vals['T'], fold=False)
    if not apply_event_selection_delphi(
            e_c=e_s[sel_c_final], e_n=e_s,
            theta_Tu=theta_thrust_p, E_reco=E_reco)['pass_reco']:
        return

    # --- nominal gen track selection (charged) ---
    gpt = np.sqrt(gpx * gpx + gpy * gpy)
    gth = np.arctan2(gpt, gpz)
    gsel = apply_track_selection_delphi(
        px_gen=gpx, py_gen=gpy, pz_gen=gpz, m_gen=gm, q_gen=gq,
        th_gen=gth, pt_gen=gpt,
        charged_p_min=charged_p_min, neutral_e_min=neutral_e_min,
    )
    sel_c_gen = gsel['sel_c_gen']

    # --- charged blocks + angular matching ---
    rec_c, ce = _charged_block(px, py, pz, m, q, pt, th, sel_c)
    gen_c, gce = _charged_block(gpx, gpy, gpz, gm, gq, gpt, gth, sel_c_gen)
    nrec, ngen = len(ce), len(gce)
    if nrec == 0 and ngen == 0:
        return
    if nrec > 0 and ngen > 0:
        ireco, igen, imiss, ifake = match_angular(
            rec_c, gen_c, match_r, require_same_charge=require_same_charge)
    else:
        ireco = np.array([], dtype=int); igen = np.array([], dtype=int)
        imiss = np.arange(ngen); ifake = np.arange(nrec)

    r_r = resp["response2d_eij_r"]; r_z = resp["response2d_eij_z"]

    # ➊ matched pairs -> response + reco2d/gen2d (+ *_match)
    for k in range(len(ireco)):
        i_rec, i_gen = ireco[k], igen[k]
        for l in range(k + 1, len(ireco)):
            j_rec, j_gen = ireco[l], igen[l]
            Eij_rec = ce[i_rec] * ce[j_rec] / E_reco**2
            Eij_gen = gce[i_gen] * gce[j_gen] / E_gen**2
            r_rec = calcAngle(rec_c.p[i_rec], rec_c.p[j_rec])
            r_gen = calcAngle(gen_c.p[i_gen], gen_c.p[j_gen])
            z_rec = (1.0 - np.cos(r_rec)) / 2.0
            z_gen = (1.0 - np.cos(r_gen)) / 2.0
            resp["reco2d_eij_r"].Fill(r_rec, Eij_rec)
            resp["reco2d_eij_r_match"].Fill(r_rec, Eij_rec)
            resp["gen2d_eij_r"].Fill(r_gen, Eij_gen)
            resp["gen2d_eij_r_match"].Fill(r_gen, Eij_gen)
            resp["reco2d_eij_z"].Fill(z_rec, Eij_rec)
            resp["reco2d_eij_z_match"].Fill(z_rec, Eij_rec)
            resp["gen2d_eij_z"].Fill(z_gen, Eij_gen)
            resp["gen2d_eij_z_match"].Fill(z_gen, Eij_gen)
            # exact-eij 1D gen+reco truth (bin-center-correction reference)
            resp["gen1d_r"].Fill(r_gen, Eij_gen)
            resp["gen1d_z"].Fill(z_gen, Eij_gen)
            resp["reco1d_r"].Fill(r_rec, Eij_rec)
            resp["reco1d_z"].Fill(z_rec, Eij_rec)
            # alt-eij-binning selected 2D (2D-binning systematic)
            resp["reco2d_eij_r_alt"].Fill(r_rec, Eij_rec)
            resp["reco2d_eij_z_alt"].Fill(z_rec, Eij_rec)
            resp["gen2d_eij_r_alt"].Fill(r_gen, Eij_gen)
            resp["gen2d_eij_z_alt"].Fill(z_gen, Eij_gen)
            r_r.Fill(array('d', [r_rec, Eij_rec, r_gen, Eij_gen]))
            r_z.Fill(array('d', [z_rec, Eij_rec, z_gen, Eij_gen]))

    # ➋ missed gen pairs (miss-miss + miss-matched) -> gen2d + gen1d (truth)
    for k, i_gen in enumerate(imiss):
        for j_gen in imiss[k + 1:]:
            Eij = gce[i_gen] * gce[j_gen] / E_gen**2
            r = calcAngle(gen_c.p[i_gen], gen_c.p[j_gen]); z = (1.0 - np.cos(r)) / 2.0
            resp["gen2d_eij_r"].Fill(r, Eij); resp["gen2d_eij_z"].Fill(z, Eij)
            resp["gen1d_r"].Fill(r, Eij); resp["gen1d_z"].Fill(z, Eij)
            resp["gen2d_eij_r_alt"].Fill(r, Eij); resp["gen2d_eij_z_alt"].Fill(z, Eij)
        for j_gen in igen:
            Eij = gce[i_gen] * gce[j_gen] / E_gen**2
            r = calcAngle(gen_c.p[i_gen], gen_c.p[j_gen]); z = (1.0 - np.cos(r)) / 2.0
            resp["gen2d_eij_r"].Fill(r, Eij); resp["gen2d_eij_z"].Fill(z, Eij)
            resp["gen1d_r"].Fill(r, Eij); resp["gen1d_z"].Fill(z, Eij)
            resp["gen2d_eij_r_alt"].Fill(r, Eij); resp["gen2d_eij_z_alt"].Fill(z, Eij)

    # ➌ fake reco pairs (fake-fake + fake-matched) -> reco2d only (measured)
    for k, i_rec in enumerate(ifake):
        for j_rec in ifake[k + 1:]:
            Eij = ce[i_rec] * ce[j_rec] / E_reco**2
            r = calcAngle(rec_c.p[i_rec], rec_c.p[j_rec]); z = (1.0 - np.cos(r)) / 2.0
            resp["reco2d_eij_r"].Fill(r, Eij); resp["reco2d_eij_z"].Fill(z, Eij)
            resp["reco1d_r"].Fill(r, Eij); resp["reco1d_z"].Fill(z, Eij)
            resp["reco2d_eij_r_alt"].Fill(r, Eij); resp["reco2d_eij_z_alt"].Fill(z, Eij)
        for j_rec in ireco:
            Eij = ce[i_rec] * ce[j_rec] / E_reco**2
            r = calcAngle(rec_c.p[i_rec], rec_c.p[j_rec]); z = (1.0 - np.cos(r)) / 2.0
            resp["reco2d_eij_r"].Fill(r, Eij); resp["reco2d_eij_z"].Fill(z, Eij)
            resp["reco1d_r"].Fill(r, Eij); resp["reco1d_z"].Fill(z, Eij)
            resp["reco2d_eij_r_alt"].Fill(r, Eij); resp["reco2d_eij_z_alt"].Fill(z, Eij)


# ============================================================================
# COVARIANCE HELPERS
# ============================================================================

def init_covariance_state(covariance_2d):
    """
    Initialize covariance accumulation state.

    If covariance_2d: accumulate over flattened 2D (r x Eij) bins.
    Otherwise: accumulate over 1D r and z bins separately.
    """
    state = {'is_2d': covariance_2d, 'n_events': 0}

    if covariance_2d:
        # 2D covariance: use template histograms to get bin counts
        template_r = ROOT.TH2D("_cov_template_r", "",
                                len(rbins) - 1, array('d', rbins),
                                len(eijbins) - 1, array('d', eijbins))
        template_z = ROOT.TH2D("_cov_template_z", "",
                                len(zbins) - 1, array('d', zbins),
                                len(eijbins) - 1, array('d', eijbins))
        nx_r = template_r.GetNbinsX() + 2
        ny_r = template_r.GetNbinsY() + 2
        nx_z = template_z.GetNbinsX() + 2
        ny_z = template_z.GetNbinsY() + 2
        total_r = nx_r * ny_r
        total_z = nx_z * ny_z

        state['template_r'] = template_r
        state['template_z'] = template_z
        state['nx_r'] = nx_r
        state['ny_r'] = ny_r
        state['nx_z'] = nx_z
        state['ny_z'] = ny_z
        state['total_r'] = total_r
        state['total_z'] = total_z
        state['sum_eec_r'] = np.zeros(total_r)
        state['sum_eec_products_r'] = np.zeros((total_r, total_r))
        state['sum_eec_z'] = np.zeros(total_z)
        state['sum_eec_products_z'] = np.zeros((total_z, total_z))
    else:
        # 1D covariance over r and z projections
        template_r = ROOT.TH1D("_cov_template_r_1d", "",
                                len(rbins) - 1, array('d', rbins))
        template_z = ROOT.TH1D("_cov_template_z_1d", "",
                                len(zbins) - 1, array('d', zbins))
        total_r = template_r.GetNbinsX() + 2
        total_z = template_z.GetNbinsX() + 2

        state['template_r'] = template_r
        state['template_z'] = template_z
        state['total_r'] = total_r
        state['total_z'] = total_z
        state['sum_eec_r'] = np.zeros(total_r)
        state['sum_eec_products_r'] = np.zeros((total_r, total_r))
        state['sum_eec_z'] = np.zeros(total_z)
        state['sum_eec_products_z'] = np.zeros((total_z, total_z))

    return state


def _accumulate_covariance(r, z, ee, iu, ju, E, state):
    """
    Accumulate per-event EEC into covariance running sums.
    """
    state['n_events'] += 1

    if state['is_2d']:
        # 2D covariance: flatten (r, Eij) -> 1D vector per event
        event_eec_r = np.zeros(state['total_r'])
        event_eec_z = np.zeros(state['total_z'])
        tmpl_r = state['template_r']
        tmpl_z = state['template_z']
        ny_r = state['ny_r']
        ny_z = state['ny_z']

        for idx_i, idx_j in zip(iu, ju):
            r_val = r[idx_i, idx_j]
            z_val = z[idx_i, idx_j]
            eij_val = ee[idx_i, idx_j]

            # Accumulate the per-event COUNT vector (one entry per pair), NOT
            # the eij-weighted sum. The covariance is handed to RooUnfold via
            # SetMeasuredCov against the unit-weight EEC2d COUNT histogram; the
            # eij weighting is applied downstream in the (z,eij)->1D projection.
            # Using eij here makes the covariance ~eij^2 too small vs the count
            # data -> absurdly small unfolded errors.
            # r covariance
            i_bin = tmpl_r.GetXaxis().FindBin(r_val)
            j_bin = tmpl_r.GetYaxis().FindBin(eij_val)
            flat_idx = i_bin * ny_r + j_bin
            if flat_idx < state['total_r']:
                event_eec_r[flat_idx] += 1.0

            # z covariance
            i_bin_z = tmpl_z.GetXaxis().FindBin(z_val)
            j_bin_z = tmpl_z.GetYaxis().FindBin(eij_val)
            flat_idx_z = i_bin_z * ny_z + j_bin_z
            if flat_idx_z < state['total_z']:
                event_eec_z[flat_idx_z] += 1.0
    else:
        # 1D covariance
        event_eec_r = np.zeros(state['total_r'])
        event_eec_z = np.zeros(state['total_z'])
        tmpl_r = state['template_r']
        tmpl_z = state['template_z']

        for idx_i, idx_j in zip(iu, ju):
            r_val = r[idx_i, idx_j]
            z_val = z[idx_i, idx_j]
            eij_val = ee[idx_i, idx_j]

            # COUNT vector (see 2D branch above): eij weighting is applied in
            # the downstream projection, not in the covariance.
            i_bin_r = tmpl_r.GetXaxis().FindBin(r_val)
            event_eec_r[i_bin_r] += 1.0

            i_bin_z = tmpl_z.GetXaxis().FindBin(z_val)
            event_eec_z[i_bin_z] += 1.0

    # Accumulate sum and sum-of-outer-products. The per-event vector is sparse
    # (only ~O(n_pairs) of the up-to-total^2 bins are non-zero), so update only
    # the non-zero sub-block instead of forming the full dense total x total
    # outer product every event (47M elements for the 2D/overflow case — the
    # dominant cost). Equivalent result, ~(total/n_nz)^2 faster.
    state['sum_eec_r'] += event_eec_r
    nz_r = np.flatnonzero(event_eec_r)
    if nz_r.size:
        v = event_eec_r[nz_r]
        state['sum_eec_products_r'][np.ix_(nz_r, nz_r)] += np.outer(v, v)

    state['sum_eec_z'] += event_eec_z
    nz_z = np.flatnonzero(event_eec_z)
    if nz_z.size:
        v = event_eec_z[nz_z]
        state['sum_eec_products_z'][np.ix_(nz_z, nz_z)] += np.outer(v, v)


def write_covariance(state, fout):
    """
    Write covariance matrices and mean EEC to ROOT file.
    """
    N = state['n_events']
    if N < 2:
        print("WARNING: Need at least 2 events for covariance, skipping.")
        return

    print(f"Writing covariance matrices (N={N} events)")

    fout.cd()

    for var in ('r', 'z'):
        total = state[f'total_{var}']
        sum_eec = state[f'sum_eec_{var}']
        sum_prod = state[f'sum_eec_products_{var}']

        # Store raw sums for later combination (hadd-safe).
        # Fill the whole bin buffer in one call via SetContent instead of
        # ~total^2 per-cell SetBinContent (47M PyROOT calls for the 2D/overflow
        # case). ROOT's global bin index is binx + (nbinsX+2)*biny, so lay
        # sum_prod into the interior (binx,biny in 1..total) of a
        # (total+2)x(total+2) buffer (rows=biny, cols=binx) and flatten C-order.
        cov_hist = ROOT.TH2D(
            f"covariance_matrix_{var}",
            f"EEC {var} Sum of Outer Products (N={N})",
            total, 0.5, total + 0.5,
            total, 0.5, total + 0.5
        )
        cov_buf = np.zeros((total + 2, total + 2), dtype=np.float64)
        cov_buf[1:total + 1, 1:total + 1] = sum_prod.T  # full[biny,binx]=sum_prod[binx,biny]
        cov_hist.SetContent(np.ascontiguousarray(cov_buf.ravel()))

        mean_hist = ROOT.TH1D(
            f"mean_eec_{var}",
            f"Sum of EEC {var} (N={N})",
            total, 0.5, total + 0.5
        )
        mean_buf = np.zeros(total + 2, dtype=np.float64)
        mean_buf[1:total + 1] = sum_eec
        mean_hist.SetContent(np.ascontiguousarray(mean_buf))

        cov_hist.Write()
        mean_hist.Write()

    # Save binning metadata
    bin_info = ROOT.TH1D("bin_info", "Covariance Binning Info", 6, 0, 6)
    bin_info.SetBinContent(1, N)
    bin_info.SetBinContent(2, state['total_r'])
    bin_info.SetBinContent(3, state['total_z'])
    bin_info.SetBinContent(4, 1 if state['is_2d'] else 0)
    if state['is_2d']:
        bin_info.SetBinContent(5, state['nx_r'])
        bin_info.SetBinContent(6, state['ny_r'])
    bin_info.Write()

    print(f"  Covariance type: {'2D' if state['is_2d'] else '1D'}")
    print(f"  r bins: {state['total_r']}, z bins: {state['total_z']}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='EEC analysis (charged-only) with systematic variations'
    )
    parser.add_argument('infile', help='Input ROOT file')
    parser.add_argument('outfile', help='Output ROOT file')
    parser.add_argument('--is-mc', action='store_true',
                        help='Process as MC (fills gen-level histograms)')
    parser.add_argument('--systematic', default='all',
                        help='Run specific systematic or "all" (default: all)')
    parser.add_argument('--compute-covariance', action='store_true',
                        help='Compute covariance matrix (nominal only)')
    parser.add_argument('--covariance-2d', action='store_true',
                        help='Use full 2D (r x Eij) covariance instead of 1D')

    args = parser.parse_args()

    # --- Determine systematics to run ---
    if args.systematic == 'all':
        systematics_to_run = SYSTEMATICS
    elif args.systematic in SYSTEMATICS:
        systematics_to_run = {args.systematic: SYSTEMATICS[args.systematic]}
    else:
        print(f"ERROR: Unknown systematic '{args.systematic}'")
        print(f"Available: {', '.join(SYSTEMATICS.keys())}")
        return 1

    print(f"Running EEC analysis on: {args.infile}")
    print(f"MC mode: {args.is_mc}")
    print(f"Systematics: {', '.join(systematics_to_run.keys())}")
    print(f"Covariance: {args.compute_covariance} (2D: {args.covariance_2d})")

    # --- Open input file ---
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

    # --- Create histograms ---
    counter = create_counter_histogram()

    gen_hists = None
    if args.is_mc:
        gen_hists = create_gen_histograms()

    reco_hists = {}
    for syst_name in systematics_to_run:
        reco_hists[syst_name] = create_reco_histograms(syst_name)

    # --- Covariance state ---
    covariance_state = None
    if args.compute_covariance:
        covariance_state = init_covariance_state(args.covariance_2d)

    # --- Event loop ---
    for iEvt in range(n_events):
        if iEvt % 1000 == 0:
            print(f"  Processing event {iEvt}/{n_events}")

        # Gen level (MC only)
        if args.is_mc and gen_hists is not None:
            process_gen_level(tree_gen_before, tree_gen, iEvt, gen_hists)

        # Reco level (all)
        counter.Fill(0.5)
        process_reco_level(
            tree_reco, iEvt, systematics_to_run,
            reco_hists, counter,
            covariance_state=covariance_state,
        )

    print(f"Processed {n_events} events")

    # --- Write output ---
    fout = ROOT.TFile(args.outfile, 'RECREATE')

    # Gen histograms
    if gen_hists is not None:
        fout.mkdir("gen")
        fout.cd("gen")
        for h in gen_hists.values():
            h.Write()

    # Reco histograms (organized by systematic in a flat "reco" directory)
    fout.mkdir("reco")
    fout.cd("reco")
    for syst_name, hists in reco_hists.items():
        for h in hists.values():
            h.Write()

    # Counter
    fout.cd()
    counter.Write()

    # Covariance
    if covariance_state is not None:
        fout.mkdir("covariance")
        fout.cd("covariance")
        write_covariance(covariance_state, fout)

    fout.Close()
    fin.Close()

    print(f"Output written to {args.outfile}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
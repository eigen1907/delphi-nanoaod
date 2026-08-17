#!/usr/bin/env python3
"""
Optimized Thrust Analysis
-------------------------
Single-pass thrust analysis with systematic variations.

For MC:
  - Fills gen-level histograms (before selection, no systematics)
  - Fills reco-level histograms (after selection AND event selection, all systematics)
  - Fills response matrices (after selection AND event selection, all systematics)

For Data:
  - Fills reco-level histograms (after selection AND event selection, all systematics)

Usage:
  MC:   python analysis_thrust.py input.root output.root --is-mc
  Data: python analysis_thrust.py input.root output.root
  Single systematic: python analysis_thrust.py input.root output.root --systematic nominal
"""

import ROOT
import numpy as np
import argparse
import math
from array import array

# Import from common modules
from common_functions import (
    thrust_axis_fast_optimized,
    thrust_theta,
    apply_momentum_bias,
    apply_fake_drop_charged,
    randomly_drop_particles,
    calc_multiplicity_weight_linear,
    missing_p,
)
from binning_and_selections import (
    tbins2,
    logtbins2,
    apply_track_selection_delphi,
    apply_event_selection_delphi,
)

from thrust_moments import (
    create_all_moment_histograms,
    fill_moments_all_variants,
    write_moment_histograms,
    TAU_MAX_PHYS,
)

# ============================================================================
# SYSTEMATIC CONFIGURATION
# ============================================================================

SYSTEMATICS = {
    'nominal': {
        'type': 'selection',
        'charged_p_min': 0.4,
        'neutral_e_min': 0.5,
    },
    
    # Momentum scale systematics
    'charged_scale_up': {
        'type': 'momentum',
        'momentum_scale_charged': 1.01,
    },
    'charged_scale_down': {
        'type': 'momentum',
        'momentum_scale_charged': 0.99,
    },
    'neutral_scale_up': {
        'type': 'momentum',
        'momentum_scale_neutral': 1.1,
    },
    'neutral_scale_down': {
        'type': 'momentum',
        'momentum_scale_neutral': 0.9,
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
    'neutral_eff': {
        'type': 'efficiency',
        'drop_neutral_fraction': 0.02,
    },
    
    # Event multiplicity reweighting
    'evt_weight': {
        'type': 'reweighting',
        'use_evt_weight': True,
    },
    
    # Selection systematics
    'neutral_e_up': {
        'type': 'selection',
        'charged_p_min': 0.4,
        'neutral_e_min': 5.0,
    },
    'neutral_e_down': {
        'type': 'selection',
        'charged_p_min': 0.4,
        'neutral_e_min': 0.3,
    },
    'charged_pt_up': {
        'type': 'selection',
        'charged_p_min': 0.5,
        'neutral_e_min': 0.5,
    },
    'charged_pt_down': {
        'type': 'selection',
        'charged_p_min': 0.3,
        'neutral_e_min': 0.5,
    },
}

# ============================================================================
# HISTOGRAM CREATION FUNCTIONS
# ============================================================================

def create_counter_histogram():
    """
    Create counter histogram to track event counts for different thrust scenarios.
    Bin 1: Total events processed
    Bin 2: Events passing P-scheme all-particle event selection  
    Bin 3: Events passing E-scheme all-particle event selection
    Bin 4: Events passing P-scheme charged-only event selection
    Bin 5: Events passing E-scheme charged-only event selection
    """
    counter = ROOT.TH1D('counter', 'Event Counter', 5, 0, 5)
    counter.GetXaxis().SetBinLabel(1, 'Total Events')
    counter.GetXaxis().SetBinLabel(2, 'P-scheme All')
    counter.GetXaxis().SetBinLabel(3, 'E-scheme All')
    counter.GetXaxis().SetBinLabel(4, 'P-scheme Charged')
    counter.GetXaxis().SetBinLabel(5, 'E-scheme Charged')
    return counter

def create_counter_weighted_histogram():
    """
    Create counter histogram to track event counts for different thrust scenarios.
    Bin 1: Total events processed
    Bin 2: Events passing P-scheme all-particle event selection  
    Bin 3: Events passing E-scheme all-particle event selection
    Bin 4: Events passing P-scheme charged-only event selection
    Bin 5: Events passing E-scheme charged-only event selection
    """
    counter_weighted = ROOT.TH1D('counter_weighted', 'Event Counter', 5, 0, 5)
    counter_weighted.GetXaxis().SetBinLabel(1, 'Total Events')
    counter_weighted.GetXaxis().SetBinLabel(2, 'P-scheme All')
    counter_weighted.GetXaxis().SetBinLabel(3, 'E-scheme All')
    counter_weighted.GetXaxis().SetBinLabel(4, 'P-scheme Charged')
    counter_weighted.GetXaxis().SetBinLabel(5, 'E-scheme Charged')
    return counter_weighted

def create_gen_histograms(suffix=''):
    """
    Create gen-level histograms (before selection, no systematics).
    Only used for MC.

    `suffix` is appended to the ROOT object names (NOT the dict keys) so a
    second, parallel gen set can be booked without an in-memory name clash —
    e.g. create_gen_histograms(suffix='_genPart') for a GenPart-based gen.
    The dict keys stay fixed so process_gen_arrays fills either set unchanged.
    """
    hists = {}

    # Standard binning (tbins2)
    hists['Thrust_before2'] = ROOT.TH1D('Thrust_before2'+suffix, '', len(tbins2)-1, array('d', tbins2))
    hists['Thrust_before2_Escheme'] = ROOT.TH1D('Thrust_before2_Escheme'+suffix, '', len(tbins2)-1, array('d', tbins2))
    hists['ThrustC_before2'] = ROOT.TH1D('ThrustC_before2'+suffix, '', len(tbins2)-1, array('d', tbins2))
    hists['ThrustC_before2_Escheme'] = ROOT.TH1D('ThrustC_before2_Escheme'+suffix, '', len(tbins2)-1, array('d', tbins2))

    # Log binning (logtbins2)
    hists['Thrust_before_log2'] = ROOT.TH1D('Thrust_before_log2'+suffix, '', len(logtbins2)-1, array('d', logtbins2))
    hists['Thrust_before_log2_Escheme'] = ROOT.TH1D('Thrust_before_log2_Escheme'+suffix, '', len(logtbins2)-1, array('d', logtbins2))
    hists['ThrustC_before_log2'] = ROOT.TH1D('ThrustC_before_log2'+suffix, '', len(logtbins2)-1, array('d', logtbins2))
    hists['ThrustC_before_log2_Escheme'] = ROOT.TH1D('ThrustC_before_log2_Escheme'+suffix, '', len(logtbins2)-1, array('d', logtbins2))

    return hists


def create_reco_histograms(syst_name):
    """
    Create reco-level histograms for one systematic.
    Used for both MC and data.
    """
    hists = {}
    
    # Standard binning (tbins2)
    hists[f'ThrustMissPNC2_{syst_name}'] = ROOT.TH1D(f'ThrustMissPNC2_{syst_name}', '', 
                                                      len(tbins2)-1, array('d', tbins2))
    hists[f'ThrustMissPNC2_Escheme_{syst_name}'] = ROOT.TH1D(f'ThrustMissPNC2_Escheme_{syst_name}', '', 
                                                              len(tbins2)-1, array('d', tbins2))
    hists[f'ThrustC2_{syst_name}'] = ROOT.TH1D(f'ThrustC2_{syst_name}', '', 
                                                len(tbins2)-1, array('d', tbins2))
    hists[f'ThrustC2_Escheme_{syst_name}'] = ROOT.TH1D(f'ThrustC2_Escheme_{syst_name}', '', 
                                                        len(tbins2)-1, array('d', tbins2))
    
    # Log binning (logtbins2)
    hists[f'ThrustMissPNCLog2_{syst_name}'] = ROOT.TH1D(f'ThrustMissPNCLog2_{syst_name}', '', 
                                                         len(logtbins2)-1, array('d', logtbins2))
    hists[f'ThrustMissPNCLog2_Escheme_{syst_name}'] = ROOT.TH1D(f'ThrustMissPNCLog2_Escheme_{syst_name}', '', 
                                                                 len(logtbins2)-1, array('d', logtbins2))
    hists[f'ThrustCLog2_{syst_name}'] = ROOT.TH1D(f'ThrustCLog2_{syst_name}', '', 
                                                   len(logtbins2)-1, array('d', logtbins2))
    hists[f'ThrustCLog2_Escheme_{syst_name}'] = ROOT.TH1D(f'ThrustCLog2_Escheme_{syst_name}', '', 
                                                           len(logtbins2)-1, array('d', logtbins2))
    
    return hists


def create_response_matrices(syst_name):
    """
    Create 2D response matrices for one systematic.
    For MC only - all systematics get response matrices.
    """
    matrices = {}
    
    # Standard binning (tbins2 x tbins2)
    matrices[f'response_thrust2_{syst_name}'] = ROOT.TH2D(
        f'response_thrust2_{syst_name}', '',
        len(tbins2)-1, array('d', tbins2),
        len(tbins2)-1, array('d', tbins2)
    )
    matrices[f'response_thrust2_Escheme_{syst_name}'] = ROOT.TH2D(
        f'response_thrust2_Escheme_{syst_name}', '',
        len(tbins2)-1, array('d', tbins2),
        len(tbins2)-1, array('d', tbins2)
    )
    matrices[f'response_thrustC2_{syst_name}'] = ROOT.TH2D(
        f'response_thrustC2_{syst_name}', '',
        len(tbins2)-1, array('d', tbins2),
        len(tbins2)-1, array('d', tbins2)
    )
    matrices[f'response_thrustC2_Escheme_{syst_name}'] = ROOT.TH2D(
        f'response_thrustC2_Escheme_{syst_name}', '',
        len(tbins2)-1, array('d', tbins2),
        len(tbins2)-1, array('d', tbins2)
    )
    
    # Log binning (logtbins2 x logtbins2)
    matrices[f'response_thrust_log2_{syst_name}'] = ROOT.TH2D(
        f'response_thrust_log2_{syst_name}', '',
        len(logtbins2)-1, array('d', logtbins2),
        len(logtbins2)-1, array('d', logtbins2)
    )
    matrices[f'response_thrust_log2_Escheme_{syst_name}'] = ROOT.TH2D(
        f'response_thrust_log2_Escheme_{syst_name}', '',
        len(logtbins2)-1, array('d', logtbins2),
        len(logtbins2)-1, array('d', logtbins2)
    )
    matrices[f'response_thrustC_log2_{syst_name}'] = ROOT.TH2D(
        f'response_thrustC_log2_{syst_name}', '',
        len(logtbins2)-1, array('d', logtbins2),
        len(logtbins2)-1, array('d', logtbins2)
    )
    matrices[f'response_thrustC_log2_Escheme_{syst_name}'] = ROOT.TH2D(
        f'response_thrustC_log2_Escheme_{syst_name}', '',
        len(logtbins2)-1, array('d', logtbins2),
        len(logtbins2)-1, array('d', logtbins2)
    )
    
    return matrices

# ============================================================================
# SYSTEMATIC VARIATION FUNCTIONS
# ============================================================================

def apply_systematic_variation(px, py, pz, m, q, pt, th, d0, z0, config):
    """
    Apply systematic variations to particle momenta and efficiency.
    Each systematic varies only ONE thing.
    
    Parameters:
    -----------
    px, py, pz, m, q, pt, th, d0, z0 : arrays
        Particle properties
    config : dict
        Systematic configuration
    
    Returns:
    --------
    dict with all particle arrays (dropped particles removed)
    """
    # Make copies
    px_var = px.copy()
    py_var = py.copy()
    pz_var = pz.copy()
    m_var = m.copy()
    q_var = q.copy()
    pt_var = pt.copy()
    th_var = th.copy()
    d0_var = d0.copy()
    z0_var = z0.copy()
    
    # Identify charged and neutral particles
    is_charged = np.abs(q_var) > 0.1
    
    # === Momentum scale systematics ===
    if config.get('momentum_scale_charged'):
        scale = config['momentum_scale_charged']
        px_var[is_charged] *= scale
        py_var[is_charged] *= scale
        pz_var[is_charged] *= scale
        # Update pt and theta
        pt_var[is_charged] = np.sqrt(px_var[is_charged]**2 + py_var[is_charged]**2)
        th_var[is_charged] = np.arctan2(pt_var[is_charged], pz_var[is_charged])
    
    if config.get('momentum_scale_neutral'):
        scale = config['momentum_scale_neutral']
        px_var[~is_charged] *= scale
        py_var[~is_charged] *= scale
        pz_var[~is_charged] *= scale
        # Update theta for neutrals
        pt_neutral = np.sqrt(px_var[~is_charged]**2 + py_var[~is_charged]**2)
        th_var[~is_charged] = np.arctan2(pt_neutral, pz_var[~is_charged])
    
    # === Efficiency systematics ===
    keep_mask = np.ones(len(px_var), dtype=bool)
    
    if config.get('apply_fake_drop'):
        charged_indices = np.where(is_charged)[0]
        if len(charged_indices) > 0:
            fake_dropped = apply_fake_drop_charged(
                px_var[is_charged], 
                py_var[is_charged], 
                pz_var[is_charged], 
                m_var[is_charged],
                seed=config.get('fake_drop_seed')
            )
            keep_mask[charged_indices] = fake_dropped['mask']
    
    if config.get('drop_charged_fraction', 0.0) > 0:
        charged_indices = np.where(is_charged)[0]
        if len(charged_indices) > 0:
            dropped = randomly_drop_particles(
                px_var[is_charged],
                py_var[is_charged],
                pz_var[is_charged],
                m_var[is_charged],
                drop_fraction=config['drop_charged_fraction'],
                seed=config.get('drop_charged_seed')
            )
            keep_mask[charged_indices] = dropped['mask']
    
    if config.get('drop_neutral_fraction', 0.0) > 0:
        neutral_indices = np.where(~is_charged)[0]
        if len(neutral_indices) > 0:
            dropped = randomly_drop_particles(
                px_var[~is_charged],
                py_var[~is_charged],
                pz_var[~is_charged],
                m_var[~is_charged],
                drop_fraction=config['drop_neutral_fraction'],
                seed=config.get('drop_neutral_seed')
            )
            keep_mask[neutral_indices] = dropped['mask']
    
    # Apply keep mask to all arrays
    return {
        'px': px_var[keep_mask],
        'py': py_var[keep_mask],
        'pz': pz_var[keep_mask],
        'm': m_var[keep_mask],
        'q': q_var[keep_mask],
        'pt': pt_var[keep_mask],
        'th': th_var[keep_mask],
        'd0': d0_var[keep_mask],
        'z0': z0_var[keep_mask],
    }


def calculate_all_thrust_variants(px, py, pz, m, q, sel_c, sel_n, include_met=True):
    """
    Calculate all thrust variants for one event.
    
    Parameters:
    -----------
    px, py, pz, m, q : arrays
        Particle properties (already selected)
    sel_c, sel_n : boolean arrays
        Masks for charged and neutral particles
    include_met : bool
        Whether to include missing momentum (True for reco, False for gen)
    
    Returns:
    --------
    dict with thrust values, log values, axis vectors, and T values
    """
    # Recalculate energy from momentum
    e = np.sqrt(px**2 + py**2 + pz**2 + m**2)
    
    # === Standard (P-scheme) Thrust ===
    p3_all = np.stack([px, py, pz], axis=1)
    axis_all, T_all = thrust_axis_fast_optimized(p3_all, include_met=include_met)
    thrust_val = 1 - T_all
    thrust_log_val = np.log(thrust_val) if thrust_val > 0 else -10.0
    
    # === E-scheme Thrust ===
    p3_norm = np.linalg.norm(p3_all, axis=1, keepdims=True)
    n = p3_all / np.where(p3_norm > 0, p3_norm, 1.0)
    p3_escheme = e[:, np.newaxis] * n
    axis_escheme, T_escheme = thrust_axis_fast_optimized(p3_escheme, include_met=include_met)
    thrust_escheme_val = 1 - T_escheme
    thrust_log_escheme_val = np.log(thrust_escheme_val) if thrust_escheme_val > 0 else -10.0
    
    # === Charged-only Thrust ===
    if np.sum(sel_c) > 0:
        p3_charged = np.stack([px[sel_c], py[sel_c], pz[sel_c]], axis=1)
        axis_charged, T_charged = thrust_axis_fast_optimized(p3_charged, include_met=include_met)
        thrust_charged_val = 1 - T_charged
        thrust_charged_log_val = np.log(thrust_charged_val) if thrust_charged_val > 0 else -10.0
    else:
        axis_charged = np.zeros(3)
        T_charged = 0.0
        thrust_charged_val = 0.0
        thrust_charged_log_val = -10.0
    
    # === Charged-only E-scheme Thrust ===
    if np.sum(sel_c) > 0:
        e_charged = e[sel_c]
        p3_charged_norm = np.linalg.norm(p3_charged, axis=1, keepdims=True)
        n_charged = p3_charged / np.where(p3_charged_norm > 0, p3_charged_norm, 1.0)
        p3_charged_escheme = e_charged[:, np.newaxis] * n_charged
        axis_charged_escheme, T_charged_escheme = thrust_axis_fast_optimized(
            p3_charged_escheme, include_met=include_met
        )
        thrust_charged_escheme_val = 1 - T_charged_escheme
        thrust_charged_log_escheme_val = np.log(thrust_charged_escheme_val) if thrust_charged_escheme_val > 0 else -10.0
    else:
        axis_charged_escheme = np.zeros(3)
        T_charged_escheme = 0.0
        thrust_charged_escheme_val = 0.0
        thrust_charged_log_escheme_val = -10.0
    
    return {
        # Thrust values
        'thrust': thrust_val,
        'thrust_log': thrust_log_val,
        'thrust_escheme': thrust_escheme_val,
        'thrust_log_escheme': thrust_log_escheme_val,
        'thrust_charged': thrust_charged_val,
        'thrust_charged_log': thrust_charged_log_val,
        'thrust_charged_escheme': thrust_charged_escheme_val,
        'thrust_charged_log_escheme': thrust_charged_log_escheme_val,
        # Axis and T values for event selection
        'axis': axis_all,
        'T': T_all,
        'axis_escheme': axis_escheme,
        'T_escheme': T_escheme,
        'axis_charged': axis_charged,
        'T_charged': T_charged,
        'axis_charged_escheme': axis_charged_escheme,
        'T_charged_escheme': T_charged_escheme,
    }

# ============================================================================
# EVENT PROCESSING FUNCTIONS
# ============================================================================

def process_gen_level(tree_gen_before, tree_gen, event_idx, gen_hists,
                      charged_p_min=0.4, neutral_e_min=0.5,
                      moment_hists=None):
    """
    Process generator-level particles (MC only). No event selection at gen level.

    When called with `gen_hists is not None` (the once-per-event "before" call),
    this function ALSO fills the gen_full moment accumulators (nominal-only).
    On per-systematic calls (gen_hists=None), only the gen-selected thrust
    values for response matrices are computed and returned.

    Returns:
        dict with gen thrust values (for response matrix filling),
        or None if invalid.
    """
    # Load from primary tree
    tree_gen_before.GetEntry(event_idx)

    # Reject off-peak energy-scan runs (keep ±1 GeV around the Z pole)
    E = tree_gen_before.Energy
    if abs(E - 91.25) > 1:
        return None

    # Load particles
    px = np.array(tree_gen_before.px)
    py = np.array(tree_gen_before.py)
    pz = np.array(tree_gen_before.pz)
    m  = np.array(tree_gen_before.mass)
    q  = np.array(tree_gen_before.charge)
    pt = np.array(tree_gen_before.pt)
    th = np.array(tree_gen_before.theta)

    if len(px) == 0:
        return None

    # Only an EXCESS of energy means a V0 showered in the calorimeter; fall
    # back to tgen (post-decay/pre-shower) so the V0 stays a single neutral.
    # A deficit means ISR — keep tgenBefore in that case (do NOT use abs()).
    e = np.sqrt(px**2 + py**2 + pz**2 + m**2)
    if np.sum(e) - E > 0.1:
        tree_gen.GetEntry(event_idx)
        px = np.array(tree_gen.px)
        py = np.array(tree_gen.py)
        pz = np.array(tree_gen.pz)
        m  = np.array(tree_gen.mass)
        q  = np.array(tree_gen.charge)
        pt = np.array(tree_gen.pt)
        th = np.array(tree_gen.theta)

    return process_gen_arrays(
        px, py, pz, m, q, pt, th, E, gen_hists,
        charged_p_min=charged_p_min, neutral_e_min=neutral_e_min,
        moment_hists=moment_hists,
    )


def process_gen_arrays(px, py, pz, m, q, pt, th, E, gen_hists,
                       charged_p_min=0.4, neutral_e_min=0.5,
                       moment_hists=None):
    """
    Pure-array gen core extracted from process_gen_level (everything after the
    particles are loaded). Reader-agnostic so the V0-aware nanoaod gen reuses
    it. When gen_hists is given (the once-per-event "before" call) it fills the
    gen 'before' histograms and the gen_full moment accumulators; then it
    applies the per-syst gen track selection and returns the selected-gen
    thrust values for response-matrix filling (or None).
    """
    if len(px) == 0:
        return None

    # ========================================================================
    # PURE GEN HISTOGRAMS + GEN_FULL MOMENT FILLS
    # No track selection, no event selection. Truth in full phase space.
    # Both share `thrust_vals_no_sel` and the same gate so they only run on
    # the once-per-event "before" call (gen_hists is not None).
    # ========================================================================
    if gen_hists is not None:
        is_charged = np.abs(q) > 0.1
        sel_c_gen = is_charged
        sel_n_gen = ~is_charged

        thrust_vals_no_sel = calculate_all_thrust_variants(
            px, py, pz, m, q,
            sel_c_gen, sel_n_gen,
            include_met=False
        )

        # Fill gen histograms (no event selection at gen level)
        gen_hists['Thrust_before2'].Fill(thrust_vals_no_sel['thrust'])
        gen_hists['Thrust_before_log2'].Fill(thrust_vals_no_sel['thrust_log'])
        gen_hists['Thrust_before2_Escheme'].Fill(thrust_vals_no_sel['thrust_escheme'])
        gen_hists['Thrust_before_log2_Escheme'].Fill(thrust_vals_no_sel['thrust_log_escheme'])

        gen_hists['ThrustC_before2'].Fill(thrust_vals_no_sel['thrust_charged'])
        gen_hists['ThrustC_before_log2'].Fill(thrust_vals_no_sel['thrust_charged_log'])
        gen_hists['ThrustC_before2_Escheme'].Fill(thrust_vals_no_sel['thrust_charged_escheme'])
        gen_hists['ThrustC_before_log2_Escheme'].Fill(thrust_vals_no_sel['thrust_charged_log_escheme'])

        # --- gen_full moment accumulators (nominal only, once per event) ---
        if moment_hists is not None:
            fill_moments_all_variants(
                moment_hists['gen_full']['nominal'],
                scenario='gen_full',
                syst_name='nominal',
                thrust_vals=thrust_vals_no_sel,
                weight=1.0,
                tau_max=TAU_MAX_PHYS,
            )

    # ========================================================================
    # FOR RESPONSE MATRICES: Apply gen track selection (per-syst cuts)
    # ========================================================================
    gen_selection = apply_track_selection_delphi(
        px_gen=px, py_gen=py, pz_gen=pz, m_gen=m, q_gen=q,
        th_gen=th, pt_gen=pt,
        charged_p_min=charged_p_min,
        neutral_e_min=neutral_e_min,
    )

    sel_gen_all = gen_selection['sel_gen']

    if not np.any(sel_gen_all):
        return None

    # Extract selected gen particles
    px_sel = px[sel_gen_all]
    py_sel = py[sel_gen_all]
    pz_sel = pz[sel_gen_all]
    m_sel  = m[sel_gen_all]
    q_sel  = q[sel_gen_all]

    sel_c_final = np.abs(q_sel) > 0.1
    sel_n_final = ~sel_c_final

    # Calculate thrust for selected gen particles (for response matrices)
    thrust_vals_selected = calculate_all_thrust_variants(
        px_sel, py_sel, pz_sel, m_sel, q_sel,
        sel_c_final, sel_n_final,
        include_met=False
    )

    return thrust_vals_selected


def process_reco_level(tree_reco, event_idx, systematics, reco_hists,
                       response_matrices,
                       thrust_gen_dict=None, is_mc=False,
                       counter=None, counter_weighted=None,
                       moment_hists=None):
    """Thin TTree wrapper: read the flat reco tree 't' for one event and
    delegate to process_reco_level_arrays (the shared, reader-agnostic core).
    Non-flat readers (EDM4hep, raw nanoaod) call the array core directly.
    """
    tree_reco.GetEntry(event_idx)
    process_reco_level_arrays(
        np.array(tree_reco.px), np.array(tree_reco.py), np.array(tree_reco.pz),
        np.array(tree_reco.mass), np.array(tree_reco.charge),
        np.array(tree_reco.pt), np.array(tree_reco.theta),
        np.array(tree_reco.d0), np.array(tree_reco.z0),
        tree_reco.Energy,
        systematics, reco_hists, response_matrices,
        thrust_gen_dict=thrust_gen_dict, is_mc=is_mc,
        counter=counter, counter_weighted=counter_weighted,
        moment_hists=moment_hists,
    )


def process_reco_level_arrays(px, py, pz, m, q, pt, th, d0, z0, E,
                              systematics, reco_hists, response_matrices,
                              thrust_gen_dict=None, is_mc=False,
                              counter=None, counter_weighted=None,
                              moment_hists=None):
    """
    Process reco-level particles for all systematics from pre-built per-event
    arrays. Identical physics to the former process_reco_level body; factored
    out so non-flat readers (EDM4hep, raw nanoaod) reuse the exact same
    systematics / event-selection / response / moments path.

    A SINGLE event selection is applied (P-scheme all-particle), and ALL
    four thrust variants (P/E scheme, all/charged) plus all four response
    matrices are gated on it. No per-variant recomputation.
    """
    # Reject off-peak energy-scan runs (keep ±1 GeV around the Z pole)
    if abs(E - 91.25) > 1:
        return

    if len(px) == 0:
        return

    # Loop over all systematics
    for syst_name, syst_config in systematics.items():
        thrust_gen = thrust_gen_dict.get(syst_name) if thrust_gen_dict else None
 
        # Apply systematic variations
        varied = apply_systematic_variation(
            px, py, pz, m, q, pt, th, d0, z0,
            syst_config
        )
 
        px_var = varied['px']
        py_var = varied['py']
        pz_var = varied['pz']
        m_var  = varied['m']
        q_var  = varied['q']
        pt_var = varied['pt']
        th_var = varied['th']
        d0_var = varied['d0']
        z0_var = varied['z0']
 
        if len(px_var) == 0:
            continue
 
        # Apply particle selection
        selection = apply_track_selection_delphi(
            px=px_var, py=py_var, pz=pz_var, m=m_var, q=q_var,
            th=th_var, pt=pt_var,
            d0=d0_var, z0=z0_var,
            charged_p_min=syst_config.get('charged_p_min', 0.4),
            charged_z0sin_max=10.0,
            neutral_e_min=syst_config.get('neutral_e_min', 0.5),
        )
 
        sel_all = selection['sel']
        if not np.any(sel_all):
            continue
 
        # Extract selected particles
        px_sel = px_var[sel_all]
        py_sel = py_var[sel_all]
        pz_sel = pz_var[sel_all]
        m_sel  = m_var[sel_all]
        q_sel  = q_var[sel_all]
 
        sel_c_final = np.abs(q_sel) > 0.1
        sel_n_final = ~sel_c_final
 
        # Event weight (only for use_evt_weight systematic)
        evt_weight = 1.0
        if syst_config.get('use_evt_weight'):
            n_total_sel = len(px_sel)
            evt_weight = calc_multiplicity_weight_linear(n_total_sel)
 
        # Calculate thrust
        thrust_vals = calculate_all_thrust_variants(
            px_sel, py_sel, pz_sel, m_sel, q_sel,
            sel_c_final, sel_n_final,
            include_met=True
        )
 
        # Energies for event selection
        e_c = np.sqrt(px_sel[sel_c_final]**2 + py_sel[sel_c_final]**2 +
                      pz_sel[sel_c_final]**2 + m_sel[sel_c_final]**2)
        e_all = np.sqrt(px_sel**2 + py_sel**2 + pz_sel**2 + m_sel**2)
 
        passed_p_all = False
        passed_e_all = False
        passed_p_charged = False
        passed_e_charged = False
 
        # ===================================================================
        # SINGLE EVENT SELECTION (P-scheme all-particle)
        # All four thrust variants and all four response matrices are gated
        # on this same selection. No per-variant recomputation.
        # ===================================================================
        theta_thrust_p = thrust_theta(thrust_vals['axis'], thrust_vals['T'], fold=False)
        event_sel = apply_event_selection_delphi(
            e_c=e_c,
            e_n=e_all,
            theta_Tu=theta_thrust_p,
            E_reco=E,
        )
 
        is_nominal = (syst_name == 'nominal')
 
        if event_sel['pass_reco']:
            passed_p_all = True
            passed_e_all = True
            passed_p_charged = True
            passed_e_charged = True
 
            h = reco_hists[syst_name]
 
            # --- P-scheme all-particle ---
            h[f'ThrustMissPNC2_{syst_name}'].Fill(thrust_vals['thrust'], evt_weight)
            h[f'ThrustMissPNCLog2_{syst_name}'].Fill(thrust_vals['thrust_log'], evt_weight)
 
            # --- E-scheme all-particle ---
            h[f'ThrustMissPNC2_Escheme_{syst_name}'].Fill(thrust_vals['thrust_escheme'], evt_weight)
            h[f'ThrustMissPNCLog2_Escheme_{syst_name}'].Fill(thrust_vals['thrust_log_escheme'], evt_weight)
 
            # --- Charged P-scheme ---
            h[f'ThrustC2_{syst_name}'].Fill(thrust_vals['thrust_charged'], evt_weight)
            h[f'ThrustCLog2_{syst_name}'].Fill(thrust_vals['thrust_charged_log'], evt_weight)
 
            # --- Charged E-scheme ---
            h[f'ThrustC2_Escheme_{syst_name}'].Fill(thrust_vals['thrust_charged_escheme'], evt_weight)
            h[f'ThrustCLog2_Escheme_{syst_name}'].Fill(thrust_vals['thrust_charged_log_escheme'], evt_weight)
 
            # --- Response matrices (MC only) ---
            if is_mc and thrust_gen is not None:
                M = response_matrices[syst_name]
                M[f'response_thrust2_{syst_name}'].Fill(
                    thrust_vals['thrust'], thrust_gen['thrust'])
                M[f'response_thrust_log2_{syst_name}'].Fill(
                    thrust_vals['thrust_log'], thrust_gen['thrust_log'])
                M[f'response_thrust2_Escheme_{syst_name}'].Fill(
                    thrust_vals['thrust_escheme'], thrust_gen['thrust_escheme'])
                M[f'response_thrust_log2_Escheme_{syst_name}'].Fill(
                    thrust_vals['thrust_log_escheme'], thrust_gen['thrust_log_escheme'])
                M[f'response_thrustC2_{syst_name}'].Fill(
                    thrust_vals['thrust_charged'], thrust_gen['thrust_charged'])
                M[f'response_thrustC_log2_{syst_name}'].Fill(
                    thrust_vals['thrust_charged_log'], thrust_gen['thrust_charged_log'])
                M[f'response_thrustC2_Escheme_{syst_name}'].Fill(
                    thrust_vals['thrust_charged_escheme'], thrust_gen['thrust_charged_escheme'])
                M[f'response_thrustC_log2_Escheme_{syst_name}'].Fill(
                    thrust_vals['thrust_charged_log_escheme'], thrust_gen['thrust_charged_log_escheme'])
 
            # --- Reco moment accumulators ---
            if moment_hists is not None:
                fill_moments_all_variants(
                    moment_hists['reco'][syst_name],
                    scenario='reco',
                    syst_name=syst_name,
                    thrust_vals=thrust_vals,
                    weight=evt_weight,
                    tau_max=TAU_MAX_PHYS,
                )
 
                # --- gen_passReco moment accumulators (MC only) ---
                # Uses the GEN-level thrust values for this same event,
                # gated by the reco event selection passing.
                if is_mc and thrust_gen is not None:
                    fill_moments_all_variants(
                        moment_hists['gen_passReco'][syst_name],
                        scenario='gen_passReco',
                        syst_name=syst_name,
                        thrust_vals=thrust_gen,
                        weight=evt_weight,
                        tau_max=TAU_MAX_PHYS,
                    )
 
        # After event_sel block, fill counter for each scenario
        if is_nominal:
            if passed_p_all and counter is not None:
                counter.Fill(1.5)  # Bin 2
            if passed_e_all and counter is not None:
                counter.Fill(2.5)  # Bin 3
            if passed_p_charged and counter is not None:
                counter.Fill(3.5)  # Bin 4
            if passed_e_charged and counter is not None:
                counter.Fill(4.5)  # Bin 5
 
        if syst_config.get('use_evt_weight'):
            if passed_p_all and counter_weighted is not None:
                counter_weighted.Fill(1.5, evt_weight)
            if passed_e_all and counter_weighted is not None:
                counter_weighted.Fill(2.5, evt_weight)
            if passed_p_charged and counter_weighted is not None:
                counter_weighted.Fill(3.5, evt_weight)
            if passed_e_charged and counter_weighted is not None:
                counter_weighted.Fill(4.5, evt_weight)

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """
    Main function to run optimized thrust analysis.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Optimized single-pass thrust analysis with systematic variations'
    )
    parser.add_argument('infile', help='Input ROOT file')
    parser.add_argument('outfile', help='Output ROOT file')
    parser.add_argument('--is-mc', action='store_true', 
                        help='Process as MC (fills gen histograms and response matrices)')
    parser.add_argument('--systematic', default='all',
                        help='Run specific systematic or "all" (default: all)')
    
    args = parser.parse_args()
    
    # Determine which systematics to run
    if args.systematic == 'all':
        systematics_to_run = SYSTEMATICS
    elif args.systematic in SYSTEMATICS:
        systematics_to_run = {args.systematic: SYSTEMATICS[args.systematic]}
    else:
        print(f"ERROR: Unknown systematic '{args.systematic}'")
        print(f"Available systematics: {', '.join(SYSTEMATICS.keys())}")
        return 1
    
    print(f"Running with systematics: {', '.join(systematics_to_run.keys())}")
    
    # Open input file
    fin = ROOT.TFile.Open(args.infile, 'READ')
    if not fin or fin.IsZombie():
        print(f"ERROR: Cannot open input file {args.infile}")
        return 1
    
    # Get trees
    tree_reco = fin.Get('t')
    if not tree_reco:
        print("ERROR: Cannot find tree 't' in input file")
        return 1
    
    tree_gen_before = None
    tree_gen = None
    if args.is_mc:
        tree_gen_before = fin.Get('tgenBefore')
        tree_gen = fin.Get('tgen')
        if not tree_gen_before or not tree_gen:
            print("ERROR: MC mode requested but cannot find 'tgenBefore' or 'tgen' trees")
            return 1
    
    n_events = tree_reco.GetEntries()
    print(f"Processing {n_events} events from {args.infile}")
    print(f"MC mode: {args.is_mc}")
    
    # Create histograms
    print("Creating histograms...")
    
    # Create counter histogram
    counter = create_counter_histogram()
    counter_weighted = create_counter_weighted_histogram()
    
    gen_hists = None
    if args.is_mc:
        gen_hists = create_gen_histograms()
    
    reco_hists = {}
    for syst_name in systematics_to_run.keys():
        reco_hists[syst_name] = create_reco_histograms(syst_name)

    moment_hists = create_all_moment_histograms(systematics_to_run)
    
    # Create response matrices (all systematics in MC)
    response_matrices = {}
    if args.is_mc:
        for syst_name in systematics_to_run.keys():
            response_matrices[syst_name] = create_response_matrices(syst_name)
        print(f"Created response matrices for all {len(systematics_to_run)} systematics")
    
    print(f"Created histograms for {len(systematics_to_run)} systematics")
    
    # Event loop
    print("Starting event loop...")
    n_processed = 0
    
    for iEvt in range(n_events):
        if iEvt % 1000 == 0:
            print(f"  Processing event {iEvt}/{n_events}")
        
        # Count total events (bin 1)
        counter.Fill(0.5)

        if args.is_mc:
            # Call once with gen_hists to fill "before" histograms
            process_gen_level(
                tree_gen_before, tree_gen, iEvt, gen_hists,
                charged_p_min=0.4,  # Doesn't matter, gen_hists filling doesn't use these
                neutral_e_min=0.5,
                moment_hists=moment_hists,
            )
        
        # Process gen level (MC only)
        thrust_gen_dict = {}
        if args.is_mc:
            for syst_name, syst_config in systematics_to_run.items():
                thrust_gen_dict[syst_name] = process_gen_level(
                    tree_gen_before, tree_gen, iEvt, None,  # gen_hists=None, don't fill again
                    charged_p_min=syst_config.get('charged_p_min', 0.4),
                    neutral_e_min=syst_config.get('neutral_e_min', 0.5),
                )
        
        # Process reco level (always)
        process_reco_level(
            tree_reco, iEvt,
            systematics_to_run,
            reco_hists,
            response_matrices,
            thrust_gen_dict=thrust_gen_dict,
            is_mc=args.is_mc,
            counter=counter,
            counter_weighted=counter_weighted,
            moment_hists=moment_hists, 
        )
        
        n_processed += 1
    
    print(f"\nProcessed {n_processed} events")
    print(f"Event counts:")
    print(f"  Total events: {int(counter.GetBinContent(1))}")
    print(f"  Events passing selection: {int(counter.GetBinContent(2))}")
    if args.is_mc:
        print(f"  Gen events passing selection: {int(counter.GetBinContent(3))}")
    
    # Write output
    print(f"\nWriting output to {args.outfile}...")
    fout = ROOT.TFile(args.outfile, 'RECREATE')
    
    # Write counter histogram at top level
    fout.cd()
    counter.Write()
    counter_weighted.Write()
    
    # Write gen histograms
    if gen_hists:
        fout.mkdir('gen')
        fout.cd('gen')
        for hist in gen_hists.values():
            hist.Write()
        fout.cd()
    
    # Write reco histograms
    fout.mkdir('reco')
    fout.cd('reco')
    for syst_name, hists in reco_hists.items():
        for hist in hists.values():
            hist.Write()
    fout.cd()
    
    # Write response matrices
    if response_matrices:
        fout.mkdir('response')
        fout.cd('response')
        for syst_name, matrices in response_matrices.items():
            for matrix in matrices.values():
                matrix.Write()
        fout.cd()

    write_moment_histograms(fout, moment_hists, top_dir='moments')
    
    fout.Close()
    fin.Close()
    
    print("Done!")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
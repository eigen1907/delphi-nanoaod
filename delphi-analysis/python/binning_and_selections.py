import ROOT
import numpy as np, dataclasses as dc
import sys
import os
import argparse
import math
import itertools
from array import array
from common_functions import *


# Theta: 0-180 degrees, 1 degree bins
theta_bins = np.linspace(0, 180, 181)

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

eijbins = [0.0, 0.0001, 0.00012589254117941674, 0.00015848931924611142, 0.00019952623149688788, 0.00025118864315095795, 0.00031622776601683794, 0.00039810717055349735, 0.0005011872336272725, 0.000630957344480193, 0.0007943282347242813, 0.001, 0.0012589254117941675, 0.001584893192461114, 0.001995262314968879, 0.002511886431509582, 0.0031622776601683794, 0.003981071705534973, 0.005011872336272725, 0.00630957344480193, 0.00794328234724282, 0.01, 0.012589254117941675, 0.01584893192461114, 0.01995262314968881, 0.025118864315095822, 0.03162277660168379, 0.039810717055349734, 0.05011872336272725, 0.06309573444801936, 0.07943282347242822, 0.1, 1]

# Alternative eij binning (28 bins) for the 2D-binning systematic uncertainty.
eijbins_alt = [0.0, 0.0001, 0.0002, 0.0005, 0.00075, 0.001, 0.00125, 0.0015, 0.00175, 0.002, 0.00225, 0.0025, 0.00275, 0.003, 0.0035, 0.004, 0.005, 0.007, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10, 0.15, 0.20, 0.3, 1]

evalues = np.linspace(0, 40, 41)
ebins = np.append(evalues, [42, 45, 200])

rbins = calcBinEdge(0.002, np.pi/2, 100)
zbins = calcBinEdge(0.000001, 0.5, 100)

tbins = np.linspace(-0.1, 0.5, 61)
logtbins = np.linspace(-10, 0, 101)

tbins2 = [-0.1, 0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10,
          0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 
          0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 
          0.4, 0.50, 1.0] 

tbinsDelphi = [-0.1, 0.0, 0.01, 0.02, 0.03, 0.04,
         0.05, 0.06, 0.07, 0.08, 0.09,
         0.10, 0.11, 0.12, 0.14, 0.16,
         0.18, 0.20, 0.25, 0.30, 0.35,
         0.40, 0.50, 1.0]

logtbins2 = [ -10., -8.,  -7., -6.5,
  -6.4, -6.3, -6.2, -6.1, -6., -5.9, -5.8, -5.7, -5.6, -5.5, -5.4, -5.3,
  -5.2, -5.1, -5., -4.9, -4.8, -4.7, -4.6, -4.5, -4.4, -4.3, -4.2, -4.1,
  -4., -3.9, -3.8, -3.7, -3.6, -3.5, -3.4, -3.3, -3.2, -3.1, -3., -2.9,
  -2.8, -2.7, -2.6, -2.5, -2.4, -2.3, -2.2, -2.1, -2., -1.9, -1.8, -1.7,
  -1.6, -1.5, -1.4, -1.3, -1.2, -1.1, -1., -0.9, -0.8, -0.7, -0.6, -0.5,
  -0.4, -0.3, -0.2, -0.1, 0.]

eijbins = np.array(eijbins, dtype=np.float64)
eijbins_alt = np.array(eijbins_alt, dtype=np.float64)
rbins = np.array(rbins, dtype=np.float64)
zbins = np.array(zbins, dtype=np.float64)
tbins = np.array(tbins, dtype=np.float64)
logtbins = np.array(logtbins, dtype=np.float64)
ebins = np.array(ebins, dtype=np.float64)

tbins2 = np.array(tbins2, dtype=np.float64)
tbinsDelphi = np.array(tbinsDelphi, dtype=np.float64)
logtbins2 = np.array(logtbins2, dtype=np.float64)

def conversion_veto_mask(eta, phi, charge, pwflag,
                         conversion_deta, conversion_dphi):
    """
    Parameters
    ----------
    eta, phi, charge, pwflag : array-like, shape (N,)
        Per-track arrays.
        - pwflag==2 marks an electron.
    conversion_deta : float
        Maximum |Δη| between candidate and its partner.
    conversion_dphi : float
        Maximum |Δφ| between candidate and its partner.
    
    Returns
    -------
    mask : ndarray of bool, shape (N,)
        True for tracks that are NOT conversion electrons,
        False for conversions.
    """
    eta   = np.asarray(eta)
    phi   = np.asarray(phi)
    charge= np.asarray(charge)
    pw    = np.asarray(pwflag)
    N     = eta.size

    # Build an array of “previous” indices,
    # with index 0 peeking at 1 so we don’t run off the front:
    prev = np.arange(N) - 1
    prev[0] = 1

    # 1) both must be electrons (pwflag==2)
    both_elec = (pw == 2) & (pw[prev] == 2)
    # 2) opposite charge
    opp_charge = (charge == -charge[prev])
    # 3) |Δη| small
    deta = np.abs(eta - eta[prev])
    pass_deta  = (deta <= conversion_deta)
    # 4) |Δφ| small, accounting for periodicity
    dphi = np.arccos(np.cos(phi - phi[prev]))
    pass_dphi  = (dphi <= conversion_dphi)

    # conversion if all four conditions hold
    is_conv = both_elec & opp_charge & pass_deta & pass_dphi

    # now propagate back onto prev[i]
    conv_prev = np.zeros_like(is_conv, dtype=bool)
    conv_prev[prev] = is_conv

    # any track that is_conv or conv_prev is a conversion
    conv_any = is_conv | conv_prev

    # final mask: True = keep (not conversion), False = drop
    return ~conv_any

def conversion_veto_mask_bypid(eta, phi, charge, pwflag,
                               conversion_deta, conversion_dphi):
    """
    Parameters
    ----------
    eta, phi, charge, pwflag : array-like, shape (N,)
        Per-track arrays.
        - pwflag==2 marks an electron.
    conversion_deta : float
        Maximum |Δη| between candidate and its partner.
    conversion_dphi : float
        Maximum |Δφ| between candidate and its partner.
    
    Returns
    -------
    mask : ndarray of bool, shape (N,)
        True for tracks that are NOT conversion electrons,
        False for conversions.
    """
    eta   = np.asarray(eta)
    phi   = np.asarray(phi)
    charge= np.asarray(charge)
    pw    = abs(np.asarray(pwflag))
    N     = eta.size
 
    # Build an array of “previous” indices,
    # with index 0 peeking at 1 so we don’t run off the front:
    prev = np.arange(N) - 1
    prev[0] = 1

    # 1) both must be electrons (pwflag==2)
    both_elec = (pw == 11) & (pw[prev] == 11)
    # 2) opposite charge
    opp_charge = (charge == -charge[prev])
    # 3) |Δη| small
    deta = np.abs(eta - eta[prev])
    pass_deta  = (deta <= conversion_deta)
    # 4) |Δφ| small, accounting for periodicity
    dphi = np.arccos(np.cos(phi - phi[prev]))
    pass_dphi  = (dphi <= conversion_dphi)

    # conversion if all four conditions hold
    is_conv = both_elec & opp_charge & pass_deta & pass_dphi

    # now propagate back onto prev[i]
    conv_prev = np.zeros_like(is_conv, dtype=bool)
    conv_prev[prev] = is_conv

    # any track that is_conv or conv_prev is a conversion
    conv_any = is_conv | conv_prev

    # final mask: True = keep (not conversion), False = drop
    return ~conv_any

def apply_track_selection_delphi(px=None, py=None, pz=None, m=None, q=None, th=None, pt=None, eta=None, phi=None, d0=None, z0=None, pwflag=None, hp=None,
                                 px_gen=None, py_gen=None, pz_gen=None, m_gen=None, q_gen=None, th_gen=None, pt_gen=None, eta_gen=None, phi_gen=None, pwflag_gen=None, hp_gen=None,
                                 charged_pt_min=0.4,
                                 charged_p_min=None,
                                 charged_d0_max=4.0,
                                 charged_z0sin_max=4.0,
                                 neutral_e_min=0.5,
                                 neutral_e_max=50.0,
                                 theta_min_deg=20.0,
                                 theta_max_deg=160.0):
    """
    Apply track selection criteria for reconstructed and/or generated particles.
    
    Parameters:
    -----------
    px, py, pz, m, q, th, pt, eta, phi, d0, z0, pwflag, hp : arrays, optional
        Reconstructed particle properties
    px_gen, py_gen, pz_gen, m_gen, q_gen, th_gen, pt_gen, eta_gen, phi_gen, pwflag_gen, hp_gen : arrays, optional
        Generated particle properties
    charged_pt_min : float, optional
        Minimum pT for charged particles (GeV). Default: 0.4
    charged_p_min : float, optional
        Minimum |p| for charged particles (GeV). If set (not None), replaces
        the pT cut with a total-momentum cut. Default: None.
    charged_d0_max : float, optional
        Maximum |d0| for charged particles (cm). Default: 4.0
    charged_z0sin_max : float, optional
        Maximum |z0*sin(theta)| for charged particles (cm). Default: 4.0
    neutral_e_min : float, optional
        Minimum energy for neutral particles (GeV). Default: 0.5
    neutral_e_max : float, optional
        Maximum energy for neutral particles (GeV). Default: 50.0
    theta_min_deg : float, optional
        Minimum polar angle (degrees). Default: 20.0
    theta_max_deg : float, optional
        Maximum polar angle (degrees). Default: 160.0
    
    Returns:
    --------
    results : dict
        Dictionary containing selection masks for available levels:
        - 'sel_c': Selection mask for reconstructed charged particles (if reco data provided)
        - 'sel_n': Selection mask for reconstructed neutral particles (if reco data provided)
        - 'sel': Selection mask for all reconstructed particles (if reco data provided)
        - 'sel_c_gen': Selection mask for generated charged particles (if gen data provided)
        - 'sel_n_gen': Selection mask for generated neutral particles (if gen data provided)
        - 'sel_gen': Selection mask for all generated particles (if gen data provided)
    """
    
    results = {}
    
    # Check if reconstructed data is provided
    has_reco = q is not None and th is not None and pt is not None and d0 is not None and z0 is not None
    
    # Check if generated data is provided
    has_gen = q_gen is not None and th_gen is not None and pt_gen is not None
    
    if not has_reco and not has_gen:
        raise ValueError("Must provide either reconstructed or generated level data (or both)")
    
    # ================================================================
    # RECONSTRUCTED LEVEL SELECTION
    # ================================================================
    if has_reco:
        # Basic kinematic cuts
        abs_c = np.abs(q)
        angle_accept = (th > np.deg2rad(theta_min_deg)) & (th < np.deg2rad(theta_max_deg))
        
        # Track quality cuts for reconstructed particles
        sin_th = np.sin(th)
        abs_d0 = np.abs(d0)
        abs_z0sin = np.abs(z0 * sin_th)

        e = np.sqrt(px**2 + py**2 + pz**2 + m**2)

        # Charged momentum cut: |p| if charged_p_min is set, else pT
        if charged_p_min is not None:
            p_mag = np.sqrt(px**2 + py**2 + pz**2)
            charged_mom_pass = (p_mag > charged_p_min)
        else:
            charged_mom_pass = (pt > charged_pt_min)

        # Charged particle selection (reconstructed)
        core_charged = (abs_c > 0.1) & charged_mom_pass & (abs_d0 < charged_d0_max) & (abs_z0sin < charged_z0sin_max)
        sel_c = core_charged & angle_accept
        
        # Neutral particle selection (reconstructed)
        core_neutral = (abs_c < 0.1) & (e > neutral_e_min) & (e < neutral_e_max)
        sel_n = core_neutral & angle_accept
        
        # Combined selection (charged + neutral, reconstructed)
        sel = (core_charged | core_neutral) & angle_accept
        
        results['sel_c'] = sel_c
        results['sel_n'] = sel_n
        results['sel'] = sel
    
    # ================================================================
    # GENERATED LEVEL SELECTION
    # ================================================================
    if has_gen:
        # Basic kinematic cuts for generated particles
        abs_c_gen = np.abs(q_gen)
        angle_accept_gen = (th_gen > np.deg2rad(theta_min_deg)) & (th_gen < np.deg2rad(theta_max_deg))

        e_gen = np.sqrt(px_gen**2 + py_gen**2 + pz_gen**2 + m_gen**2)
        
        # Charged particle selection (generated)
        # Note: No track quality cuts (d0, z0) at generator level
        if charged_p_min is not None:
            p_mag_gen = np.sqrt(px_gen**2 + py_gen**2 + pz_gen**2)
            charged_mom_pass_gen = (p_mag_gen > charged_p_min)
        else:
            charged_mom_pass_gen = (pt_gen > charged_pt_min)
        core_charged_gen = (abs_c_gen > 0.1) & charged_mom_pass_gen
        sel_c_gen = core_charged_gen & angle_accept_gen
        
        # Neutral particle selection (generated)
        core_neutral_gen = (abs_c_gen < 0.1) & (e_gen > neutral_e_min) & (e_gen < neutral_e_max)
        sel_n_gen = core_neutral_gen & angle_accept_gen
        
        # Combined selection (charged + neutral, generated)
        sel_gen = (core_charged_gen | core_neutral_gen) & angle_accept_gen
        
        results['sel_c_gen'] = sel_c_gen
        results['sel_n_gen'] = sel_n_gen
        results['sel_gen'] = sel_gen
    
    return results

def apply_track_selection_aleph(px=None, py=None, pz=None, m=None, q=None, th=None, pt=None, eta=None, phi=None, d0=None, z0=None, pwflag=None, hp=None, 
                                px_gen=None, py_gen=None, pz_gen=None, m_gen=None, q_gen=None, th_gen=None, pt_gen=None, eta_gen=None, phi_gen=None, pwflag_gen=None, hp_gen=None):
    """
    Apply track selection criteria for reconstructed and/or generated particles.
    
    Parameters:
    -----------
    px, py, pz, m, q, th, pt, eta, phi, d0, z0, pwflag, hp : arrays, optional
        Reconstructed particle properties
    px_gen, py_gen, pz_gen, m_gen, q_gen, th_gen, pt_gen, eta_gen, phi_gen, pwflag_gen, hp_gen : arrays, optional
        Generated particle properties
    
    Returns:
    --------
    results : dict
        Dictionary containing selection masks for available levels:
        - 'sel_c': Selection mask for reconstructed charged particles (if reco data provided)
        - 'sel_n': Selection mask for reconstructed neutral particles (if reco data provided)
        - 'sel': Selection mask for all reconstructed particles (if reco data provided)
        - 'sel_c_gen': Selection mask for generated charged particles (if gen data provided)
        - 'sel_n_gen': Selection mask for generated neutral particles (if gen data provided)
        - 'sel_gen': Selection mask for all generated particles (if gen data provided)
    """
    
    results = {}
    
    # Check if reconstructed data is provided
    has_reco = q is not None and th is not None and pt is not None 
    
    # Check if generated data is provided
    has_gen = q_gen is not None and th_gen is not None and pt_gen is not None and hp_gen is not None
    
    if not has_reco and not has_gen:
        raise ValueError("Must provide either reconstructed or generated level data (or both)")
    
    # ================================================================
    # RECONSTRUCTED LEVEL SELECTION
    # ================================================================
    if has_reco:
        # Basic kinematic cuts
        abs_c = np.abs(q)
        angle_accept_c = (th > np.deg2rad(20)) & (th < np.deg2rad(160))
        angle_accept_n = (th > np.deg2rad(11)) & (th < np.deg2rad(169))
        
        core_charged = (abs_c > 0.1) & (pt > 0.2) & (hp > 0.5)
        sel_c = core_charged & angle_accept_c
        
        # Neutral particle selection (reconstructed)
        core_neutral = (abs_c < 0.1) & (pt > 0.4)
        sel_n = core_neutral & angle_accept_n
        
        # Combined selection (charged + neutral, reconstructed)
        sel = sel_c | sel_n
        
        results['sel_c'] = sel_c
        results['sel_n'] = sel_n
        results['sel'] = sel
    
    # ================================================================
    # GENERATED LEVEL SELECTION
    # ================================================================
    if has_gen:
        # Basic kinematic cuts for generated particles
        abs_c_gen = np.abs(q_gen)
        
        # Charged particle selection (generated)
        # Note: No track quality cuts (d0, z0) at generator level
        core_charged_gen = (abs_c_gen > 0.1) & (hp_gen > 0.5)
        sel_c_gen = core_charged_gen
        
        # Neutral particle selection (generated)
        core_neutral_gen = (abs_c_gen < 0.1) & (pt_gen > 0.4)
        sel_n_gen = core_neutral_gen
        
        # Combined selection (charged + neutral, generated)
        sel_gen = sel_c_gen | sel_n_gen
        
        results['sel_c_gen'] = sel_c_gen
        results['sel_n_gen'] = sel_n_gen
        results['sel_gen'] = sel_gen
    
    return results

def apply_event_selection_delphi(e_c=None, e_n=None, e_gen_c=None, e_gen_n=None, theta_Tu=None, theta_gen_Tu=None, E_reco=None, E_gen=None):
    """
    Apply event-level selection criteria for reconstructed and/or generated events.
    
    Parameters:
    -----------
    e_c, e_n : arrays, optional
        Energies of charged and all selected reconstructed particles
    e_gen_c, e_gen_n : arrays, optional
        Energies of charged and all selected generated particles  
    theta_Tu, theta_gen_Tu : float, optional
        Thrust angles for reconstructed and generated events
    E_reco, E_gen : float, optional
        Total beam energies for reconstructed and generated events
        
    Returns:
    --------
    results : dict
        Dictionary containing:
        - 'pass_reco': Whether reconstructed event passes selection (if reco data provided)
        - 'pass_gen': Whether generated event passes selection (if gen data provided)
    """
    
    results = {}
    
    # Check if reconstructed data is provided
    has_reco = e_c is not None and e_n is not None and theta_Tu is not None and E_reco is not None
    
    # Check if generated data is provided
    has_gen = e_gen_c is not None and e_gen_n is not None and theta_gen_Tu is not None and E_gen is not None
    
    if not has_reco and not has_gen:
        raise ValueError("Must provide either reconstructed or generated level data (or both)")
    
    # Event selection criteria
    min_charged_tracks = 7
    min_energy_fraction = 0.5
    min_thrust_angle = np.deg2rad(30)
    max_thrust_angle = np.deg2rad(150)
    
    # Reconstructed level event selection
    if has_reco:
        pass_reco = (
            (len(e_c) >= min_charged_tracks) &
            (np.sum(e_n) >= min_energy_fraction * E_reco) &
            (theta_Tu >= min_thrust_angle) &
            (theta_Tu <= max_thrust_angle)
        )
        results['pass_reco'] = pass_reco
    
    # Generated level event selection
    if has_gen:
        pass_gen = (
            (len(e_gen_c) >= min_charged_tracks) &
            (np.sum(e_gen_n) >= min_energy_fraction * E_gen) &
            (theta_gen_Tu >= min_thrust_angle) &
            (theta_gen_Tu <= max_thrust_angle)
        )
        results['pass_gen'] = pass_gen
    
    return results

def apply_event_selection_aleph(e_c=None, e_nc=None, e_gen_c=None, e_gen_nc=None, sphericity=None, sphericity_gen=None):
    """
    Apply event-level selection criteria for reconstructed and/or generated events.
    
    Parameters:
    -----------
    e_c, e_n : arrays, optional
        Energies of charged and all selected reconstructed particles
    e_gen_c, e_gen_n : arrays, optional
        Energies of charged and all selected generated particles  
    sphericity, sphericity_gen : float, optional
        Sphericity values for reconstructed and generated events
        
    Returns:
    --------
    results : dict
        Dictionary containing:
        - 'pass_reco': Whether reconstructed event passes selection (if reco data provided)
        - 'pass_gen': Whether generated event passes selection (if gen data provided)
    """
    
    results = {}
    
    # Check if reconstructed data is provided
    has_reco = e_c is not None and sphericity is not None and e_nc is not None
    
    # Check if generated data is provided
    has_gen = e_gen_c is not None and sphericity_gen is not None and e_gen_nc is not None
    
    if not has_reco and not has_gen:
        raise ValueError("Must provide either reconstructed or generated level data (or both)")
    
    # Event selection criteria
    min_charged_tracks = 5
    min_charged_energy = 15
    min_particles = 13
    min_s_angle = -0.82
    max_s_angle = 0.82
    
    # Reconstructed level event selection
    if has_reco:
        pass_reco = (
            (len(e_c) >= min_charged_tracks) &
            (np.sum(e_c) >= min_charged_energy) &
            (len(e_nc) >= min_particles) &
            (abs(sphericity) >= min_s_angle) &
            (abs(sphericity) <= max_s_angle)
        )
        results['pass_reco'] = pass_reco
    
    # Generated level event selection
    if has_gen:
        pass_gen = (
            (len(e_gen_c) >= min_charged_tracks) &
            (np.sum(e_gen_c) >= min_charged_energy) &
            (len(e_gen_nc) >= min_particles) &
            (abs(sphericity_gen) >= min_s_angle) &
            (abs(sphericity_gen) <= max_s_angle)
        )
        results['pass_gen'] = pass_gen
    
    return results

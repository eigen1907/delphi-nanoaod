#!/usr/bin/env python3
"""
Thrust Moments Infrastructure
==============================
Per-event moment computation and histogram bookkeeping for the thrust analysis.

Three families of moments are computed for each thrust variant:

  1. Mellin-log unified table:
        f(N, M) = < tau^N * log^M(tau) >
     with N in [0, N_MAX] and M in [0, M_MAX]. Setting M=0 gives pure Mellin
     moments; N=0 gives pure log moments. Convention: log(tau) literally
     (signed, negative for tau < 1).

  2. Three Legendre flavors, ell in [0, LEG_MAX]:
       - "raw"  : <P_ell(tau)>            argument tau on [0, 1]
       - "eec"  : <P_ell(1 - 2*tau)>      maps [0,1] -> [-1,1]
       - "phys" : <P_ell(1 - 2*tau/tau_max)>  maps [0, tau_max] -> [-1,1]

For each, the bin contents store sum_i w_i * (moment integrand), and a
companion "cov" histogram stores the cross-term sums needed to build the
statistical covariance matrix in post-processing.

Mellin-log uses an index-sum compression for the cov: the (N1,M1)-(N2,M2)
covariance only depends on (N1+N2, M1+M2), so a single 2D histogram of size
(2*N_MAX+1) x (2*M_MAX+1) captures all 36*37/2 unique entries.

Legendre cov accumulators store the full ell1 x ell2 matrix directly, since
no analogous compression exists.
"""

import numpy as np
import ROOT
from numpy.polynomial.legendre import legval

# ============================================================================
# CONFIGURATION
# ============================================================================

N_MAX = 5      # Mellin power: tau^N for N in 0..N_MAX
M_MAX = 5      # Log power:    log^M(tau) for M in 0..M_MAX
LEG_MAX = 20   # Legendre order: ell in 0..LEG_MAX

TAU_MAX_PHYS = 0.5  # kinematic max of tau = 1-T (used by "phys" Legendre flavor)

# Convenient sizes
N_MELLIN = N_MAX + 1                 # 6
N_MELLIN_COV = 2 * N_MAX + 1         # 11
N_LOG = M_MAX + 1                    # 6
N_LOG_COV = 2 * M_MAX + 1            # 11
N_LEG = LEG_MAX + 1                  # 21

# The four thrust variants we fill moments for
THRUST_VARIANTS = ['p_all', 'e_all', 'p_charged', 'e_charged']

# Map variant -> (key for the tau value, key for the log(tau) value)
# These keys come from the dict returned by calculate_all_thrust_variants().
VARIANT_TAU_KEYS = {
    'p_all':     ('thrust',                   'thrust_log'),
    'e_all':     ('thrust_escheme',           'thrust_log_escheme'),
    'p_charged': ('thrust_charged',           'thrust_charged_log'),
    'e_charged': ('thrust_charged_escheme',   'thrust_charged_log_escheme'),
}

# Scenarios
#   reco         - reco particles, per syst
#   gen_passReco - gen particles on events that passed reco selection, per syst
#   gen_full     - gen particles on all gen events, nominal only
SCENARIOS = ['reco', 'gen_passReco', 'gen_full']


# ============================================================================
# PER-EVENT MOMENT COMPUTATION
# ============================================================================

# Pre-compute integer power grids once at import time so the per-event call is
# just a couple of broadcasts.
_N_GRID = np.arange(N_MELLIN)              # shape (6,)   for tau^N
_M_GRID = np.arange(N_LOG)                 # shape (6,)   for log^M
_NSUM_GRID = np.arange(N_MELLIN_COV)       # shape (11,)  for tau^(N1+N2)
_MSUM_GRID = np.arange(N_LOG_COV)          # shape (11,)  for log^(M1+M2)

# Legendre coefficient identity matrix: row ell is [0, ..., 0, 1, 0, ...]
# legval(x, _LEG_COEFFS[ell]) = P_ell(x)
_LEG_COEFFS = np.eye(N_LEG)


def _safe_log(tau):
    """Return log(tau), with a floor to avoid -inf for pathological tau<=0."""
    if tau <= 0.0:
        return -50.0
    return np.log(tau)


def calculate_moment_contributions(tau, weight=1.0, tau_max=TAU_MAX_PHYS):
    """
    Compute this event's contribution to all moment accumulators.

    Parameters
    ----------
    tau : float
        The thrust value tau = 1 - T for this event/variant.
    weight : float
        Event weight (for systematic reweighting variations).
    tau_max : float
        Kinematic max of tau, used by the "phys" Legendre flavor.

    Returns
    -------
    dict with keys:
        'mellin_nm'         : 2D array (N_MELLIN, N_LOG)
                              w * tau^N * log^M(tau), the moment values
        'mellin_nm_cov'     : 2D array (N_MELLIN_COV, N_LOG_COV)
                              w^2 * tau^(N1+N2) * log^(M1+M2)(tau)
                              (compressed: indices are the SUM of the pair)
        'leg_raw', 'leg_eec', 'leg_phys'      : 1D array (N_LEG,)
                              w * P_ell(arg) for ell in 0..LEG_MAX
        'leg_raw_cov', 'leg_eec_cov', 'leg_phys_cov' : 2D array (N_LEG, N_LEG)
                              w^2 * P_ell1(arg) * P_ell2(arg)
        'norm'              : float, w
    """
    log_tau = _safe_log(tau)

    # Powers of tau and log(tau) for the moment table
    tau_pow_N = tau ** _N_GRID                       # shape (N_MELLIN,)
    log_pow_M = log_tau ** _M_GRID                   # shape (N_LOG,)
    mellin_nm = weight * np.outer(tau_pow_N, log_pow_M)   # (6,6)

    # Powers for the compressed covariance (uses w^2 and SUMS of indices)
    tau_pow_NSUM = tau ** _NSUM_GRID                 # shape (N_MELLIN_COV,)
    log_pow_MSUM = log_tau ** _MSUM_GRID             # shape (N_LOG_COV,)
    mellin_nm_cov = (weight * weight) * np.outer(tau_pow_NSUM, log_pow_MSUM)  # (11,11)

    # Legendre - three flavors, three different arguments
    arg_raw  = tau
    arg_eec  = 1.0 - 2.0 * tau
    arg_phys = 1.0 - 2.0 * (tau / tau_max if tau_max > 0 else 0.0)

    # Evaluate P_0..P_LEG_MAX at each argument in one shot.
    # legval(x, identity_matrix) returns shape (N_LEG,) with entries P_ell(x).
    P_raw  = legval(arg_raw,  _LEG_COEFFS)           # shape (N_LEG,)
    P_eec  = legval(arg_eec,  _LEG_COEFFS)
    P_phys = legval(arg_phys, _LEG_COEFFS)

    leg_raw  = weight * P_raw
    leg_eec  = weight * P_eec
    leg_phys = weight * P_phys

    leg_raw_cov  = (weight * weight) * np.outer(P_raw,  P_raw)
    leg_eec_cov  = (weight * weight) * np.outer(P_eec,  P_eec)
    leg_phys_cov = (weight * weight) * np.outer(P_phys, P_phys)

    return {
        'mellin_nm':     mellin_nm,
        'mellin_nm_cov': mellin_nm_cov,
        'leg_raw':       leg_raw,
        'leg_eec':       leg_eec,
        'leg_phys':      leg_phys,
        'leg_raw_cov':   leg_raw_cov,
        'leg_eec_cov':   leg_eec_cov,
        'leg_phys_cov':  leg_phys_cov,
        'norm':          float(weight),
    }


# ============================================================================
# HISTOGRAM BOOKKEEPING
# ============================================================================
#
# Naming convention for moment histograms:
#
#   {kind}_{variant}_{scenario}_{syst}
#
# where
#   kind     in { Mellin_NM, Mellin_NM_cov,
#                 Legendre_raw,  Legendre_raw_cov,
#                 Legendre_eec,  Legendre_eec_cov,
#                 Legendre_phys, Legendre_phys_cov,
#                 MomentNorm }
#   variant  in THRUST_VARIANTS
#   scenario in SCENARIOS
#   syst     is the systematic name (always 'nominal' for gen_full)
#
# Bin indexing (ROOT is 1-indexed):
#   Mellin_NM     : x bin = N+1, y bin = M+1, with N in 0..N_MAX, M in 0..M_MAX
#   Mellin_NM_cov : x bin = (N1+N2)+1, y bin = (M1+M2)+1
#   Legendre_*    : bin = ell+1, with ell in 0..LEG_MAX
#   Legendre_*_cov: x bin = ell1+1, y bin = ell2+1
#   MomentNorm    : single bin (1) holding sum of weights
# ============================================================================


def _name(kind, variant, scenario, syst):
    return f'{kind}_{variant}_{scenario}_{syst}'


def create_moment_histograms(scenario, syst_name):
    """
    Create the full set of moment accumulator histograms for one
    (scenario, systematic) combination, across all 4 thrust variants.

    Returns a flat dict {hist_name: TH1/TH2 object}.
    """
    hists = {}

    for variant in THRUST_VARIANTS:

        # --- Mellin-log values: TH2D with axes (N, M) ---
        n = _name('Mellin_NM', variant, scenario, syst_name)
        hists[n] = ROOT.TH2D(n, ';N;M',
                             N_MELLIN, -0.5, N_MELLIN - 0.5,
                             N_LOG,    -0.5, N_LOG    - 0.5)

        # --- Mellin-log covariance (compressed): TH2D with axes (N1+N2, M1+M2) ---
        n = _name('Mellin_NM_cov', variant, scenario, syst_name)
        hists[n] = ROOT.TH2D(n, ';N1+N2;M1+M2',
                             N_MELLIN_COV, -0.5, N_MELLIN_COV - 0.5,
                             N_LOG_COV,    -0.5, N_LOG_COV    - 0.5)

        # --- Legendre values and covariances, three flavors ---
        for flavor in ('raw', 'eec', 'phys'):
            nv = _name(f'Legendre_{flavor}', variant, scenario, syst_name)
            hists[nv] = ROOT.TH1D(nv, ';#ell',
                                  N_LEG, -0.5, N_LEG - 0.5)

            nc = _name(f'Legendre_{flavor}_cov', variant, scenario, syst_name)
            hists[nc] = ROOT.TH2D(nc, ';#ell_{1};#ell_{2}',
                                  N_LEG, -0.5, N_LEG - 0.5,
                                  N_LEG, -0.5, N_LEG - 0.5)

        # --- Norm: single bin ---
        n = _name('MomentNorm', variant, scenario, syst_name)
        hists[n] = ROOT.TH1D(n, ';;#sum w', 1, 0.0, 1.0)

    return hists


def create_all_moment_histograms(systematics):
    """
    Build the full set of moment histograms for the analysis.

    Layout:
      moment_hists['reco'][syst_name]         -> dict of histograms
      moment_hists['gen_passReco'][syst_name] -> dict of histograms
      moment_hists['gen_full']['nominal']     -> dict of histograms

    Parameters
    ----------
    systematics : dict
        The SYSTEMATICS dict from analysis_thrust.py (keys are systematic names).
    """
    moment_hists = {scenario: {} for scenario in SCENARIOS}

    for syst_name in systematics.keys():
        moment_hists['reco'][syst_name] = create_moment_histograms('reco', syst_name)
        moment_hists['gen_passReco'][syst_name] = create_moment_histograms(
            'gen_passReco', syst_name)

    # gen_full: nominal only
    moment_hists['gen_full']['nominal'] = create_moment_histograms(
        'gen_full', 'nominal')

    return moment_hists


# ============================================================================
# FILL HELPERS
# ============================================================================

def _fill_moment_histograms_for_variant(hists, variant, scenario, syst_name, contrib):
    """
    Fill the 9 moment histograms for one variant from a single event's
    moment contributions dict (output of calculate_moment_contributions).

    Uses AddBinContent so that what's stored is exactly the running sum of the
    contributions, with no Sumw2 double counting. The accumulators carry their
    own statistical information through the dedicated _cov histograms.
    """
    # Mellin-log values
    h = hists[_name('Mellin_NM', variant, scenario, syst_name)]
    nm = contrib['mellin_nm']
    for iN in range(N_MELLIN):
        for iM in range(N_LOG):
            b = h.GetBin(iN + 1, iM + 1)
            h.AddBinContent(b, float(nm[iN, iM]))

    # Mellin-log covariance (compressed)
    h = hists[_name('Mellin_NM_cov', variant, scenario, syst_name)]
    nmc = contrib['mellin_nm_cov']
    for iN in range(N_MELLIN_COV):
        for iM in range(N_LOG_COV):
            b = h.GetBin(iN + 1, iM + 1)
            h.AddBinContent(b, float(nmc[iN, iM]))

    # Legendre, three flavors
    for flavor in ('raw', 'eec', 'phys'):
        # values
        h = hists[_name(f'Legendre_{flavor}', variant, scenario, syst_name)]
        v = contrib[f'leg_{flavor}']
        for ell in range(N_LEG):
            h.AddBinContent(ell + 1, float(v[ell]))

        # covariance
        h = hists[_name(f'Legendre_{flavor}_cov', variant, scenario, syst_name)]
        c = contrib[f'leg_{flavor}_cov']
        for i in range(N_LEG):
            for j in range(N_LEG):
                b = h.GetBin(i + 1, j + 1)
                h.AddBinContent(b, float(c[i, j]))

    # Normalization
    h = hists[_name('MomentNorm', variant, scenario, syst_name)]
    h.AddBinContent(1, contrib['norm'])


def fill_moments_all_variants(hists, scenario, syst_name, thrust_vals, weight,
                              tau_max=TAU_MAX_PHYS):
    """
    Compute and fill moments for all 4 thrust variants from one event.

    Parameters
    ----------
    hists : dict
        The moment histogram dict for this (scenario, syst), from
        create_moment_histograms().
    scenario : str
        One of SCENARIOS.
    syst_name : str
        Systematic name.
    thrust_vals : dict
        Output of calculate_all_thrust_variants(), containing
        thrust/thrust_escheme/thrust_charged/thrust_charged_escheme.
    weight : float
        Event weight.
    tau_max : float
        Kinematic max for the "phys" Legendre flavor.
    """
    for variant in THRUST_VARIANTS:
        tau_key, _log_key = VARIANT_TAU_KEYS[variant]
        tau = float(thrust_vals[tau_key])

        # Guard against the rare degenerate case where tau is unphysical.
        # tau == 0 also catches the "no charged particles" sentinel set by
        # calculate_all_thrust_variants when sel_c is empty (the charged
        # thrust value defaults to 0.0 there); without this skip, log(tau)
        # is floored to -50 and the pure-log moments get polluted.
        if not np.isfinite(tau) or tau <= 0.0:
            continue

        contrib = calculate_moment_contributions(tau, weight=weight, tau_max=tau_max)
        _fill_moment_histograms_for_variant(
            hists, variant, scenario, syst_name, contrib)


# ============================================================================
# OUTPUT WRITING
# ============================================================================

def write_moment_histograms(fout, moment_hists, top_dir='moments'):
    """
    Write all moment histograms to a TFile under fout/top_dir/, organized as

      moments/reco/<syst>/<all 9*4 hists>
      moments/gen_passReco/<syst>/<all 9*4 hists>
      moments/gen_full/nominal/<all 9*4 hists>

    Parameters
    ----------
    fout : ROOT.TFile
        Open output file (caller is responsible for opening/closing).
    moment_hists : dict
        Output of create_all_moment_histograms.
    top_dir : str
        Top-level directory name in the file.
    """
    fout.mkdir(top_dir)
    for scenario, syst_dict in moment_hists.items():
        scen_dir = f'{top_dir}/{scenario}'
        fout.mkdir(scen_dir)
        for syst_name, hdict in syst_dict.items():
            d = f'{scen_dir}/{syst_name}'
            fout.mkdir(d)
            fout.cd(d)
            for h in hdict.values():
                h.Write()
    fout.cd()
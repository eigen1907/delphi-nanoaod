#!/usr/bin/env python3
"""
FBCEC + FBCC analysis on DelphiNanoAOD (*_ttree.root) input.

NanoAOD counterpart to analysis_fbcec.py. The physics — histogram booking,
systematic variations, FBCEC/FBCC fill, gen-reco matching, covariance —
is unchanged and re-used from analysis_fbcec via the shared
process_gen_arrays / process_reco_arrays entry points. Only the I/O
layer differs.

Gen particles are produced by the SHARED gen builder build_gen_leaf_redecayed
imported from analysis_thrust_nanoaod (the same one thrust/EEC use): V0-aware
select_leaves + Pythia re-decay of any kept strange-V0 leaf, so the gen final
state is in the Pythia V0-decay convention and matches the reco V0 daughters
(2026-07-03 fix, propagated to fbcec 2026-07-08). The energy-balance fallback
('if abs(np.sum(e) - E) > 0.1: fall back to tgen') in
analysis_fbcec.load_gen_particles is no longer needed. Because select_leaves
is used (not copied), this walk descends BOTH strange V0s AND prompt
heavy-flavor (charm/bottom) hadrons — DESCEND_DELPHI_CODES = V0 | heavy-flavor
in analysis_thrust_nanoaod. So the B/D-hadron daughter finding (which recovers
the displaced charged daughters of D/B hadrons that DELSIM tracks separately
in the official kk2f4146_qqpy sample, ~+2.8 charged/event) is active here too;
any earlier note that this reader only did V0 recovery is stale.

Reco-level uses Part_* for kinematics+charge and Trac_* (linked via
Trac_partIdx) for d0/z0 (Trac_impParToVertexRPhi / Trac_impParToVertexZ).

Usage:
  MC:   python analysis_fbcec_nanoaod.py input_ttree.root output.root --is-mc
  Data: python analysis_fbcec_nanoaod.py input_ttree.root output.root
"""

import os
import sys
import argparse
import numpy as np
import uproot
import awkward as ak
import ROOT

from analysis_fbcec import (
    SYSTEMATICS,
    create_counter_histogram,
    create_gen_histograms,
    create_reco_histograms,
    create_cov_accumulators,
    finalize_errors,
    process_gen_arrays,
    process_reco_arrays,
    fill_gen_evtsel,
    build_systematics_to_run,
    cov_target_systs,
    species_codes,
)
from analysis_thrust_nanoaod import select_leaves, build_gen_leaf_redecayed
from vertex_reweight import (
    VertexReweighter3D,
    VertexResolutionSmearer,
    detect_period as rw_detect_period,
    find_3d_table,
    find_resolution_table,
)


# ============================================================================
# BRANCH NAMES
# ============================================================================

_PX = "_fourMomentum_fCoordinates_fX"
_PY = "_fourMomentum_fCoordinates_fY"
_PZ = "_fourMomentum_fCoordinates_fZ"
_E  = "_fourMomentum_fCoordinates_fT"

RECO_BRANCHES = [
    "Event_cmEnergy",
    "Part_charge",
    "Part_lock",           # for the Part_lock == 0 filter (mirrors TPCNtuple writer)
    f"Part{_PX}", f"Part{_PY}", f"Part{_PZ}", f"Part{_E}",
    "Trac_partIdx",
    "Trac_impParToVertexRPhi", "Trac_impParToVertexZ",
    # dpp = sqrt(1/W11) / |1/R|. Empirical layout (verified per slot, also
    # used by analysis_tau.py): curvature is per[:, 1], its weight is wm[:, 1].
    # The writer's comment at nanoaod_writer.cpp:1541-1554 (per[:,4], wm[:,14])
    # is wrong — those slots are φ and an off-diagonal weight element.
    "Trac_perigee_fArray",
    "Trac_weightMatrix_fRep_fArray",
    # Primary-vertex diagnostics (analysis_v0 convention: PV = first Vtx
    # with status != 4 (decay); status=0 fitted, status=1 default IP).
    "Vtx_position_fCoordinates_fX",
    "Vtx_position_fCoordinates_fY",
    "Vtx_position_fCoordinates_fZ",
    "Vtx_status",
    "Vtx_nOutgoing",
]

GEN_BRANCHES = [
    "SimPart_originVtxIdx", "SimPart_decayVtxIdx",
    "SimPart_pdgId", "SimPart_charge",
    f"SimPart{_PX}", f"SimPart{_PY}", f"SimPart{_PZ}", f"SimPart{_E}",
    # GenPart = JETSET generator record (STANDARD PDG codes, unlike SimPart's
    # DELPHI codes) with a parent/child tree. Used for the event-level primary
    # Z->qqbar flavor tag (primary_zqq_flavor) and the Z 4-momentum that gives
    # SumE (primary_z_4mom).
    "GenPart_pdgId", "GenPart_firstChildIdx", "GenPart_lastChildIdx",
    "GenPart_vector_fCoordinates_fT", "GenPart_vector_fCoordinates_fX",
    "GenPart_vector_fCoordinates_fY", "GenPart_vector_fCoordinates_fZ",
]


# ============================================================================
# READERS: NanoAOD → flat per-event numpy arrays
# ============================================================================

def build_reco_arrays(arr, iEv):
    """Build per-particle reco numpy arrays from one event's NanoAOD record.

    Applies the Part_lock == 0 filter to mirror the TPCNtuple writer
    (nanoaod_writer.cpp:1465) so the resulting track set matches the
    v70-era baseline (skelana's LVLOCK = standard hadronic-track cuts,
    incl. implicit dpp<1, track length>30 cm, etc.).

    Returns (E_event, px, py, pz, m, q, pt, th, d0, z0, dpp) covering the
    unlocked Part_ entries. d0/z0/dpp are scattered from Trac_* via
    Trac_partIdx onto the full-length per-Part arrays first, then the
    lock filter is applied to all parallel arrays together so the Trac
    indexing stays consistent.

    dpp = sqrt(1/W11) / |curv|. Empirical layout (verified per slot, also
    used by analysis_tau.py): per[:, 1] is the curvature 1/R, wm[:, 1] is
    the corresponding diagonal weight. The writer's comment claiming
    per[:, 4]/wm[:, 14] is wrong — those slots are φ and an off-diagonal
    weight, not curvature.

    Returns None when the event has no unlocked particles.
    """
    px = ak.to_numpy(arr[f"Part{_PX}"][iEv]).astype(np.float64)
    nPart = len(px)
    if nPart == 0:
        return None

    py = ak.to_numpy(arr[f"Part{_PY}"][iEv]).astype(np.float64)
    pz = ak.to_numpy(arr[f"Part{_PZ}"][iEv]).astype(np.float64)
    en = ak.to_numpy(arr[f"Part{_E}"][iEv]).astype(np.float64)
    q  = ak.to_numpy(arr["Part_charge"][iEv]).astype(np.float64)
    lock = ak.to_numpy(arr["Part_lock"][iEv]).astype(np.int64)

    pt = np.sqrt(px*px + py*py)
    th = np.arctan2(pt, pz)
    p2 = px*px + py*py + pz*pz
    m  = np.sqrt(np.clip(en*en - p2, 0.0, None))

    # Scatter Trac-level quantities (d0, z0, dpp) into per-Part arrays of
    # length nPart so that the lock filter can later be applied uniformly.
    d0  = np.zeros(nPart, dtype=np.float64)
    z0  = np.zeros(nPart, dtype=np.float64)
    dpp = np.full(nPart, -1.0, dtype=np.float64)  # sentinel = invalid
    tpi = ak.to_numpy(arr["Trac_partIdx"][iEv]).astype(np.int64)
    nTrac = len(tpi)
    if nTrac > 0:
        valid = (tpi >= 0) & (tpi < nPart)
        if valid.any():
            d0_t = ak.to_numpy(arr["Trac_impParToVertexRPhi"][iEv]).astype(np.float64)
            z0_t = ak.to_numpy(arr["Trac_impParToVertexZ"][iEv]).astype(np.float64)
            d0[tpi[valid]] = d0_t[valid]
            z0[tpi[valid]] = z0_t[valid]
            # dpp uses empirically-correct slots:
            #   per[:, 1] = curvature 1/R   (per[:, 4] is φ, not 1/R)
            #   wm[:, 1]  = its diagonal W11 (real diagonals at {1,3,6,10};
            #              wm[:, 14] is an off-diagonal element)
            per = ak.to_numpy(arr["Trac_perigee_fArray"][iEv]).astype(np.float64)
            wm  = ak.to_numpy(arr["Trac_weightMatrix_fRep_fArray"][iEv]).astype(np.float64)
            if per.size == 5 * nTrac and wm.size == 15 * nTrac:
                curv = per.reshape(nTrac, 5)[:, 1]
                W11  = wm.reshape(nTrac, 15)[:, 1]
                ok = (np.abs(curv) > 1e-10) & (W11 > 0.0)
                dpp_t = np.full(nTrac, -1.0, dtype=np.float64)
                dpp_t[ok] = np.sqrt(1.0 / W11[ok]) / np.abs(curv[ok])
                dpp[tpi[valid]] = dpp_t[valid]

    # Apply the LVLOCK / Part_lock == 0 filter to all parallel arrays.
    keep = (lock == 0)
    if not keep.any():
        return None
    px  = px[keep];  py  = py[keep];  pz = pz[keep];  en = en[keep]
    q   = q[keep];   pt  = pt[keep];  th = th[keep]
    m   = m[keep];   d0  = d0[keep];  z0 = z0[keep];  dpp = dpp[keep]

    E = float(arr["Event_cmEnergy"][iEv])
    return E, px, py, pz, m, q, pt, th, d0, z0, dpp


def build_gen_data(arr, iEv, E_event):
    """Build gen-level particle data in the Pythia V0-decay convention.

    Uses the SHARED gen builder build_gen_leaf_redecayed (same as thrust/EEC):
    V0-aware select_leaves, then Pythia re-decay of any kept strange-V0 leaf so
    the gen final state matches the reco V0 daughters (2026-07-03 fix, now
    propagated to fbcec). Returns dict {px, py, pz, m, q, e, E, species} over all
    final particles, or None if SimPart is empty / no leaves. E is the event cm
    energy (passed in).
    """
    ovtx = ak.to_numpy(arr["SimPart_originVtxIdx"][iEv]).astype(np.int64)
    if len(ovtx) == 0:
        return None
    dvtx = ak.to_numpy(arr["SimPart_decayVtxIdx"][iEv]).astype(np.int64)
    pdg  = ak.to_numpy(arr["SimPart_pdgId"][iEv]).astype(np.int64)
    q_a  = ak.to_numpy(arr["SimPart_charge"][iEv]).astype(np.float64)
    Px   = ak.to_numpy(arr[f"SimPart{_PX}"][iEv]).astype(np.float64)
    Py   = ak.to_numpy(arr[f"SimPart{_PY}"][iEv]).astype(np.float64)
    Pz   = ak.to_numpy(arr[f"SimPart{_PZ}"][iEv]).astype(np.float64)
    E_a  = ak.to_numpy(arr[f"SimPart{_E}"][iEv]).astype(np.float64)

    res = build_gen_leaf_redecayed(ovtx, dvtx, pdg, q_a, Px, Py, Pz, E_a,
                                   return_codes=True)
    if res is None:
        return None
    px, py, pz, m, q, codes, redec = res
    # Drop neutrinos so the gen leaf set is the VISIBLE final state (matches the
    # reco Part collection, which never sees neutrinos): DELPHI internal codes
    # |1|/|5|/|9| (nu_e/mu/tau) for kept leaves, standard PDG |12|/|14|/|16| for
    # re-decayed daughters. Charged observables are unaffected (nu are neutral);
    # this only trims the neutral energy sums (SumE / SumE_recosel).
    _ac = np.abs(codes)
    _isnu = ((~redec) & np.isin(_ac, (1, 5, 9))) | (redec & np.isin(_ac, (12, 14, 16)))
    if _isnu.any():
        _k = ~_isnu
        px, py, pz, m, q, codes, redec = px[_k], py[_k], pz[_k], m[_k], q[_k], codes[_k], redec[_k]
    e = np.sqrt(px*px + py*py + pz*pz + m*m)
    # Species: kept leaves carry DELPHI internal codes; the Pythia-re-decayed
    # V0 daughters carry standard PDG codes -> map each group with its own
    # species_codes convention.
    species = np.zeros(len(codes), dtype=np.int8)
    if (~redec).any():
        species[~redec] = species_codes(codes[~redec], convention='delphi')
    if redec.any():
        species[redec] = species_codes(codes[redec], convention='pdg')
    return {'px': px, 'py': py, 'pz': pz, 'm': m, 'q': q, 'e': e, 'E': E_event,
            'species': species}


def primary_zqq_flavor(arr, iEv):
    """Event-level primary Z->qqbar flavor tag from the GenPart JETSET record.

    Returns 'bb' (b), 'cc' (c), 'light' (u/d/s), or None if no quark is found.
    GenPart uses STANDARD PDG codes (not SimPart's DELPHI codes). Rule: take the
    decaying Z (pdgId==23) whose firstChild..lastChild entries are quarks and
    read their flavor (heaviest wins); fall back to the first quark in the
    record. Validated to reproduce the Z hadronic branching fractions on both
    pythia8_isrbk and kk2f4146_qqpy. MC only (data has no GenPart).
    """
    pid = ak.to_numpy(arr["GenPart_pdgId"][iEv]).astype(np.int64)
    if len(pid) == 0:
        return None
    fc = ak.to_numpy(arr["GenPart_firstChildIdx"][iEv]).astype(np.int64)
    lc = ak.to_numpy(arr["GenPart_lastChildIdx"][iEv]).astype(np.int64)
    fl = 0
    for zi in np.flatnonzero(pid == 23)[::-1]:   # last (decaying) Z first
        if fc[zi] >= 0 and lc[zi] >= fc[zi]:
            kids = pid[fc[zi]:lc[zi] + 1]
            akids = np.abs(kids)
            qk = akids[(akids >= 1) & (akids <= 6)]
            if len(qk):
                fl = int(qk.max())
                break
    if fl == 0:
        aq = np.abs(pid)
        q_all = aq[(aq >= 1) & (aq <= 6)]
        if len(q_all):
            fl = int(q_all[0])   # first quark in record
    if fl == 5:
        return 'bb'
    if fl == 4:
        return 'cc'
    if fl in (1, 2, 3):
        return 'light'
    return None


def primary_z_4mom(arr, iEv):
    """Lab 4-momentum of the decaying Z from GenPart -> (E_Z, m_Z), or None.

    E_Z IS the sum of particle energies with neutrinos INCLUDED and ISR photons
    EXCLUDED: the ISR is radiated off the incoming e+-, so everything the Z
    decays to (hadrons, FSR, decay neutrinos) sums to the Z 4-momentum, and
    energy conservation gives E_Z = sqrt(s) - E_ISR. m_Z = sqrt(s'), the
    effective collision energy.

    This is computed from the Z rather than by tagging and subtracting ISR
    photons because the e+--parent photon tag only works on pythia8: kk2f
    stores a PRUNED GenPart and the tag silently returns zero ISR photons
    there. The Z (pdgId 23) is present in both. MC only.
    """
    pid = ak.to_numpy(arr["GenPart_pdgId"][iEv]).astype(np.int64)
    z = np.flatnonzero(pid == 23)
    if len(z) == 0:
        return None
    j = int(z[-1])   # the decaying Z (last copy)
    E  = float(ak.to_numpy(arr["GenPart_vector_fCoordinates_fT"][iEv])[j])
    px = float(ak.to_numpy(arr["GenPart_vector_fCoordinates_fX"][iEv])[j])
    py = float(ak.to_numpy(arr["GenPart_vector_fCoordinates_fY"][iEv])[j])
    pz = float(ak.to_numpy(arr["GenPart_vector_fCoordinates_fZ"][iEv])[j])
    return E, float(np.sqrt(max(E * E - px * px - py * py - pz * pz, 0.0)))


def fill_pv_diagnostics(arr, iEv, h_nominal):
    """Fill per-event primary-vertex histograms (`pv_*`, `n_pv_candidates`)
    on the nominal reco_hists dict. PV is selected as the first Vtx_*
    entry with status != 4 (decay); status=0/1 are fitted/default-IP.
    Matches the convention in analysis_v0.
    """
    try:
        vtx_x = ak.to_numpy(arr["Vtx_position_fCoordinates_fX"][iEv]).astype(np.float64)
        vtx_y = ak.to_numpy(arr["Vtx_position_fCoordinates_fY"][iEv]).astype(np.float64)
        vtx_z = ak.to_numpy(arr["Vtx_position_fCoordinates_fZ"][iEv]).astype(np.float64)
        vtx_status = ak.to_numpy(arr["Vtx_status"][iEv]).astype(np.int64)
        n_out      = ak.to_numpy(arr["Vtx_nOutgoing"][iEv]).astype(np.int64)
    except (KeyError, IndexError):
        return
    if len(vtx_x) == 0:
        return
    non_decay = np.flatnonzero(vtx_status != 4)
    h_nominal['n_pv_candidates'].Fill(len(non_decay))
    if len(non_decay) == 0:
        return
    pv = int(non_decay[0])
    pvx, pvy, pvz = float(vtx_x[pv]), float(vtx_y[pv]), float(vtx_z[pv])
    pvrho = float(np.sqrt(pvx * pvx + pvy * pvy))
    h_nominal['pv_x'].Fill(pvx)
    h_nominal['pv_y'].Fill(pvy)
    h_nominal['pv_z'].Fill(pvz)
    h_nominal['pv_rho'].Fill(pvrho)
    h_nominal['pv_xy'].Fill(pvx, pvy)
    h_nominal['pv_z_vs_rho'].Fill(pvrho, pvz)
    h_nominal['pv_n_outgoing'].Fill(int(n_out[pv]))


# ============================================================================
# MAIN
# ============================================================================

def main():
    ap = argparse.ArgumentParser(
        description='FBCEC + FBCC analysis on DelphiNanoAOD (*_ttree.root) input'
    )
    ap.add_argument('infile', help='Input NanoAOD _ttree.root file')
    ap.add_argument('outfile', help='Output ROOT file')
    ap.add_argument('--is-mc', action='store_true',
                    help='Process as MC (fills gen-level histograms and gen-reco matching)')
    ap.add_argument('--systematic', default='all',
                    help='Run specific systematic or "all" (default: all)')
    ap.add_argument('--compute-covariance', action='store_true', default=None,
                    help='Compute covariance matrix and 2D histograms (nominal only). '
                         'Default: enabled for data, disabled for MC.')
    ap.add_argument('--no-covariance', action='store_true',
                    help='Disable covariance computation even for data.')
    ap.add_argument('--period', choices=['auto', '94c', '95d'], default='auto',
                    help='Data-taking period for vertex reweighting; '
                         'auto-detects from infile path if "auto" (default).')
    ap.add_argument('--reweight-dir', default='.',
                    help='Directory containing vertex_reweight_<period>_v73_3d.root '
                         '(default: cwd).')
    ap.add_argument('--max-events', type=int, default=-1,
                    help='Cap on events processed (default: all)')
    args = ap.parse_args()

    # Systematics. Each base entry produces its histogram set; additionally
    # every entry gets a `_dpp0p5` partner (dpp<0.5 cut) and a `_d0z0loose`
    # partner (loose IP acceptance |d0|<4 / |z0sinθ|<10). Both are cut
    # variants, not uncertainties, and additive — see
    # build_systematics_to_run in analysis_fbcec.
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

    print(f"Running FBCEC NanoAOD analysis on: {args.infile}")
    print(f"MC mode: {args.is_mc}")
    print(f"Systematics: {', '.join(systematics_to_run.keys())}")
    print(f"Covariance + 2D: {do_2d}")

    # Vertex reweighter (MC only) — v73 PDF-ratio weights for the
    # _rw_z0sin_charge companions.
    period = args.period if args.period != 'auto' else rw_detect_period(args.infile)
    vtx_rw = None
    smearer = None
    rng = None
    if args.is_mc:
        if period is None:
            print("WARNING: cannot determine period from infile; "
                  "vertex reweighting / smearing disabled")
        else:
            table_path = find_3d_table(period, search_dir=args.reweight_dir)
            if table_path is None:
                print(f"WARNING: vertex_reweight_{period}_v73_3d.root not found "
                      f"in {args.reweight_dir}; v73 reweighting disabled")
            else:
                try:
                    vtx_rw = VertexReweighter3D(table_path)
                    print(f"Vertex reweighter loaded from {table_path}; period={period}")
                except Exception as exc:
                    print(f"WARNING: vertex reweighter disabled — {exc}")
                    vtx_rw = None

            # v78 resolution smearer (separate file, additive Gaussian
            # on d0 and z0·sinθ; feeds smear_d0 / smear_z0sin systematics).
            sm_path = find_resolution_table(period, search_dir=args.reweight_dir)
            if sm_path is None:
                print(f"WARNING: vertex_resolution_{period}_v78_3d.root not found "
                      f"in {args.reweight_dir}; smear_d0/smear_z0sin systematics disabled")
            else:
                try:
                    smearer = VertexResolutionSmearer(sm_path)
                    # Deterministic RNG seeded by input file path so noise is
                    # reproducible per file (different across files).
                    seed = abs(hash(os.path.basename(args.infile))) & 0xFFFFFFFF
                    rng = np.random.default_rng(seed)
                    print(f"Resolution smearer loaded from {sm_path}; period={period}; rng seed={seed}")
                except Exception as exc:
                    print(f"WARNING: resolution smearer disabled — {exc}")
                    smearer = None

    # Open + bulk-read the file with uproot
    try:
        tree = uproot.open(args.infile)["t"]
    except Exception as exc:
        print(f"ERROR: cannot open {args.infile}: {exc}")
        return 1

    branches = list(RECO_BRANCHES)
    if args.is_mc:
        branches += GEN_BRANCHES

    arr = tree.arrays(branches)
    n_events = len(arr["Event_cmEnergy"])
    if args.max_events > 0:
        n_events = min(n_events, args.max_events)
    print(f"Processing {n_events} events")

    # Booking
    counter    = create_counter_histogram()
    gen_hists  = create_gen_histograms(do_2d=do_2d) if args.is_mc else None
    reco_hists = {s: create_reco_histograms(s, do_2d=do_2d, is_mc=args.is_mc)
                  for s in systematics_to_run}
    cov_accum  = create_cov_accumulators() if do_2d else None

    # Event loop
    for iEv in range(n_events):
        if iEv % 1000 == 0:
            print(f"  Processing event {iEv}/{n_events}")

        E_event = float(arr["Event_cmEnergy"][iEv])
        on_pole = abs(E_event - 91.25) <= 1  # DELPHI E=91.25±1 gate (cf. analysis_fbcec)

        # Primary Z->qqbar flavor tag (MC truth, once per event). The SAME
        # event-level flag is applied to both the gen and reco flavor-split
        # FBCEC/FBCC histograms. None for data (no GenPart).
        flavor = primary_zqq_flavor(arr, iEv) if (args.is_mc and on_pole) else None
        # Z 4-momentum -> SumE (= sqrt(s) - E_ISR) and sqrt(s') (= m_Z).
        z_4mom = primary_z_4mom(arr, iEv) if (args.is_mc and on_pole) else None

        # Gen
        gen_data = None
        if args.is_mc and on_pole:
            gen_data = build_gen_data(arr, iEv, E_event)
            if gen_data is not None and gen_hists is not None:
                process_gen_arrays(gen_data['px'], gen_data['py'], gen_data['pz'],
                                   gen_data['q'], gen_data['e'],
                                   gen_hists, do_2d=do_2d,
                                   species=gen_data.get('species'), flavor=flavor,
                                   z_4mom=z_4mom)

        # Reco (counter.Fill(0.5) is done unconditionally to mirror analysis_fbcec.main)
        counter.Fill(0.5)
        if not on_pole:
            continue
        reco = build_reco_arrays(arr, iEv)
        if reco is None:
            continue
        E_r, px, py, pz, m, q, pt, th, d0, z0, dpp = reco

        # Per-event primary-vertex diagnostics (nominal only). PV = first
        # Vtx with status != 4 (status=4 is decay). Matches analysis_v0.
        if 'nominal' in reco_hists:
            fill_pv_diagnostics(arr, iEv, reco_hists['nominal'])

        passed_evtsel = process_reco_arrays(
                            E_r, px, py, pz, m, q, pt, th, d0, z0,
                            systematics_to_run, reco_hists, counter,
                            cov_accum=cov_accum, do_2d=do_2d,
                            gen_data=gen_data, vtx_rw=vtx_rw, period=period,
                            dpp=dpp, smearer=smearer, rng=rng, flavor=flavor)

        # Gen FBCEC/FBCC gated by the RECO event selection (pT>2), for the
        # matched gen/reco (acceptance x efficiency) comparison. Filled only
        # when the reco event passed the nominal event selection.
        if args.is_mc and gen_data is not None and gen_hists is not None and passed_evtsel:
            fill_gen_evtsel(gen_data['px'], gen_data['py'], gen_data['pz'],
                            gen_data['q'], gen_data['e'], gen_hists, flavor=flavor,
                            z_4mom=z_4mom)

    print(f"Processed {n_events} events")

    # Write output (same layout as analysis_fbcec)
    fout = ROOT.TFile(args.outfile, "RECREATE")

    if gen_hists is not None:
        fout.mkdir("gen"); fout.cd("gen")
        for h in gen_hists.values():
            h.Write()

    fout.mkdir("reco"); fout.cd("reco")
    for hd in reco_hists.values():
        for h in hd.values():
            h.Write()

    fout.cd()
    counter.Write()

    if cov_accum is not None:
        finalize_errors(reco_hists, cov_accum, n_events, save_cov_matrices=True)
        fout.cd("reco")
        # Re-write nominal + its _dpp0p5 / _d0z0loose partners (now with
        # correct errors + new cov TH2Ds)
        for syst in cov_target_systs():
            for h in reco_hists.get(syst, {}).values():
                h.Write(h.GetName(), ROOT.TObject.kOverwrite)

    fout.Close()
    print(f"Output written to {args.outfile}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

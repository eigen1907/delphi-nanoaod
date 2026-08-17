#!/usr/bin/env python3
"""
Phi(1020) -> K+K- Charge Mis-Tagging Analysis
===============================================
Uses the phi(1020) -> K+K- resonance as a control region to measure
charge mis-identification rates in hadronic Z events.

Method:
  - Loop over all charged track pairs in hadronic events
  - Assign kaon mass (493.677 MeV) to both tracks, compute invariant mass
  - OS pairs: real phi signal lives here (denominator)
  - SS pairs: any phi peak = charge mis-ID (numerator)
  - Fill m_KK spectra per (theta, pT cut) and per (p, pT cut) for OS and SS
  - Signal/sideband windows defined later at analysis stage

Event selection: same as analysis_fbcec.py baseline hadronic selection
Track selection: same as correction_fbcc.py (|d0|<0.6, |z0*sin|<1)

For each pT threshold on BOTH tracks in the pair:
  pT > 0.4, 1.0, 2.0, 4.0, 10.0, 20.0 GeV

Per-track theta bins (20 deg): [20,40), [40,60), ..., [140,160)  (7 bins)
Per-track momentum bins: 0.5, 1, 2, 3, 5, 8, 12, 20, 30, 45, 60 GeV

Each track in a pair fills the histogram for its own theta/p bin.

Usage:
  python analysis_phi.py input_ttree.root output.root [--max-events N]
"""

import ROOT
import uproot
import numpy as np
import argparse
import math
import sys
from array import array

from common_functions import (
    thrust_axis_fast,
    thrust_theta,
)
from binning_and_selections import (
    apply_track_selection_delphi,
    apply_event_selection_delphi,
)

# ============================================================================
# CONSTANTS
# ============================================================================

M_KAON = 0.493677  # GeV

# pT thresholds
PT_CUTS = [0.4, 1.0, 2.0, 4.0, 10.0, 20.0]

# Theta bins (degrees): 7 bins of 20 degrees
THETA_EDGES = np.array([20, 40, 60, 80, 100, 120, 140, 160], dtype=np.float64)
N_THETA_BINS = len(THETA_EDGES) - 1

# Mass spectrum binning (GeV): 0.98 to 1.08, 0.5 MeV bins
MASS_BINS = np.linspace(0.98, 1.08, 201, dtype=np.float64)


# ============================================================================
# HISTOGRAMS
# ============================================================================

def pt_label(pcut):
    return f"pt{pcut:.1f}".replace('.', 'p')


def book_histograms():
    h = {}

    # Cutflow
    h['cutflow'] = ROOT.TH1D('cutflow', '', 10, 0, 10)
    for i, l in enumerate(['Total', 'Loaded', 'TrackSel', 'EventSel',
                            'HasPairs', 'bin6', 'bin7', 'bin8', 'bin9', 'bin10']):
        h['cutflow'].GetXaxis().SetBinLabel(i + 1, l)

    # Event-level
    h['n_charged'] = ROOT.TH1D('n_charged', ';N_{charged};Events', 100, 0, 100)

    mb = array('d', MASS_BINS)
    nmb = len(MASS_BINS) - 1

    # Book two sets: with kaon ID ("kid_") and without ("noid_")
    for prefix in ["kid_", "noid_"]:
        # Opening angle histograms (filled before angle cut, in phi mass window)
        h[f'{prefix}angle_os'] = ROOT.TH1D(
            f'{prefix}angle_os', ';Opening angle [rad] (OS, m_{{KK}}#in[1.01,1.03]);Pairs',
            100, 0, math.pi)
        h[f'{prefix}angle_ss'] = ROOT.TH1D(
            f'{prefix}angle_ss', ';Opening angle [rad] (SS, m_{{KK}}#in[1.01,1.03]);Pairs',
            100, 0, math.pi)

        for pcut in PT_CUTS:
            t = pt_label(pcut)

            # ---- Inclusive mass spectra ----
            h[f'{prefix}mass_os_{t}'] = ROOT.TH1D(
                f'{prefix}mass_os_{t}', f';m_{{KK}} [GeV];Pairs', nmb, mb)
            h[f'{prefix}mass_ss_{t}'] = ROOT.TH1D(
                f'{prefix}mass_ss_{t}', f';m_{{KK}} [GeV];Pairs', nmb, mb)
            h[f'{prefix}mass_sspp_{t}'] = ROOT.TH1D(
                f'{prefix}mass_sspp_{t}', f';m_{{KK}} [GeV];Pairs (++)', nmb, mb)
            h[f'{prefix}mass_ssmm_{t}'] = ROOT.TH1D(
                f'{prefix}mass_ssmm_{t}', f';m_{{KK}} [GeV];Pairs (--)', nmb, mb)

            # ---- Mass spectra per theta bin ----
            for ith in range(N_THETA_BINS):
                lo = int(THETA_EDGES[ith])
                hi = int(THETA_EDGES[ith + 1])
                h[f'{prefix}mass_os_theta{lo}_{t}'] = ROOT.TH1D(
                    f'{prefix}mass_os_theta{lo}_{t}',
                    f';m_{{KK}} [GeV] (OS, #theta#in[{lo},{hi})#circ);Tracks',
                    nmb, mb)
                h[f'{prefix}mass_ss_theta{lo}_{t}'] = ROOT.TH1D(
                    f'{prefix}mass_ss_theta{lo}_{t}',
                    f';m_{{KK}} [GeV] (SS, #theta#in[{lo},{hi})#circ);Tracks',
                    nmb, mb)
                h[f'{prefix}mass_sspp_theta{lo}_{t}'] = ROOT.TH1D(
                    f'{prefix}mass_sspp_theta{lo}_{t}',
                    f';m_{{KK}} [GeV] (++, #theta#in[{lo},{hi})#circ);Tracks',
                    nmb, mb)
                h[f'{prefix}mass_ssmm_theta{lo}_{t}'] = ROOT.TH1D(
                    f'{prefix}mass_ssmm_theta{lo}_{t}',
                    f';m_{{KK}} [GeV] (--, #theta#in[{lo},{hi})#circ);Tracks',
                    nmb, mb)

            # ---- 2D: mass vs theta (per-track) ----
            h[f'{prefix}mass_theta_os_{t}'] = ROOT.TH2D(
                f'{prefix}mass_theta_os_{t}', f';m_{{KK}} [GeV];#theta [deg]',
                nmb, mb,
                N_THETA_BINS, array('d', THETA_EDGES))
            h[f'{prefix}mass_theta_ss_{t}'] = ROOT.TH2D(
                f'{prefix}mass_theta_ss_{t}', f';m_{{KK}} [GeV];#theta [deg]',
                nmb, mb,
                N_THETA_BINS, array('d', THETA_EDGES))
            h[f'{prefix}mass_theta_sspp_{t}'] = ROOT.TH2D(
                f'{prefix}mass_theta_sspp_{t}', f';m_{{KK}} [GeV];#theta [deg]',
                nmb, mb,
                N_THETA_BINS, array('d', THETA_EDGES))
            h[f'{prefix}mass_theta_ssmm_{t}'] = ROOT.TH2D(
                f'{prefix}mass_theta_ssmm_{t}', f';m_{{KK}} [GeV];#theta [deg]',
                nmb, mb,
                N_THETA_BINS, array('d', THETA_EDGES))

    return h


# ============================================================================
# PAIR PROCESSING
# ============================================================================

def find_theta_bin(theta_deg):
    """Return the lower edge of the theta bin, or -1 if outside acceptance."""
    for ith in range(N_THETA_BINS):
        if THETA_EDGES[ith] <= theta_deg < THETA_EDGES[ith + 1]:
            return int(THETA_EDGES[ith])
    return -1


def process_pairs(px_c, py_c, pz_c, q_c, pt_c, th_c, p_c, is_kaon, hists, prefix, max_angle=999.):
    """
    Loop over all charged track pairs, compute KK invariant mass,
    fill OS and SS mass spectra: inclusive, per-theta, and 2D.
    Only pairs where BOTH tracks pass is_kaon are considered.
    max_angle: maximum opening angle between the two tracks (radians).
    """
    n = len(px_c)
    if n < 2:
        return

    # Precompute kaon energies
    e_k = np.sqrt(px_c**2 + py_c**2 + pz_c**2 + M_KAON**2)

    # Track-level theta in degrees
    theta_deg = np.degrees(th_c)

    for i in range(n):
        if not is_kaon[i]:
            continue
        for j in range(i + 1, n):
            if not is_kaon[j]:
                continue

            # Opening angle: cos(alpha) = (p_i . p_j) / (|p_i| |p_j|)
            dot = px_c[i]*px_c[j] + py_c[i]*py_c[j] + pz_c[i]*pz_c[j]
            cos_alpha = dot / (p_c[i] * p_c[j]) if p_c[i] > 0 and p_c[j] > 0 else 1.0
            cos_alpha = max(-1.0, min(1.0, cos_alpha))
            alpha = math.acos(cos_alpha)

            # Invariant mass with kaon hypothesis
            m_kk2 = ((e_k[i] + e_k[j])**2
                      - (px_c[i] + px_c[j])**2
                      - (py_c[i] + py_c[j])**2
                      - (pz_c[i] + pz_c[j])**2)
            if m_kk2 < 0:
                continue
            m_kk = math.sqrt(m_kk2)

            # Only fill within the broad mass window
            if m_kk < MASS_BINS[0] or m_kk > MASS_BINS[-1]:
                continue

            is_os = (q_c[i] * q_c[j] < 0)

            # Fill opening angle histograms (before angle cut, in phi mass region)
            if 1.01 <= m_kk <= 1.03:
                if is_os:
                    hists[f'{prefix}angle_os'].Fill(alpha)
                else:
                    hists[f'{prefix}angle_ss'].Fill(alpha)

            # Apply opening angle cut
            if alpha > max_angle:
                continue

            # Loop over pT thresholds
            for pcut in PT_CUTS:
                if pt_c[i] < pcut or pt_c[j] < pcut:
                    continue

                t = pt_label(pcut)

                # ---- Inclusive mass ----
                if is_os:
                    hists[f'{prefix}mass_os_{t}'].Fill(m_kk)
                else:
                    hists[f'{prefix}mass_ss_{t}'].Fill(m_kk)
                    if q_c[i] > 0 and q_c[j] > 0:
                        hists[f'{prefix}mass_sspp_{t}'].Fill(m_kk)
                    else:
                        hists[f'{prefix}mass_ssmm_{t}'].Fill(m_kk)

                # ---- Per-track distributions ----
                cat = 'os' if is_os else 'ss'

                # Determine charge-split SS category
                if not is_os:
                    if q_c[i] > 0 and q_c[j] > 0:
                        sscat = 'sspp'
                    else:
                        sscat = 'ssmm'
                else:
                    sscat = None

                for k in (i, j):
                    # Per-theta mass spectrum
                    th_bin = find_theta_bin(theta_deg[k])
                    if th_bin >= 0:
                        hists[f'{prefix}mass_{cat}_theta{th_bin}_{t}'].Fill(m_kk)
                        if sscat is not None:
                            hists[f'{prefix}mass_{sscat}_theta{th_bin}_{t}'].Fill(m_kk)

                    # 2D: mass vs theta
                    hists[f'{prefix}mass_theta_{cat}_{t}'].Fill(m_kk, theta_deg[k])
                    if sscat is not None:
                        hists[f'{prefix}mass_theta_{sscat}_{t}'].Fill(m_kk, theta_deg[k])


# ============================================================================
# DATA LOADING (uproot, same pattern as analysis_tau.py)
# ============================================================================

def load_all_events(infile, max_events=-1):
    f = uproot.open(infile)
    tree = f["t"]
    keys = tree.keys()

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
        print(f"ERROR: Cannot find Part_fourMomentum. Available: {[k for k in keys if 'Part' in k][:20]}")
        sys.exit(1)

    branches = [px_key, py_key, pz_key, en_key, "Part_charge", "Part_lock"]
    opt = ["Event_cmEnergy",
           "Trac_partIdx", "Trac_impParToVertexRPhi", "Trac_impParToVertexZ"]
    # Kaon ID branches (try several taggers)
    kaon_branches = ["Haidr_kaonTag", "Haide_kaonTag", "Haidc_kaonTag",
                     "Haid_kaonRich", "Haid_kaonDedx", "Haid_kaonCombined"]
    for k in opt + kaon_branches:
        if k in keys:
            branches.append(k)

    data = tree.arrays(branches, entry_stop=max_events if max_events > 0 else None)
    result = {"px": data[px_key], "py": data[py_key], "pz": data[pz_key],
              "en": data[en_key], "charge": data["Part_charge"], "lock": data["Part_lock"]}
    for k in opt + kaon_branches:
        if k in data.fields:
            result[k] = data[k]
    if "Trac_impParToVertexRPhi" in data.fields:
        result["Trac_d0"] = data["Trac_impParToVertexRPhi"]
        result["Trac_z0"] = data["Trac_impParToVertexZ"]
        result["Trac_partIdx"] = data["Trac_partIdx"]
    return result


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Phi->KK charge mis-tagging analysis'
    )
    parser.add_argument('infile', help='Input ROOT file (treefy\'d NanoAOD)')
    parser.add_argument('outfile', help='Output ROOT file')
    parser.add_argument('--max-events', type=int, default=-1,
                        help='Maximum events to process (-1 = all)')
    parser.add_argument('--kaon-id', default='Haid_kaonCombined',
                        help='Kaon ID branch name (default: Haid_kaonCombined). '
                             'Use "none" to disable kaon ID requirement.')
    parser.add_argument('--kaon-id-min', type=float, default=0.5,
                        help='Minimum kaon tag value to accept (default: 0.5)')
    parser.add_argument('--max-opening-angle', type=float, default=0.8,
                        help='Maximum opening angle between track pairs in radians '
                             '(default: 0.8)')

    args = parser.parse_args()

    print(f"=== Phi->KK Charge Mis-Tagging Analysis ===")
    print(f"Input:  {args.infile}")
    print(f"Output: {args.outfile}")
    print(f"pT thresholds: {PT_CUTS}")
    print(f"Theta bins: {list(THETA_EDGES)}")
    if args.max_opening_angle < 10:
        print(f"Max opening angle: {args.max_opening_angle:.2f} rad")
    else:
        print(f"Max opening angle: none")

    # Load all events via uproot
    data = load_all_events(args.infile, args.max_events)
    nevt = len(data["px"])
    has_trac = "Trac_d0" in data

    # Kaon ID setup
    use_kaon_id = args.kaon_id.lower() != 'none'
    kaon_id_branch = args.kaon_id if use_kaon_id else None
    if use_kaon_id:
        if kaon_id_branch not in data:
            print(f"ERROR: Kaon ID branch '{kaon_id_branch}' not found in file.")
            print(f"  Available Haid branches: {[k for k in data.keys() if 'Haid' in k or 'haid' in k.lower()]}")
            return 1
        print(f"Kaon ID: {kaon_id_branch} >= {args.kaon_id_min}")
    else:
        print(f"Kaon ID: disabled (no PID requirement)")

    print(f"Loaded {nevt} events (has Trac: {has_trac})")

    # Book histograms
    hists = book_histograms()
    n_pass = 0

    # ================================================================
    # Event loop
    # ================================================================
    for iEvt in range(nevt):
        if iEvt % 5000 == 0:
            print(f"  Event {iEvt}/{nevt}  (passed: {n_pass})")

        hists['cutflow'].Fill(0.5)  # Total

        E = 91.25  # default Z-pole

        px     = np.asarray(data["px"][iEvt], dtype=np.float64)
        py     = np.asarray(data["py"][iEvt], dtype=np.float64)
        pz     = np.asarray(data["pz"][iEvt], dtype=np.float64)
        en     = np.asarray(data["en"][iEvt], dtype=np.float64)
        charge = np.asarray(data["charge"][iEvt], dtype=np.float64)
        lock   = np.asarray(data["lock"][iEvt], dtype=np.int32)

        npart = len(px)
        if npart == 0:
            continue

        # Filter out locked particles
        good = lock == 0
        if np.sum(good) < 2:
            continue

        px = px[good]
        py = py[good]
        pz = pz[good]
        en = en[good]
        q  = charge[good]

        # Compute derived quantities
        pt = np.sqrt(px**2 + py**2)
        p  = np.sqrt(px**2 + py**2 + pz**2)
        m  = np.sqrt(np.maximum(en**2 - p**2, 0))
        th = np.arctan2(pt, pz)

        # d0, z0 from Trac banks
        d0 = np.full(len(px), 0.0)
        z0 = np.full(len(px), 0.0)
        if has_trac:
            ti = np.asarray(data["Trac_partIdx"][iEvt], dtype=np.int32)
            td = np.asarray(data["Trac_d0"][iEvt], dtype=np.float64)
            tz = np.asarray(data["Trac_z0"][iEvt], dtype=np.float64)
            gi = np.where(good)[0]
            gm = {int(o): ig for ig, o in enumerate(gi)}
            for k in range(len(ti)):
                oi = int(ti[k])
                if oi in gm:
                    d0[gm[oi]] = td[k]
                    z0[gm[oi]] = tz[k]

        hists['cutflow'].Fill(1.5)  # Loaded

        if len(px) == 0:
            continue

        # ---- Track selection (same as correction_fbcc.py) ----
        selection = apply_track_selection_delphi(
            px=px, py=py, pz=pz, m=m, q=q,
            th=th, pt=pt, d0=d0, z0=z0,
            charged_pt_min=0.4,
            charged_d0_max=0.6,
            charged_z0sin_max=1.0,
        )

        sel_all = selection['sel']
        if not np.any(sel_all):
            continue
        hists['cutflow'].Fill(2.5)  # TrackSel

        # Selected particles
        px_sel = px[sel_all]
        py_sel = py[sel_all]
        pz_sel = pz[sel_all]
        m_sel  = m[sel_all]
        q_sel  = q[sel_all]

        sel_c = np.abs(q_sel) > 0.1

        # ---- Event selection (same as analysis_fbcec.py) ----
        e_sel = np.sqrt(px_sel**2 + py_sel**2 + pz_sel**2 + m_sel**2)
        p3_all = np.stack([px_sel, py_sel, pz_sel], axis=1)
        axis, T = thrust_axis_fast(p3_all, include_met=True)

        theta_thrust = thrust_theta(axis, T, fold=False)
        pass_sel = apply_event_selection_delphi(
            e_c=e_sel[sel_c], e_n=e_sel,
            theta_Tu=theta_thrust, E_reco=E
        )['pass_reco']

        if not pass_sel:
            continue
        hists['cutflow'].Fill(3.5)  # EventSel
        n_pass += 1

        # ---- Extract charged tracks ----
        px_c = px_sel[sel_c]
        py_c = py_sel[sel_c]
        pz_c = pz_sel[sel_c]
        q_c  = q_sel[sel_c]
        pt_c = np.sqrt(px_c**2 + py_c**2)
        p_c  = np.sqrt(px_c**2 + py_c**2 + pz_c**2)
        th_c = np.arctan2(pt_c, pz_c)

        # Kaon ID for charged tracks
        if use_kaon_id:
            kid_all = np.asarray(data[kaon_id_branch][iEvt], dtype=np.float64)
            kid_good = kid_all[good]
            kid_sel = kid_good[sel_all]
            kid_c = kid_sel[sel_c]
            is_kaon = kid_c >= args.kaon_id_min
        else:
            is_kaon = np.ones(len(px_c), dtype=bool)

        n_ch = len(px_c)
        hists['n_charged'].Fill(n_ch)

        if n_ch < 2:
            continue
        hists['cutflow'].Fill(4.5)  # HasPairs

        # ---- Process pairs: with kaon ID ----
        if use_kaon_id:
            process_pairs(px_c, py_c, pz_c, q_c, pt_c, th_c, p_c, is_kaon, hists, "kid_",
                          max_angle=args.max_opening_angle)

        # ---- Process pairs: no PID (all charged) ----
        all_true = np.ones(len(px_c), dtype=bool)
        process_pairs(px_c, py_c, pz_c, q_c, pt_c, th_c, p_c, all_true, hists, "noid_",
                      max_angle=args.max_opening_angle)

    # ================================================================
    # Summary
    # ================================================================
    print(f"\n=== Results ===")
    print(f"Events: {nevt}  Passed event selection: {n_pass}")

    print(f"\nCutflow:")
    cf = hists['cutflow']
    for i in range(1, cf.GetNbinsX() + 1):
        c = int(cf.GetBinContent(i))
        if c:
            print(f"  {cf.GetXaxis().GetBinLabel(i):>12}: {c:>8}")

    # Quick inclusive yields
    for prefix, label in [("kid_", "With Kaon ID"), ("noid_", "No PID")]:
        print(f"\nInclusive phi mass yields - {label} (raw counts in [1.01, 1.03] GeV):")
        for pcut in PT_CUTS:
            t = pt_label(pcut)
            h_os = hists[f'{prefix}mass_os_{t}']
            h_ss = hists[f'{prefix}mass_ss_{t}']
            b_lo = h_os.FindBin(1.010)
            b_hi = h_os.FindBin(1.030)
            n_os = h_os.Integral(b_lo, b_hi)
            n_ss = h_ss.Integral(b_lo, b_hi)
            if n_os > 0:
                print(f"  pT>{pcut:5.1f}:  OS={n_os:10.0f}   SS={n_ss:10.0f}   "
                      f"ratio={n_ss/n_os:.6f}")
            else:
                print(f"  pT>{pcut:5.1f}:  OS={n_os:10.0f}   SS={n_ss:10.0f}")

    # Per-theta yields (kaon ID only)
    print(f"\nPer-theta yields in [1.01, 1.03] GeV (pT>0.4, with Kaon ID):")
    t = pt_label(0.4)
    for ith in range(N_THETA_BINS):
        lo = int(THETA_EDGES[ith])
        hi = int(THETA_EDGES[ith + 1])
        h_os = hists[f'kid_mass_os_theta{lo}_{t}']
        h_ss = hists[f'kid_mass_ss_theta{lo}_{t}']
        b_lo = h_os.FindBin(1.010)
        b_hi = h_os.FindBin(1.030)
        n_os = h_os.Integral(b_lo, b_hi)
        n_ss = h_ss.Integral(b_lo, b_hi)
        if n_os > 0:
            print(f"  theta [{lo:3d},{hi:3d}):  OS={n_os:8.0f}  SS={n_ss:8.0f}  "
                  f"ratio={n_ss/n_os:.6f}")
        else:
            print(f"  theta [{lo:3d},{hi:3d}):  OS={n_os:8.0f}  SS={n_ss:8.0f}")

    # ================================================================
    # Write output
    # ================================================================
    fout = ROOT.TFile(args.outfile, 'RECREATE')
    for h in hists.values():
        h.Write()
    fout.Close()

    print(f"\nOutput: {args.outfile}")
    print(f"Total histograms: {len(hists)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
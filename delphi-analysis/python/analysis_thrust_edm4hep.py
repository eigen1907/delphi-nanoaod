#!/usr/bin/env python3
"""
Thrust analysis on DELPHI **EDM4hep** packaged files (S. Qian's workflow),
written to validate the EDM4hep production against the DelphiNanoAOD result.

It reproduces the *nominal* reco histograms of analysis_thrust.py
(ThrustMissPNC2 / ThrustC2, P- and E-scheme, linear + log) so the EDM4hep
output can be overlaid 1:1 against the old nanoaod output. For MC it also
fills the gen-level "before" histograms (Thrust_before2 / ThrustC_before2).

Physics is imported verbatim from the existing modules
(calculate_all_thrust_variants, apply_track_selection_delphi,
apply_event_selection_delphi, thrust_theta, tbins2/logtbins2) so the only
thing that changes vs analysis_thrust.py is the *input reader*.

EDM4hep specifics handled here
------------------------------
* Reco particles live in `fDST_MAIN_Particles` / `sDST_MAIN_Particles`
  (edm4hep::ReconstructedParticle): momentum.{x,y,z}, energy, mass, charge.
  Charged tracks carry the pion mass; PDG is not filled. Pick the collection
  with --dst {fdst,sdst} (identical in the packaged files as of 2026-06).
* Per-event beam energy is the GP param `sDST_EVT_ECMS` (== nanoaod `Energy`).
* DATA events are stored **3x consecutively, byte-identical** (a packaging
  artifact of the two-pass SDST+FDST union); MC events are unique. We collapse
  consecutive entries with an identical (run, event, npart, sum-E) fingerprint,
  which removes the data triples and leaves MC untouched.
* The d0/z0 track-quality cut of the nanoaod selection CANNOT be applied:
  MAIN_Particles carries no per-track impact parameter. We treat MAIN_Particles
  as already track-selected (it is DELPHI's official final state) and feed
  d0=z0=0 so that cut is a no-op. All other cuts (charge, |p|, neutral E,
  20-160 deg) are identical to analysis_thrust.py nominal.
* Gen final state (MC) = generatorStatus==1, neutrinos (|PDG| in 12,14,16)
  removed. NOTE: this is a plain status-1 final state and does NOT reproduce
  the nanoaod V0-aware leaf recovery; treat the gen overlay as indicative.

Usage
-----
  Data: python analysis_thrust_edm4hep.py 'data/*.edm4hep.root' out.root
  MC:   python analysis_thrust_edm4hep.py 'mc/*.edm4hep.root'  out.root --is-mc
        (optional: --dst sdst, --max-events N)
"""

import sys
import glob
import argparse
from array import array

import numpy as np
import awkward as ak
import uproot
import ROOT  # noqa: F401

from binning_and_selections import (
    tbins2,
    logtbins2,
    apply_track_selection_delphi,
    apply_event_selection_delphi,
)
from common_functions import thrust_theta
from analysis_thrust import calculate_all_thrust_variants

# EDM4hep collection names
DST_COLLECTION = {"fdst": "fDST_MAIN_Particles", "sdst": "sDST_MAIN_Particles"}
GEN_COLLECTION = "sDST_LUJ_GenParticles"
ECMS_KEY = "sDST_EVT_ECMS"
RUN_KEY = "sDST_EVT_runNumber"
EVT_KEY = "sDST_EVT_eventNumber"

NEUTRINO_PDG = {12, 14, 16}

# Nominal selection (matches analysis_thrust.py SYSTEMATICS['nominal'] + reco call)
CHARGED_P_MIN = 0.4
NEUTRAL_E_MIN = 0.5
CHARGED_Z0SIN_MAX = 10.0  # as passed in analysis_thrust.process_reco_level


# ---------------------------------------------------------------------------
# Histogram booking (names/dirs mirror analysis_thrust.py for direct overlay)
# ---------------------------------------------------------------------------
def book_reco_hists():
    h = {}
    h["ThrustMissPNC2_nominal"] = ROOT.TH1D("ThrustMissPNC2_nominal", "", len(tbins2) - 1, array("d", tbins2))
    h["ThrustMissPNC2_Escheme_nominal"] = ROOT.TH1D("ThrustMissPNC2_Escheme_nominal", "", len(tbins2) - 1, array("d", tbins2))
    h["ThrustC2_nominal"] = ROOT.TH1D("ThrustC2_nominal", "", len(tbins2) - 1, array("d", tbins2))
    h["ThrustC2_Escheme_nominal"] = ROOT.TH1D("ThrustC2_Escheme_nominal", "", len(tbins2) - 1, array("d", tbins2))
    h["ThrustMissPNCLog2_nominal"] = ROOT.TH1D("ThrustMissPNCLog2_nominal", "", len(logtbins2) - 1, array("d", logtbins2))
    h["ThrustMissPNCLog2_Escheme_nominal"] = ROOT.TH1D("ThrustMissPNCLog2_Escheme_nominal", "", len(logtbins2) - 1, array("d", logtbins2))
    h["ThrustCLog2_nominal"] = ROOT.TH1D("ThrustCLog2_nominal", "", len(logtbins2) - 1, array("d", logtbins2))
    h["ThrustCLog2_Escheme_nominal"] = ROOT.TH1D("ThrustCLog2_Escheme_nominal", "", len(logtbins2) - 1, array("d", logtbins2))
    return h


def book_gen_hists():
    h = {}
    h["Thrust_before2"] = ROOT.TH1D("Thrust_before2", "", len(tbins2) - 1, array("d", tbins2))
    h["Thrust_before2_Escheme"] = ROOT.TH1D("Thrust_before2_Escheme", "", len(tbins2) - 1, array("d", tbins2))
    h["ThrustC_before2"] = ROOT.TH1D("ThrustC_before2", "", len(tbins2) - 1, array("d", tbins2))
    h["ThrustC_before2_Escheme"] = ROOT.TH1D("ThrustC_before2_Escheme", "", len(tbins2) - 1, array("d", tbins2))
    h["Thrust_before_log2"] = ROOT.TH1D("Thrust_before_log2", "", len(logtbins2) - 1, array("d", logtbins2))
    h["Thrust_before_log2_Escheme"] = ROOT.TH1D("Thrust_before_log2_Escheme", "", len(logtbins2) - 1, array("d", logtbins2))
    h["ThrustC_before_log2"] = ROOT.TH1D("ThrustC_before_log2", "", len(logtbins2) - 1, array("d", logtbins2))
    h["ThrustC_before_log2_Escheme"] = ROOT.TH1D("ThrustC_before_log2_Escheme", "", len(logtbins2) - 1, array("d", logtbins2))
    return h


def book_counter():
    c = ROOT.TH1D("counter", "Event Counter", 5, 0, 5)
    c.GetXaxis().SetBinLabel(1, "Total Events")
    c.GetXaxis().SetBinLabel(2, "Pass reco selection")
    return c


# ---------------------------------------------------------------------------
# GP (general-parameter) column reader
# ---------------------------------------------------------------------------
def gp_column(tree, kind, key):
    """Return a 1D numpy array (one value per entry) for GP `key`."""
    keys = [str(k) for k in tree[f"GP{kind}Keys"].array(entry_stop=1)[0]]
    if key not in keys:
        return None
    idx = keys.index(key)
    vals = tree[f"GP{kind}Values"].array()
    return ak.to_numpy(vals[:, idx]).reshape(-1)


# ---------------------------------------------------------------------------
# Track impact parameters d0/z0 from the AtIP track state (optional)
# ---------------------------------------------------------------------------
# edm4hep TrackState.location enum: AtIP == 1. d0/z0 are stored on the AtIP
# state of each track (edm4hep::Track has no direct d0; navigate
# ReconstructedParticle -> track -> getTrackState(AtIP)). Lengths are in mm;
# we convert to cm to match the nanoaod impact-parameter cut thresholds.
# NOTE: the AtIP state is referenced to the origin, not the fitted primary
# vertex (the packaged PV is a 0,0,0 placeholder), so this is the
# origin-referenced impact parameter -- close to, but not identical to, the
# nanoaod impParToVertex used in analysis_thrust.py.
ATIP_LOCATION = 1
MM_TO_CM = 0.1


def track_d0z0_branches(t, dst_coll):
    """Read (per file) the awkward arrays needed to navigate
    ReconstructedParticle -> track -> AtIP TrackState. Returns a dict, or None
    if any required branch is missing."""
    prefix = dst_coll.split("_")[0]                 # 'fDST_MAIN_Particles' -> 'fDST'
    trac = f"{prefix}_TRAC_Tracks"
    names = {
        "tb": f"{dst_coll}.tracks_begin",
        "te": f"{dst_coll}.tracks_end",
        "rel": f"_{dst_coll}_tracks.index",
        "tsb": f"{trac}.trackStates_begin",
        "tse": f"{trac}.trackStates_end",
        "D0": f"_{trac}_trackStates.D0",
        "Z0": f"_{trac}_trackStates.Z0",
        "loc": f"_{trac}_trackStates.location",
    }
    try:
        return {k: t[v].array() for k, v in names.items()}
    except Exception as exc:
        print(f"  track d0/z0 unavailable ({exc}); falling back to d0=z0=0")
        return None


def entry_track_d0z0(br, i, nmain):
    """d0,z0 (cm) per MAIN particle for entry i, via the AtIP track state.
    Neutrals / tracks without an AtIP state get 0 (cut is then a no-op for
    them, matching the neutral treatment)."""
    d0 = np.zeros(nmain, dtype=np.float64)
    z0 = np.zeros(nmain, dtype=np.float64)
    tb = ak.to_numpy(br["tb"][i]); te = ak.to_numpy(br["te"][i])
    rel = ak.to_numpy(br["rel"][i])
    tsb = ak.to_numpy(br["tsb"][i]); tse = ak.to_numpy(br["tse"][i])
    D0 = ak.to_numpy(br["D0"][i]); Z0 = ak.to_numpy(br["Z0"][i])
    loc = np.round(ak.to_numpy(br["loc"][i])).astype(int)
    ntrk = len(tsb)
    for p in range(nmain):
        if te[p] <= tb[p]:
            continue                                # no associated track (neutral)
        k = int(rel[tb[p]])                         # first track (local index)
        if k < 0 or k >= ntrk:
            continue
        for s in range(int(tsb[k]), int(tse[k])):
            if loc[s] == ATIP_LOCATION:
                d0[p] = D0[s] * MM_TO_CM
                z0[p] = Z0[s] * MM_TO_CM
                break
    return d0, z0


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------
def process_file(fp, dst_coll, is_mc, reco_hists, gen_hists, counter, max_events, n_done,
                 track_d0z0=False):
    try:
        t = uproot.open(fp)["events"]
    except Exception as exc:
        print(f"  skip {fp}: {exc}")
        return n_done

    n_entries = t.num_entries
    if n_entries == 0:
        return n_done

    # --- reco particle arrays for the chosen DST ---
    px = t[f"{dst_coll}.momentum.x"].array()
    py = t[f"{dst_coll}.momentum.y"].array()
    pz = t[f"{dst_coll}.momentum.z"].array()
    en = t[f"{dst_coll}.energy"].array()
    ms = t[f"{dst_coll}.mass"].array()
    ch = t[f"{dst_coll}.charge"].array()

    # --- track impact parameters from the AtIP track state (optional) ---
    trk_br = track_d0z0_branches(t, dst_coll) if track_d0z0 else None

    # --- per-event scalars + dedup fingerprint ---
    ecms = gp_column(t, "Float", ECMS_KEY)
    run = gp_column(t, "Int", RUN_KEY)
    evno = gp_column(t, "Int", EVT_KEY)
    if ecms is None:
        ecms = np.full(n_entries, 91.25)  # 94c is Z-pole; fall back if GP missing
    if run is None:
        run = np.zeros(n_entries, dtype=np.int64)
    if evno is None:
        evno = np.arange(n_entries)

    npart = ak.to_numpy(ak.num(ch))
    sumE = ak.to_numpy(ak.sum(en, axis=1))
    # collapse consecutive byte-identical entries (data triples; MC untouched)
    fp_key = np.stack([run.astype(np.float64), evno.astype(np.float64),
                       npart.astype(np.float64), np.round(sumE, 3)], axis=1)
    keep = np.ones(n_entries, dtype=bool)
    keep[1:] = np.any(fp_key[1:] != fp_key[:-1], axis=1)

    # --- gen arrays (MC) ---
    if is_mc:
        g_px = t[f"{GEN_COLLECTION}.momentum.x"].array()
        g_py = t[f"{GEN_COLLECTION}.momentum.y"].array()
        g_pz = t[f"{GEN_COLLECTION}.momentum.z"].array()
        g_ms = t[f"{GEN_COLLECTION}.mass"].array()
        g_ch = t[f"{GEN_COLLECTION}.charge"].array()
        g_st = t[f"{GEN_COLLECTION}.generatorStatus"].array()
        g_pdg = t[f"{GEN_COLLECTION}.PDG"].array()

    for i in range(n_entries):
        if not keep[i]:
            continue
        if max_events > 0 and n_done >= max_events:
            break
        E = float(ecms[i])
        # off-peak energy-scan rejection (no-op for 94c pole; kept for fidelity)
        if abs(E - 91.25) > 1:
            continue

        counter.Fill(0.5)
        n_done += 1

        if is_mc:
            fill_gen(g_px[i], g_py[i], g_pz[i], g_ms[i], g_ch[i], g_st[i], g_pdg[i], gen_hists)

        d0 = z0 = None
        if trk_br is not None:
            d0, z0 = entry_track_d0z0(trk_br, i, len(ms[i]))

        fill_reco(px[i], py[i], pz[i], ms[i], ch[i], E, reco_hists, counter, d0=d0, z0=z0)

    return n_done


def fill_reco(px, py, pz, ms, ch, E, h, counter, d0=None, z0=None):
    px = ak.to_numpy(px).astype(np.float64)
    py = ak.to_numpy(py).astype(np.float64)
    pz = ak.to_numpy(pz).astype(np.float64)
    m = ak.to_numpy(ms).astype(np.float64)
    q = ak.to_numpy(ch).astype(np.float64)
    if len(px) == 0:
        return

    pt = np.sqrt(px * px + py * py)
    th = np.arctan2(pt, pz)
    # d0/z0 from the AtIP track state when provided (--track-d0z0); otherwise
    # zeros -> impact-parameter cut is a no-op (the original behaviour).
    d0 = np.zeros_like(px) if d0 is None else np.asarray(d0, dtype=np.float64)
    z0 = np.zeros_like(px) if z0 is None else np.asarray(z0, dtype=np.float64)

    selection = apply_track_selection_delphi(
        px=px, py=py, pz=pz, m=m, q=q, th=th, pt=pt, d0=d0, z0=z0,
        charged_p_min=CHARGED_P_MIN,
        charged_z0sin_max=CHARGED_Z0SIN_MAX,
        neutral_e_min=NEUTRAL_E_MIN,
    )
    sel = selection["sel"]
    if not np.any(sel):
        return

    px_s, py_s, pz_s = px[sel], py[sel], pz[sel]
    m_s, q_s = m[sel], q[sel]
    sel_c = np.abs(q_s) > 0.1
    sel_n = ~sel_c

    tv = calculate_all_thrust_variants(px_s, py_s, pz_s, m_s, q_s, sel_c, sel_n, include_met=True)

    e_c = np.sqrt(px_s[sel_c] ** 2 + py_s[sel_c] ** 2 + pz_s[sel_c] ** 2 + m_s[sel_c] ** 2)
    e_all = np.sqrt(px_s ** 2 + py_s ** 2 + pz_s ** 2 + m_s ** 2)
    theta_T = thrust_theta(tv["axis"], tv["T"], fold=False)

    ev = apply_event_selection_delphi(e_c=e_c, e_n=e_all, theta_Tu=theta_T, E_reco=E)
    if not ev["pass_reco"]:
        return

    counter.Fill(1.5)
    h["ThrustMissPNC2_nominal"].Fill(tv["thrust"])
    h["ThrustMissPNCLog2_nominal"].Fill(tv["thrust_log"])
    h["ThrustMissPNC2_Escheme_nominal"].Fill(tv["thrust_escheme"])
    h["ThrustMissPNCLog2_Escheme_nominal"].Fill(tv["thrust_log_escheme"])
    h["ThrustC2_nominal"].Fill(tv["thrust_charged"])
    h["ThrustCLog2_nominal"].Fill(tv["thrust_charged_log"])
    h["ThrustC2_Escheme_nominal"].Fill(tv["thrust_charged_escheme"])
    h["ThrustCLog2_Escheme_nominal"].Fill(tv["thrust_charged_log_escheme"])


def fill_gen(px, py, pz, ms, ch, st, pdg, h):
    st = ak.to_numpy(st)
    pdg = ak.to_numpy(pdg)
    fs = (st == 1) & ~np.isin(np.abs(pdg), list(NEUTRINO_PDG))
    if not np.any(fs):
        return
    px = ak.to_numpy(px).astype(np.float64)[fs]
    py = ak.to_numpy(py).astype(np.float64)[fs]
    pz = ak.to_numpy(pz).astype(np.float64)[fs]
    m = ak.to_numpy(ms).astype(np.float64)[fs]
    q = ak.to_numpy(ch).astype(np.float64)[fs]

    sel_c = np.abs(q) > 0.1
    sel_n = ~sel_c
    tv = calculate_all_thrust_variants(px, py, pz, m, q, sel_c, sel_n, include_met=False)

    h["Thrust_before2"].Fill(tv["thrust"])
    h["Thrust_before_log2"].Fill(tv["thrust_log"])
    h["Thrust_before2_Escheme"].Fill(tv["thrust_escheme"])
    h["Thrust_before_log2_Escheme"].Fill(tv["thrust_log_escheme"])
    h["ThrustC_before2"].Fill(tv["thrust_charged"])
    h["ThrustC_before_log2"].Fill(tv["thrust_charged_log"])
    h["ThrustC_before2_Escheme"].Fill(tv["thrust_charged_escheme"])
    h["ThrustC_before_log2_Escheme"].Fill(tv["thrust_charged_log_escheme"])


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("infile", help="EDM4hep file, glob, or a .txt/.list of file paths (one per line)")
    ap.add_argument("outfile")
    ap.add_argument("--is-mc", action="store_true",
                    help="also fill gen-level 'before' histograms from sDST_LUJ_GenParticles")
    ap.add_argument("--dst", choices=["fdst", "sdst"], default="fdst",
                    help="which ReconstructedParticle collection to use (identical in current files)")
    ap.add_argument("--max-events", type=int, default=-1)
    ap.add_argument("--track-d0z0", action="store_true",
                    help="apply the d0/z0 cut using the AtIP track state "
                         "(navigate ReconstructedParticle -> track -> getTrackState(AtIP); "
                         "mm->cm). Default off keeps the original d0=z0=0 no-op behaviour.")
    args = ap.parse_args()

    if args.infile.endswith((".txt", ".list")):
        with open(args.infile) as fh:
            files = [ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")]
    elif any(c in args.infile for c in "*?["):
        files = sorted(glob.glob(args.infile))
    else:
        files = [args.infile]
    if not files:
        print(f"No files match: {args.infile}", file=sys.stderr)
        sys.exit(1)
    dst_coll = DST_COLLECTION[args.dst]
    print(f"Reading {len(files)} file(s); DST={dst_coll}; is_mc={args.is_mc}; "
          f"track_d0z0={args.track_d0z0}")

    reco_hists = book_reco_hists()
    gen_hists = book_gen_hists() if args.is_mc else None
    counter = book_counter()

    n_done = 0
    for k, fp in enumerate(files):
        if args.max_events > 0 and n_done >= args.max_events:
            break
        n_done = process_file(fp, dst_coll, args.is_mc, reco_hists, gen_hists,
                              counter, args.max_events, n_done,
                              track_d0z0=args.track_d0z0)
        print(f"  [{k + 1}/{len(files)}] {fp.split('/')[-1]}  -> {n_done} events kept")

    print(f"Total: {int(counter.GetBinContent(1))} events, "
          f"{int(counter.GetBinContent(2))} pass reco selection")

    fout = ROOT.TFile(args.outfile, "RECREATE")
    fout.cd()
    counter.Write()
    if gen_hists:
        fout.mkdir("gen"); fout.cd("gen")
        for hh in gen_hists.values():
            hh.Write()
        fout.cd()
    fout.mkdir("reco"); fout.cd("reco")
    for hh in reco_hists.values():
        hh.Write()
    fout.Close()
    print(f"Wrote {args.outfile}")


if __name__ == "__main__":
    main()

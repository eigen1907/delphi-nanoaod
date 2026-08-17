#!/usr/bin/env python3
"""
EEC analysis (charged-only) on **raw DelphiNanoAOD** ttree files — RECO + GEN.

NanoAOD counterpart to analysis_eec.py. The physics — histogram booking,
systematic variations, EEC/thrust fill, covariance — is unchanged and
re-used from analysis_eec via the shared process_gen_arrays /
process_reco_arrays entry points. Only the I/O layer differs.

Why this exists
---------------
analysis_eec.py reads a pre-flattened skim (the `t`/`tgenBefore`/`tgen` trees
with flat px/py/pz/d0/z0 branches found in the TPCNtuple files). Its gen-level
particles come from tgenBefore with an energy-balance fallback to tgen — which
does NOT recover the correct final state when V0s (K0_S, Lambda, ...) or prompt
heavy-flavor hadrons decay in the tracker, so the GEN EEC is biased. Here we
read the raw DelphiNanoAOD `t` tree directly and build the GEN final state with
the V0/heavy-flavor-aware SimPart leaf walk (analysis_thrust_nanoaod.
select_leaves — the SAME input + leaf logic the thrust nanoaod analyzer uses).
This yields the CORRECT gen EEC, identical in definition to the thrust nanoaod.

Input format: DelphiNanoAOD flat `t` TTree
------------------------------------------
Reads the DelphiNanoAOD `*_ttree.root` files — a single TTree **"t"** holding
the raw per-event collections as flat branches
(`Part_fourMomentum_fCoordinates_{fX,fY,fZ,fT}`, `Part_lock`, `Trac_*`,
`SimPart_*`, `Event_cmEnergy`) — exactly the tree/branch layout
analysis_thrust_nanoaod reads. This is the canonical analysis nanoaod
(`MC/<period>/DelphiNanoAOD/<stream>/<date>/` and `Data/<period>/DelphiNanoAOD/
<date>/`), NOT the TPCNtuple `Events` RNTuple (a separate copy) nor the
TPCNtuple flat skim.

Reader specifics (same input + leaf logic as analysis_thrust_nanoaod)
--------------------------------------------------------------------
* RECO final state = Part with Part_lock == 0 (LVLOCK standard selection).
  px,py,pz,energy <- Part_fourMomentum_fCoordinates_{fX,fY,fZ,fT}; mass from
  E^2 - p^2; charge <- Part_charge. Impact parameters d0,z0 are scattered into
  the per-Part arrays from Trac_impParToVertexRPhi / Trac_impParToVertexZ via
  Trac_partIdx (neutrals keep d0=z0=0). The Trac impact parameters are in CM
  (SKELANA QTRAC(38/39), DELPHI's native length unit). Because real d0/z0 ARE
  available, the charged |d0| / |z0*sin(theta)| cuts in
  apply_track_selection_delphi are ACTIVE here.
* GEN final state = V0-aware SimPart leaf recovery (select_leaves). This is the
  CORRECT gen definition; it deliberately does NOT use analysis_eec.py's flat
  tgenBefore->tgen energy fallback.

Baseline charged-track cut is |p| > 0.4 GeV (charged_p_min), inherited from
analysis_eec.SYSTEMATICS — matching the thrust analyzer.

Usage
-----
  MC:   python analysis_eec_nanoaod.py 'mc/*.root'   out.root --is-mc
  Data: python analysis_eec_nanoaod.py 'data/*.root' out.root
        (optional: --systematic nominal, --compute-covariance,
         --covariance-2d, --max-events N)
"""

import sys
import glob
import argparse

import numpy as np
import uproot
import awkward as ak
import ROOT  # noqa: F401

from analysis_eec import (
    SYSTEMATICS,
    create_counter_histogram,
    create_gen_histograms,
    create_reco_histograms,
    process_gen_arrays,
    process_reco_arrays,
    init_covariance_state,
    write_covariance,
    create_response_objects,
    process_response_arrays,
)
from analysis_thrust_nanoaod import select_leaves, build_gen_leaf_redecayed

# DelphiNanoAOD stores the raw collections in a flat TTree "t" (same tree +
# flat branch names that analysis_thrust_nanoaod reads). Four-momentum
# components are flat branches Part/SimPart_fourMomentum_fCoordinates_{fX..fT}.
TREE = "t"
_PX = "_fourMomentum_fCoordinates_fX"
_PY = "_fourMomentum_fCoordinates_fY"
_PZ = "_fourMomentum_fCoordinates_fZ"
_E = "_fourMomentum_fCoordinates_fT"

RECO_BRANCHES = [
    "Event_cmEnergy",
    "Part_charge", "Part_lock",
    f"Part{_PX}", f"Part{_PY}", f"Part{_PZ}", f"Part{_E}",
    "Trac_partIdx", "Trac_impParToVertexRPhi", "Trac_impParToVertexZ",
]
GEN_BRANCHES = [
    "SimPart_originVtxIdx", "SimPart_decayVtxIdx", "SimPart_pdgId", "SimPart_charge",
    f"SimPart{_PX}", f"SimPart{_PY}", f"SimPart{_PZ}", f"SimPart{_E}",
]


# ---------------------------------------------------------------------------
# Per-event readers: raw nanoaod collections -> per-event physics arrays
# ---------------------------------------------------------------------------
def build_reco_arrays(arr, iEv):
    """RECO final state from Part with Part_lock == 0.

    Returns (E, px, py, pz, m, q, pt, th, d0, z0) [GeV / cm], or None if the
    event has no Part entries / no unlocked particles. E is the event cm
    energy. d0/z0 are scattered from Trac_* via Trac_partIdx (neutrals keep
    d0=z0=0) before the lock filter so the Trac indexing stays consistent.
    """
    px = ak.to_numpy(arr[f"Part{_PX}"][iEv]).astype(np.float64)
    nPart = len(px)
    if nPart == 0:
        return None
    py = ak.to_numpy(arr[f"Part{_PY}"][iEv]).astype(np.float64)
    pz = ak.to_numpy(arr[f"Part{_PZ}"][iEv]).astype(np.float64)
    en = ak.to_numpy(arr[f"Part{_E}"][iEv]).astype(np.float64)
    q = ak.to_numpy(arr["Part_charge"][iEv]).astype(np.float64)
    lock = ak.to_numpy(arr["Part_lock"][iEv]).astype(np.int64)

    pt = np.sqrt(px * px + py * py)
    th = np.arctan2(pt, pz)
    m = np.sqrt(np.clip(en * en - (px * px + py * py + pz * pz), 0.0, None))

    d0 = np.zeros(nPart, dtype=np.float64)
    z0 = np.zeros(nPart, dtype=np.float64)
    tpi = ak.to_numpy(arr["Trac_partIdx"][iEv]).astype(np.int64)
    if len(tpi) > 0:
        valid = (tpi >= 0) & (tpi < nPart)
        if valid.any():
            d0_t = ak.to_numpy(arr["Trac_impParToVertexRPhi"][iEv]).astype(np.float64)
            z0_t = ak.to_numpy(arr["Trac_impParToVertexZ"][iEv]).astype(np.float64)
            d0[tpi[valid]] = d0_t[valid]
            z0[tpi[valid]] = z0_t[valid]

    keep = (lock == 0)
    if not keep.any():
        return None

    E = float(arr["Event_cmEnergy"][iEv])
    return (E, px[keep], py[keep], pz[keep], m[keep], q[keep],
            pt[keep], th[keep], d0[keep], z0[keep])


def build_gen_arrays(arr, iEv):
    """GEN final state via V0-aware SimPart leaf recovery, in the Pythia V0-decay
    convention (kept-undecayed / interacted V0s are re-decayed; shared with the
    thrust analyzer via build_gen_leaf_redecayed).

    Returns (px, py, pz, m, q) [GeV], or None if no leaves. Charge is the raw
    SimPart_charge of each leaf (re-decay daughters carry Pythia charges). Energy
    is recomputed by the EEC core from the momenta + mass, so it is not returned.
    """
    o = ak.to_numpy(arr["SimPart_originVtxIdx"][iEv])
    if len(o) == 0:
        return None
    d = ak.to_numpy(arr["SimPart_decayVtxIdx"][iEv])
    pdg = ak.to_numpy(arr["SimPart_pdgId"][iEv])
    Px = ak.to_numpy(arr[f"SimPart{_PX}"][iEv]).astype(np.float64)
    Py = ak.to_numpy(arr[f"SimPart{_PY}"][iEv]).astype(np.float64)
    Pz = ak.to_numpy(arr[f"SimPart{_PZ}"][iEv]).astype(np.float64)
    E4 = ak.to_numpy(arr[f"SimPart{_E}"][iEv]).astype(np.float64)
    charge = ak.to_numpy(arr["SimPart_charge"][iEv]).astype(np.float64)
    return build_gen_leaf_redecayed(o, d, pdg, charge, Px, Py, Pz, E4)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("infile", help="raw DelphiNanoAOD file or glob")
    ap.add_argument("outfile")
    ap.add_argument("--is-mc", action="store_true",
                    help="fill gen-level (V0-aware) EEC + thrust histograms")
    ap.add_argument("--systematic", default="all",
                    help='run one systematic or "all" (default: all)')
    ap.add_argument("--compute-covariance", action="store_true",
                    help="compute covariance matrix (nominal only)")
    ap.add_argument("--covariance-2d", action="store_true",
                    help="use full 2D (r x Eij) covariance instead of 1D")
    ap.add_argument("--no-response", action="store_true",
                    help="skip the flattened gen<->reco response build (MC only; "
                         "on by default for MC). Produces response2d_eij_{r,z} "
                         "(4D THnD) + reco2d/gen2d/_match for unfold_2dflatten*.")
    ap.add_argument("--match-r", type=float, default=0.05,
                    help="gen<->reco angular matching radius (rad) for the "
                         "response build (default 0.05; systematic variations "
                         "e.g. 0.01 / 0.1 require rebuilding the response)")
    ap.add_argument("--match-same-charge", action="store_true",
                    help="require SAME charge in the gen<->reco response matching. "
                         "Default OFF: the EEC is charge-blind, so matching by angle "
                         "only avoids losing charge-misID'd tracks. (fbcec keeps its "
                         "own same-charge matching independently; unaffected.)")
    ap.add_argument("--max-events", type=int, default=-1)
    args = ap.parse_args()

    build_response = args.is_mc and not args.no_response

    if args.systematic == "all":
        systematics_to_run = SYSTEMATICS
    elif args.systematic in SYSTEMATICS:
        systematics_to_run = {args.systematic: SYSTEMATICS[args.systematic]}
    else:
        print(f"ERROR: unknown systematic '{args.systematic}'", file=sys.stderr)
        print(f"Available: {', '.join(SYSTEMATICS.keys())}", file=sys.stderr)
        return 1

    files = (sorted(glob.glob(args.infile))
             if any(c in args.infile for c in "*?[") else [args.infile])
    if not files:
        print(f"No files match: {args.infile}", file=sys.stderr)
        return 1
    print(f"Running EEC NanoAOD analysis on {len(files)} file(s); "
          f"is_mc={args.is_mc}")
    print(f"Systematics: {', '.join(systematics_to_run.keys())}")
    print(f"Covariance: {args.compute_covariance} (2D: {args.covariance_2d})")
    print(f"Response (flattened, unfold_2dflatten): {build_response}")

    # --- Histograms (mirror analysis_eec.py main) ---
    counter = create_counter_histogram()
    gen_hists = create_gen_histograms() if args.is_mc else None
    reco_hists = {s: create_reco_histograms(s) for s in systematics_to_run}
    covariance_state = (init_covariance_state(args.covariance_2d)
                        if args.compute_covariance else None)
    resp = create_response_objects() if build_response else None

    read_branches = list(RECO_BRANCHES) + (list(GEN_BRANCHES) if args.is_mc else [])

    n_done = 0
    stop = False
    for k, fp in enumerate(files):
        if stop:
            break
        try:
            t = uproot.open(fp)[TREE]
        except Exception as exc:
            print(f"  skip {fp}: {exc}")
            continue
        n_entries = t.num_entries
        if n_entries == 0:
            continue
        arr = t.arrays(read_branches)

        for iEv in range(n_entries):
            if args.max_events > 0 and n_done >= args.max_events:
                stop = True
                break

            # Count every processed event (bin 1), like analysis_eec.py main.
            counter.Fill(0.5)
            n_done += 1

            E = float(arr["Event_cmEnergy"][iEv])
            on_pole = abs(E - 91.25) <= 1  # DELPHI Z-pole gate (load-bearing)

            # --- GEN (MC only, on-pole): V0-aware SimPart leaves ---
            gen = None
            if args.is_mc and on_pole:
                gen = build_gen_arrays(arr, iEv)
                if gen is not None and gen_hists is not None:
                    gpx, gpy, gpz, gm, gq = gen
                    process_gen_arrays(gpx, gpy, gpz, gm, gq, E, gen_hists)

            # --- RECO (always): all systematics, event selection, EEC,
            #     and (nominal) covariance. The core re-applies the E gate. ---
            reco = build_reco_arrays(arr, iEv)
            if reco is None:
                continue
            E_r, rpx, rpy, rpz, rm, rq, rpt, rth, rd0, rz0 = reco
            process_reco_arrays(
                rpx, rpy, rpz, rm, rq, rpt, rth, rd0, rz0, E_r,
                systematics_to_run, reco_hists, counter,
                covariance_state=covariance_state,
            )

            # --- RESPONSE (MC, on-pole): single-pass gen<->reco migration,
            #     same selection + V0-aware gen as above. Flattened THnD for
            #     unfold_2dflatten*. Gated on reco event selection inside. ---
            if resp is not None and gen is not None:
                gpx, gpy, gpz, gm, gq = gen
                process_response_arrays(
                    rpx, rpy, rpz, rm, rq, rpt, rth, rd0, rz0, E_r,
                    gpx, gpy, gpz, gm, gq, E, resp,
                    match_r=args.match_r,
                    require_same_charge=args.match_same_charge,
                )

        print(f"  [{k + 1}/{len(files)}] {fp.split('/')[-1]}  -> {n_done} events processed")

    print(f"Total: {int(counter.GetBinContent(1))} events, "
          f"{int(counter.GetBinContent(2))} pass P-scheme selection")

    # --- Write output (same structure as analysis_eec.py) ---
    fout = ROOT.TFile(args.outfile, "RECREATE")

    if gen_hists is not None:
        fout.mkdir("gen"); fout.cd("gen")
        for h in gen_hists.values():
            h.Write()

    fout.mkdir("reco"); fout.cd("reco")
    for hists in reco_hists.values():
        for h in hists.values():
            h.Write()

    fout.cd()
    counter.Write()

    # Surface the nominal reco EEC2d at file root as EEC2d_{r,z} (no suffix).
    # unfold_2dflatten*.py reads the *data* measurement as `EEC2d_{jacobian}`
    # at root (data2dname); the per-syst copies still live in reco/.
    if 'nominal' in reco_hists:
        for var in ('r', 'z'):
            src = reco_hists['nominal'].get(f'EEC2d_{var}_nominal')
            if src is not None:
                src.Clone(f'EEC2d_{var}').Write()

    # Flattened response objects at file root (where unfold_2dflatten* reads
    # reco2d_eij_{r,z}[_match], gen2d_eij_{r,z}[_match], response2d_eij_{r,z}).
    if resp is not None:
        fout.cd()
        for h in resp.values():
            h.Write()

    if covariance_state is not None:
        fout.mkdir("covariance"); fout.cd("covariance")
        write_covariance(covariance_state, fout)

    fout.Close()
    print(f"Wrote {args.outfile}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

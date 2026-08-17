#!/usr/bin/env python3
"""
Per-event dump for joint thrust bootstrap — RAW DelphiNanoAOD reader.

Nanoaod counterpart of analysis_thrust_joint_data.py: identical data-side
nominal selection and per-event observables, but reads the raw DelphiNanoAOD
tree 't' (Part_*/Trac_*/Event_*) instead of the pre-flattened skim. The reco
final state is built with analysis_thrust_nanoaod.build_reco_arrays (Part_lock
== 0, d0/z0 scattered from Trac_* via Trac_partIdx) — the SAME validated reader
analysis_thrust_nanoaod.py uses (v69 reproduced the flat-skim reco exactly).
The selection / thrust / event-selection lines below are byte-identical to
analysis_thrust_joint_data.py and to the nominal branch of
analysis_thrust.process_reco_level_arrays, so the per-event values here are
bit-consistent with the v70 nominal reco histograms and reproduce the
flat-skim dump.

Output: numpy npz archive (written verbatim to the given path, numpy's auto
".npz" suffix suppressed via a file handle) with arrays
    thrust2              float64  1 - T,        P-scheme all-particles
    thrust2_Escheme      float64  1 - T_E,      E-scheme all-particles
    thrust_log2          float64  log(thrust2)
    thrust_log2_Escheme  float64  log(thrust2_Escheme)
    evt_weight           float64  per-event weight (1.0 under nominal)
Schema identical to analysis_thrust_joint_data.py, so hadd_npz_thrust_joint.py
and unfold_thrust_bootstrap_joint.py consume it unchanged.

Usage:
    python analysis_thrust_joint_data_nanoaod.py input_ttree.root output_path
"""

import sys
import argparse
import numpy as np
import uproot
import awkward as ak  # noqa: F401  (build_reco_arrays uses awkward)
import ROOT  # noqa: F401

from common_functions import thrust_theta
from binning_and_selections import (
    apply_track_selection_delphi,
    apply_event_selection_delphi,
)
from analysis_thrust import calculate_all_thrust_variants
from analysis_thrust_nanoaod import build_reco_arrays, RECO_BRANCHES


def process_file(path, store):
    """Append per-event values from one raw-nanoaod ROOT file to `store`."""
    try:
        t = uproot.open(path)["t"]
    except Exception as exc:
        print(f"  ERROR: cannot open {path}: {exc}")
        return 0, 0

    n = t.num_entries
    if n == 0:
        return 0, 0
    arr = t.arrays(RECO_BRANCHES)

    n_pass = 0
    for iEvt in range(n):
        E = float(arr["Event_cmEnergy"][iEvt])
        if abs(E - 91.25) > 1:
            continue

        # Reco final state from the raw nanoaod (Part_lock == 0; d0/z0 from Trac_*)
        reco = build_reco_arrays(arr, iEvt)
        if reco is None:
            continue
        px, py, pz, m, q, pt, th, d0, z0 = reco

        sel = apply_track_selection_delphi(
            px=px, py=py, pz=pz, m=m, q=q, th=th, pt=pt, d0=d0, z0=z0,
            charged_p_min=0.4, charged_z0sin_max=10.0, neutral_e_min=0.5,
        )['sel']
        if not np.any(sel):
            continue

        px_s = px[sel]; py_s = py[sel]; pz_s = pz[sel]
        m_s  = m[sel];  q_s  = q[sel]
        sel_c = np.abs(q_s) > 0.1
        sel_n = ~sel_c

        tv = calculate_all_thrust_variants(
            px_s, py_s, pz_s, m_s, q_s, sel_c, sel_n, include_met=True,
        )

        e_c = np.sqrt(px_s[sel_c]**2 + py_s[sel_c]**2 +
                      pz_s[sel_c]**2 + m_s[sel_c]**2)
        e_all = np.sqrt(px_s**2 + py_s**2 + pz_s**2 + m_s**2)
        theta_T = thrust_theta(tv['axis'], tv['T'], fold=False)
        es = apply_event_selection_delphi(
            e_c=e_c, e_n=e_all, theta_Tu=theta_T, E_reco=E,
        )
        if not es['pass_reco']:
            continue

        store['thrust2'].append(tv['thrust'])
        store['thrust2_Escheme'].append(tv['thrust_escheme'])
        store['thrust_log2'].append(tv['thrust_log'])
        store['thrust_log2_Escheme'].append(tv['thrust_log_escheme'])
        store['evt_weight'].append(1.0)
        n_pass += 1

    return n, n_pass


def main():
    parser = argparse.ArgumentParser(
        description='Per-event dump for joint thrust bootstrap (raw nanoaod, data, nominal).'
    )
    parser.add_argument('infile',  help='Input raw DelphiNanoAOD ttree file')
    parser.add_argument('outfile', help='Output path (npz contents written verbatim)')
    args = parser.parse_args()

    store = {
        'thrust2': [],
        'thrust2_Escheme': [],
        'thrust_log2': [],
        'thrust_log2_Escheme': [],
        'evt_weight': [],
    }

    print(f"Processing {args.infile}")
    n_seen, n_pass = process_file(args.infile, store)
    print(f"Events seen:    {n_seen}")
    print(f"Events passing: {n_pass}")

    arrays = {k: np.asarray(v, dtype=np.float64) for k, v in store.items()}
    with open(args.outfile, 'wb') as fh:
        np.savez(fh, **arrays)
    print(f"Wrote {args.outfile} (npz contents, {n_pass} events)")
    return 0


if __name__ == '__main__':
    sys.exit(main())

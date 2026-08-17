#!/usr/bin/env python3
"""
Per-event dump for joint thrust bootstrap (data, nominal, all-particles only).

Reproduces the data-side selection used by analysis_thrust.py for the nominal
systematic, and dumps per-event arrays of the four all-particles thrust
variants (+ evt_weight). The output feeds a Poisson-bootstrap that resamples
events ONCE per toy and projects the same resample into all four observables,
so the off-diagonal blocks of the joint covariance are statistically meaningful.

Output is a numpy npz archive with arrays:
    thrust2              float64  1 - T,         P-scheme all-particles
    thrust2_Escheme      float64  1 - T_E,       E-scheme all-particles
    thrust_log2          float64  log(thrust2)
    thrust_log2_Escheme  float64  log(thrust2_Escheme)
    evt_weight           float64  per-event weight (1.0 under nominal)

Calling convention matches analysis_thrust.py (input, output) so the script
can be dispatched through condor/submit_ana.py + condor/condor.sub unchanged.
The condor pipeline names the output with a .root extension; this script
writes npz *contents* to that path verbatim (numpy's auto-append of ".npz" is
suppressed via a file handle). The downstream concat step loads the per-job
files with np.load (extension-agnostic) and writes the final .npz.

Usage:
    python analysis_thrust_joint_data.py input.root output_path
"""

import sys
import argparse
import numpy as np
import ROOT

from common_functions import thrust_theta
from binning_and_selections import (
    apply_track_selection_delphi,
    apply_event_selection_delphi,
)
from analysis_thrust import calculate_all_thrust_variants


def process_file(path, store):
    """Append per-event values from one input ROOT file to `store`."""
    fin = ROOT.TFile.Open(path, 'READ')
    if not fin or fin.IsZombie():
        print(f"  ERROR: cannot open {path}")
        return 0, 0

    tree = fin.Get('t')
    if not tree:
        print(f"  ERROR: no tree 't' in {path}")
        fin.Close()
        return 0, 0

    n = tree.GetEntries()
    n_pass = 0
    for iEvt in range(n):
        tree.GetEntry(iEvt)

        E = tree.Energy
        if abs(E - 91.25) > 1:
            continue

        px = np.array(tree.px); py = np.array(tree.py); pz = np.array(tree.pz)
        m  = np.array(tree.mass); q  = np.array(tree.charge)
        pt = np.array(tree.pt); th = np.array(tree.theta)
        d0 = np.array(tree.d0); z0 = np.array(tree.z0)

        if len(px) == 0:
            continue

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

    fin.Close()
    return n, n_pass


def main():
    parser = argparse.ArgumentParser(
        description='Per-event dump for joint thrust bootstrap (data, nominal).'
    )
    parser.add_argument('infile',  help='Input ROOT file')
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

#!/usr/bin/env python3
"""
Bootstrap covariance matrix for thrust unfolding
-------------------------------------------------
Computes the statistical covariance matrix of the unfolded, normalized
thrust distribution using Poisson toy fluctuations.

Method (following ALEPH hep-ex/0512066):
  For each toy:
    1. Poisson-fluctuate each bin of the raw folded data histogram
    2. Unfold with RooUnfoldBayes (response matrix fixed)
    3. Divide by bin width
    4. Divide by norm_toy = toy_folded.Integral()  (cross-section normalized)
    5. Store the normalized array
  Then compute the sample covariance matrix from the ensemble.

Only for:
  - kk2f_qqpy (nominal response matrix)
  - nominal systematic
  - Both datasets: 94c, 95d
  - All 8 thrust modes

Usage:
  python unfold_thrust_bootstrap.py --version v49 --iterations 5 --ntoys 10000
"""

import ROOT
import numpy as np
import os
import sys
import argparse
import time
import contextlib

from ROOT import RooUnfoldResponse, RooUnfoldBayes


@contextlib.contextmanager
def _suppress_c_stdout():
    """fd-level suppression of stdout (catches printf/cout from RooUnfold C++)."""
    sys.stdout.flush()
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_fd = os.dup(1)
    os.dup2(devnull_fd, 1)
    try:
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved_fd, 1)
        os.close(saved_fd)
        os.close(devnull_fd)


def convert_to_roounfold_response(response_hist, name="response"):
    """Convert TH2 response matrix to RooUnfoldResponse object.
    
    Uses projections of the response matrix itself for the 1D distributions,
    consistent with unfold_thrust_cross.py.
    """
    gen_from_response = response_hist.ProjectionY(f"{name}_gen_proj")
    reco_from_response = response_hist.ProjectionX(f"{name}_reco_proj")
    
    response = ROOT.RooUnfoldResponse(
        reco_from_response,
        gen_from_response,
        response_hist,
        name,
        name,
        False
    )
    return response


def th1_to_array(h):
    """Extract bin contents from TH1 as numpy array."""
    nbins = h.GetNbinsX()
    return np.array([h.GetBinContent(i + 1) for i in range(nbins)])


def get_bin_widths(h):
    """Extract bin widths from TH1 as numpy array."""
    nbins = h.GetNbinsX()
    return np.array([h.GetBinWidth(i + 1) for i in range(nbins)])


def get_bin_edges(h):
    """Extract bin edges from TH1 as numpy array."""
    nbins = h.GetNbinsX()
    edges = np.array([h.GetBinLowEdge(i + 1) for i in range(nbins + 1)])
    return edges


def poisson_fluctuate_th1(h_original, rng):
    """Create a Poisson-fluctuated clone of a TH1.
    
    Parameters
    ----------
    h_original : ROOT.TH1
        Original histogram with integer event counts
    rng : np.random.Generator
        Numpy random generator for reproducibility
    
    Returns
    -------
    ROOT.TH1 : Fluctuated clone
    """
    h_toy = h_original.Clone("h_toy")
    h_toy.SetDirectory(0)
    
    for i in range(1, h_original.GetNbinsX() + 1):
        mu = h_original.GetBinContent(i)
        if mu > 0:
            new_val = rng.poisson(mu)
        else:
            new_val = 0
        h_toy.SetBinContent(i, float(new_val))
        h_toy.SetBinError(i, np.sqrt(float(new_val)))
    
    return h_toy


def run_bootstrap_for_mode(data_hist, response_2d, n_toys, n_iterations, rng):
    """Run bootstrap toys for a single thrust mode.
    
    Parameters
    ----------
    data_hist : ROOT.TH1D
        Raw folded data histogram (event counts)
    response_2d : ROOT.TH2D
        2D response matrix
    n_toys : int
        Number of toy replicas
    n_iterations : int
        Bayesian unfolding iterations
    rng : np.random.Generator
        Random generator
    
    Returns
    -------
    dict with:
        'ensemble' : ndarray (n_toys, n_bins) - normalized unfolded toys
        'nominal'  : ndarray (n_bins,) - nominal unfolded result
        'mean'     : ndarray (n_bins,) - mean of toys
        'cov'      : ndarray (n_bins, n_bins) - covariance matrix
        'corr'     : ndarray (n_bins, n_bins) - correlation matrix
        'bin_edges': ndarray (n_bins+1,)
    """
    nbins = data_hist.GetNbinsX()
    bin_widths = get_bin_widths(data_hist)
    bin_edges = get_bin_edges(data_hist)
    
    # Build RooUnfoldResponse (fixed for all toys)
    response = convert_to_roounfold_response(response_2d, name="bootstrap_response")
    
    # --- Nominal unfolding (for validation) ---
    norm_data = data_hist.Integral()
    with _suppress_c_stdout():
        unfold_nom = RooUnfoldBayes(response, data_hist, n_iterations)
        h_nom = unfold_nom.Hunfold()
    nominal = th1_to_array(h_nom) / bin_widths / norm_data

    # --- Toy loop ---
    ensemble = np.zeros((n_toys, nbins))

    t0 = time.time()
    with _suppress_c_stdout():
        for k in range(n_toys):
            if (k + 1) % 1000 == 0:
                elapsed = time.time() - t0
                rate = (k + 1) / elapsed
                eta = (n_toys - k - 1) / rate
                # Progress to stderr so it survives the stdout suppression
                print(f"      Toy {k+1}/{n_toys}  ({rate:.0f} toys/s, ETA {eta:.0f}s)",
                      file=sys.stderr, flush=True)

            # 1. Poisson fluctuate
            h_toy = poisson_fluctuate_th1(data_hist, rng)

            # 2. Get norm from fluctuated histogram
            norm_toy = h_toy.Integral()
            if norm_toy <= 0:
                # Pathological fluctuation — skip (extremely rare)
                ensemble[k, :] = np.nan
                continue

            # 3. Unfold
            unfold_toy = RooUnfoldBayes(response, h_toy, n_iterations)
            h_unf = unfold_toy.Hunfold()

            # 4. Divide by bin width and normalize
            toy_result = th1_to_array(h_unf) / bin_widths / norm_toy

            # 5. Store
            ensemble[k, :] = toy_result

            # Clean up
            del h_toy, unfold_toy, h_unf

    elapsed = time.time() - t0
    print(f"      Completed {n_toys} toys in {elapsed:.1f}s ({n_toys/elapsed:.0f} toys/s)")
    
    # Remove any NaN rows (pathological fluctuations)
    valid_mask = ~np.any(np.isnan(ensemble), axis=1)
    n_valid = np.sum(valid_mask)
    if n_valid < n_toys:
        print(f"      WARNING: {n_toys - n_valid} pathological toys removed")
    ensemble_valid = ensemble[valid_mask]
    
    # --- Compute statistics ---
    mean = np.mean(ensemble_valid, axis=0)
    
    # Sample covariance: Cov[i,j] = 1/(N-1) * sum_k (x_ki - mean_i)(x_kj - mean_j)
    cov = np.cov(ensemble_valid, rowvar=False)
    
    # Correlation matrix
    std = np.sqrt(np.diag(cov))
    with np.errstate(divide='ignore', invalid='ignore'):
        corr = cov / np.outer(std, std)
        corr = np.nan_to_num(corr, nan=0.0)
    
    return {
        'ensemble': ensemble_valid,
        'nominal': nominal,
        'mean': mean,
        'cov': cov,
        'corr': corr,
        'bin_edges': bin_edges,
        'bin_widths': bin_widths,
        'n_valid': n_valid,
    }


# ============================================================================
# Thrust mode definitions
# ============================================================================

THRUST_MODES = [
    {
        'name': 'thrust2',
        'response_base': 'response_thrust2',
        'data_name': 'ThrustMissPNC2_nominal',
        'norm_bin': 2,  # P-scheme All
        'label': 'P-scheme All, tau',
    },
    {
        'name': 'thrust_log2',
        'response_base': 'response_thrust_log2',
        'data_name': 'ThrustMissPNCLog2_nominal',
        'norm_bin': 2,  # P-scheme All
        'label': 'P-scheme All, log(tau)',
    },
    {
        'name': 'thrust2_Escheme',
        'response_base': 'response_thrust2_Escheme',
        'data_name': 'ThrustMissPNC2_Escheme_nominal',
        'norm_bin': 3,  # E-scheme All
        'label': 'E-scheme All, tau',
    },
    {
        'name': 'thrust_log2_Escheme',
        'response_base': 'response_thrust_log2_Escheme',
        'data_name': 'ThrustMissPNCLog2_Escheme_nominal',
        'norm_bin': 3,  # E-scheme All
        'label': 'E-scheme All, log(tau)',
    },
    {
        'name': 'thrustC2',
        'response_base': 'response_thrustC2',
        'data_name': 'ThrustC2_nominal',
        'norm_bin': 4,  # P-scheme Charged
        'label': 'P-scheme Charged, tau',
    },
    {
        'name': 'thrustC_log2',
        'response_base': 'response_thrustC_log2',
        'data_name': 'ThrustCLog2_nominal',
        'norm_bin': 4,  # P-scheme Charged
        'label': 'P-scheme Charged, log(tau)',
    },
    {
        'name': 'thrustC2_Escheme',
        'response_base': 'response_thrustC2_Escheme',
        'data_name': 'ThrustC2_Escheme_nominal',
        'norm_bin': 5,  # E-scheme Charged
        'label': 'E-scheme Charged, tau',
    },
    {
        'name': 'thrustC_log2_Escheme',
        'response_base': 'response_thrustC_log2_Escheme',
        'data_name': 'ThrustCLog2_Escheme_nominal',
        'norm_bin': 5,  # E-scheme Charged
        'label': 'E-scheme Charged, log(tau)',
    },
]

DATASETS = [
    {'suffix': '94c', 'year': '1994'},
    {'suffix': '95d', 'year': '1995'},
]

SYST = 'nominal'
GENERATOR = 'kk2f4146_qqpy'


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        description='Bootstrap covariance matrix for thrust unfolding')
    parser.add_argument('--version', type=str, default='v49',
                        help='Version tag for input files (e.g., v49)')
    parser.add_argument('--iterations', type=int, default=5,
                        help='Number of Bayesian unfolding iterations')
    parser.add_argument('--ntoys', type=int, default=10000,
                        help='Number of bootstrap toy replicas')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--mode', type=str, default=None,
                        help='Run only a specific thrust mode (e.g., thrust2)')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Run only a specific dataset (e.g., 94c)')
    
    args = parser.parse_args()
    VERSION = args.version
    
    # Random generator
    rng = np.random.default_rng(args.seed)
    
    # Filter modes/datasets if requested
    modes_to_run = THRUST_MODES
    if args.mode:
        modes_to_run = [m for m in THRUST_MODES if m['name'] == args.mode]
        if not modes_to_run:
            print(f"ERROR: Unknown mode '{args.mode}'")
            print(f"Available: {[m['name'] for m in THRUST_MODES]}")
            sys.exit(1)
    
    datasets_to_run = DATASETS
    if args.dataset:
        datasets_to_run = [d for d in DATASETS if d['suffix'] == args.dataset]
        if not datasets_to_run:
            print(f"ERROR: Unknown dataset '{args.dataset}'")
            sys.exit(1)
    
    # Output file
    filenameout = f"bootstrap_covariance_thrust_{VERSION}_iter{args.iterations}_ntoys{args.ntoys}.root"
    
    print(f"{'='*70}")
    print(f"BOOTSTRAP COVARIANCE FOR THRUST UNFOLDING")
    print(f"{'='*70}")
    print(f"  Version:    {VERSION}")
    print(f"  Iterations: {args.iterations}")
    print(f"  N toys:     {args.ntoys}")
    print(f"  Seed:       {args.seed}")
    print(f"  Generator:  {GENERATOR} (nominal only)")
    print(f"  Modes:      {len(modes_to_run)}")
    print(f"  Datasets:   {[d['suffix'] for d in datasets_to_run]}")
    print(f"  Output:     {filenameout}")
    print(f"{'='*70}\n")
    
    # Open output ROOT file
    fout = ROOT.TFile(filenameout, 'RECREATE')
    
    # Also collect results for npz output
    all_results = {}
    
    total_start = time.time()
    
    for ds in datasets_to_run:
        ds_suffix = ds['suffix']
        
        # Input files
        data_file = f"thrust_data_{ds_suffix}_{VERSION}.root"
        mc_file = f"thrust_{GENERATOR}_91.25_{ds_suffix}_{VERSION}.root"
        
        print(f"\n{'#'*70}")
        print(f"# Dataset: {ds_suffix} ({ds['year']})")
        print(f"{'#'*70}")
        print(f"  Data: {data_file}")
        print(f"  MC:   {mc_file}")
        
        # Check files exist
        if not os.path.exists(data_file):
            print(f"  ERROR: Data file not found: {data_file}")
            continue
        if not os.path.exists(mc_file):
            print(f"  ERROR: MC file not found: {mc_file}")
            continue
        
        fdata = ROOT.TFile.Open(data_file, 'READ')
        fmc = ROOT.TFile.Open(mc_file, 'READ')
        
        # Validate counter
        counter_data = fdata.Get("counter")
        if counter_data:
            print(f"  Counter bins:")
            for ib in range(1, counter_data.GetNbinsX() + 1):
                print(f"    Bin {ib} ({counter_data.GetXaxis().GetBinLabel(ib)}): "
                      f"{counter_data.GetBinContent(ib):.0f}")
        
        for mode in modes_to_run:
            mode_name = mode['name']
            norm_bin = mode['norm_bin']
            
            print(f"\n  --- Mode: {mode_name} ({mode['label']}) ---")
            
            # Get data histogram
            data_path = f"reco/{mode['data_name']}"
            h_data = fdata.Get(data_path)
            if not h_data:
                print(f"    WARNING: Data not found: {data_path}")
                continue
            h_data = h_data.Clone(f"data_{mode_name}_{ds_suffix}")
            h_data.SetDirectory(0)
            
            # Validate: compare histogram integral with counter
            hist_integral = h_data.Integral()
            counter_val = counter_data.GetBinContent(norm_bin) if counter_data else -1
            print(f"    Data histogram integral: {hist_integral:.0f}")
            print(f"    Counter bin {norm_bin}:          {counter_val:.0f}")
            if abs(hist_integral - counter_val) > 1:
                print(f"    NOTE: Small difference ({hist_integral - counter_val:.0f}) "
                      f"likely from events outside binning range")
            
            # Get response matrix
            response_path = f"response/{mode['response_base']}_{SYST}"
            h_response = fmc.Get(response_path)
            if not h_response:
                print(f"    WARNING: Response not found: {response_path}")
                continue
            h_response = h_response.Clone(f"response_{mode_name}_{ds_suffix}")
            h_response.SetDirectory(0)
            
            print(f"    Response matrix: {h_response.GetNbinsX()}x{h_response.GetNbinsY()}, "
                  f"integral={h_response.Integral():.0f}")
            
            # Run bootstrap
            print(f"    Running {args.ntoys} bootstrap toys...")
            result = run_bootstrap_for_mode(
                h_data, h_response,
                n_toys=args.ntoys,
                n_iterations=args.iterations,
                rng=rng
            )
            
            # Validation: compare nominal with toy mean
            rel_diff = np.abs(result['nominal'] - result['mean']) / np.where(
                result['nominal'] != 0, np.abs(result['nominal']), 1.0)
            max_rel_diff = np.max(rel_diff)
            print(f"    Nominal vs toy mean: max relative diff = {max_rel_diff:.4f}")
            if max_rel_diff > 0.05:
                print(f"    WARNING: Large difference between nominal and toy mean!")
            
            # Print diagonal errors for comparison
            toy_errors = np.sqrt(np.diag(result['cov']))
            print(f"    Bootstrap stat errors (first 5 bins): "
                  f"{toy_errors[:5]}")
            
            # Store key name
            key = f"{mode_name}_qqpy_{ds_suffix}"
            
            # --- Write to ROOT file ---
            fout.cd()
            
            # Covariance matrix as TH2D
            nbins = len(result['nominal'])
            edges = result['bin_edges']
            from array import array as arr
            
            h_cov = ROOT.TH2D(
                f"cov_{key}", f"Covariance {mode['label']} {ds_suffix}",
                nbins, arr('d', edges.tolist()),
                nbins, arr('d', edges.tolist())
            )
            for i in range(nbins):
                for j in range(nbins):
                    h_cov.SetBinContent(i + 1, j + 1, result['cov'][i, j])
            h_cov.Write()
            
            # Correlation matrix as TH2D
            h_corr = ROOT.TH2D(
                f"corr_{key}", f"Correlation {mode['label']} {ds_suffix}",
                nbins, arr('d', edges.tolist()),
                nbins, arr('d', edges.tolist())
            )
            for i in range(nbins):
                for j in range(nbins):
                    h_corr.SetBinContent(i + 1, j + 1, result['corr'][i, j])
            h_corr.Write()
            
            # Nominal unfolded as TH1D
            h_nom = ROOT.TH1D(
                f"nominal_{key}", f"Nominal unfolded {mode['label']} {ds_suffix}",
                nbins, arr('d', edges.tolist())
            )
            for i in range(nbins):
                h_nom.SetBinContent(i + 1, result['nominal'][i])
                h_nom.SetBinError(i + 1, toy_errors[i])
            h_nom.Write()
            
            # Toy mean as TH1D (for validation)
            h_mean = ROOT.TH1D(
                f"toy_mean_{key}", f"Toy mean {mode['label']} {ds_suffix}",
                nbins, arr('d', edges.tolist())
            )
            for i in range(nbins):
                h_mean.SetBinContent(i + 1, result['mean'][i])
                h_mean.SetBinError(i + 1, toy_errors[i])
            h_mean.Write()
            
            # Save for npz
            all_results[key] = {
                'nominal': result['nominal'],
                'mean': result['mean'],
                'cov': result['cov'],
                'corr': result['corr'],
                'bin_edges': result['bin_edges'],
                'bin_widths': result['bin_widths'],
                'n_valid': result['n_valid'],
            }
            
            print(f"    ✓ Written: cov_{key}, corr_{key}, nominal_{key}")
        
        fdata.Close()
        fmc.Close()
    
    fout.Close()
    
    # --- Save npz for easy Python access ---
    npz_file = filenameout.replace('.root', '.npz')
    npz_dict = {}
    for key, res in all_results.items():
        npz_dict[f"{key}_nominal"] = res['nominal']
        npz_dict[f"{key}_mean"] = res['mean']
        npz_dict[f"{key}_cov"] = res['cov']
        npz_dict[f"{key}_corr"] = res['corr']
        npz_dict[f"{key}_bin_edges"] = res['bin_edges']
    np.savez(npz_file, **npz_dict)
    
    total_elapsed = time.time() - total_start
    
    print(f"\n{'='*70}")
    print(f"Bootstrap completed!")
    print(f"  Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print(f"  ROOT output: {filenameout}")
    print(f"  NPZ output:  {npz_file}")
    print(f"  Results: {len(all_results)} mode×dataset combinations")
    print(f"{'='*70}")
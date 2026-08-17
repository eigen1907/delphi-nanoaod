"""
Unfolding script for thrust analysis with systematic variations
----------------------------------------------------------------
Reads output from analysis_thrust.py which has structure:
  - /reco/ThrustMissPNC2_{systematic}, etc.
  - /gen/Thrust_before2, etc. (no systematics at gen level)
  - /response/response_thrust2_{systematic}, etc.

Unfolds data using response matrices for each systematic variation.
"""

import ROOT
import numpy as np
import sys
import os
import argparse
from array import array

from ROOT import RooUnfoldResponse
from ROOT import RooUnfold
from ROOT import RooUnfoldBayes
from common_functions import *
from binning_and_selections import *

def convert_to_roounfold_response(reco_hist, gen_hist, response_hist, name="response"):
    # Project response matrix to get true gen distribution (y-axis)
    gen_from_response = response_hist.ProjectionY(f"{name}_gen_projection")
    
    # Project response matrix to get reco distribution (x-axis)  
    reco_from_response = response_hist.ProjectionX(f"{name}_reco_projection")
    
    response = ROOT.RooUnfoldResponse(
        reco_from_response,  # Use projection from response
        gen_from_response,   # Use projection from response
        response_hist,
        name,
        name,
        False
    )
    return response

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='RooUnfold unfolding script with systematics')
    parser.add_argument('--iterations', type=int, default=6,
                        help='Number of iterations for Bayesian unfolding')
    parser.add_argument('--version', type=str, default='v70',
                        help='Version tag for output + the nominal (kk2f) response + data (e.g., v70)')
    parser.add_argument('--syst-version', type=str, default=None,
                        help='Version for the systematics generators (pythia8_default/'
                             'pythia8_vincia/sherpa). Defaults to --version. kk2f (nominal) '
                             'and data always use --version. Use e.g. --version v71 '
                             '--syst-version v70 to swap only kk2f+data to v71.')
    parser.add_argument('--systematic', type=str, default='all',
                        help='Which systematic to unfold: all, nominal, or specific systematic name')

    args = parser.parse_args()

    VERSION = args.version
    SYST_VERSION = args.syst_version or VERSION

    # List of systematics to process
    # These should match the SYSTEMATICS dictionary in analysis_thrust.py
    ALL_SYSTEMATICS = [
        'nominal',
#        'charged_scale_up',
#        'charged_scale_down',
#        'neutral_scale_up',
#        'neutral_scale_down',
#        'fake_drop',
#        'charged_eff',
#        'neutral_eff',
#        'evt_weight',
        'neutral_e_up',
#        'neutral_e_down',
#        'charged_pt_up',
#        'charged_pt_down',
    ]
    
    # Determine which systematics to process
    if args.systematic == 'all':
        systematics_to_unfold = ALL_SYSTEMATICS
    elif args.systematic in ALL_SYSTEMATICS:
        systematics_to_unfold = [args.systematic]
    else:
        print(f"ERROR: Unknown systematic '{args.systematic}'")
        print(f"Available: {', '.join(ALL_SYSTEMATICS)}")
        sys.exit(1)

    # Define thrust modes to unfold
    # Maps internal names to histogram names in the ROOT files
    thrust_modes = [
        {
            'name': 'thrust2',
            'response_base': 'response_thrust2',         # response/response_thrust2_{syst}
            'reco_base': 'ThrustMissPNC2',              # reco/ThrustMissPNC2_{syst}
            'gen_name': 'Thrust_before2',               # gen/Thrust_before2
            'data_name': 'ThrustMissPNC2_nominal'       # reco/ThrustMissPNC2_nominal (from data file)
        },
        {
            'name': 'thrust_log2',
            'response_base': 'response_thrust_log2',
            'reco_base': 'ThrustMissPNCLog2',
            'gen_name': 'Thrust_before_log2',
            'data_name': 'ThrustMissPNCLog2_nominal'
        },
        {
            'name': 'thrust2_Escheme',
            'response_base': 'response_thrust2_Escheme',
            'reco_base': 'ThrustMissPNC2_Escheme',
            'gen_name': 'Thrust_before2_Escheme',
            'data_name': 'ThrustMissPNC2_Escheme_nominal'
        },
        {
            'name': 'thrust_log2_Escheme',
            'response_base': 'response_thrust_log2_Escheme',
            'reco_base': 'ThrustMissPNCLog2_Escheme',
            'gen_name': 'Thrust_before_log2_Escheme',
            'data_name': 'ThrustMissPNCLog2_Escheme_nominal'
        },
        {
            'name': 'thrustC2',
            'response_base': 'response_thrustC2',
            'reco_base': 'ThrustC2',
            'gen_name': 'ThrustC_before2',
            'data_name': 'ThrustC2_nominal'
        },
        {
            'name': 'thrustC_log2',
            'response_base': 'response_thrustC_log2',
            'reco_base': 'ThrustCLog2',
            'gen_name': 'ThrustC_before_log2',
            'data_name': 'ThrustCLog2_nominal'
        },
        {
            'name': 'thrustC2_Escheme',
            'response_base': 'response_thrustC2_Escheme',
            'reco_base': 'ThrustC2_Escheme',
            'gen_name': 'ThrustC_before2_Escheme',
            'data_name': 'ThrustC2_Escheme_nominal'
        },
        {
            'name': 'thrustC_log2_Escheme',
            'response_base': 'response_thrustC_log2_Escheme',
            'reco_base': 'ThrustCLog2_Escheme',
            'gen_name': 'ThrustC_before_log2_Escheme',
            'data_name': 'ThrustCLog2_Escheme_nominal'
        }
    ]

    # Define MC generator configurations.
    # Unfold the SAME data with each generator's response matrix, to compare
    # the four models (kk2f, pythia8_default, pythia8_vincia, sherpa) and assess
    # the generator/model dependence of the unfolding. Files are the hadded v70
    # thrust_nanoaod outputs.
    GENERATORS = [
        ('kk2f',            'thrust_kk2f4146_qqpy_91.25_{ds}_{ver}.root'),  # KK2f4146 qqpy (nominal)
        ('pythia8_default', 'thrust_pythia8_default_isron_{ds}_{ver}.root'),
        ('pythia8_vincia',  'thrust_pythia8_vincia_isron_{ds}_{ver}.root'),
        ('sherpa',          'thrust_sherpa_isron_{ds}_{ver}.root'),
    ]
    # kk2f (the nominal response) + data use --version; the alternative-model
    # systematics generators use --syst-version (defaults to --version).
    mc_configs = [
        {
            'name': f'{gen}_{ds}',
            'mc_file': tmpl.format(ds=ds, ver=(VERSION if gen == 'kk2f' else SYST_VERSION)),
            'data_file': f'thrust_data_{ds}_{VERSION}.root',
            'dataset': ds,
        }
        for ds in ('94c', '95d')
        for gen, tmpl in GENERATORS
    ]

    # Output file
    filenameout = f"unfolded_thrust_systematics_{VERSION}_iter{args.iterations}.root"
    fout = ROOT.TFile(filenameout, 'recreate')
    
    print(f"Configuration:")
    print(f"  Version (output + kk2f + data): {VERSION}")
    print(f"  Syst-generator version (pythia8/vincia/sherpa): {SYST_VERSION}")
    print(f"  Iterations: {args.iterations}")
    print(f"  Systematics: {', '.join(systematics_to_unfold)}")
    print(f"  Output file: {filenameout}")
    print(f"  Processing {len(thrust_modes)} thrust modes x {len(mc_configs)} MC generators x {len(systematics_to_unfold)} systematics")
    print(f"  Total combinations: {len(thrust_modes) * len(mc_configs) * len(systematics_to_unfold)}\n")

    # Track counters we've already written
    data_counters_written = set()  # dataset only

    # Dictionary to cache opened files
    opened_mc_files = {}
    opened_data_files = {}

    # Loop over all MC configurations FIRST (outer loop)
    for config in mc_configs:
        mc_name = config['name']
        mc_file = config['mc_file']
        data_file = config['data_file']
        dataset = config['dataset']
        
        print(f"\n{'='*60}")
        print(f"MC Generator: {mc_name}")
        print(f"  MC file: {mc_file}")
        print(f"  Data file: {data_file}")
        print(f"  Dataset: {dataset}")
        print(f"{'='*60}")
        
        # Check if files exist
        if not os.path.exists(mc_file):
            print(f"WARNING: MC file {mc_file} not found. Skipping...")
            continue
        
        if not os.path.exists(data_file):
            print(f"WARNING: Data file {data_file} not found. Skipping...")
            continue
        
        # Open files ONCE per MC config (cache them)
        if mc_file not in opened_mc_files:
            opened_mc_files[mc_file] = ROOT.TFile.Open(mc_file, 'READ')
        fmc = opened_mc_files[mc_file]
        
        if data_file not in opened_data_files:
            opened_data_files[data_file] = ROOT.TFile.Open(data_file, 'READ')
        fdata = opened_data_files[data_file]
        
        # Copy data counter ONCE per dataset
        if dataset not in data_counters_written:
            print(f"DEBUG: Writing counter for {dataset} for the FIRST time")
            counter_data = fdata.Get("counter")
            if counter_data:
                print(f"DEBUG: Counter integral from file: {counter_data.Integral()}")
                counter_clone = counter_data.Clone(f"counter_data_{dataset}")
                fout.cd()
                counter_clone.Write()
                data_counters_written.add(dataset)
        else:
            print(f"DEBUG: SKIPPING counter for {dataset} (already written)")
        
        # Loop over all thrust modes (inner loop)
        for mode in thrust_modes:
            mode_name = mode['name']
            response_base = mode['response_base']
            reco_base = mode['reco_base']
            gen_name = mode['gen_name']
            data_name = mode['data_name']
            
            print(f"\n{'#'*70}")
            print(f"# THRUST MODE: {mode_name}")
            print(f"{'#'*70}")
            
            # Get gen-level histogram (no systematics at gen level)
            gen_hist_path = f"gen/{gen_name}"
            gen_hist = fmc.Get(gen_hist_path)
            
            if not gen_hist:
                print(f"WARNING: Gen histogram {gen_hist_path} not found in {mc_file}. Skipping...")
                continue
            
            # Loop over systematics
            for syst in systematics_to_unfold:
                combo_name = f"{mode_name}_{mc_name}_{syst}"
                
                print(f"\n  Processing systematic: {syst}")

                # Load data histogram for THIS systematic
                data_hist_name_syst = data_name.replace("nominal", syst)  #
                data_hist_path = f"reco/{data_hist_name_syst}"            #
                data_hist = fdata.Get(data_hist_path)                     #
                
                if not data_hist:                                          #
                    print(f"    WARNING: Data histogram {data_hist_path} not found. Skipping...")  #
                    continue                                               #
                
                data = data_hist.Clone(f"data_{combo_name}")              #
                print(f"    Data histogram: {data_hist_path} (integral: {data.Integral():.1f})")  #
                
                # Build histogram names with systematic suffix
                response_name = f"response/{response_base}_{syst}"
                reco_name = f"reco/{reco_base}_{syst}"
                
                # Get histograms
                response_hist = fmc.Get(response_name)
                reco_hist = fmc.Get(reco_name)
                
                if not response_hist:
                    print(f"    WARNING: {response_name} not found. Skipping...")
                    continue
                
                if not reco_hist:
                    print(f"    WARNING: {reco_name} not found. Skipping...")
                    continue
                
                # Clone histograms
                response_2d = response_hist.Clone(f"response_{combo_name}")
                reco = reco_hist.Clone(f"reco_{combo_name}")
                gen = gen_hist.Clone(f"gen_{combo_name}")
                
                print(f"    Response integral: {response_2d.Integral():.1f}")
                print(f"    Reco integral: {reco.Integral():.1f}")
                print(f"    Gen integral: {gen.Integral():.1f}")
                print(f"    Data integral: {data.Integral():.1f}")
                
                # Create RooUnfoldResponse object
                response = convert_to_roounfold_response(reco, gen, response_2d, name=f"response_{combo_name}")
                
                # Check response matrix condition
                RESPONSE = response.Mresponse()
                singular = ROOT.TDecompSVD(RESPONSE)
                sig_values = singular.GetSig()
                n_sig = sig_values.GetNrows()
                if n_sig > 0 and sig_values[n_sig-1] > 0:
                    cond_num = sig_values[0] / sig_values[n_sig-1]
                    print(f"    Response matrix condition number: {cond_num:.2e}")
                else:
                    print(f"    WARNING: Response matrix is singular or has zero singular values!")


                print(f"    Data integral BEFORE unfolding: {data.Integral():.1f}")
                print(f"    Reco (MC) integral: {reco.Integral():.1f}")
                print(f"    Gen (MC) integral: {gen.Integral():.1f}")
                print(f"    Response integral: {response_2d.Integral():.1f}")
                
                data_clone = data.Clone(f"data_clone_{combo_name}")
                # Perform unfolding
                print(f"    Performing Bayesian unfolding with {args.iterations} iterations...")
                unfold = ROOT.RooUnfoldBayes(response, data_clone, args.iterations)
                
                hUnf = unfold.Hunfold().Clone(f"unfolded_{combo_name}")
                
                print(f"    Unfolded integral: {hUnf.Integral():.1f}")
                
                # Get error histogram (diagonal errors from covariance matrix)
                hErr = unfold.Eunfold(ROOT.RooUnfold.kErrors).Clone(f"errors_{combo_name}")
                
                # Write to output file with unique names
                fout.cd()
                response_2d.Write(f"response_{combo_name}")
                reco.Write(f"reco_{combo_name}")
                data.Write(f"data_{combo_name}") 
                gen.Write(f"gen_{combo_name}")
                hUnf.Write(f"unfolded_{combo_name}")
                hErr.Write(f"errors_{combo_name}")
                
                print(f"    ✓ Completed unfolding for {combo_name}")

    # Close all opened files
    print("\nClosing input files...")
    for f in opened_mc_files.values():
        f.Close()
    for f in opened_data_files.values():
        f.Close()

    # Close output file
    fout.Close()

    print(f"\n{'#'*70}")
    print(f"All unfolding completed successfully!")
    print(f"Output saved to: {filenameout}")
    print(f"{'#'*70}")
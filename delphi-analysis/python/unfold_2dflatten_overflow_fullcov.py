import ROOT
import numpy as np
import sys
import os
import argparse
from array import array

# Explicitly load the locally-built RooUnfold before importing its classes.
# ROOT's autoload (via rootmap) is fragile on lxplus, where nodes carry
# inconsistent ROOT versions and stray RooUnfold rootmaps; a path-based load of
# our own build (matched to the ROOT we source) is robust across nodes.
_ru = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "RooUnfold", "build", "libRooUnfold")
if ROOT.gSystem.Load(_ru) < 0:
    raise RuntimeError("failed to load %s.so -- rebuild RooUnfold against the "
                       "running ROOT (root-config --version)" % _ru)

from ROOT import RooUnfoldResponse
from ROOT import RooUnfold
from ROOT import RooUnfoldBayes

from common_functions import *
from binning_and_selections import *

def Proj2D_X(h,ymin,ymax,hname="XXX",Debug=False):
    # project 2D histogram into 1D along X
    imin=h.GetYaxis().FindBin(ymin)
    imax=h.GetYaxis().FindBin(ymax)-1

    proj_x=h.ProjectionX(hname, imin, imax)
    ROOT.SetOwnership(proj_x,True)

    return proj_x

def flatten_2d_histogram(hist_2d, name_suffix="_flattened", include_overflow=False):
    """
    Convert 2D histogram to 1D by flattening.
    
    Parameters:
    -----------
    hist_2d : ROOT.TH2
        Input 2D histogram
    name_suffix : str
        Suffix for the output histogram name
    include_overflow : bool
        If True, include overflow/underflow bins
    """
    
    if include_overflow:
        # Include overflow/underflow bins
        nx = hist_2d.GetNbinsX() + 2  # +2 for under/overflow
        ny = hist_2d.GetNbinsY() + 2  # +2 for under/overflow
        i_start, i_end = 0, nx
        j_start, j_end = 0, ny
        print(f"Including overflow: {nx}x{ny} bins (with under/overflow)")
    else:
        # Only nominal bins
        nx = hist_2d.GetNbinsX()
        ny = hist_2d.GetNbinsY()
        i_start, i_end = 1, nx + 1
        j_start, j_end = 1, ny + 1
        print(f"Nominal bins only: {nx}x{ny} bins")
    
    total_bins = nx * ny
    
    print(f"Flattening 2D histogram -> 1D with {total_bins} bins")
    
    # Create 1D histogram
    flat_name = hist_2d.GetName() + name_suffix
    hist_1d = ROOT.TH1D(flat_name, hist_2d.GetTitle(), total_bins, 0.5, total_bins + 0.5)
    
    if include_overflow:
        # Extract bin edges for overflow bins
        # Get nominal bin edges first
        x_nominal_edges = []
        y_nominal_edges = []
        
        for i in range(hist_2d.GetNbinsX() + 1):
            x_nominal_edges.append(hist_2d.GetXaxis().GetBinLowEdge(i + 1))
        
        for j in range(hist_2d.GetNbinsY() + 1):
            y_nominal_edges.append(hist_2d.GetYaxis().GetBinLowEdge(j + 1))
        
        # Create extended edges: [underflow_edge] + [nominal_edges] + [overflow_edge]
        x_first_width = x_nominal_edges[1] - x_nominal_edges[0]
        x_last_width = x_nominal_edges[-1] - x_nominal_edges[-2]
        y_first_width = y_nominal_edges[1] - y_nominal_edges[0]
        y_last_width = y_nominal_edges[-1] - y_nominal_edges[-2]
        
        # Build extended edge arrays
        x_edges = [x_nominal_edges[0] - x_first_width] + x_nominal_edges + [x_nominal_edges[-1] + x_last_width]
        y_edges = [y_nominal_edges[0] - y_first_width] + y_nominal_edges + [y_nominal_edges[-1] + y_last_width]
        
        print(f"Extended X edges: {len(x_edges)} total ({len(x_nominal_edges)} nominal + 2 overflow)")
        print(f"Extended Y edges: {len(y_edges)} total ({len(y_nominal_edges)} nominal + 2 overflow)")
        
        # Enhanced mapping information
        mapping = {
            'nx': nx,
            'ny': ny,
            'include_overflow': True,
            'x_edges': x_edges,
            'y_edges': y_edges,
            'x_min': hist_2d.GetXaxis().GetXmin(),
            'x_max': hist_2d.GetXaxis().GetXmax(),
            'y_min': hist_2d.GetYaxis().GetXmin(),
            'y_max': hist_2d.GetYaxis().GetXmax()
        }
    else:
        # For nominal bins only
        x_edges = [hist_2d.GetXaxis().GetBinLowEdge(i) for i in range(1, nx + 2)]
        y_edges = [hist_2d.GetYaxis().GetBinLowEdge(j) for j in range(1, ny + 2)]
        
        mapping = {
            'nx': nx,
            'ny': ny,
            'include_overflow': False,
            'x_edges': x_edges,
            'y_edges': y_edges,
            'x_min': hist_2d.GetXaxis().GetXmin(),
            'x_max': hist_2d.GetXaxis().GetXmax(),
            'y_min': hist_2d.GetYaxis().GetXmin(),
            'y_max': hist_2d.GetYaxis().GetXmax()
        }
    
    # Fill 1D histogram
    for i in range(i_start, i_end):
        for j in range(j_start, j_end):
            content = hist_2d.GetBinContent(i, j)
            error = hist_2d.GetBinError(i, j)
            
            # Calculate flat bin index
            if include_overflow:
                flat_bin = i * ny + j + 1  # i and j start from 0
            else:
                flat_bin = (i - 1) * ny + (j - 1) + 1  # i and j start from 1
            
            hist_1d.SetBinContent(flat_bin, content)
            hist_1d.SetBinError(flat_bin, error)
    
    print(f"Original integral: {hist_2d.Integral():.6f}")
    print(f"Flattened integral: {hist_1d.Integral():.6f}")
    
    if include_overflow:
        # Check overflow/underflow content
        overflow_content = 0.0
        for i in [0, nx-1]:  # First and last i (underflow/overflow)
            for j in range(ny):
                overflow_content += hist_1d.GetBinContent(i * ny + j + 1)
        for j in [0, ny-1]:  # First and last j (underflow/overflow) 
            for i in range(1, nx-1):  # Avoid double counting corners
                overflow_content += hist_1d.GetBinContent(i * ny + j + 1)
        
        print(f"Overflow/underflow content: {overflow_content:.6f}")
    
    return hist_1d, mapping

def flatten_2d_histogram_with_overflow(hist_2d, name_suffix="_flattened"):
    """
    Convert 2D histogram to 1D by flattening, including overflow/underflow.
    Convenience wrapper for flatten_2d_histogram with include_overflow=True.
    """
    return flatten_2d_histogram(hist_2d, name_suffix, include_overflow=True)

def unflatten_to_2d(hist_1d, mapping, name_suffix="_unflattened"):
    """
    Convert flattened 1D histogram back to 2D.
    
    Parameters:
    -----------
    hist_1d : ROOT.TH1D
        Flattened 1D histogram
    mapping : dict
        Mapping dictionary from flatten_2d_histogram
    name_suffix : str
        Suffix for output histogram name
    """
    
    nx = mapping['nx']
    ny = mapping['ny']
    include_overflow = mapping['include_overflow']
    
    nbins_1d = hist_1d.GetNbinsX()
    expected_bins = nx * ny
    
    print(f"Unflattening 1D -> 2D: {nx}x{ny} bins (overflow={include_overflow})")
    print(f"1D histogram has {nbins_1d} bins, expected {expected_bins}")
    
    # Validate dimensions
    if nbins_1d != expected_bins:
        print(f"⚠️  WARNING: 1D histogram size mismatch!")
        print(f"   1D bins: {nbins_1d}")
        print(f"   Expected (nx*ny): {expected_bins}")
        print(f"   Proceeding with caution...")
    
    # Create 2D histogram
    unflatten_name = hist_1d.GetName() + name_suffix
    
    x_edges_array = array('d', mapping['x_edges'])
    y_edges_array = array('d', mapping['y_edges'])
    
    hist_2d = ROOT.TH2D(
        unflatten_name, hist_1d.GetTitle(),
        nx, x_edges_array,
        ny, y_edges_array
    )
    
    # Fill 2D histogram - only up to actual number of bins available
    max_bins = min(nbins_1d, expected_bins)
    
    filled_count = 0
    skipped_count = 0
    
    for flat_bin in range(1, max_bins + 1):
        content = hist_1d.GetBinContent(flat_bin)
        error = hist_1d.GetBinError(flat_bin)
        
        # Reverse the flattening
        linear_index = flat_bin - 1  # Convert to 0-based
        
        if include_overflow:
            # For overflow: indices are already 0-based in the flat array
            i = linear_index // ny  # 0-based
            j = linear_index % ny   # 0-based
            
            # Bounds check
            if i >= 0 and i < nx and j >= 0 and j < ny:
                hist_2d.SetBinContent(i, j, content)
                hist_2d.SetBinError(i, j, error)
                if content != 0:
                    filled_count += 1
            else:
                skipped_count += 1
        else:
            # For non-overflow: need to convert to 1-based ROOT indexing
            # The flattening was: flat_bin = (i - 1) * ny + (j - 1) + 1
            # So: linear_index = (i - 1) * ny + (j - 1)
            # Therefore: i - 1 = linear_index // ny, so i = linear_index // ny + 1
            #            j - 1 = linear_index % ny, so j = linear_index % ny + 1
            i = linear_index // ny + 1  # 1-based
            j = linear_index % ny + 1   # 1-based
            
            # Bounds check (1-based)
            if i >= 1 and i <= nx and j >= 1 and j <= ny:
                hist_2d.SetBinContent(i, j, content)
                hist_2d.SetBinError(i, j, error)
                if content != 0:
                    filled_count += 1
            else:
                skipped_count += 1
                if skipped_count <= 5:  # Only print first few
                    print(f"⚠️  Skipping out-of-bounds bin: flat_bin={flat_bin}, i={i}, j={j} (expected i:1-{nx}, j:1-{ny})")
    
    print(f"Filled {filled_count} non-zero bins, skipped {skipped_count} bins")
    
    if nbins_1d != expected_bins:
        print(f"⚠️  Filled {max_bins} bins out of {expected_bins} expected")
    
    print(f"Unflattened integral: {hist_2d.Integral():.6f}")
    
    return hist_2d

def unflatten_to_2d_nominal(hist_1d, mapping, name_suffix="_unflattened"):
    """
    Convert flattened 1D histogram back to 2D, keeping ONLY nominal range bins.
    Overflow/underflow bins are captured in the response matrix but ignored in final result.
    """
    
    nx = mapping['nx']
    ny = mapping['ny']
    
    # Original histogram dimensions (without overflow)
    nx_nominal = nx - 2
    ny_nominal = ny - 2
    print(f"Unflattening with overflow -> nominal 2D {nx_nominal}x{ny_nominal} (ignoring overflow bins)")
    
    # Create 2D histogram with ONLY nominal range
    unflatten_name = hist_1d.GetName() + name_suffix
    
    x_edges_array = array('d', mapping['x_edges'])
    y_edges_array = array('d', mapping['y_edges'])
    
    if len(x_edges_array) != nx + 1 or len(y_edges_array) != ny + 1:
        print(f"WARNING: Edge array size mismatch. X: {len(x_edges_array)} vs {nx+1}, Y: {len(y_edges_array)} vs {ny+1}")
    
    # Extract nominal edges: skip first (underflow) and last (overflow)
    x_nominal_edges = array('d', x_edges_array[1:nx_nominal+2])  # [1:nx-1] = nominal edges
    y_nominal_edges = array('d', y_edges_array[1:ny_nominal+2])  # [1:ny-1] = nominal edges
    
    print(f"Extracted {len(x_nominal_edges)-1} nominal X bins from {len(x_edges_array)-1} total")
    print(f"Extracted {len(y_nominal_edges)-1} nominal Y bins from {len(y_edges_array)-1} total")
    print(f"X range: [{x_nominal_edges[0]:.6f}, {x_nominal_edges[-1]:.6f}]")
    print(f"Y range: [{y_nominal_edges[0]:.6f}, {y_nominal_edges[-1]:.6f}]")
    
    hist_2d = ROOT.TH2D(
        unflatten_name, hist_1d.GetTitle(),
        nx_nominal, x_nominal_edges,
        ny_nominal, y_nominal_edges
    )
    
    # Fill 2D histogram - ONLY nominal bins
    nominal_content = 0.0
    overflow_content = 0.0
    
    for flat_bin in range(1, hist_1d.GetNbinsX() + 1):
        content = hist_1d.GetBinContent(flat_bin)
        error = hist_1d.GetBinError(flat_bin)
        
        # Reverse the flattening
        linear_index = flat_bin - 1
        
        i = linear_index // ny  # 0-based, includes overflow
        j = linear_index % ny   # 0-based, includes overflow
        
        # Only fill nominal bins (exclude overflow: i=0, i=nx-1, j=0, j=ny-1)
        if i > 0 and i < nx-1 and j > 0 and j < ny-1:
            # Convert to 1-based nominal histogram coordinates
            i_nominal = i         # i=1..nx-2 maps to 1..nx_nominal  
            j_nominal = j         # j=1..ny-2 maps to 1..ny_nominal
            hist_2d.SetBinContent(i_nominal, j_nominal, content)
            hist_2d.SetBinError(i_nominal, j_nominal, error)
            nominal_content += content
        else:
            # This is overflow/underflow content - just track it
            overflow_content += content
    
    print(f"Nominal range integral: {nominal_content:.6f}")
    if overflow_content > 0:
        print(f"Overflow/underflow content ignored: {overflow_content:.6f}")
        print(f"Fraction kept in nominal range: {nominal_content/(nominal_content + overflow_content):.3f}")
    
    print(f"Final 2D histogram integral: {hist_2d.Integral():.6f}")
    
    return hist_2d

def unflatten_errors_to_2d_nominal(error_1d, mapping, name_suffix="_error_2d"):
    """
    Convert flattened 1D error histogram to 2D nominal range.
    """
    
    nx_full = mapping['nx']  # Includes overflow
    ny_full = mapping['ny']  # Includes overflow
    nx_nominal = nx_full - 2     # Nominal bins only
    ny_nominal = ny_full - 2     # Nominal bins only
    
    print(f"Unflattening errors: {error_1d.GetNbinsX()} -> {nx_nominal}x{ny_nominal}")
    
    # Create 2D error histogram for nominal range
    error_name = error_1d.GetName() + name_suffix
    x_edges_array = array('d', mapping['x_edges'][1:-1])  # Skip overflow edges
    y_edges_array = array('d', mapping['y_edges'][1:-1])  # Skip overflow edges
    
    error_2d = ROOT.TH2D(
        error_name, "Per-bin Errors (2D)",
        nx_nominal, x_edges_array,
        ny_nominal, y_edges_array
    )
    
    # Map from 1D nominal error bins to 2D nominal bins
    for i in range(1, nx_nominal + 1):
        for j in range(1, ny_nominal + 1):
            # Calculate flat bin in nominal space
            flat_bin = (i - 1) * ny_nominal + j
            
            if flat_bin <= error_1d.GetNbinsX():
                error_value = error_1d.GetBinContent(flat_bin)
                error_2d.SetBinContent(i, j, error_value)
    
    print(f"Error 2D integral after unflattening: {error_2d.Integral():.6f}")
    
    return error_2d

def create_2d_response_matrix_from_thnf(thnf_4d, name_suffix="_response_2d", include_overflow=False):
    """
    Convert 4D THnF to 2D TH2 response matrix.
    
    Parameters:
    -----------
    thnf_4d : ROOT.THnSparseF
        4D response histogram
    name_suffix : str
        Suffix for output name
    include_overflow : bool
        If True, include overflow/underflow bins
    """
    
    print(f"Converting 4D THnF to 2D response matrix (overflow={include_overflow})...")
    
    if thnf_4d.GetNdimensions() != 4:
        raise ValueError(f"Expected 4D THnF, got {thnf_4d.GetNdimensions()}D")
    
    # Get dimensions
    axis_reco_x = thnf_4d.GetAxis(0)
    axis_reco_y = thnf_4d.GetAxis(1)
    axis_gen_x = thnf_4d.GetAxis(2)
    axis_gen_y = thnf_4d.GetAxis(3)
    
    if include_overflow:
        # Include overflow/underflow bins
        nx_reco = axis_reco_x.GetNbins() + 2
        ny_reco = axis_reco_y.GetNbins() + 2
        nx_gen = axis_gen_x.GetNbins() + 2
        ny_gen = axis_gen_y.GetNbins() + 2
        
        reco_start, reco_end = 0, nx_reco
        gen_start, gen_end = 0, nx_gen
        reco_j_start, reco_j_end = 0, ny_reco
        gen_j_start, gen_j_end = 0, ny_gen
    else:
        # Nominal bins only
        nx_reco = axis_reco_x.GetNbins()
        ny_reco = axis_reco_y.GetNbins()
        nx_gen = axis_gen_x.GetNbins()
        ny_gen = axis_gen_y.GetNbins()
        
        reco_start, reco_end = 1, nx_reco + 1
        gen_start, gen_end = 1, nx_gen + 1
        reco_j_start, reco_j_end = 1, ny_reco + 1
        gen_j_start, gen_j_end = 1, ny_gen + 1
    
    total_reco_bins = nx_reco * ny_reco
    total_gen_bins = nx_gen * ny_gen
    
    print(f"THnF dimensions: reco({nx_reco}x{ny_reco}) -> gen({nx_gen}x{ny_gen})")
    print(f"Flattened response matrix: {total_reco_bins} x {total_gen_bins}")
    
    # Create 2D response matrix
    response_name = thnf_4d.GetName() + name_suffix
    response_2d = ROOT.TH2D(
        response_name, "2D Response Matrix",
        total_reco_bins, 0.5, total_reco_bins + 0.5,  # Reco (X-axis)
        total_gen_bins, 0.5, total_gen_bins + 0.5      # Gen (Y-axis)
    )
    
    print("Filling 2D response matrix...")
    
    filled_bins = 0
    coords = array('i', [0, 0, 0, 0])
    total_content = 0.0
    overflow_content = 0.0
    
    # Iterate through all bins
    for i_reco in range(reco_start, reco_end):
        for j_reco in range(reco_j_start, reco_j_end):
            # Calculate flat reco bin
            if include_overflow:
                flat_reco_bin = i_reco * ny_reco + j_reco + 1
            else:
                flat_reco_bin = (i_reco - 1) * ny_reco + (j_reco - 1) + 1
            
            for i_gen in range(gen_start, gen_end):
                for j_gen in range(gen_j_start, gen_j_end):
                    # Calculate flat gen bin
                    if include_overflow:
                        flat_gen_bin = i_gen * ny_gen + j_gen + 1
                    else:
                        flat_gen_bin = (i_gen - 1) * ny_gen + (j_gen - 1) + 1
                    
                    if include_overflow:
                        # Map 0-based indices to THnF overflow convention
                        if i_reco == 0:
                            coords[0] = 0  # Underflow
                        elif i_reco == nx_reco - 1:
                            coords[0] = axis_reco_x.GetNbins() + 1  # Overflow
                        else:
                            coords[0] = i_reco  # Normal bin
                        
                        if j_reco == 0:
                            coords[1] = 0
                        elif j_reco == ny_reco - 1:
                            coords[1] = axis_reco_y.GetNbins() + 1
                        else:
                            coords[1] = j_reco
                            
                        if i_gen == 0:
                            coords[2] = 0
                        elif i_gen == nx_gen - 1:
                            coords[2] = axis_gen_x.GetNbins() + 1
                        else:
                            coords[2] = i_gen
                            
                        if j_gen == 0:
                            coords[3] = 0
                        elif j_gen == ny_gen - 1:
                            coords[3] = axis_gen_y.GetNbins() + 1
                        else:
                            coords[3] = j_gen
                    else:
                        # Normal 1-based bin indices
                        coords[0] = i_reco
                        coords[1] = j_reco
                        coords[2] = i_gen
                        coords[3] = j_gen
                    
                    try:
                        content = thnf_4d.GetBinContent(coords)
                        if content > 0:
                            response_2d.SetBinContent(flat_reco_bin, flat_gen_bin, content)
                            filled_bins += 1
                            total_content += content
                            
                            if include_overflow:
                                # Track overflow content
                                is_overflow = (i_reco == 0 or i_reco == nx_reco-1 or 
                                             j_reco == 0 or j_reco == ny_reco-1 or
                                             i_gen == 0 or i_gen == nx_gen-1 or
                                             j_gen == 0 or j_gen == ny_gen-1)
                                if is_overflow:
                                    overflow_content += content
                        else:
                            response_2d.SetBinContent(flat_reco_bin, flat_gen_bin, 0)
                    except:
                        continue
        
        if i_reco % 20 == 0:
            print(f"  Processed reco slice {i_reco}/{reco_end - 1}")
    
    print(f"Filled {filled_bins} response matrix elements")
    print(f"Total response content: {total_content:.6f}")
    if include_overflow:
        print(f"Overflow/underflow content: {overflow_content:.6f} ({100*overflow_content/total_content:.1f}%)")
    
    # Create mapping information
    mapping = {
        'reco_nx': nx_reco, 'reco_ny': ny_reco,
        'gen_nx': nx_gen, 'gen_ny': ny_gen,
        'include_overflow': include_overflow,
        'reco_x_min': axis_reco_x.GetBinLowEdge(1),
        'reco_x_max': axis_reco_x.GetBinUpEdge(axis_reco_x.GetNbins()),
        'reco_y_min': axis_reco_y.GetBinLowEdge(1),
        'reco_y_max': axis_reco_y.GetBinUpEdge(axis_reco_y.GetNbins()),
        'gen_x_min': axis_gen_x.GetBinLowEdge(1),
        'gen_x_max': axis_gen_x.GetBinUpEdge(axis_gen_x.GetNbins()),
        'gen_y_min': axis_gen_y.GetBinLowEdge(1),
        'gen_y_max': axis_gen_y.GetBinUpEdge(axis_gen_y.GetNbins())
    }
    
    return response_2d, mapping

def create_2d_response_matrix_from_thnf_with_overflow(thnf_4d, name_suffix="_response_2d"):
    """
    Convert 4D THnF to 2D TH2 response matrix, including overflow/underflow.
    Convenience wrapper for create_2d_response_matrix_from_thnf with include_overflow=True.
    """
    return create_2d_response_matrix_from_thnf(thnf_4d, name_suffix, include_overflow=True)

def extract_covariance_and_errors(unfolder, gen_mapping):
    """
    Extract covariance matrix and errors, handling overflow and extracting nominal range.
    Returns nominal-range covariance and errors.
    
    This function assumes the unfolder was built with overflow bins included.
    """
    
    try:
        cov_obj = unfolder.Eunfold()
        
        if cov_obj is None:
            print("No covariance matrix returned")
            return None, None, 'none'
        
        # Convert covariance object to 2D histogram
        cov_hist_full = None
        
        # If it's already a 2D histogram, use as-is
        if hasattr(cov_obj, 'GetNbinsX') and hasattr(cov_obj, 'GetNbinsY'):
            print(f"Covariance is 2D histogram: {cov_obj.GetNbinsX()}x{cov_obj.GetNbinsY()}")
            cov_hist_full = cov_obj
        
        # If it's a matrix, convert to 2D histogram
        elif hasattr(cov_obj, 'GetNrows'):
            print(f"Converting matrix to 2D histogram: {cov_obj.GetNrows()}x{cov_obj.GetNcols()}")
            
            nrows = cov_obj.GetNrows()
            ncols = cov_obj.GetNcols()
            
            cov_hist_full = ROOT.TH2D(
                "covariance_matrix_full", "Full Covariance Matrix",
                nrows, 0.5, nrows + 0.5,
                ncols, 0.5, ncols + 0.5
            )
            
            # Copy matrix values to histogram
            for i in range(nrows):
                for j in range(ncols):
                    cov_value = cov_obj(i, j)
                    cov_hist_full.SetBinContent(i + 1, j + 1, cov_value)
            
            print(f"Created full covariance histogram: {cov_hist_full.GetNbinsX()}x{cov_hist_full.GetNbinsY()}")
        
        # If it's a 1D histogram (diagonal only), convert to 2D diagonal
        elif hasattr(cov_obj, 'GetNbinsX'):
            print(f"Converting 1D diagonal to 2D: {cov_obj.GetNbinsX()} bins")
            
            nbins = cov_obj.GetNbinsX()
            
            cov_hist_full = ROOT.TH2D(
                "covariance_diagonal_full", "Full Diagonal Covariance Matrix",
                nbins, 0.5, nbins + 0.5,
                nbins, 0.5, nbins + 0.5
            )
            
            # Fill diagonal elements only
            for i in range(1, nbins + 1):
                error = cov_obj.GetBinContent(i)
                variance = error * error  # Convert error to variance
                cov_hist_full.SetBinContent(i, i, variance)
            
            print(f"Created full diagonal covariance: {cov_hist_full.GetNbinsX()}x{cov_hist_full.GetNbinsY()}")
        
        else:
            print(f"Unknown covariance type: {type(cov_obj)}")
            return None, None, 'unknown'
        
        if cov_hist_full is None:
            return None, None, 'failed'
        
        # Check if this is an overflow case
        if not gen_mapping.get('include_overflow', False):
            print("WARNING: extract_covariance_and_errors called for non-overflow case!")
            # Just return the covariance as-is
            return cov_hist_full, None, 'success_non_overflow'
        
        # Extract diagonal errors from full matrix and convert to nominal range
        print("Extracting errors and nominal covariance...")
        
        # Get dimensions
        nx_full = gen_mapping['nx']  # Includes overflow
        ny_full = gen_mapping['ny']  # Includes overflow
        nx_nominal = nx_full - 2     # Nominal bins only
        ny_nominal = ny_full - 2     # Nominal bins only
        
        total_full_bins = nx_full * ny_full
        total_nominal_bins = nx_nominal * ny_nominal
        
        print(f"Full dimensions: {nx_full}x{ny_full} = {total_full_bins} bins")
        print(f"Nominal dimensions: {nx_nominal}x{ny_nominal} = {total_nominal_bins} bins")
        print(f"Covariance matrix dimensions: {cov_hist_full.GetNbinsX()}x{cov_hist_full.GetNbinsY()}")
        
        # Check if covariance is already nominal-only (RooUnfold might strip overflow)
        cov_is_nominal_only = (cov_hist_full.GetNbinsX() == total_nominal_bins)
        
        if cov_is_nominal_only:
            print("⚠️  Covariance matrix is ALREADY nominal-only (no overflow bins)!")
            print("    RooUnfold has already stripped overflow bins from the covariance.")
            print("    Returning covariance as-is without extraction.")
            
            # Extract errors from diagonal
            error_hist_nominal = ROOT.TH1D(
                "errors_nominal", "Nominal Per-bin Errors",
                total_nominal_bins, 0.5, total_nominal_bins + 0.5
            )
            
            total_variance = 0.0
            for i in range(1, total_nominal_bins + 1):
                variance = cov_hist_full.GetBinContent(i, i)
                error = np.sqrt(max(0.0, variance))
                error_hist_nominal.SetBinContent(i, error)
                total_variance += variance
            
            print(f"Total variance (diagonal sum): {total_variance:.6f}")
            print(f"Error histogram integral: {error_hist_nominal.Integral():.6f}")
            
            return cov_hist_full, error_hist_nominal, 'success_already_nominal'
        
        # Verify dimensions match for overflow case
        if cov_hist_full.GetNbinsX() != total_full_bins or cov_hist_full.GetNbinsY() != total_full_bins:
            print(f"⚠️  WARNING: Covariance dimensions don't match expected full dimensions!")
            print(f"  Expected: {total_full_bins}x{total_full_bins}")
            print(f"  Got: {cov_hist_full.GetNbinsX()}x{cov_hist_full.GetNbinsY()}")
            print(f"  This might cause indexing errors!")
            # Proceed anyway, but with caution
        
        # Create nominal covariance matrix
        cov_hist_nominal = ROOT.TH2D(
            "covariance_nominal", "Nominal Covariance Matrix",
            total_nominal_bins, 0.5, total_nominal_bins + 0.5,
            total_nominal_bins, 0.5, total_nominal_bins + 0.5
        )
        
        # Create nominal error histogram
        error_hist_nominal = ROOT.TH1D(
            "errors_nominal", "Nominal Per-bin Errors",
            total_nominal_bins, 0.5, total_nominal_bins + 0.5
        )
        
        # Map nominal bins from full matrix to nominal matrix
        # In the overflow-inclusive flattening: flat_bin = i * ny + j (with i,j starting from 0)
        # i=0, j=anything: underflow in X
        # i=nx-1, j=anything: overflow in X  
        # i=anything, j=0: underflow in Y
        # i=anything, j=ny-1: overflow in Y
        # Nominal bins: i=1..nx-2, j=1..ny-2 (in 0-based indexing)
        
        nominal_bin_map = []  # Maps nominal index -> full flat bin index (0-based)
        
        print("Building bin map...")
        for i in range(1, nx_nominal + 1):  # Loop over nominal X bins (1-based)
            for j in range(1, ny_nominal + 1):  # Loop over nominal Y bins (1-based)
                # In 0-based indexing, these are i-1 and j-1
                # But in the overflow-inclusive array, nominal bins start at index 1 (after underflow at 0)
                i_in_full = i  # Nominal bin 1 -> index 1 in full array (after underflow at 0)
                j_in_full = j  # Nominal bin 1 -> index 1 in full array (after underflow at 0)
                
                # Calculate flat index in 0-based indexing
                full_flat_bin = i_in_full * ny_full + j_in_full
                nominal_bin_map.append(full_flat_bin)
        
        print(f"Created mapping for {len(nominal_bin_map)} nominal bins")
        print(f"Bin map range: {min(nominal_bin_map)} to {max(nominal_bin_map)}")
        print(f"Should be within [0, {total_full_bins-1}]")
        
        print(f"Created mapping for {len(nominal_bin_map)} nominal bins")
        
        # Extract the nominal submatrix and errors
        extracted_elements = 0
        total_variance = 0.0
        
        for nom_i, full_i in enumerate(nominal_bin_map):
            for nom_j, full_j in enumerate(nominal_bin_map):
                # ROOT histograms use 1-based indexing, so add 1 to our 0-based flat indices
                root_bin_i = full_i + 1
                root_bin_j = full_j + 1
                
                # Bounds check
                if root_bin_i > cov_hist_full.GetNbinsX() or root_bin_j > cov_hist_full.GetNbinsY():
                    print(f"ERROR: Trying to access bin ({root_bin_i}, {root_bin_j}) but matrix is {cov_hist_full.GetNbinsX()}x{cov_hist_full.GetNbinsY()}")
                    continue
                
                # Get covariance element from full matrix
                cov_value = cov_hist_full.GetBinContent(root_bin_i, root_bin_j)
                
                # Set in nominal matrix (nom_i and nom_j are 0-based, so add 1 for ROOT)
                cov_hist_nominal.SetBinContent(nom_i + 1, nom_j + 1, cov_value)
                
                if nom_i == nom_j:  # Diagonal element
                    total_variance += cov_value
                    # Extract error (square root of variance)
                    error = np.sqrt(np.maximum(0.0, cov_value))
                    error_hist_nominal.SetBinContent(nom_i + 1, error)
                
                if cov_value != 0:
                    extracted_elements += 1
        
        print(f"Extracted {extracted_elements} non-zero covariance elements")
        print(f"Total variance (diagonal sum): {total_variance:.6f}")
        print(f"Error histogram integral: {error_hist_nominal.Integral():.6f}")
        
        return cov_hist_nominal, error_hist_nominal, 'success'
        
    except Exception as e:
        print(f"Error extracting covariance and errors: {e}")
        import traceback
        traceback.print_exc()
        return None, None, 'failed'

def extract_covariance_simple(unfolder, gen_mapping):
    """
    Simple covariance extraction for non-overflow case.
    Returns covariance as TMatrixD or TH2D, and errors as TH1D.
    """
    
    try:
        cov_obj = unfolder.Eunfold()
        
        if cov_obj is None:
            print("No covariance matrix returned")
            return None, None
        
        # Extract errors from diagonal
        error_1d = None
        
        if hasattr(cov_obj, 'GetNrows'):  # It's a TMatrixD
            nrows = cov_obj.GetNrows()
            ncols = cov_obj.GetNcols()
            print(f"Covariance is TMatrixD: {nrows}x{ncols}")
            
            # Sanity check
            nx = gen_mapping['nx']
            ny = gen_mapping['ny']
            expected_bins = nx * ny
            print(f"Expected bins from mapping: {nx}x{ny} = {expected_bins}")
            
            if nrows != expected_bins:
                print(f"⚠️  WARNING: Matrix size ({nrows}) doesn't match expected ({expected_bins})")
                print(f"  Difference: {nrows - expected_bins} bins")
                print(f"  Using actual matrix size: {nrows}")
            
            # Use the ACTUAL matrix dimensions, not the expected ones
            nbins = nrows
            
            # Update the mapping to reflect actual dimensions if needed
            if nrows != expected_bins:
                print(f"⚠️  Updating gen_mapping to match actual covariance size")
                # This is a discrepancy we need to handle
            
            error_1d = ROOT.TH1D("errors_1d", "Per-bin Errors", nbins, 0.5, nbins + 0.5)
            
            for i in range(nbins):
                try:
                    variance = cov_obj(i, i)
                    error = np.sqrt(max(0.0, variance))
                    error_1d.SetBinContent(i + 1, error)
                except:
                    print(f"Error accessing matrix element ({i}, {i})")
                    raise
            
            print(f"Extracted {nbins} errors from diagonal")
            
        elif hasattr(cov_obj, 'GetNbinsX') and hasattr(cov_obj, 'GetNbinsY'):
            print(f"Covariance is TH2D: {cov_obj.GetNbinsX()}x{cov_obj.GetNbinsY()}")
            
            nbins = cov_obj.GetNbinsX()
            error_1d = ROOT.TH1D("errors_1d", "Per-bin Errors", nbins, 0.5, nbins + 0.5)
            
            for i in range(1, nbins + 1):
                variance = cov_obj.GetBinContent(i, i)
                error = np.sqrt(max(0.0, variance))
                error_1d.SetBinContent(i, error)
            
            print(f"Extracted {nbins} errors from diagonal")
            
        elif hasattr(cov_obj, 'GetNbinsX'):  # It's a 1D histogram (errors only)
            print(f"Covariance is TH1D (errors): {cov_obj.GetNbinsX()} bins")
            error_1d = cov_obj.Clone("errors_1d")
            
            # Convert to covariance matrix (diagonal)
            nbins = error_1d.GetNbinsX()
            cov_matrix = ROOT.TMatrixD(nbins, nbins)
            
            for i in range(nbins):
                error = error_1d.GetBinContent(i + 1)
                variance = error * error
                cov_matrix[i][i] = variance
            
            cov_obj = cov_matrix
            print(f"Created diagonal covariance matrix from errors")
        
        return cov_obj, error_1d
        
    except Exception as e:
        print(f"Error in simple covariance extraction: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def perform_2d_unfolding_via_flattening(reco_2d, gen_2d, response_thnf, data_2d, 
                                       n_iterations=4, method="Bayes", include_overflow=False,
                                       h_covariance=None):
    """
    2D EEC unfolding with optional overflow handling.
    
    Parameters:
    -----------
    include_overflow : bool
        If True, include overflow bins in response matrix but return only nominal range
        If False, use only nominal bins throughout
    """
    
    print("="*60)
    print(f"PERFORMING 2D UNFOLDING")
    print(f"Overflow handling: {include_overflow}")
    print("="*60)
    
    # Step 1: Flatten histograms
    print("\nStep 1: Flattening 2D histograms...")
    reco_1d, reco_mapping = flatten_2d_histogram(reco_2d, "_reco_flat", include_overflow)
    gen_1d, gen_mapping = flatten_2d_histogram(gen_2d, "_gen_flat", include_overflow)
    data_1d, data_mapping = flatten_2d_histogram(data_2d, "_data_flat", include_overflow)
    
    # Step 2: Create response matrix
    print(f"\nStep 2: Creating response matrix...")
    response_2d, response_mapping = create_2d_response_matrix_from_thnf(
        response_thnf, include_overflow=include_overflow
    )
    
    # Step 3: RooUnfold setup
    print("\nStep 3: Creating RooUnfoldResponse...")
    roounfold_response = ROOT.RooUnfoldResponse(
        reco_1d, gen_1d, response_2d,
        "response_2d", "2D Response", False
    )

    # Step 4: Perform unfolding
    if method == "Bayes":
        print(f"\nStep 4: Performing {method} unfolding with {n_iterations} iterations...")
        unfolder = ROOT.RooUnfoldBayes(roounfold_response, data_1d, n_iterations)
        
        if h_covariance:
            print(f"✅ Setting combined data+fake covariance matrix: {h_covariance.GetNrows()}x{h_covariance.GetNcols()}")
            unfolder.SetMeasuredCov(h_covariance)
        else:
            print("ℹ️ No data covariance matrix provided. Using Poisson errors from data histogram.")
    elif method == "BinByBin":
        print(f"\nStep 4: Performing {method} unfolding...")
        unfolder = ROOT.RooUnfoldBinByBin(roounfold_response, data_1d)
    elif method == "Invert":
        print(f"\nStep 4: Performing {method} unfolding...")
        unfolder = ROOT.RooUnfoldInvert(roounfold_response, data_1d)
    else:
        raise ValueError(f"Unknown method: {method}")
        
    unfolded_1d = unfolder.Hunfold()
    
    print(f"1D unfolding completed")
    print(f"  Data integral: {data_1d.Integral():.6f}")
    print(f"  Unfolded integral: {unfolded_1d.Integral():.6f}")
    
    # Step 5: Extract covariance and errors
    print("\nStep 5: Extracting covariance matrix and errors...")
    if include_overflow:
        cov_2d, error_1d, cov_status = extract_covariance_and_errors(unfolder, gen_mapping)
        
        # Convert 1D errors to 2D if available
        error_2d = None
        if error_1d is not None:
            error_2d = unflatten_errors_to_2d_nominal(error_1d, gen_mapping, "_error_2d")
        else:
            print("❌ No errors extracted")
    else:
        # For non-overflow case, use simpler extraction
        cov_2d, error_1d = extract_covariance_simple(unfolder, gen_mapping)
        
        # Convert 1D errors to 2D if available
        error_2d = None
        if error_1d is not None:
            error_2d = unflatten_to_2d(error_1d, gen_mapping, "_error_2d")
        else:
            print("❌ No errors extracted")
    
    # Step 6: Unflatten to 2D
    print("\nStep 6: Unflattening to 2D...")
    if include_overflow:
        unfolded_2d = unflatten_to_2d_nominal(unfolded_1d, gen_mapping, "_unfolded_2d")
    else:
        unfolded_2d = unflatten_to_2d(unfolded_1d, gen_mapping, "_unfolded_2d")
    
    print(f"\n=== UNFOLDING SUMMARY ===")
    print(f"Input data: {data_2d.Integral():.6f}")
    print(f"Unfolded: {unfolded_2d.Integral():.6f}")
    
    return unfolded_2d, error_2d, cov_2d, response_2d

def perform_2d_unfolding_via_flattening_with_overflow(reco_2d, gen_2d, response_thnf, data_2d, 
                                                      n_iterations=4, method="Bayes", h_covariance=None):
    """
    2D EEC unfolding that includes overflow in response matrix but returns only nominal range.
    Convenience wrapper for perform_2d_unfolding_via_flattening with include_overflow=True.
    """
    return perform_2d_unfolding_via_flattening(
        reco_2d, gen_2d, response_thnf, data_2d,
        n_iterations, method, include_overflow=True, h_covariance=h_covariance
    )

def apply_efficiency_correction_hist(hUnf2d, eff_corr):
    """
    Apply efficiency correction while preserving the relative error pattern from unfolding
    """
    
    print("Applying efficiency correction while preserving relative errors...")
    
    for i in range(1, hUnf2d.GetNbinsX() + 1):
        for j in range(1, hUnf2d.GetNbinsY() + 1):
            # Get current values
            content = hUnf2d.GetBinContent(i, j)
            error = hUnf2d.GetBinError(i, j)
            eff_factor = eff_corr.GetBinContent(i, j)
            
            # Calculate relative error BEFORE correction
            rel_error = error / content if content > 0 else 0
            
            # Apply efficiency correction to content only
            new_content = content * eff_factor
            
            # Calculate new error to preserve the same relative error
            new_error = new_content * rel_error
            
            # Set the corrected values
            hUnf2d.SetBinContent(i, j, new_content)
            hUnf2d.SetBinError(i, j, new_error)
    
    print("✅ Efficiency correction applied with preserved relative errors")

def unfold_2d_eec(reco_2d_hist, gen_2d_hist, response_4d_thnf, data_2d_hist, 
                 n_iterations=4, unfolding_method="Invert", include_overflow=False,
                 h_covariance=None):
    """
    2D EEC unfolding with optional overflow handling.
    
    Parameters:
    -----------
    include_overflow : bool
        If True, include overflow bins in response matrix but return only nominal range
        If False, use only nominal bins throughout
    
    Returns:
    --------
    unfolded_2d : ROOT.TH2D
        Unfolded result in original 2D coordinates
    error_2d : ROOT.TH2D  
        Per-bin errors in original 2D coordinates (optional, can be None)
    cov_2d : ROOT.TH2D or ROOT.TMatrixD
        Covariance matrix in flattened coordinate space
    """
    
    try:
        unfolded_2d, error_2d, cov_2d, resp_2d = perform_2d_unfolding_via_flattening(
            reco_2d_hist, gen_2d_hist, response_4d_thnf, data_2d_hist, 
            n_iterations, unfolding_method, include_overflow, h_covariance
        )
    
        return unfolded_2d, error_2d, cov_2d, resp_2d
        
    except Exception as e:
        print(f"ERROR in unfold_2d_eec: {e}")
        return None, None, None, None

def unfold_2d_eec_with_overflow(reco_2d_hist, gen_2d_hist, response_4d_thnf, data_2d_hist, 
                                n_iterations=4, unfolding_method="Invert", h_covariance=None):
    """
    2D EEC unfolding that includes overflow in response matrix but returns only nominal range.
    Convenience wrapper for unfold_2d_eec with include_overflow=True.
    """
    return unfold_2d_eec(
        reco_2d_hist, gen_2d_hist, response_4d_thnf, data_2d_hist,
        n_iterations, unfolding_method, include_overflow=True, h_covariance=h_covariance
    )

def th1d_to_numpy_simple(hist):
    """Convert ROOT TH1D to numpy array"""
    n_bins = hist.GetNbinsX()
    array = np.zeros(n_bins)
    for i in range(1, n_bins + 1):
        array[i-1] = hist.GetBinContent(i)
    return array

def th2d_to_numpy_simple(hist):
    """Convert ROOT TH2D to numpy array"""
    nx = hist.GetNbinsX()
    ny = hist.GetNbinsY()
    array = np.zeros((nx, ny))
    for i in range(1, nx + 1):
        for j in range(1, ny + 1):
            array[i-1, j-1] = hist.GetBinContent(i, j)
    return array

def numpy_to_tmatrixd(numpy_array):
    """
    Converts a 2D NumPy array into a ROOT TMatrixD.

    Args:
        numpy_array (np.ndarray): The 2D NumPy array to convert.

    Returns:
        ROOT.TMatrixD: The converted TMatrixD object.
    """
    if numpy_array.ndim != 2:
        raise ValueError("Input must be a 2D NumPy array.")
    
    rows, cols = numpy_array.shape
    tmatrix = ROOT.TMatrixD(rows, cols)
    
    # Loop over elements and fill the TMatrixD
    # NumPy and TMatrixD are both 0-indexed, so the mapping is direct.
    for i in range(rows):
        for j in range(cols):
            tmatrix[i][j] = numpy_array[i, j]
            
    return tmatrix

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='2D Unfolding with Optional Overflow Handling')
    parser.add_argument('--method', type=str, default="Bayes",
                       help='Unfolding method to use (default: Bayes)')
    parser.add_argument('--niter', type=int, default=4,
                       help='number of iterations')
    parser.add_argument('--jacobian', type=str, default="r",
                       help='Unfold r or z distribution')
    parser.add_argument('--overflow', action='store_true',
                       help='Include overflow bins in response matrix')
    parser.add_argument('--mc', type=str, default=None,
                       help='MC response file (overrides hardcoded filenamein)')
    parser.add_argument('--data', type=str, default=None,
                       help='Data file (overrides hardcoded datafile)')
    parser.add_argument('--out', type=str, default=None,
                       help='Output file (overrides hardcoded filenameout)')
    args = parser.parse_args()

    if args.method == "Bayes":
        surfix = f'_niter{args.niter}'
    else:
        surfix = ''
    
    if args.overflow:
        surfix += '_overflow'
    
    surfix += f'_{args.jacobian}'

    filenamein = 'response_kk2f4146_qqpy_91.25_v16_zcov.root'
    datafile = 'h_94c_v8_cov.root'
    filenameout = 'unfolded_data_kk2f4146_qqpy_91.25_v6_niter4_overflow_z.root'
    #filenamein = 'response_ALEPHMC_LEP1MC1994_recons_aftercut.v3.root'
    #datafile = 'h_aleph_v3.root'
    #filenameout = f'unfolded_data_aleph_v3{surfix}.root'

    # CLI overrides (preferred): --mc / --data / --out
    if args.mc:   filenamein  = args.mc
    if args.data: datafile    = args.data
    if args.out:  filenameout = args.out
    
    data2dname = f'EEC2d_{args.jacobian}'
    
    fin = ROOT.TFile.Open(filenamein,'r')
    h_response = fin.Get(f"response2d_eij_{args.jacobian}")
    h_reco_matched = fin.Get(f'reco2d_eij_{args.jacobian}_match')
    h_gen_matched = fin.Get(f"gen2d_eij_{args.jacobian}_match")

    h_reco_total_mc = fin.Get(f'reco2d_eij_{args.jacobian}')
    h_gen_total_mc = fin.Get(f'gen2d_eij_{args.jacobian}')

    mc_event_count = fin.Get("counter").GetBinContent(2)

    # Open data file
    fdata = ROOT.TFile.Open(datafile, 'r')
    h_data_raw = fdata.Get(data2dname).Clone('h_data_raw')
    try:
        data_event_count = fdata.Get('counter').GetBinContent(2)
        counter = fdata.Get("counter").Clone("N")
    except:
        data_event_count = fdata.Get('N').GetBinContent(2)
        counter = fdata.Get("N").Clone("N")

    print(f"MC Events: {mc_event_count}, Data Events: {data_event_count}")

    scale_factor = float(data_event_count) / mc_event_count
    print(f"MC-to-Data Scale Factor: {scale_factor}")

    # The absolute number of fakes in the MC is (total reco) - (reco from matched gen)
    h_fakes_mc_abs = h_reco_total_mc.Clone("h_fakes_mc_abs")
    h_fakes_mc_abs.Add(h_reco_matched, -1)

    # Now, scale this absolute MC prediction to the data luminosity
    h_fakes_scaled = h_fakes_mc_abs.Clone("h_fakes_scaled")
    h_fakes_scaled.Scale(scale_factor)

    h_data_subtracted = h_data_raw.Clone("h_data_subtracted")

    # Subtract the scaled, absolute fake prediction
    h_data_subtracted.Add(h_fakes_scaled, -1)

    try: 
        sum_of_eec_products = th2d_to_numpy_simple(fdata.Get(f"covariance_matrix_{args.jacobian}"))
        sum_of_eecs = th1d_to_numpy_simple(fdata.Get(f"mean_eec_{args.jacobian}"))
        numerator = data_event_count * sum_of_eec_products - np.outer(sum_of_eecs, sum_of_eecs)
        data_covariance_matrix = numerator / (data_event_count - 1)

        sum_of_eec_products = th2d_to_numpy_simple(fin.Get(f"covariance_matrix_{args.jacobian}"))
        sum_of_eecs = th1d_to_numpy_simple(fin.Get(f"mean_eec_{args.jacobian}"))
        numerator = mc_event_count * sum_of_eec_products -  np.outer(sum_of_eecs, sum_of_eecs)
        fake_covariance_matrix = numerator / (mc_event_count - 1)
        
        total_covariance_numpy = data_covariance_matrix + fake_covariance_matrix*(scale_factor**2)
        
        # Check dimensions
        cov_size = total_covariance_numpy.shape[0]
        data_nx = h_data_subtracted.GetNbinsX()
        data_ny = h_data_subtracted.GetNbinsY()
        expected_size_no_overflow = data_nx * data_ny
        expected_size_with_overflow = (data_nx + 2) * (data_ny + 2)
        
        print(f"\n=== COVARIANCE DIMENSION CHECK ===")
        print(f"Data histogram: {data_nx}x{data_ny} = {expected_size_no_overflow} bins (nominal)")
        print(f"With overflow: {data_nx+2}x{data_ny+2} = {expected_size_with_overflow} bins")
        print(f"Covariance matrix: {cov_size}x{cov_size} bins")
        
        # If covariance includes overflow but we're not using overflow, extract nominal submatrix
        if cov_size == expected_size_with_overflow and not args.overflow:
            print(f"⚠️  Covariance includes overflow bins, but --overflow flag not set!")
            print(f"   Extracting nominal {expected_size_no_overflow}x{expected_size_no_overflow} submatrix...")
            
            # Build mapping from nominal bins to full (overflow-inclusive) bins
            ny_full = data_ny + 2
            nominal_indices = []
            for i in range(1, data_nx + 1):  # Skip underflow (0) and overflow (nx+1)
                for j in range(1, data_ny + 1):  # Skip underflow (0) and overflow (ny+1)
                    flat_idx = i * ny_full + j
                    nominal_indices.append(flat_idx)
            
            # Extract submatrix
            nominal_cov = total_covariance_numpy[np.ix_(nominal_indices, nominal_indices)]
            print(f"✅ Extracted nominal covariance: {nominal_cov.shape[0]}x{nominal_cov.shape[1]}")
            total_covariance_numpy = nominal_cov
            
        elif cov_size == expected_size_with_overflow and args.overflow:
            print(f"✅ Covariance includes overflow - matches --overflow flag")
        elif cov_size == expected_size_no_overflow:
            print(f"✅ Covariance is nominal-only - matches data dimensions")
        else:
            print(f"❌ WARNING: Covariance size mismatch!")
            print(f"   Expected either {expected_size_no_overflow} or {expected_size_with_overflow}, got {cov_size}")
        
        print(f"=================================\n")

        total_covariance_tmatrix = numpy_to_tmatrixd(total_covariance_numpy)
        
    except Exception as e:
        total_covariance_tmatrix = None
        print(f"❌ Could not find input data covariance matrix: {e}")
        import traceback
        traceback.print_exc()

    print("Performing unfolding...")
    hUnf2d, hErr2d, hCov2d, hResp2d = unfold_2d_eec(
        h_reco_matched, h_gen_matched, h_response, h_data_subtracted,
        n_iterations=args.niter,
        unfolding_method=args.method,
        include_overflow=args.overflow,
        h_covariance=total_covariance_tmatrix
    )

    h_eff = h_gen_total_mc.Clone("h_eff")
    h_eff.Divide(h_gen_matched)

    # Save output
    fout = ROOT.TFile(filenameout, 'recreate')
    fout.cd()

    hResp2d.Write("response_2d")
    hUnf2d.Write("unfolded_2d_noeff")

    apply_efficiency_correction_hist(hUnf2d, h_eff)
    
    eec_reco = Proj2D_X(h_data_raw, eijbins[1], eijbins[-1], f"RECO_EEC")
    eec_gen = Proj2D_X(h_gen_total_mc, eijbins[1], eijbins[-1], f"GEN_EEC")
    eec_unfold = Proj2D_X(hUnf2d, eijbins[1], eijbins[-1], f"UNFOLD_EEC")
    eec_reco.SetDirectory(0)
    eec_gen.SetDirectory(0)
    eec_unfold.SetDirectory(0)

    hUnf2d.Write("unfolded_2d")

    if hErr2d is not None:
        hErr2d.Write("errors_2d")
    
    if hCov2d is not None:
        # Debug: check what we're about to write
        if hasattr(hCov2d, 'GetNrows'):
            print(f"Writing covariance TMatrixD: {hCov2d.GetNrows()}x{hCov2d.GetNcols()}")
        elif hasattr(hCov2d, 'GetNbinsX'):
            print(f"Writing covariance TH2D: {hCov2d.GetNbinsX()}x{hCov2d.GetNbinsY()}")
        
        hCov2d.Write("covariance_2d")

    counter.Write("N")
        
    h_gen_total_mc.Write("gen_2d")
    h_reco_total_mc.Write("reco_2d")
    h_data_raw.Write("data_2d")

    h_fakes_scaled.Write("fake_corr")
    h_eff.Write("eff_corr")

    h_gen_matched.Write("gen_2d_match")
    h_reco_matched.Write("reco_2d_match")

    eec_reco.Write()
    eec_gen.Write()
    eec_unfold.Write()

    fout.Close()
    
    print(f"\n=== UNFOLDING COMPLETE ===")
    print(f"Method: {args.method}")
    if args.method == "Bayes":
        print(f"Iterations: {args.niter}")
    print(f"Overflow handling: {args.overflow}")
    print(f"Output file: {filenameout}")
    print(f"Final unfolded integral: {hUnf2d.Integral():.6f}")
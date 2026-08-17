"""
Per-track vertex reweighting helper for the FBCC/FBCEC and tau analyses.

v74+ uses 3D weight tables w(pT, θ, z0sinθ) per track charge, stored as TH3F
in `vertex_reweight_<period>_v73_3d.root` files under `weights/`. Per-track
lookup gives one weight (charge-asymmetric, z0sinθ-binned).

The legacy 2D-table loader is kept for backward compat (older v72 production
used it) and exposed as `VertexReweighter`.

Common helpers:
    detect_period(path)  → '94c' / '95d' / None
"""

from __future__ import annotations

import os
import numpy as np

# Legacy 2D-table layout (kept for v72 compatibility)
PT_EDGES = np.array([0.0, 1.0, 2.0, 4.0, 6.0, 8.0, 50.0])
IP_EDGES = np.linspace(-1.0, 1.0, 41)
W_CLIP_LO, W_CLIP_HI = 0.5, 2.0

MODES = ('d0_avg', 'd0_charge', 'z0sin_avg', 'z0sin_charge')
PERIODS = ('94c', '95d')


def detect_period(path: str | None) -> str | None:
    """Return '94c' / '95d' if the path contains it, else None."""
    if not path:
        return None
    s = str(path).lower()
    if '94c' in s:
        return '94c'
    if '95d' in s:
        return '95d'
    return None


# ----------------------------------------------------------------------------
# Legacy 2D table loader (v72 era — kept for backward compat)
# ----------------------------------------------------------------------------

class VertexReweighter:
    """Loads 2D (pT, IP) weight tables from a Python-syntax .txt file.

    Per-track lookup keyed on (pT, IP, q, mode, period). Modes: see MODES.
    Used by v72 production. v73+ uses VertexReweighter3D instead.
    """

    def __init__(self, table_path: str):
        if not os.path.isfile(table_path):
            raise FileNotFoundError(f"reweight table not found: {table_path}")

        import ast as _ast
        import re as _re
        ns: dict = {}
        with open(table_path, 'r') as f:
            text = f.read()
        for m in _re.finditer(r'^(w_\w+_2d)\s*=\s*\[', text, _re.MULTILINE):
            name = m.group(1)
            depth = 0
            i = m.end() - 1
            while i < len(text):
                c = text[i]
                if c == '[':
                    depth += 1
                elif c == ']':
                    depth -= 1
                    if depth == 0:
                        body = text[m.end() - 1: i + 1]
                        ns[name] = _ast.literal_eval(body)
                        break
                i += 1

        self._tables: dict[tuple[str, str, str], np.ndarray] = {}
        for period in PERIODS:
            for ipvar in ('d0', 'z0sin'):
                for charge in ('avg', 'pos', 'neg'):
                    key = f'w_{period}_{ipvar}_{charge}_2d'
                    if key not in ns:
                        raise KeyError(f"{key} missing from {table_path}")
                    arr = np.asarray(ns[key], dtype=np.float64)
                    expected_shape = (len(PT_EDGES) - 1, len(IP_EDGES) - 1)
                    if arr.shape != expected_shape:
                        raise ValueError(
                            f"{key} shape {arr.shape} != expected {expected_shape}"
                        )
                    self._tables[(period, ipvar, charge)] = arr

    @staticmethod
    def _bin_indices(pt: np.ndarray, ip: np.ndarray):
        ipt = np.searchsorted(PT_EDGES, pt, side='right') - 1
        ipt = np.clip(ipt, 0, len(PT_EDGES) - 2)
        iip = np.searchsorted(IP_EDGES, ip, side='right') - 1
        iip = np.clip(iip, 0, len(IP_EDGES) - 2)
        return ipt.astype(np.intp), iip.astype(np.intp)

    def weights(self, pt, ip, q, mode: str, period: str) -> np.ndarray:
        if period not in PERIODS:
            return np.ones_like(pt, dtype=np.float64)
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        pt = np.asarray(pt, dtype=np.float64)
        ip = np.asarray(ip, dtype=np.float64)
        q = np.asarray(q, dtype=np.float64)
        ipvar, scheme = mode.split('_', 1)
        ipt, iip = self._bin_indices(pt, ip)
        if scheme == 'avg':
            w = self._tables[(period, ipvar, 'avg')][ipt, iip]
        else:
            tab_pos = self._tables[(period, ipvar, 'pos')]
            tab_neg = self._tables[(period, ipvar, 'neg')]
            w = np.where(q > 0, tab_pos[ipt, iip], tab_neg[ipt, iip])
        return np.clip(w, W_CLIP_LO, W_CLIP_HI)


# ----------------------------------------------------------------------------
# v74+ 3D table loader — w(pT, θ, z0sinθ, q) from TH3F in ROOT file
# ----------------------------------------------------------------------------

class VertexReweighter3D:
    """Per-track w(pT, θ, z0sinθ, q) lookup from the v73 3D weight tables.

    Parameters
    ----------
    root_file_path : str
        Path to `vertex_reweight_<period>_v73_3d.root`. Reads the
        `weights/z0sin_vtx_pos_3d` and `weights/z0sin_vtx_neg_3d` TH3F
        histograms, copies the bin contents into numpy arrays, and exposes a
        vectorised `weights(...)` method.

    Bins outside the histogram coverage clip into the boundary bin. Bins
    where the table value is ≤ 0 default to 1.0. The final weight is clipped
    to [0.5, 2.0] per the upstream recommendation.
    """

    CLIP_LO = 0.5
    CLIP_HI = 2.0

    def __init__(self, root_file_path: str):
        if not os.path.isfile(root_file_path):
            raise FileNotFoundError(f"3D reweight ROOT file not found: {root_file_path}")
        # Defer ROOT import so this module can be imported in non-ROOT envs
        import ROOT  # noqa: F401  (used below)
        f = ROOT.TFile.Open(root_file_path, 'READ')
        if not f or f.IsZombie():
            raise RuntimeError(f"could not open {root_file_path}")
        try:
            h_pos = f.Get('weights/z0sin_vtx_pos_3d')
            h_neg = f.Get('weights/z0sin_vtx_neg_3d')
            if not h_pos or not h_neg:
                raise KeyError("weights/z0sin_vtx_{pos,neg}_3d missing from ROOT file")

            # Sanity: same axes for pos/neg
            self._x_edges = self._axis_edges(h_pos.GetXaxis())
            self._y_edges = self._axis_edges(h_pos.GetYaxis())
            self._z_edges = self._axis_edges(h_pos.GetZaxis())
            self._tab_pos = self._th3_to_numpy(h_pos)
            self._tab_neg = self._th3_to_numpy(h_neg)
        finally:
            f.Close()

    @staticmethod
    def _axis_edges(axis) -> np.ndarray:
        n = axis.GetNbins()
        edges = np.empty(n + 1, dtype=np.float64)
        for i in range(n):
            edges[i] = axis.GetBinLowEdge(i + 1)
        edges[n] = axis.GetBinUpEdge(n)
        return edges

    @staticmethod
    def _th3_to_numpy(h) -> np.ndarray:
        nx = h.GetNbinsX()
        ny = h.GetNbinsY()
        nz = h.GetNbinsZ()
        arr = np.empty((nx, ny, nz), dtype=np.float64)
        for ix in range(1, nx + 1):
            for iy in range(1, ny + 1):
                for iz in range(1, nz + 1):
                    arr[ix - 1, iy - 1, iz - 1] = h.GetBinContent(ix, iy, iz)
        return arr

    def _bin_indices(self, pt, theta_deg, z0sin):
        ix = np.searchsorted(self._x_edges, pt, side='right') - 1
        iy = np.searchsorted(self._y_edges, theta_deg, side='right') - 1
        iz = np.searchsorted(self._z_edges, z0sin, side='right') - 1
        ix = np.clip(ix, 0, len(self._x_edges) - 2)
        iy = np.clip(iy, 0, len(self._y_edges) - 2)
        iz = np.clip(iz, 0, len(self._z_edges) - 2)
        return ix.astype(np.intp), iy.astype(np.intp), iz.astype(np.intp)

    def weights(self, pt, theta_deg, z0sin, q) -> np.ndarray:
        """Vectorised per-track weight lookup.

        All inputs are arrays of equal length. Returns a float array of the
        same length, with weights clipped to [0.5, 2.0] and bins where the
        table value is ≤ 0 set to 1.0.
        """
        pt        = np.asarray(pt,        dtype=np.float64)
        theta_deg = np.asarray(theta_deg, dtype=np.float64)
        z0sin     = np.asarray(z0sin,     dtype=np.float64)
        q         = np.asarray(q,         dtype=np.float64)
        ix, iy, iz = self._bin_indices(pt, theta_deg, z0sin)
        w = np.where(q > 0,
                     self._tab_pos[ix, iy, iz],
                     self._tab_neg[ix, iy, iz])
        # Empty / negative bins fall back to 1.0
        w = np.where(w > 0, w, 1.0)
        return np.clip(w, self.CLIP_LO, self.CLIP_HI)


def find_3d_table(period: str, search_dir: str = '.') -> str | None:
    """Find the v73 3D weight ROOT file for a given period.

    Looks for `vertex_reweight_<period>_v73_3d.root` in `search_dir`.
    Returns the path or None if not found.
    """
    candidate = os.path.join(search_dir, f'vertex_reweight_{period}_v73_3d.root')
    return candidate if os.path.isfile(candidate) else None


# ============================================================================
# v75: resolution-smearing reweighter (additive Gaussian on MC d0 and z0sinθ)
# ============================================================================

class VertexResolutionSmearer:
    """Per-track Gaussian smearing of MC d0 / z0·sinθ to match data resolution.

    Reads `vertex_resolution_<period>_v78_3d.root` produced upstream, takes
    the per-(pT, θ) sigma TH2Fs from the `sigmas/` subdirectory, and at each
    call produces independent Gaussian noise samples whose width is
    Δ = sqrt(max(σ_data² − σ_mc², 0)) for the requested coordinate.

    `kind='d0'`     uses `d0_sigma_data_avg`    vs `d0_sigma_mc_avg`.
    `kind='z0sin'`  uses `z0sin_sigma_data_avg` vs `z0sin_sigma_mc_avg`.

    Binning is read from the TH2F itself (currently 6 pT bins × 18 θ bins of
    10°). Out-of-range pT/θ folds into the first/last bin via np.clip.
    """

    def __init__(self, path: str):
        import ROOT  # noqa: F401 (lazy: avoid module-level ROOT import)
        f = ROOT.TFile.Open(path)
        if not f or f.IsZombie():
            raise RuntimeError(f"VertexResolutionSmearer: cannot open {path}")
        try:
            sd_d0   = f.Get('sigmas/d0_sigma_data_avg')
            sm_d0   = f.Get('sigmas/d0_sigma_mc_avg')
            sd_z0s  = f.Get('sigmas/z0sin_sigma_data_avg')
            sm_z0s  = f.Get('sigmas/z0sin_sigma_mc_avg')
            if not all((sd_d0, sm_d0, sd_z0s, sm_z0s)):
                raise RuntimeError(
                    f"VertexResolutionSmearer: missing one of "
                    f"sigmas/{{d0,z0sin}}_sigma_{{data,mc}}_avg in {path}")
            self.delta_d0    = self._build_delta(sd_d0, sm_d0)
            self.delta_z0sin = self._build_delta(sd_z0s, sm_z0s)
            # Common axes (assume all four hists share binning)
            self.pt_edges    = np.array(
                [sd_d0.GetXaxis().GetBinLowEdge(i+1) for i in range(sd_d0.GetNbinsX()+1)],
                dtype=np.float64)
            self.theta_edges = np.array(
                [sd_d0.GetYaxis().GetBinLowEdge(i+1) for i in range(sd_d0.GetNbinsY()+1)],
                dtype=np.float64)
        finally:
            f.Close()

    @staticmethod
    def _build_delta(h_data, h_mc):
        nx = h_data.GetNbinsX(); ny = h_data.GetNbinsY()
        delta = np.zeros((nx, ny), dtype=np.float64)
        for ix in range(1, nx + 1):
            for iy in range(1, ny + 1):
                sd = h_data.GetBinContent(ix, iy)
                sm = h_mc.GetBinContent(ix, iy)
                if sd > 0 and sm > 0:
                    diff2 = sd * sd - sm * sm
                    if diff2 > 0:
                        delta[ix - 1, iy - 1] = np.sqrt(diff2)
        return delta

    def _bin(self, pt, theta_deg):
        ix = np.clip(np.searchsorted(self.pt_edges,    pt,        side='right') - 1,
                     0, len(self.pt_edges)    - 2)
        iy = np.clip(np.searchsorted(self.theta_edges, theta_deg, side='right') - 1,
                     0, len(self.theta_edges) - 2)
        return ix, iy

    def widths(self, kind: str, pt, theta_deg) -> np.ndarray:
        """Return per-track smear width Δ (cm). 0 in bins where σ_data ≤ σ_mc
        or either is missing."""
        pt = np.asarray(pt, dtype=np.float64)
        theta_deg = np.asarray(theta_deg, dtype=np.float64)
        ix, iy = self._bin(pt, theta_deg)
        table = self.delta_d0 if kind == 'd0' else self.delta_z0sin
        return table[ix, iy]

    def sample(self, kind: str, pt, theta_deg, rng) -> np.ndarray:
        """Draw N independent Gaussian samples with per-track Δ width."""
        w = self.widths(kind, pt, theta_deg)
        if len(w) == 0:
            return np.zeros(0, dtype=np.float64)
        return rng.normal(0.0, 1.0, size=len(w)) * w


def find_resolution_table(period: str, search_dir: str = '.') -> str | None:
    """Find the v75 3D resolution ROOT file for a given period."""
    candidate = os.path.join(search_dir, f'vertex_resolution_{period}_v78_3d.root')
    return candidate if os.path.isfile(candidate) else None

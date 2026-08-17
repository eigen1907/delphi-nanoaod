#!/usr/bin/env python3
"""
Tau Tag-and-Probe: Charge-Dependent Tracking Efficiency
========================================================
Uses uproot. Runs on treefy'd miniAOD.

For 1+1 events, the tag charge tells you the TRUE probe charge:
  tag=- → probe should be + : fill denominator tag_neg(θ), numerator probe_pos_given_tag_neg(θ)
  tag=+ → probe should be - : fill denominator tag_pos(θ), numerator probe_neg_given_tag_pos(θ)

Efficiency: ε(+,θ) = probe_pos_given_tag_neg / tag_neg  (tracks found with correct charge)
Mis-ID:     f(+,θ) = probe_pos_given_tag_pos / tag_pos  (tracks found with wrong charge = SS)

All split by pT thresholds on the probe track.

Usage:
  python analysis_tau.py input_ttree.root output.root [--is-mc] [--max-events N]
"""

import ROOT
import uproot
import awkward as ak
import numpy as np
import argparse
import math
import sys
from array import array

# Vertex reweighting (per-track w(pT, z0sinθ, q) — applied to MC at fill time
# so reweighted "_rw_z0sin_charge" companion histograms can be compared to
# data/MC ratios. Only used when --is-mc and the period is detected.
from vertex_reweight import VertexReweighter3D, detect_period as rw_detect_period, find_3d_table

RW_TAG = '_rw_z0sin_charge'

# Impact-parameter diagnostic binning — identical to analysis_fbcec.py
# (D0_VTX / Z0_VTX / Z0SIN_VTX) so the d0 / z0 / z0·sinθ distributions can be
# overlaid directly against the FBCEC ones.
D0_VTX_NBINS,    D0_VTX_RANGE    = 1000, (-1.0, 1.0)
Z0_VTX_NBINS,    Z0_VTX_RANGE    = 1000, (-3.0, 3.0)
Z0SIN_VTX_NBINS, Z0SIN_VTX_RANGE = 1000, (-1.2, 1.2)


_PDG_DB = ROOT.TDatabasePDG.Instance()
_CHARGE_CACHE = {}


def pdg_charge(pid):
    pid = int(pid)
    if pid in _CHARGE_CACHE:
        return _CHARGE_CACHE[pid]
    p = _PDG_DB.GetParticle(pid)
    # GetParticle returns a NULL TParticlePDG* (falsy but NOT Python None) for
    # PDG ids absent from the table — `not p` catches both None and nullptr,
    # avoiding a null-deref ReferenceError on p.Charge() (seen on rare gen
    # particles in some pythia8_isrbk events).
    q = 0.0 if not p else p.Charge() / 3.0
    _CHARGE_CACHE[pid] = q
    return q


M_PI_GEV = 0.13957

# Angular matching cutoff for reco-gen matching (radians, 3D opening angle).
# Mirrors `MATCH_DR` in analysis_fbcec.py so the tau and hadronic analyses
# share the same gen-match criterion.
MATCH_DR = 0.05


# ============================================================================
# LOAD
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
    opt = ["Event_cmEnergy", "Muid_tag", "Muid_partIdx", "Elid_tag", "Elid_partIdx",
           "Trac_partIdx", "Trac_impParToVertexRPhi", "Trac_impParToVertexZ",
           "Trac_perigee_fArray", "Trac_weightMatrix_fRep_fArray"]
    for k in opt:
        if k in keys: branches.append(k)

    gen_px_key = None
    if "GenPart_pdgId" in keys:
        for gc in ["GenPart_vector_fCoordinates_fX",
                    "GenPart_vector.fCoordinates.fX",
                    "GenPart_vector/GenPart_vector.fCoordinates.fX"]:
            if gc in keys:
                gen_px_key = gc
                branches += [gc, gc.replace("fX","fY"), gc.replace("fX","fZ")]
                break
        for k in ["GenPart_pdgId", "GenPart_status"]:
            if k in keys: branches.append(k)

    data = tree.arrays(branches, entry_stop=max_events if max_events > 0 else None)
    result = {"px": data[px_key], "py": data[py_key], "pz": data[pz_key],
              "en": data[en_key], "charge": data["Part_charge"], "lock": data["Part_lock"]}
    for k in opt:
        if k in data.fields: result[k] = data[k]
    if "Trac_impParToVertexRPhi" in data.fields:
        result["Trac_d0"] = data["Trac_impParToVertexRPhi"]
        result["Trac_z0"] = data["Trac_impParToVertexZ"]
        result["Trac_partIdx"] = data["Trac_partIdx"]
    if gen_px_key and gen_px_key in data.fields:
        result["gen_px"] = data[gen_px_key]
        result["gen_py"] = data[gen_px_key.replace("fX","fY")]
        result["gen_pz"] = data[gen_px_key.replace("fX","fZ")]
    for k in ["GenPart_pdgId", "GenPart_status"]:
        if k in data.fields: result[k] = data[k]
    return result


# ============================================================================
# THRUST
# ============================================================================

def compute_thrust(px, py, pz):
    p3 = np.stack([px, py, pz], axis=1)
    n = len(px)
    pt = np.sum(np.sqrt(px**2 + py**2 + pz**2))
    if n == 0 or pt < 1e-10:
        return np.array([0., 0., 1.]), 0.0
    if n <= 10:
        best, bax = 0.0, np.array([0., 0., 1.])
        for m in range(1 << (n - 1)):
            s = np.ones(n)
            for j in range(n - 1):
                if m & (1 << j): s[j] = -1.0
            t = np.sum(p3 * s[:, None], axis=0)
            tm = np.linalg.norm(t)
            if tm > best: best, bax = tm, t / tm
        return bax, best / pt
    else:
        ax = np.sum(p3, axis=0); ax /= np.linalg.norm(ax)
        for _ in range(20):
            s = np.sign(p3 @ ax); s[s == 0] = 1.0
            na = np.sum(p3 * s[:, None], axis=0); na /= np.linalg.norm(na)
            if abs(np.dot(na, ax)) > 1 - 1e-10: break
            ax = na
        return ax, np.sum(np.abs(p3 @ ax)) / pt


# ============================================================================
# HISTOGRAMS
# ============================================================================

THETA_BINS = np.linspace(0, 180, 181)  # 180 x 1deg, full [0, 180]
# cos(θ) bin edges: 140 uniform bins inside |cosθ| < cos(20°) = 0.9397
# (θ ∈ [20°, 160°] fiducial). Decoupled from THETA_BINS — different bin
# count and different range. Tracks outside this fiducial go to TH1D
# under/overflow.
COS_THETA_BINS = np.linspace(-0.9397, 0.9397, 141)
P_BINS = np.array([0.5, 1, 2, 3, 5, 8, 12, 20, 30, 45, 60], dtype=np.float64)
PT_CUTS = [0.4, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 20.0]
PMAG_CUTS = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 20.0]


def _cos_th(theta_deg):
    """Convert θ in degrees to cos(θ) (scalar or array)."""
    return np.cos(np.deg2rad(theta_deg))


class _NoOpHist:
    """Stand-in for a histogram that absorbs Fill calls silently. Returned
    by _SuffixedDict when no suffixed partner exists for a key (e.g. for
    `cutflow`, `dpp_status`, `tag_match_status` — categorical hists that
    don't get cloned into a `_dpp0p5` variant)."""
    def Fill(self, *args, **kwargs):
        return None


_NOOP_HIST = _NoOpHist()


class _SuffixedDict:
    """Lightweight dict wrapper that appends a fixed suffix to every key
    on lookup. Used to thread the `_dpp0p5` variant through the fill
    functions without editing every individual `h['name']` access.

    If the suffixed key isn't present, returns a no-op hist so the call
    site's `.Fill(...)` is silently absorbed. That lets us skip variant
    fills for non-cloned categorical hists (cutflow, dpp_status, etc.)
    without conditional checks at each Fill site."""
    __slots__ = ('_d', '_sfx')
    def __init__(self, d, suffix):
        self._d = d
        self._sfx = suffix
    def __getitem__(self, key):
        try:
            return self._d[key + self._sfx]
        except KeyError:
            return _NOOP_HIST
    def __contains__(self, key):
        return (key + self._sfx) in self._d


def book_histograms():
    h = {}
    h['cutflow'] = ROOT.TH1D('cutflow', '', 10, 0, 10)
    for i, l in enumerate(['Total', 'Good>=2', 'Nch<=4', 'Evis', 'Thrust',
                            'Acol', 'Prongs', 'Pass', '1p+1p', '1p+0p']):
        h['cutflow'].GetXaxis().SetBinLabel(i + 1, l)

    # Event-level
    h['thrust']    = ROOT.TH1D('thrust',    ';Thrust;Events', 100, 0.9, 1.0)
    h['acol']      = ROOT.TH1D('acol',      ';Acollinearity [deg];Events', 50, 0, 50)
    h['evis_frac'] = ROOT.TH1D('evis_frac', ';E_{vis}/E_{cm};Events', 50, 0, 1.0)
    h['n_charged'] = ROOT.TH1D('n_charged', ';N_{charged};Events', 10, 0, 10)
    h['mvis']               = ROOT.TH1D('mvis',               ';M_{vis} [GeV]', 100, 0, 100)
    h['mvis_before_lepveto']= ROOT.TH1D('mvis_before_lepveto',';M_{vis} [GeV]', 100, 0, 100)
    h['mvis_before_prongs'] = ROOT.TH1D('mvis_before_prongs', ';M_{vis} [GeV]', 100, 0, 100)
    h['mvis_before_thrust'] = ROOT.TH1D('mvis_before_thrust', ';M_{vis} [GeV]', 100, 0, 100)
    # MC truth-topo splits of post-cuts mvis (filled only for reco 1+1 with gen info).
    for gtopo in ('_gen1p1p', '_gen1p3p', '_genother'):
        h[f'mvis{gtopo}'] = ROOT.TH1D(f'mvis{gtopo}', ';M_{vis} [GeV]', 100, 0, 100)
    h['charge_sum']    = ROOT.TH1D('charge_sum',    ';Q_{tag}+Q_{probe}', 9, -4.5, 4.5)

    # MC truth topology label for reco 1+1 events.
    # Gen prong = number of status==1 charged FS particles per hemisphere
    # (split by reco thrust axis). Filled once per selected reco 1+1 event.
    h['reco_1p1p_gen_topo'] = ROOT.TH1D('reco_1p1p_gen_topo',
                                        ';gen topology;reco 1+1 events',
                                        3, 0, 3)
    for i, l in enumerate(['1p1p', '1p3p', 'other']):
        h['reco_1p1p_gen_topo'].GetXaxis().SetBinLabel(i + 1, l)

    # ---- Tag-Probe conditional histograms (1+1 events only) ----
    # For each pT threshold on the probe:
    #   tag_pos(θ)                  = denominator for ε(-) measurement
    #   tag_neg(θ)                  = denominator for ε(+) measurement
    #   probe_pos_given_tag_neg(θ)  = numerator for ε(+) (correct charge: OS)
    #   probe_neg_given_tag_pos(θ)  = numerator for ε(-) (correct charge: OS)
    #   probe_pos_given_tag_pos(θ)  = SS mis-ID for + (tag=+, probe=+ → wrong)
    #   probe_neg_given_tag_neg(θ)  = SS mis-ID for - (tag=-, probe=- → wrong)
    # Inclusive ('') + per-gen-topo splits (MC only, but always booked).
    GEN_TOPO_SUFFIXES = ('', '_gen1p1p', '_gen1p3p', '_genother')
    th_nbin = len(THETA_BINS) - 1
    th_edge = array('d', THETA_BINS)
    co_nbin = len(COS_THETA_BINS) - 1
    co_edge = array('d', COS_THETA_BINS)
    for pcut in PT_CUTS:
        t = f"pt{pcut:.1f}".replace('.', 'p')
        for gtopo in GEN_TOPO_SUFFIXES:
            # Tag denominators (filled once per 1+1 pair, for each tag-probe assignment)
            h[f'tag_pos_{t}{gtopo}']     = ROOT.TH1D(f'tag_pos_{t}{gtopo}',     '', th_nbin, th_edge)
            h[f'tag_neg_{t}{gtopo}']     = ROOT.TH1D(f'tag_neg_{t}{gtopo}',     '', th_nbin, th_edge)
            h[f'tag_pos_{t}{gtopo}_cos'] = ROOT.TH1D(f'tag_pos_{t}{gtopo}_cos', '', co_nbin, co_edge)
            h[f'tag_neg_{t}{gtopo}_cos'] = ROOT.TH1D(f'tag_neg_{t}{gtopo}_cos', '', co_nbin, co_edge)
            # Probe numerators: probe charge | given tag charge
            for pc in ('pos', 'neg'):
                for tc in ('pos', 'neg'):
                    h[f'probe_{pc}_tag_{tc}_{t}{gtopo}']     = ROOT.TH1D(
                        f'probe_{pc}_tag_{tc}_{t}{gtopo}',     '', th_nbin, th_edge)
                    h[f'probe_{pc}_tag_{tc}_{t}{gtopo}_cos'] = ROOT.TH1D(
                        f'probe_{pc}_tag_{tc}_{t}{gtopo}_cos', '', co_nbin, co_edge)

    # 1+1 pmag cuts (baseline pT > 0.4 always applied)
    for pcut in PMAG_CUTS:
        t = f"pmag{pcut:.1f}".replace('.', 'p')
        for gtopo in GEN_TOPO_SUFFIXES:
            h[f'tag_pos_{t}{gtopo}']     = ROOT.TH1D(f'tag_pos_{t}{gtopo}',     '', th_nbin, th_edge)
            h[f'tag_neg_{t}{gtopo}']     = ROOT.TH1D(f'tag_neg_{t}{gtopo}',     '', th_nbin, th_edge)
            h[f'tag_pos_{t}{gtopo}_cos'] = ROOT.TH1D(f'tag_pos_{t}{gtopo}_cos', '', co_nbin, co_edge)
            h[f'tag_neg_{t}{gtopo}_cos'] = ROOT.TH1D(f'tag_neg_{t}{gtopo}_cos', '', co_nbin, co_edge)
            for pc in ('pos', 'neg'):
                for tc in ('pos', 'neg'):
                    h[f'probe_{pc}_tag_{tc}_{t}{gtopo}']     = ROOT.TH1D(
                        f'probe_{pc}_tag_{tc}_{t}{gtopo}',     '', th_nbin, th_edge)
                    h[f'probe_{pc}_tag_{tc}_{t}{gtopo}_cos'] = ROOT.TH1D(
                        f'probe_{pc}_tag_{tc}_{t}{gtopo}_cos', '', co_nbin, co_edge)

    # MC gen-level
    for sign in ('pos', 'neg'):
        h[f'gen_tau_theta_{sign}']     = ROOT.TH1D(
            f'gen_tau_theta_{sign}',     '', th_nbin, th_edge)
        h[f'gen_tau_theta_{sign}_cos'] = ROOT.TH1D(
            f'gen_tau_theta_{sign}_cos', '', co_nbin, co_edge)

    # Tag/probe charge mis-ID truth-match (MC only, 1+1 only).
    # Inclusive (no pT threshold) and TNP-style per-cut: each pT/|p| cut is
    # applied to the side's OWN pT (or |p|) and the histogram is binned in
    # the side's OWN theta. Denominator = matched side with gen charge `sign`,
    # numerator = same with reco != gen.
    nb_t = len(THETA_BINS) - 1
    barr_t = array('d', THETA_BINS)
    nb_c = co_nbin
    barr_c = co_edge
    for side in ('tag', 'probe'):
        for sign in ('pos', 'neg'):
            for gtopo in GEN_TOPO_SUFFIXES:
                h[f'{side}_match_total_{sign}{gtopo}']     = ROOT.TH1D(
                    f'{side}_match_total_{sign}{gtopo}',
                    f';#theta_{{{side}}} [deg];matched {side}s',
                    nb_t, barr_t)
                h[f'{side}_match_misid_{sign}{gtopo}']     = ROOT.TH1D(
                    f'{side}_match_misid_{sign}{gtopo}',
                    f';#theta_{{{side}}} [deg];mis-ID {side}s',
                    nb_t, barr_t)
                h[f'{side}_match_total_{sign}{gtopo}_cos'] = ROOT.TH1D(
                    f'{side}_match_total_{sign}{gtopo}_cos',
                    f';cos#theta_{{{side}}};matched {side}s',
                    nb_c, barr_c)
                h[f'{side}_match_misid_{sign}{gtopo}_cos'] = ROOT.TH1D(
                    f'{side}_match_misid_{sign}{gtopo}_cos',
                    f';cos#theta_{{{side}}};mis-ID {side}s',
                    nb_c, barr_c)
                for pcut in PT_CUTS:
                    t = f"pt{pcut:.1f}".replace('.', 'p')
                    h[f'{side}_match_total_{sign}_{t}{gtopo}']     = ROOT.TH1D(
                        f'{side}_match_total_{sign}_{t}{gtopo}',     '', nb_t, barr_t)
                    h[f'{side}_match_misid_{sign}_{t}{gtopo}']     = ROOT.TH1D(
                        f'{side}_match_misid_{sign}_{t}{gtopo}',     '', nb_t, barr_t)
                    h[f'{side}_match_total_{sign}_{t}{gtopo}_cos'] = ROOT.TH1D(
                        f'{side}_match_total_{sign}_{t}{gtopo}_cos', '', nb_c, barr_c)
                    h[f'{side}_match_misid_{sign}_{t}{gtopo}_cos'] = ROOT.TH1D(
                        f'{side}_match_misid_{sign}_{t}{gtopo}_cos', '', nb_c, barr_c)
                for pcut in PMAG_CUTS:
                    t = f"pmag{pcut:.1f}".replace('.', 'p')
                    h[f'{side}_match_total_{sign}_{t}{gtopo}']     = ROOT.TH1D(
                        f'{side}_match_total_{sign}_{t}{gtopo}',     '', nb_t, barr_t)
                    h[f'{side}_match_misid_{sign}_{t}{gtopo}']     = ROOT.TH1D(
                        f'{side}_match_misid_{sign}_{t}{gtopo}',     '', nb_t, barr_t)
                    h[f'{side}_match_total_{sign}_{t}{gtopo}_cos'] = ROOT.TH1D(
                        f'{side}_match_total_{sign}_{t}{gtopo}_cos', '', nb_c, barr_c)
                    h[f'{side}_match_misid_{sign}_{t}{gtopo}_cos'] = ROOT.TH1D(
                        f'{side}_match_misid_{sign}_{t}{gtopo}_cos', '', nb_c, barr_c)
        h[f'{side}_match_status'] = ROOT.TH1D(
            f'{side}_match_status', ';status;1+1 events', 3, 0, 3)
        for i, l in enumerate(['total_1p1p', 'matched', 'unmatched']):
            h[f'{side}_match_status'].GetXaxis().SetBinLabel(i + 1, l)

    # Tag and probe pT distributions (no pT threshold, inclusive)
    PT_HIST_BINS = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 5, 7, 10, 15, 20, 30, 45, 60], dtype=np.float64)
    for side in ('tag', 'probe'):
        for sign in ('pos', 'neg'):
            h[f'{side}_pt_{sign}'] = ROOT.TH1D(
                f'{side}_pt_{sign}', f';p_{{T}} [GeV];{side} ({sign})',
                len(PT_HIST_BINS)-1, array('d', PT_HIST_BINS))
            h[f'{side}_p_{sign}'] = ROOT.TH1D(
                f'{side}_p_{sign}', f';p [GeV];{side} ({sign})',
                len(PT_HIST_BINS)-1, array('d', PT_HIST_BINS))
            h[f'{side}_theta_{sign}']     = ROOT.TH1D(
                f'{side}_theta_{sign}',     f';#theta [deg];{side} ({sign})',
                nb_t, barr_t)
            h[f'{side}_theta_{sign}_cos'] = ROOT.TH1D(
                f'{side}_theta_{sign}_cos', f';cos#theta;{side} ({sign})',
                nb_c, barr_c)
            # 1+1 kinematic shapes (single-track tag and probe).
            topo = '1p1p'
            h[f'{side}_pt_{sign}_{topo}'] = ROOT.TH1D(
                f'{side}_pt_{sign}_{topo}', f';p_{{T}} [GeV];{side} ({sign}, {topo})',
                len(PT_HIST_BINS)-1, array('d', PT_HIST_BINS))
            h[f'{side}_p_{sign}_{topo}'] = ROOT.TH1D(
                f'{side}_p_{sign}_{topo}', f';p [GeV];{side} ({sign}, {topo})',
                len(PT_HIST_BINS)-1, array('d', PT_HIST_BINS))
            h[f'{side}_theta_{sign}_{topo}']     = ROOT.TH1D(
                f'{side}_theta_{sign}_{topo}',     f';#theta [deg];{side} ({sign}, {topo})',
                nb_t, barr_t)
            h[f'{side}_theta_{sign}_{topo}_cos'] = ROOT.TH1D(
                f'{side}_theta_{sign}_{topo}_cos', f';cos#theta;{side} ({sign}, {topo})',
                nb_c, barr_c)
            # MC truth-topo splits of the 1+1 kinematic shapes (and the
            # equivalent inclusive shapes, which take identical values in
            # fill_1p1p). Filled only when gen_topo is known.
            for gtopo in ('_gen1p1p', '_gen1p3p', '_genother'):
                for base in ('', '_1p1p'):
                    h[f'{side}_pt_{sign}{base}{gtopo}'] = ROOT.TH1D(
                        f'{side}_pt_{sign}{base}{gtopo}', f';p_{{T}} [GeV];{side} ({sign}{base}{gtopo})',
                        len(PT_HIST_BINS)-1, array('d', PT_HIST_BINS))
                    h[f'{side}_p_{sign}{base}{gtopo}'] = ROOT.TH1D(
                        f'{side}_p_{sign}{base}{gtopo}', f';p [GeV];{side} ({sign}{base}{gtopo})',
                        len(PT_HIST_BINS)-1, array('d', PT_HIST_BINS))
                    h[f'{side}_theta_{sign}{base}{gtopo}']     = ROOT.TH1D(
                        f'{side}_theta_{sign}{base}{gtopo}',     f';#theta [deg];{side} ({sign}{base}{gtopo})',
                        nb_t, barr_t)
                    h[f'{side}_theta_{sign}{base}{gtopo}_cos'] = ROOT.TH1D(
                        f'{side}_theta_{sign}{base}{gtopo}_cos', f';cos#theta;{side} ({sign}{base}{gtopo})',
                        nb_c, barr_c)

    # 1+0 control region kinematics — only the surviving prong (no probe).
    # Used as denominator augmentation for the per-charge tracking efficiency
    # (combined offline with the 1+1 tag/probe denominators).
    for sign in ('pos', 'neg'):
        h[f'tag_pt_{sign}_1p0p'] = ROOT.TH1D(
            f'tag_pt_{sign}_1p0p', f';p_{{T}} [GeV];tag ({sign}, 1+0)',
            len(PT_HIST_BINS)-1, array('d', PT_HIST_BINS))
        h[f'tag_p_{sign}_1p0p'] = ROOT.TH1D(
            f'tag_p_{sign}_1p0p', f';p [GeV];tag ({sign}, 1+0)',
            len(PT_HIST_BINS)-1, array('d', PT_HIST_BINS))
        h[f'tag_theta_{sign}_1p0p']     = ROOT.TH1D(
            f'tag_theta_{sign}_1p0p',     f';#theta [deg];tag ({sign}, 1+0)',
            nb_t, barr_t)
        h[f'tag_theta_{sign}_1p0p_cos'] = ROOT.TH1D(
            f'tag_theta_{sign}_1p0p_cos', f';cos#theta;tag ({sign}, 1+0)',
            nb_c, barr_c)

    # 1+0 per-pT and per-|p| cut histograms, theta-binned (parallel to the
    # 1+1 `tag_<sign>_<t>` denominators). Cut applied on the SURVIVING track's
    # pT (used as a back-to-back proxy for the missed probe pT).
    for pcut in PT_CUTS:
        t = f"pt{pcut:.1f}".replace('.', 'p')
        h[f'tag_pos_1p0p_{t}']     = ROOT.TH1D(f'tag_pos_1p0p_{t}',     '', nb_t, barr_t)
        h[f'tag_neg_1p0p_{t}']     = ROOT.TH1D(f'tag_neg_1p0p_{t}',     '', nb_t, barr_t)
        h[f'tag_pos_1p0p_{t}_cos'] = ROOT.TH1D(f'tag_pos_1p0p_{t}_cos', '', nb_c, barr_c)
        h[f'tag_neg_1p0p_{t}_cos'] = ROOT.TH1D(f'tag_neg_1p0p_{t}_cos', '', nb_c, barr_c)
    for pcut in PMAG_CUTS:
        t = f"pmag{pcut:.1f}".replace('.', 'p')
        h[f'tag_pos_1p0p_{t}']     = ROOT.TH1D(f'tag_pos_1p0p_{t}',     '', nb_t, barr_t)
        h[f'tag_neg_1p0p_{t}']     = ROOT.TH1D(f'tag_neg_1p0p_{t}',     '', nb_t, barr_t)
        h[f'tag_pos_1p0p_{t}_cos'] = ROOT.TH1D(f'tag_pos_1p0p_{t}_cos', '', nb_c, barr_c)
        h[f'tag_neg_1p0p_{t}_cos'] = ROOT.TH1D(f'tag_neg_1p0p_{t}_cos', '', nb_c, barr_c)

    # 1+0 per-charge truth match (MC only). Surviving prong is matched to
    # nearest gen charged FS within ΔR < MATCH_DR (3D opening angle).
    # Naming mirrors the 1+1 `tag_match_*` convention but with `_1p0p` in
    # the gen-topo slot (no gen-topo split for 1+0).
    for sign in ('pos', 'neg'):
        h[f'tag_match_total_{sign}_1p0p']     = ROOT.TH1D(
            f'tag_match_total_{sign}_1p0p',     '', nb_t, barr_t)
        h[f'tag_match_misid_{sign}_1p0p']     = ROOT.TH1D(
            f'tag_match_misid_{sign}_1p0p',     '', nb_t, barr_t)
        h[f'tag_match_total_{sign}_1p0p_cos'] = ROOT.TH1D(
            f'tag_match_total_{sign}_1p0p_cos', '', nb_c, barr_c)
        h[f'tag_match_misid_{sign}_1p0p_cos'] = ROOT.TH1D(
            f'tag_match_misid_{sign}_1p0p_cos', '', nb_c, barr_c)
        for pcut in PT_CUTS:
            t = f"pt{pcut:.1f}".replace('.', 'p')
            h[f'tag_match_total_{sign}_1p0p_{t}']     = ROOT.TH1D(
                f'tag_match_total_{sign}_1p0p_{t}',     '', nb_t, barr_t)
            h[f'tag_match_misid_{sign}_1p0p_{t}']     = ROOT.TH1D(
                f'tag_match_misid_{sign}_1p0p_{t}',     '', nb_t, barr_t)
            h[f'tag_match_total_{sign}_1p0p_{t}_cos'] = ROOT.TH1D(
                f'tag_match_total_{sign}_1p0p_{t}_cos', '', nb_c, barr_c)
            h[f'tag_match_misid_{sign}_1p0p_{t}_cos'] = ROOT.TH1D(
                f'tag_match_misid_{sign}_1p0p_{t}_cos', '', nb_c, barr_c)
        for pcut in PMAG_CUTS:
            t = f"pmag{pcut:.1f}".replace('.', 'p')
            h[f'tag_match_total_{sign}_1p0p_{t}']     = ROOT.TH1D(
                f'tag_match_total_{sign}_1p0p_{t}',     '', nb_t, barr_t)
            h[f'tag_match_misid_{sign}_1p0p_{t}']     = ROOT.TH1D(
                f'tag_match_misid_{sign}_1p0p_{t}',     '', nb_t, barr_t)
            h[f'tag_match_total_{sign}_1p0p_{t}_cos'] = ROOT.TH1D(
                f'tag_match_total_{sign}_1p0p_{t}_cos', '', nb_c, barr_c)
            h[f'tag_match_misid_{sign}_1p0p_{t}_cos'] = ROOT.TH1D(
                f'tag_match_misid_{sign}_1p0p_{t}_cos', '', nb_c, barr_c)
    h['tag_match_status_1p0p'] = ROOT.TH1D(
        'tag_match_status_1p0p', ';status;1+0 events', 3, 0, 3)
    for i, l in enumerate(['total_1p0p', 'matched', 'unmatched']):
        h['tag_match_status_1p0p'].GetXaxis().SetBinLabel(i + 1, l)

    # ---- dpp sanity-check histograms (1+1 events only) ----
    # Typical LEP charged-track dpp ~ 0.005 to 0.05; LVLOCK imposes an
    # effective upper bound at dpp ~ 1 (ΔE/E < 100%), so the histogram
    # range is extended to 1.0 to show the full distribution including
    # the high-dpp tail that the Part_lock filter caps.
    h['dpp_all_1p1p'] = ROOT.TH1D(
        'dpp_all_1p1p', ';#sigma(p)/p;tracks (1+1)', 200, 0.0, 1.0)
    h['dpp_tag']   = ROOT.TH1D('dpp_tag',   ';#sigma(p)/p;tag tracks',   200, 0.0, 1.0)
    h['dpp_probe'] = ROOT.TH1D('dpp_probe', ';#sigma(p)/p;probe tracks', 200, 0.0, 1.0)
    # vs theta: will show the TPC crack at 90 deg if the computation is correct
    h['dpp_vs_theta_all_1p1p']     = ROOT.TH2D(
        'dpp_vs_theta_all_1p1p',     ';#theta [deg];#sigma(p)/p',
        nb_t, barr_t, 200, 0.0, 1.0)
    h['dpp_vs_theta_tag']           = ROOT.TH2D(
        'dpp_vs_theta_tag',           ';#theta [deg];#sigma(p)/p (tag)',
        nb_t, barr_t, 200, 0.0, 1.0)
    h['dpp_vs_theta_probe']         = ROOT.TH2D(
        'dpp_vs_theta_probe',         ';#theta [deg];#sigma(p)/p (probe)',
        nb_t, barr_t, 200, 0.0, 1.0)
    h['dpp_vs_theta_all_1p1p_cos'] = ROOT.TH2D(
        'dpp_vs_theta_all_1p1p_cos', ';cos#theta;#sigma(p)/p',
        nb_c, barr_c, 200, 0.0, 1.0)
    h['dpp_vs_theta_tag_cos']       = ROOT.TH2D(
        'dpp_vs_theta_tag_cos',       ';cos#theta;#sigma(p)/p (tag)',
        nb_c, barr_c, 200, 0.0, 1.0)
    h['dpp_vs_theta_probe_cos']     = ROOT.TH2D(
        'dpp_vs_theta_probe_cos',     ';cos#theta;#sigma(p)/p (probe)',
        nb_c, barr_c, 200, 0.0, 1.0)
    # Count of events rejected because both dpp invalid, and fraction of valid dpp
    h['dpp_status'] = ROOT.TH1D('dpp_status', ';status;1+1 events', 4, 0, 4)
    for i, l in enumerate(['total_1p1p', 'both_valid', 'one_invalid', 'both_invalid']):
        h['dpp_status'].GetXaxis().SetBinLabel(i + 1, l)

    # --- Impact-parameter diagnostics (d0 / z0 / z0·sinθ) ---
    # Per selected charged track, split by charge (pos / neg / all). Same
    # binning as analysis_fbcec.py (d0_vtx / z0_vtx / z0sin_vtx) for direct
    # overlay. MC also gets the z0sin_charge-reweighted companions for d0 and
    # z0sinθ (suffix `_rw_z0sin_charge`) — mirrors the FBCEC closure set. These
    # are NOT caught by the `tag_`/`probe_` auto-clone below, so they're booked
    # explicitly here.
    for sign in ('pos', 'neg', 'all'):
        h[f'd0_vtx_{sign}']    = ROOT.TH1D(f'd0_vtx_{sign}',    ';d_{0} [cm];tracks',
            D0_VTX_NBINS,    D0_VTX_RANGE[0],    D0_VTX_RANGE[1])
        h[f'z0_vtx_{sign}']    = ROOT.TH1D(f'z0_vtx_{sign}',    ';z_{0} [cm];tracks',
            Z0_VTX_NBINS,    Z0_VTX_RANGE[0],    Z0_VTX_RANGE[1])
        h[f'z0sin_vtx_{sign}'] = ROOT.TH1D(f'z0sin_vtx_{sign}', ';z_{0}sin#theta [cm];tracks',
            Z0SIN_VTX_NBINS, Z0SIN_VTX_RANGE[0], Z0SIN_VTX_RANGE[1])
        h[f'd0_vtx_{sign}{RW_TAG}']    = ROOT.TH1D(f'd0_vtx_{sign}{RW_TAG}',    ';d_{0} [cm];tracks',
            D0_VTX_NBINS,    D0_VTX_RANGE[0],    D0_VTX_RANGE[1])
        h[f'z0sin_vtx_{sign}{RW_TAG}'] = ROOT.TH1D(f'z0sin_vtx_{sign}{RW_TAG}', ';z_{0}sin#theta [cm];tracks',
            Z0SIN_VTX_NBINS, Z0SIN_VTX_RANGE[0], Z0SIN_VTX_RANGE[1])

    # --- z0sin-charge-reweighted companions for every reco-level MC quantity ---
    # Per-track w(pT, z0sinθ, q) is multiplied at fill time. Suffix `_rw_z0sin_charge`.
    # Cloned 1:1 from the existing TH1D histograms whose names start with
    # "tag_" or "probe_" — i.e. the per-track and tag/probe quantities. We
    # skip the categorical `*_match_status` histograms (3-bin status flags
    # don't need reweighting).
    for nm in list(h.keys()):
        if not (nm.startswith('tag_') or nm.startswith('probe_')):
            continue
        if nm.endswith('_match_status'):
            continue
        if not isinstance(h[nm], ROOT.TH1D):
            continue
        rw_nm = nm + RW_TAG
        h[rw_nm] = h[nm].Clone(rw_nm)
        h[rw_nm].Reset()
        h[rw_nm].SetDirectory(0)

    # --- dpp<0.5 cut variant: clone every histogram into a `_dpp0p5` partner ---
    # The second analysis pass uses the dpp<0.5-restricted track set; the
    # partner hists are filled in that pass with the `suffix='_dpp0p5'` arg
    # threaded through the fill functions. Skip status/label hists and the
    # already-cloned partners.
    SKIP_DPP_CLONE = {'cutflow', 'dpp_status',
                      'tag_match_status', 'probe_match_status',
                      'tag_match_status_1p0p'}
    for nm in list(h.keys()):
        if nm in SKIP_DPP_CLONE:
            continue
        # IP diagnostics are filled directly in the main loop (not threaded
        # through the `_dpp0p5` suffixed-dict), so a cloned partner would stay
        # empty — skip them.
        if nm.startswith(('d0_vtx_', 'z0_vtx_', 'z0sin_vtx_')):
            continue
        if nm.endswith('_dpp0p5'):
            continue
        dpp_nm = nm + '_dpp0p5'
        h[dpp_nm] = h[nm].Clone(dpp_nm)
        h[dpp_nm].Reset()
        h[dpp_nm].SetDirectory(0)

    return h


# ============================================================================
# TAG-AND-PROBE FILLING
# ============================================================================

def fill_1p1p(h, tag_q, tag_theta, tag_pt, tag_p, probe_q, probe_theta, probe_pt, probe_p,
              gen_topo=None, w_tag=None, w_probe=None, suffix=''):
    """Fill 1+1 tag-probe conditional histograms.
    Tag is required to be in barrel [40, 140] before calling this function.
    Probe can be at any angle.

    gen_topo (optional): one of '1p1p', '1p3p', 'other' (MC only). When given,
    TNP histograms with the matching `_gen{topo}` suffix are also filled.

    w_tag, w_probe (optional, MC only): per-track z0sin_charge weights. If
    BOTH are given, the parallel `_rw_z0sin_charge` companion histograms are
    filled. Tag-only histos use w_tag, probe-only use w_probe, joint
    (tag-probe denominator/numerator binned in probe theta) use w_tag*w_probe.

    suffix (optional): appended to every hist name (e.g. `_dpp0p5`) so the
    same code path drives the parallel `_dpp0p5` variant.
    """
    if suffix:
        h = _SuffixedDict(h, suffix)
    ts = 'pos' if tag_q > 0 else 'neg'
    ps = 'pos' if probe_q > 0 else 'neg'
    do_rw = (w_tag is not None and w_probe is not None)
    w_tp = (w_tag * w_probe) if do_rw else 0.0  # joint weight
    tag_ct   = _cos_th(tag_theta)
    probe_ct = _cos_th(probe_theta)

    # Inclusive kinematic histograms (no pT threshold)
    h[f'tag_pt_{ts}'].Fill(tag_pt)
    h[f'tag_p_{ts}'].Fill(tag_p)
    h[f'tag_theta_{ts}'].Fill(tag_theta)
    h[f'tag_theta_{ts}_cos'].Fill(tag_ct)
    h[f'probe_pt_{ps}'].Fill(probe_pt)
    h[f'probe_p_{ps}'].Fill(probe_p)
    h[f'probe_theta_{ps}'].Fill(probe_theta)
    h[f'probe_theta_{ps}_cos'].Fill(probe_ct)
    if do_rw:
        h[f'tag_pt_{ts}{RW_TAG}'].Fill(tag_pt, w_tag)
        h[f'tag_p_{ts}{RW_TAG}'].Fill(tag_p, w_tag)
        h[f'tag_theta_{ts}{RW_TAG}'].Fill(tag_theta, w_tag)
        h[f'tag_theta_{ts}_cos{RW_TAG}'].Fill(tag_ct, w_tag)
        h[f'probe_pt_{ps}{RW_TAG}'].Fill(probe_pt, w_probe)
        h[f'probe_p_{ps}{RW_TAG}'].Fill(probe_p, w_probe)
        h[f'probe_theta_{ps}{RW_TAG}'].Fill(probe_theta, w_probe)
        h[f'probe_theta_{ps}_cos{RW_TAG}'].Fill(probe_ct, w_probe)

    # Topology-specific
    h[f'tag_pt_{ts}_1p1p'].Fill(tag_pt)
    h[f'tag_p_{ts}_1p1p'].Fill(tag_p)
    h[f'tag_theta_{ts}_1p1p'].Fill(tag_theta)
    h[f'tag_theta_{ts}_1p1p_cos'].Fill(tag_ct)
    h[f'probe_pt_{ps}_1p1p'].Fill(probe_pt)
    h[f'probe_p_{ps}_1p1p'].Fill(probe_p)
    h[f'probe_theta_{ps}_1p1p'].Fill(probe_theta)
    h[f'probe_theta_{ps}_1p1p_cos'].Fill(probe_ct)
    if do_rw:
        h[f'tag_pt_{ts}_1p1p{RW_TAG}'].Fill(tag_pt, w_tag)
        h[f'tag_p_{ts}_1p1p{RW_TAG}'].Fill(tag_p, w_tag)
        h[f'tag_theta_{ts}_1p1p{RW_TAG}'].Fill(tag_theta, w_tag)
        h[f'tag_theta_{ts}_1p1p_cos{RW_TAG}'].Fill(tag_ct, w_tag)
        h[f'probe_pt_{ps}_1p1p{RW_TAG}'].Fill(probe_pt, w_probe)
        h[f'probe_p_{ps}_1p1p{RW_TAG}'].Fill(probe_p, w_probe)
        h[f'probe_theta_{ps}_1p1p{RW_TAG}'].Fill(probe_theta, w_probe)
        h[f'probe_theta_{ps}_1p1p_cos{RW_TAG}'].Fill(probe_ct, w_probe)

    # Gen-topology splits of the per-track kinematic shapes (inclusive +
    # _1p1p, identical values; see booking comment). MC only.
    if gen_topo is not None:
        gs = f'_gen{gen_topo}'
        for base in ('', '_1p1p'):
            h[f'tag_pt_{ts}{base}{gs}'].Fill(tag_pt)
            h[f'tag_p_{ts}{base}{gs}'].Fill(tag_p)
            h[f'tag_theta_{ts}{base}{gs}'].Fill(tag_theta)
            h[f'tag_theta_{ts}{base}{gs}_cos'].Fill(tag_ct)
            h[f'probe_pt_{ps}{base}{gs}'].Fill(probe_pt)
            h[f'probe_p_{ps}{base}{gs}'].Fill(probe_p)
            h[f'probe_theta_{ps}{base}{gs}'].Fill(probe_theta)
            h[f'probe_theta_{ps}{base}{gs}_cos'].Fill(probe_ct)

    gtopo_suffixes = ('',) if gen_topo is None else ('', f'_gen{gen_topo}')

    for pcut in PT_CUTS:
        if probe_pt > pcut:
            t = f"pt{pcut:.1f}".replace('.', 'p')
            for gs in gtopo_suffixes:
                h[f'tag_{ts}_{t}{gs}'].Fill(probe_theta)
                h[f'probe_{ps}_tag_{ts}_{t}{gs}'].Fill(probe_theta)
                h[f'tag_{ts}_{t}{gs}_cos'].Fill(probe_ct)
                h[f'probe_{ps}_tag_{ts}_{t}{gs}_cos'].Fill(probe_ct)
                if do_rw:
                    h[f'tag_{ts}_{t}{gs}{RW_TAG}'].Fill(probe_theta, w_tp)
                    h[f'probe_{ps}_tag_{ts}_{t}{gs}{RW_TAG}'].Fill(probe_theta, w_tp)
                    h[f'tag_{ts}_{t}{gs}_cos{RW_TAG}'].Fill(probe_ct, w_tp)
                    h[f'probe_{ps}_tag_{ts}_{t}{gs}_cos{RW_TAG}'].Fill(probe_ct, w_tp)

    for pcut in PMAG_CUTS:
        if probe_pt > 0.4 and probe_p > pcut:
            t = f"pmag{pcut:.1f}".replace('.', 'p')
            for gs in gtopo_suffixes:
                h[f'tag_{ts}_{t}{gs}'].Fill(probe_theta)
                h[f'probe_{ps}_tag_{ts}_{t}{gs}'].Fill(probe_theta)
                h[f'tag_{ts}_{t}{gs}_cos'].Fill(probe_ct)
                h[f'probe_{ps}_tag_{ts}_{t}{gs}_cos'].Fill(probe_ct)
                if do_rw:
                    h[f'tag_{ts}_{t}{gs}{RW_TAG}'].Fill(probe_theta, w_tp)
                    h[f'probe_{ps}_tag_{ts}_{t}{gs}{RW_TAG}'].Fill(probe_theta, w_tp)
                    h[f'tag_{ts}_{t}{gs}_cos{RW_TAG}'].Fill(probe_ct, w_tp)
                    h[f'probe_{ps}_tag_{ts}_{t}{gs}_cos{RW_TAG}'].Fill(probe_ct, w_tp)


def fill_1p0p(h, q_surv, theta_surv_deg, pt_surv, p_surv, w_surv=None, suffix=''):
    """Fill 1+0 control region histograms (events with exactly one charged
    track passing track selection — i.e. one tau prong reconstructed, the
    other side missing).

    Stores only the SURVIVING prong's kinematics (per charge). Combined
    offline with the 1+1 `tag_<sign>_<t>` denominators to give a per-charge
    tracking-efficiency control region.

    The pT/|p| cuts are applied to the surviving track itself; under the
    back-to-back tau-pair assumption this is a reasonable proxy for the
    missed probe momentum (exact for τ→πν, smeared for ρ/a₁).

    If `w_surv` is given (MC z0sin_charge per-track weight), the parallel
    `_rw_z0sin_charge` companion histograms are also filled.

    suffix (optional): appended to every hist name (`_dpp0p5` variant).
    """
    if suffix:
        h = _SuffixedDict(h, suffix)
    ts = 'pos' if q_surv > 0 else 'neg'
    ct = _cos_th(theta_surv_deg)

    # Inclusive kinematics
    h[f'tag_pt_{ts}_1p0p'].Fill(pt_surv)
    h[f'tag_p_{ts}_1p0p'].Fill(p_surv)
    h[f'tag_theta_{ts}_1p0p'].Fill(theta_surv_deg)
    h[f'tag_theta_{ts}_1p0p_cos'].Fill(ct)
    if w_surv is not None:
        h[f'tag_pt_{ts}_1p0p{RW_TAG}'].Fill(pt_surv, w_surv)
        h[f'tag_p_{ts}_1p0p{RW_TAG}'].Fill(p_surv, w_surv)
        h[f'tag_theta_{ts}_1p0p{RW_TAG}'].Fill(theta_surv_deg, w_surv)
        h[f'tag_theta_{ts}_1p0p_cos{RW_TAG}'].Fill(ct, w_surv)

    # Per-pT cut, theta-binned
    for pcut in PT_CUTS:
        if pt_surv > pcut:
            t = f"pt{pcut:.1f}".replace('.', 'p')
            h[f'tag_{ts}_1p0p_{t}'].Fill(theta_surv_deg)
            h[f'tag_{ts}_1p0p_{t}_cos'].Fill(ct)
            if w_surv is not None:
                h[f'tag_{ts}_1p0p_{t}{RW_TAG}'].Fill(theta_surv_deg, w_surv)
                h[f'tag_{ts}_1p0p_{t}_cos{RW_TAG}'].Fill(ct, w_surv)

    # Per-|p| cut (with baseline pT > 0.4), theta-binned
    for pcut in PMAG_CUTS:
        if pt_surv > 0.4 and p_surv > pcut:
            t = f"pmag{pcut:.1f}".replace('.', 'p')
            h[f'tag_{ts}_1p0p_{t}'].Fill(theta_surv_deg)
            h[f'tag_{ts}_1p0p_{t}_cos'].Fill(ct)
            if w_surv is not None:
                h[f'tag_{ts}_1p0p_{t}{RW_TAG}'].Fill(theta_surv_deg, w_surv)
                h[f'tag_{ts}_1p0p_{t}_cos{RW_TAG}'].Fill(ct, w_surv)


def fill_1p0p_match(h, q_reco, theta_s_deg, phi_s_deg, pt_s, p_s,
                    gen_q, gen_theta, gen_phi, suffix=''):
    """Per-charge truth match for the 1+0 surviving prong (MC only).

    Matches the surviving track to the nearest gen charged FS within
    ΔR < MATCH_DR (3D opening angle). Fills `tag_match_*_1p0p` histos
    parallel to the 1+1 per-charge truth-match convention: sign label uses
    the GEN charge, denominator = all matched tracks of that gen sign,
    numerator = subset with reco_q * gen_q < 0.

    suffix (optional): appended to every hist name (`_dpp0p5` variant).
    """
    if suffix:
        h = _SuffixedDict(h, suffix)
    h['tag_match_status_1p0p'].Fill(0.5)  # total_1p0p
    gi = _match_to_gen(theta_s_deg, phi_s_deg, gen_theta, gen_phi)
    if gi < 0:
        h['tag_match_status_1p0p'].Fill(2.5)  # unmatched
        return
    h['tag_match_status_1p0p'].Fill(1.5)  # matched
    gen_sign = 'pos' if gen_q[gi] > 0 else 'neg'
    misid = q_reco * gen_q[gi] < 0
    ct_s = _cos_th(theta_s_deg)

    h[f'tag_match_total_{gen_sign}_1p0p'].Fill(theta_s_deg)
    h[f'tag_match_total_{gen_sign}_1p0p_cos'].Fill(ct_s)
    if misid:
        h[f'tag_match_misid_{gen_sign}_1p0p'].Fill(theta_s_deg)
        h[f'tag_match_misid_{gen_sign}_1p0p_cos'].Fill(ct_s)

    for pcut in PT_CUTS:
        if pt_s > pcut:
            t = f"pt{pcut:.1f}".replace('.', 'p')
            h[f'tag_match_total_{gen_sign}_1p0p_{t}'].Fill(theta_s_deg)
            h[f'tag_match_total_{gen_sign}_1p0p_{t}_cos'].Fill(ct_s)
            if misid:
                h[f'tag_match_misid_{gen_sign}_1p0p_{t}'].Fill(theta_s_deg)
                h[f'tag_match_misid_{gen_sign}_1p0p_{t}_cos'].Fill(ct_s)

    for pcut in PMAG_CUTS:
        if pt_s > 0.4 and p_s > pcut:
            t = f"pmag{pcut:.1f}".replace('.', 'p')
            h[f'tag_match_total_{gen_sign}_1p0p_{t}'].Fill(theta_s_deg)
            h[f'tag_match_total_{gen_sign}_1p0p_{t}_cos'].Fill(ct_s)
            if misid:
                h[f'tag_match_misid_{gen_sign}_1p0p_{t}'].Fill(theta_s_deg)
                h[f'tag_match_misid_{gen_sign}_1p0p_{t}_cos'].Fill(ct_s)


def _match_to_gen(theta_deg, phi_deg, gen_theta_deg, gen_phi_deg, max_dr=MATCH_DR):
    """Nearest-neighbour match by 3D opening angle (radians). Returns gen
    index of the closest gen partner if its opening angle is below `max_dr`,
    else -1. Mirrors the cap convention used by `match_angular` in
    common_functions.py."""
    n = len(gen_theta_deg)
    if n == 0:
        return -1
    th_r = math.radians(theta_deg)
    ph_r = math.radians(phi_deg)
    s, c = math.sin(th_r), math.cos(th_r)
    x, y, z = s * math.cos(ph_r), s * math.sin(ph_r), c
    gth = np.deg2rad(np.asarray(gen_theta_deg, dtype=np.float64))
    gph = np.deg2rad(np.asarray(gen_phi_deg,   dtype=np.float64))
    gs  = np.sin(gth)
    gx, gy, gz = gs * np.cos(gph), gs * np.sin(gph), np.cos(gth)
    cosang = np.clip(x * gx + y * gy + z * gz, -1.0, 1.0)
    angs = np.arccos(cosang)
    i_best = int(np.argmin(angs))
    if angs[i_best] >= max_dr:
        return -1
    return i_best


def fill_tag_and_probe(h, nf, nb, fq, ft, fp, fpt, fdpp, bq, bt, bp, bpt, bdpp,
                       fpx=None, fpy=None, fpz=None,
                       bpx=None, bpy=None, bpz=None,
                       is_mc=False,
                       gen_q=None, gen_theta=None, gen_phi=None,
                       gen_topo=None,
                       fw_zc=None, bw_zc=None,
                       suffix=''):
    """Fill 1+1 T&P. The `suffix` arg is forwarded so the same code path
    drives the `_dpp0p5` variant."""
    if suffix:
        h = _SuffixedDict(h, suffix)
    if nf == 1 and nb == 1:
        h['cutflow'].Fill(8.5)
        h['charge_sum'].Fill(fq[0] + bq[0])

        # dpp sanity: fill for every 1+1 track regardless of validity
        h['dpp_status'].Fill(0.5)  # total_1p1p
        f_ct = _cos_th(ft[0])
        b_ct = _cos_th(bt[0])
        if fdpp[0] >= 0:
            h['dpp_all_1p1p'].Fill(fdpp[0])
            h['dpp_vs_theta_all_1p1p'].Fill(ft[0], fdpp[0])
            h['dpp_vs_theta_all_1p1p_cos'].Fill(f_ct, fdpp[0])
        if bdpp[0] >= 0:
            h['dpp_all_1p1p'].Fill(bdpp[0])
            h['dpp_vs_theta_all_1p1p'].Fill(bt[0], bdpp[0])
            h['dpp_vs_theta_all_1p1p_cos'].Fill(b_ct, bdpp[0])
        nvalid = int(fdpp[0] >= 0) + int(bdpp[0] >= 0)
        if nvalid == 2:
            h['dpp_status'].Fill(1.5)  # both_valid
        elif nvalid == 1:
            h['dpp_status'].Fill(2.5)  # one_invalid
        else:
            h['dpp_status'].Fill(3.5)  # both_invalid

        # New control region:
        #   (1) At least one track must be in the barrel [40, 140]. With
        #       acollinearity < 20 deg, this forces the other track into
        #       [20, 160], where the 1p/3p topology classification is still
        #       reliable (very low probability of losing 2 of 3 prongs).
        #   (2) Among the two tracks, whichever has the smaller (better) dpp
        #       is the tag; the other is the probe. dpp < 0 means the fit
        #       didn't constrain the curvature (pathological) and is treated
        #       as worst quality so the other track wins automatically.
        f_in_barrel = 40 <= ft[0] <= 140
        b_in_barrel = 40 <= bt[0] <= 140

        if not (f_in_barrel or b_in_barrel):
            return  # neither in barrel: topology not trustworthy, skip

        # Pick tag = smaller dpp. Invalid dpp (<0) treated as +infinity.
        fdpp0 = fdpp[0] if fdpp[0] >= 0 else float('inf')
        bdpp0 = bdpp[0] if bdpp[0] >= 0 else float('inf')

        if fdpp0 == float('inf') and bdpp0 == float('inf'):
            return  # both pathological, skip

        tag_is_forward = fdpp0 <= bdpp0
        if tag_is_forward:
            # forward track is the tag (smaller dpp wins)
            if fdpp[0] >= 0:
                h['dpp_tag'].Fill(fdpp[0])
                h['dpp_vs_theta_tag'].Fill(ft[0], fdpp[0])
                h['dpp_vs_theta_tag_cos'].Fill(f_ct, fdpp[0])
            if bdpp[0] >= 0:
                h['dpp_probe'].Fill(bdpp[0])
                h['dpp_vs_theta_probe'].Fill(bt[0], bdpp[0])
                h['dpp_vs_theta_probe_cos'].Fill(b_ct, bdpp[0])
            w_tag_zc   = float(fw_zc[0]) if fw_zc is not None else None
            w_probe_zc = float(bw_zc[0]) if bw_zc is not None else None
            fill_1p1p(h, fq[0], ft[0], fpt[0], fp[0], bq[0], bt[0], bpt[0], bp[0],
                      gen_topo=gen_topo, w_tag=w_tag_zc, w_probe=w_probe_zc)
            tag_q_reco = fq[0]; tag_theta = ft[0]; tag_pt = fpt[0]; tag_p_mag = fp[0]
            tag_px = fpx[0] if fpx is not None else 0.0
            tag_py = fpy[0] if fpy is not None else 0.0
            probe_q_reco = bq[0]; probe_theta = bt[0]; probe_pt = bpt[0]; probe_p_mag = bp[0]
            probe_px = bpx[0] if bpx is not None else 0.0
            probe_py = bpy[0] if bpy is not None else 0.0
        else:
            # backward track is the tag (smaller dpp wins)
            if bdpp[0] >= 0:
                h['dpp_tag'].Fill(bdpp[0])
                h['dpp_vs_theta_tag'].Fill(bt[0], bdpp[0])
                h['dpp_vs_theta_tag_cos'].Fill(b_ct, bdpp[0])
            if fdpp[0] >= 0:
                h['dpp_probe'].Fill(fdpp[0])
                h['dpp_vs_theta_probe'].Fill(ft[0], fdpp[0])
                h['dpp_vs_theta_probe_cos'].Fill(f_ct, fdpp[0])
            w_tag_zc   = float(bw_zc[0]) if bw_zc is not None else None
            w_probe_zc = float(fw_zc[0]) if fw_zc is not None else None
            fill_1p1p(h, bq[0], bt[0], bpt[0], bp[0], fq[0], ft[0], fpt[0], fp[0],
                      gen_topo=gen_topo, w_tag=w_tag_zc, w_probe=w_probe_zc)
            tag_q_reco = bq[0]; tag_theta = bt[0]; tag_pt = bpt[0]; tag_p_mag = bp[0]
            tag_px = bpx[0] if bpx is not None else 0.0
            tag_py = bpy[0] if bpy is not None else 0.0
            probe_q_reco = fq[0]; probe_theta = ft[0]; probe_pt = fpt[0]; probe_p_mag = fp[0]
            probe_px = fpx[0] if fpx is not None else 0.0
            probe_py = fpy[0] if fpy is not None else 0.0

        # MC tag/probe charge truth match: angular match each side to a gen
        # charged final-state particle and record reco vs gen charge.
        if is_mc and gen_q is not None and len(gen_q) > 0:
            gtopo_suffixes = ('',) if gen_topo is None else ('', f'_gen{gen_topo}')
            for side, q_reco, theta_s, pt_s, p_s, px_s, py_s in (
                ('tag',   tag_q_reco,   tag_theta,   tag_pt,   tag_p_mag,   tag_px,   tag_py),
                ('probe', probe_q_reco, probe_theta, probe_pt, probe_p_mag, probe_px, probe_py),
            ):
                h[f'{side}_match_status'].Fill(0.5)  # total_1p1p
                phi_s = math.degrees(math.atan2(py_s, px_s))
                gi = _match_to_gen(theta_s, phi_s, gen_theta, gen_phi)
                if gi >= 0:
                    h[f'{side}_match_status'].Fill(1.5)  # matched
                    gen_sign = 'pos' if gen_q[gi] > 0 else 'neg'
                    misid = q_reco * gen_q[gi] < 0
                    ct_s = _cos_th(theta_s)
                    for gs in gtopo_suffixes:
                        h[f'{side}_match_total_{gen_sign}{gs}'].Fill(theta_s)
                        h[f'{side}_match_total_{gen_sign}{gs}_cos'].Fill(ct_s)
                        if misid:
                            h[f'{side}_match_misid_{gen_sign}{gs}'].Fill(theta_s)
                            h[f'{side}_match_misid_{gen_sign}{gs}_cos'].Fill(ct_s)
                    # TNP-style per-cut: gate on side's own pT/|p|, bin in side theta
                    for pcut in PT_CUTS:
                        if pt_s > pcut:
                            t = f"pt{pcut:.1f}".replace('.', 'p')
                            for gs in gtopo_suffixes:
                                h[f'{side}_match_total_{gen_sign}_{t}{gs}'].Fill(theta_s)
                                h[f'{side}_match_total_{gen_sign}_{t}{gs}_cos'].Fill(ct_s)
                                if misid:
                                    h[f'{side}_match_misid_{gen_sign}_{t}{gs}'].Fill(theta_s)
                                    h[f'{side}_match_misid_{gen_sign}_{t}{gs}_cos'].Fill(ct_s)
                    for pcut in PMAG_CUTS:
                        if pt_s > 0.4 and p_s > pcut:
                            t = f"pmag{pcut:.1f}".replace('.', 'p')
                            for gs in gtopo_suffixes:
                                h[f'{side}_match_total_{gen_sign}_{t}{gs}'].Fill(theta_s)
                                h[f'{side}_match_total_{gen_sign}_{t}{gs}_cos'].Fill(ct_s)
                                if misid:
                                    h[f'{side}_match_misid_{gen_sign}_{t}{gs}'].Fill(theta_s)
                                    h[f'{side}_match_misid_{gen_sign}_{t}{gs}_cos'].Fill(ct_s)
                else:
                    h[f'{side}_match_status'].Fill(2.5)  # unmatched



# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('infile')
    parser.add_argument('outfile')
    parser.add_argument('--is-mc', action='store_true')
    parser.add_argument('--max-events', type=int, default=-1)
    parser.add_argument('--max-nch', type=int, default=4)
    parser.add_argument('--min-thrust', type=float, default=0.985)
    parser.add_argument('--max-acol', type=float, default=20.0)
    parser.add_argument('--max-evis', type=float, default=0.80)
    parser.add_argument('--min-evis', type=float, default=0.10)
    parser.add_argument('--min-mvis', type=float, default=15.0)
    parser.add_argument('--max-mvis', type=float, default=55.0)
    parser.add_argument('--period', choices=['auto', '94c', '95d'], default='auto',
                        help='Data-taking period for vertex reweighting; auto-detects from infile path.')
    parser.add_argument('--reweight-dir', default='.',
                        help='Directory containing vertex_reweight_<period>_v73_3d.root '
                             '(loaded only on MC).')
    args = parser.parse_args()

    cuts = dict(max_charged_mult=args.max_nch, min_thrust=args.min_thrust,
                max_acollinearity=args.max_acol,
                max_evis_frac=args.max_evis, min_evis_frac=args.min_evis,
                min_mvis=args.min_mvis, max_mvis=args.max_mvis)

    print(f"=== Tau Tag-and-Probe Analysis ===")
    print(f"Input:  {args.infile}\nOutput: {args.outfile}\nMC: {args.is_mc}")
    print(f"Cuts: {cuts}")

    # Vertex reweighter (MC only). Per-track w(pT, θ, z0sinθ, q) from the
    # v73 3D weight TH3F tables. Used to fill `_rw_z0sin_charge` companion
    # histograms for data/MC closure study.
    period = args.period if args.period != 'auto' else rw_detect_period(args.infile)
    vtx_rw = None
    if args.is_mc:
        if period is None:
            print("WARNING: cannot detect period from infile; vertex reweighting disabled.")
        else:
            table_path = find_3d_table(period, search_dir=args.reweight_dir)
            if table_path is None:
                print(f"WARNING: vertex_reweight_{period}_v73_3d.root not found "
                      f"in {args.reweight_dir}; reweighting disabled")
            else:
                try:
                    vtx_rw = VertexReweighter3D(table_path)
                    print(f"Vertex reweighter loaded from {table_path}; period={period}")
                except Exception as exc:
                    print(f"WARNING: vertex reweighter disabled — {exc}")
                    vtx_rw = None

    data = load_all_events(args.infile, args.max_events)
    nevt = len(data["px"])
    print(f"Loaded {nevt} events")

    ecm_arr = np.asarray(data.get("Event_cmEnergy", np.full(nevt, 91.25)))
    has_trac = "Trac_d0" in data
    has_gen = "GenPart_pdgId" in data
    has_muid = "Muid_tag" in data and "Muid_partIdx" in data
    has_elid = "Elid_tag" in data and "Elid_partIdx" in data

    hists = book_histograms()
    n_pass = 0
    n_zpole_cut = 0

    for ievt in range(nevt):
        if ievt % 5000 == 0:
            print(f"  Event {ievt}/{nevt}  (passed: {n_pass})")

        ecm = float(ecm_arr[ievt])

        # Z-pole only: reject off-peak runs (LEP1 scan points, LEP2, etc.)
        # Applied before cutflow so off-pole events are not counted at all.
        if abs(ecm - 91.25) > 1.0:
            n_zpole_cut += 1
            continue

        hists['cutflow'].Fill(0.5)

        px     = np.asarray(data["px"][ievt], dtype=np.float64)
        py     = np.asarray(data["py"][ievt], dtype=np.float64)
        pz     = np.asarray(data["pz"][ievt], dtype=np.float64)
        en     = np.asarray(data["en"][ievt], dtype=np.float64)
        charge = np.asarray(data["charge"][ievt], dtype=np.float64)
        lock   = np.asarray(data["lock"][ievt], dtype=np.int32)

        npart = len(px)
        if npart == 0: continue

        good = lock == 0
        if np.sum(good) < 2: continue
        hists['cutflow'].Fill(1.5)

        px_g, py_g, pz_g, en_g, q_g = px[good], py[good], pz[good], en[good], charge[good]
        pt_g = np.sqrt(px_g**2 + py_g**2)
        p_g  = np.sqrt(px_g**2 + py_g**2 + pz_g**2)
        th_g = np.arctan2(pt_g, pz_g)

        # d0/z0 from Trac banks, and dpp computed from perigee + weight matrix.
        # DELPHI storage (empirical): the 5-element "perigee" has per[0]=0
        # (d0 fixed at perigee), and the 4 real parameters are per[1..4].
        # The 15-element weightMatrix is a lower-triangle row-major
        # representation whose real diagonals sit at indices 1, 3, 6, 10
        # (corresponding to the 4 real parameters).
        #
        # The CURVATURE parameter is per[:, 1] and its weight is wm[:, 1]:
        # identified by strong negative log-log correlation with |p| and a
        # dpp median around ~0.007 (physical for LEP1 charged tracks).
        #
        #   dpp = sigma(curvature) / |curvature|
        #       = (1/sqrt(W11)) / |per[1]|
        #
        # Invalid (fit not available / dummy) flagged as -1.
        # uproot reshapes these as (nTrac, 5) and (nTrac, 15).
        d0_g  = np.full(len(px_g), 999.0)
        z0_g  = np.full(len(px_g), 999.0)
        dpp_g = np.full(len(px_g), -1.0)
        has_dpp = ("Trac_perigee_fArray" in data and
                   "Trac_weightMatrix_fRep_fArray" in data)
        if has_trac:
            ti = np.asarray(data["Trac_partIdx"][ievt], dtype=np.int32)
            td = np.asarray(data["Trac_d0"][ievt], dtype=np.float64)
            tz = np.asarray(data["Trac_z0"][ievt], dtype=np.float64)
            ntrac = len(ti)
            if has_dpp and ntrac > 0:
                per = np.asarray(data["Trac_perigee_fArray"][ievt], dtype=np.float64)
                wm  = np.asarray(data["Trac_weightMatrix_fRep_fArray"][ievt], dtype=np.float64)
                if per.ndim == 1 and per.size == 5 * ntrac:
                    per = per.reshape(ntrac, 5)
                if wm.ndim == 1 and wm.size == 15 * ntrac:
                    wm = wm.reshape(ntrac, 15)
                if per.shape == (ntrac, 5) and wm.shape == (ntrac, 15):
                    curv = per[:, 1]     # curvature parameter
                    w_cc = wm[:, 1]      # diagonal weight for curvature
                    t_dpp = np.full(ntrac, -1.0)
                    ok = (np.abs(curv) > 1e-10) & (w_cc > 0)
                    t_dpp[ok] = (1.0 / np.sqrt(w_cc[ok])) / np.abs(curv[ok])
                else:
                    t_dpp = np.full(ntrac, -1.0)
            else:
                t_dpp = np.full(ntrac, -1.0)

            gi = np.where(good)[0]
            gm = {int(o): ig for ig, o in enumerate(gi)}
            for k in range(len(ti)):
                oi = int(ti[k])
                if oi in gm:
                    d0_g[gm[oi]]  = td[k]
                    z0_g[gm[oi]]  = tz[k]
                    dpp_g[gm[oi]] = t_dpp[k]

        z0sin_g = np.abs(z0_g * np.sin(th_g))
        charged = (np.abs(q_g) > 0.5) & (pt_g > 0.4) & (np.abs(d0_g) < 0.6) & (z0sin_g < 1.0)
        # dpp<0.5 cut variant: tighter charged-track selection that drops
        # tracks with poor momentum resolution. Drives the `_dpp0p5` parallel
        # histogram set via a second-pass topology+fill dispatch below.
        charged_dpp0p5 = charged & (dpp_g >= 0) & (dpp_g < 0.5)
        n_ch = int(np.sum(charged))

        # Lower bound 1 (was 2) to keep 1+0 control-region candidates;
        # they branch off after the topology check below.
        if n_ch > cuts['max_charged_mult'] or n_ch < 1: continue
        hists['cutflow'].Fill(2.5)

        e_vis = float(np.sum(en_g))
        evis_frac = e_vis / ecm if ecm > 0 else 0
        if evis_frac > cuts['max_evis_frac'] or evis_frac < cuts['min_evis_frac']: continue
        hists['cutflow'].Fill(3.5)

        mvis2 = e_vis**2 - float(np.sum(px_g))**2 - float(np.sum(py_g))**2 - float(np.sum(pz_g))**2
        mvis_val = math.sqrt(max(mvis2, 0))
        hists['mvis_before_thrust'].Fill(mvis_val)

        thrust_axis, thrust_val = compute_thrust(px_g, py_g, pz_g)
        if thrust_val < cuts['min_thrust']: continue
        hists['cutflow'].Fill(4.5)

        proj = px_g * thrust_axis[0] + py_g * thrust_axis[1] + pz_g * thrust_axis[2]
        hemi_fwd, hemi_bwd = proj > 0, proj <= 0
        nf = int(np.sum(charged & hemi_fwd))
        nb = int(np.sum(charged & hemi_bwd))

        # --- Impact-parameter diagnostics over all selected charged tracks ---
        # d0 / z0 / z0·sinθ (signed) split by charge (pos/neg/all), same binning
        # as analysis_fbcec.py. Filled for every preselected event (after the
        # n_ch / E_vis / thrust cuts), independent of the TNP topology — the
        # FBCEC analog of "all selected charged tracks after event selection".
        # MC additionally fills the z0sin_charge-reweighted companions (d0,
        # z0sinθ) using the per-track w(pT, θ, z0sinθ, q) lookup.
        if np.any(charged):
            d0_ch    = d0_g[charged]
            z0_ch    = z0_g[charged]
            z0sin_ch = z0_g[charged] * np.sin(th_g[charged])
            q_ch     = q_g[charged]
            w_ch = None
            if vtx_rw is not None:
                w_ch = vtx_rw.weights(pt_g[charged], np.degrees(th_g[charged]),
                                      z0sin_ch, q_ch)
            for k in range(len(q_ch)):
                sgn = 'pos' if q_ch[k] > 0 else 'neg'
                hists['d0_vtx_all'].Fill(d0_ch[k])
                hists[f'd0_vtx_{sgn}'].Fill(d0_ch[k])
                hists['z0_vtx_all'].Fill(z0_ch[k])
                hists[f'z0_vtx_{sgn}'].Fill(z0_ch[k])
                hists['z0sin_vtx_all'].Fill(z0sin_ch[k])
                hists[f'z0sin_vtx_{sgn}'].Fill(z0sin_ch[k])
                if w_ch is not None:
                    wk = float(w_ch[k])
                    hists['d0_vtx_all'    + RW_TAG].Fill(d0_ch[k], wk)
                    hists[f'd0_vtx_{sgn}' + RW_TAG].Fill(d0_ch[k], wk)
                    hists['z0sin_vtx_all'    + RW_TAG].Fill(z0sin_ch[k], wk)
                    hists[f'z0sin_vtx_{sgn}' + RW_TAG].Fill(z0sin_ch[k], wk)

        # Gen final-state extraction (MC). Used by both the 1+0 truth match
        # and the 1+1 gen-topo classification + per-side match (the latter
        # via fill_tag_and_probe). Computed once here so both topology paths
        # share the same gen arrays.
        gen_q_fs = gen_th_fs = gen_phi_fs = None
        gpx_fs = gpy_fs = gpz_fs = None
        if args.is_mc and has_gen:
            gpx_a = np.asarray(data["gen_px"][ievt], dtype=np.float64)
            gpy_a = np.asarray(data["gen_py"][ievt], dtype=np.float64)
            gpz_a = np.asarray(data["gen_pz"][ievt], dtype=np.float64)
            gpdg_a = np.asarray(data["GenPart_pdgId"][ievt], dtype=np.int32)
            gst_a  = np.asarray(data["GenPart_status"][ievt], dtype=np.int32)
            gq_a = np.array([pdg_charge(p) for p in gpdg_a], dtype=np.float64)
            gen_fs = (gst_a == 1) & (np.abs(gq_a) > 0.5)
            gpx_fs = gpx_a[gen_fs]; gpy_fs = gpy_a[gen_fs]; gpz_fs = gpz_a[gen_fs]
            gen_q_fs = gq_a[gen_fs]
            gpt_fs = np.sqrt(gpx_fs*gpx_fs + gpy_fs*gpy_fs)
            gen_th_fs = np.degrees(np.arctan2(gpt_fs, gpz_fs))
            gen_phi_fs = np.degrees(np.arctan2(gpy_fs, gpx_fs))

        # ----- 1+0 control region (parallel path) -----
        # Events with one charged tag and a missing probe. Skip the acol /
        # hemisphere-mass / mvis cuts (which assume two reconstructed sides
        # with comparable visible momentum).
        if (nf, nb) in ((1, 0), (0, 1)):
            # Apply electron veto only — kills τ→eνν and Z→ee with one missed.
            # No muon veto here (existing veto is gated on n_ch == 2).
            veto_1p0 = False
            if has_elid:
                el_tag = np.asarray(data["Elid_tag"][ievt], dtype=np.int32)
                n_el_loose = sum(1 for k in range(len(el_tag)) if int(el_tag[k]) >= 3)
                if n_el_loose >= 1:
                    veto_1p0 = True
            if not veto_1p0:
                surv = charged & (hemi_fwd if nf == 1 else hemi_bwd)
                idx = int(np.argmax(surv))  # exactly one True
                # z0sin_charge weight (MC, period detected) — single-track
                # 3D lookup w(pT, θ, z0sinθ, q).
                w_surv = None
                if vtx_rw is not None:
                    z0sin_idx     = float(z0_g[idx] * np.sin(th_g[idx]))
                    theta_deg_idx = float(math.degrees(th_g[idx]))
                    w_surv = float(vtx_rw.weights(
                        np.array([float(pt_g[idx])]),
                        np.array([theta_deg_idx]),
                        np.array([z0sin_idx]),
                        np.array([float(q_g[idx])]),
                    )[0])
                fill_1p0p(
                    hists,
                    q_surv=float(q_g[idx]),
                    theta_surv_deg=math.degrees(th_g[idx]),
                    pt_surv=float(pt_g[idx]),
                    p_surv=float(p_g[idx]),
                    w_surv=w_surv,
                )
                hists['cutflow'].Fill(9.5)  # '1p+0p' bin

                # Per-charge truth match for the surviving prong (MC only).
                if args.is_mc and gen_q_fs is not None and len(gen_q_fs) > 0:
                    phi_idx = math.degrees(math.atan2(
                        float(py_g[idx]), float(px_g[idx])))
                    fill_1p0p_match(
                        hists,
                        q_reco=float(q_g[idx]),
                        theta_s_deg=math.degrees(th_g[idx]),
                        phi_s_deg=phi_idx,
                        pt_s=float(pt_g[idx]),
                        p_s=float(p_g[idx]),
                        gen_q=gen_q_fs, gen_theta=gen_th_fs, gen_phi=gen_phi_fs,
                    )

                # --- dpp<0.5 variant for 1+0 ---
                # If the surviving prong also passes dpp<0.5, fill the
                # `_dpp0p5` partner. Otherwise (prong fails dpp), the
                # event has 0 charged tracks at all in the dpp variant —
                # no fill in either 1+0 or 1+1+ blocks.
                surv_dpp = charged_dpp0p5 & (hemi_fwd if nf == 1 else hemi_bwd)
                if np.any(surv_dpp):
                    idx_d = int(np.argmax(surv_dpp))
                    w_surv_d = None
                    if vtx_rw is not None:
                        z0sin_d = float(z0_g[idx_d] * np.sin(th_g[idx_d]))
                        theta_d = float(math.degrees(th_g[idx_d]))
                        w_surv_d = float(vtx_rw.weights(
                            np.array([float(pt_g[idx_d])]),
                            np.array([theta_d]),
                            np.array([z0sin_d]),
                            np.array([float(q_g[idx_d])]),
                        )[0])
                    fill_1p0p(
                        hists,
                        q_surv=float(q_g[idx_d]),
                        theta_surv_deg=math.degrees(th_g[idx_d]),
                        pt_surv=float(pt_g[idx_d]),
                        p_surv=float(p_g[idx_d]),
                        w_surv=w_surv_d,
                        suffix='_dpp0p5',
                    )
                    if args.is_mc and gen_q_fs is not None and len(gen_q_fs) > 0:
                        phi_id = math.degrees(math.atan2(
                            float(py_g[idx_d]), float(px_g[idx_d])))
                        fill_1p0p_match(
                            hists,
                            q_reco=float(q_g[idx_d]),
                            theta_s_deg=math.degrees(th_g[idx_d]),
                            phi_s_deg=phi_id,
                            pt_s=float(pt_g[idx_d]),
                            p_s=float(p_g[idx_d]),
                            gen_q=gen_q_fs, gen_theta=gen_th_fs, gen_phi=gen_phi_fs,
                            suffix='_dpp0p5',
                        )
            continue

        p3f = np.array([np.sum(px_g[hemi_fwd]), np.sum(py_g[hemi_fwd]), np.sum(pz_g[hemi_fwd])])
        p3b = np.array([np.sum(px_g[hemi_bwd]), np.sum(py_g[hemi_bwd]), np.sum(pz_g[hemi_bwd])])
        mf, mb = np.linalg.norm(p3f), np.linalg.norm(p3b)
        if mf < 1e-6 or mb < 1e-6: continue
        acol = 180.0 - np.degrees(np.arccos(np.clip(np.dot(p3f, p3b) / (mf * mb), -1, 1)))
        if acol > cuts['max_acollinearity']: continue
        hists['cutflow'].Fill(5.5)
        hists['mvis_before_prongs'].Fill(mvis_val)

        if nf not in {1, 3} or nb not in {1, 3}: continue
        hists['cutflow'].Fill(6.5)
        hists['mvis_before_lepveto'].Fill(mvis_val)

        # Lepton veto
        veto = False
        # Muon veto: 2-track events with 2 standard muons
        if has_muid and n_ch == 2:
            mu_tag = np.asarray(data["Muid_tag"][ievt], dtype=np.int32)
            n_mu = sum(1 for k in range(len(mu_tag)) if int(mu_tag[k]) & (1 << 2))
            if n_mu >= 2: veto = True
        # Electron veto: ANY track with Elid >= 3 (loose electron)
        # This kills Z->ee (both tracks are electrons) and tau->e nu nu
        if has_elid and not veto:
            el_tag = np.asarray(data["Elid_tag"][ievt], dtype=np.int32)
            n_el = sum(1 for k in range(len(el_tag)) if int(el_tag[k]) >= 3)
            if n_el >= 1: veto = True
        if veto: continue

        if mvis_val < cuts['min_mvis'] or mvis_val > cuts['max_mvis']: continue
        hists['cutflow'].Fill(7.5)

        n_pass += 1
        hists['thrust'].Fill(thrust_val)
        hists['acol'].Fill(acol)
        hists['evis_frac'].Fill(evis_frac)
        hists['n_charged'].Fill(n_ch)
        hists['mvis'].Fill(mvis_val)

        fc, bc = charged & hemi_fwd, charged & hemi_bwd

        # Gen-topo classification for reco 1+1 events (uses gen FS arrays
        # extracted earlier above, before the 1+0 branch).
        gen_topo = None
        if args.is_mc and has_gen and gpx_fs is not None:
            # Gen topology label for reco 1+1 events: project gen charged FS
            # onto the reco thrust axis to count gen prongs per hemisphere.
            if nf == 1 and nb == 1:
                gproj = (gpx_fs * thrust_axis[0]
                         + gpy_fs * thrust_axis[1]
                         + gpz_fs * thrust_axis[2])
                gen_nf = int(np.sum(gproj > 0))
                gen_nb = int(np.sum(gproj <= 0))
                if gen_nf == 1 and gen_nb == 1:
                    gen_topo = '1p1p'
                    hists['reco_1p1p_gen_topo'].Fill(0.5)  # 1p1p
                elif (gen_nf, gen_nb) in ((1, 3), (3, 1)):
                    gen_topo = '1p3p'
                    hists['reco_1p1p_gen_topo'].Fill(1.5)  # 1p3p
                else:
                    gen_topo = 'other'
                    hists['reco_1p1p_gen_topo'].Fill(2.5)  # other
                hists[f'mvis_gen{gen_topo}'].Fill(mvis_val)

        # Per-charged-track z0sin_charge weights (MC, period detected) for
        # the forward and backward charged-track sets. Used by fill_1p1p
        # (and forwarded by fill_tag_and_probe) to populate the
        # _rw_z0sin_charge companion histograms. Single 3D-table mode
        # w(pT, θ, z0sinθ, q).
        fw_zc = bw_zc = None
        if vtx_rw is not None:
            fc_pt        = pt_g[fc]
            fc_theta_deg = np.degrees(th_g[fc])
            fc_z0sin     = z0_g[fc] * np.sin(th_g[fc])
            fc_q         = q_g[fc]
            bc_pt        = pt_g[bc]
            bc_theta_deg = np.degrees(th_g[bc])
            bc_z0sin     = z0_g[bc] * np.sin(th_g[bc])
            bc_q         = q_g[bc]
            fw_zc = vtx_rw.weights(fc_pt, fc_theta_deg, fc_z0sin, fc_q)
            bw_zc = vtx_rw.weights(bc_pt, bc_theta_deg, bc_z0sin, bc_q)

        fill_tag_and_probe(hists, nf, nb,
            q_g[fc], np.degrees(th_g[fc]), p_g[fc], pt_g[fc], dpp_g[fc],
            q_g[bc], np.degrees(th_g[bc]), p_g[bc], pt_g[bc], dpp_g[bc],
            fpx=px_g[fc], fpy=py_g[fc], fpz=pz_g[fc],
            bpx=px_g[bc], bpy=py_g[bc], bpz=pz_g[bc],
            is_mc=args.is_mc,
            gen_q=gen_q_fs, gen_theta=gen_th_fs, gen_phi=gen_phi_fs,
            gen_topo=gen_topo,
            fw_zc=fw_zc, bw_zc=bw_zc)

        # --- dpp<0.5 variant: second-pass topology + fill ---
        # Event-level cuts already passed (same `_g` arrays used to compute
        # thrust/evis/mvis/acol, so they're identical for both passes).
        # Only the per-charged-track selection differs.
        fc_d = charged_dpp0p5 & hemi_fwd
        bc_d = charged_dpp0p5 & hemi_bwd
        nf_d = int(np.sum(fc_d))
        nb_d = int(np.sum(bc_d))
        if (nf_d, nb_d) in ((1, 0), (0, 1)):
            # The dpp cut dropped one prong → variant sees this as 1+0.
            surv_d = fc_d if nf_d == 1 else bc_d
            idx_d = int(np.argmax(surv_d))
            w_surv_d = None
            if vtx_rw is not None:
                z0sin_d = float(z0_g[idx_d] * np.sin(th_g[idx_d]))
                theta_d = float(math.degrees(th_g[idx_d]))
                w_surv_d = float(vtx_rw.weights(
                    np.array([float(pt_g[idx_d])]),
                    np.array([theta_d]),
                    np.array([z0sin_d]),
                    np.array([float(q_g[idx_d])]),
                )[0])
            fill_1p0p(
                hists,
                q_surv=float(q_g[idx_d]),
                theta_surv_deg=math.degrees(th_g[idx_d]),
                pt_surv=float(pt_g[idx_d]),
                p_surv=float(p_g[idx_d]),
                w_surv=w_surv_d,
                suffix='_dpp0p5',
            )
            if args.is_mc and gen_q_fs is not None and len(gen_q_fs) > 0:
                phi_id = math.degrees(math.atan2(
                    float(py_g[idx_d]), float(px_g[idx_d])))
                fill_1p0p_match(
                    hists,
                    q_reco=float(q_g[idx_d]),
                    theta_s_deg=math.degrees(th_g[idx_d]),
                    phi_s_deg=phi_id,
                    pt_s=float(pt_g[idx_d]),
                    p_s=float(p_g[idx_d]),
                    gen_q=gen_q_fs, gen_theta=gen_th_fs, gen_phi=gen_phi_fs,
                    suffix='_dpp0p5',
                )
        elif nf_d in {1, 3} and nb_d in {1, 3}:
            # Full TNP analysis with dpp-tightened track set.
            fw_zc_d = bw_zc_d = None
            if vtx_rw is not None:
                fw_zc_d = vtx_rw.weights(
                    pt_g[fc_d], np.degrees(th_g[fc_d]),
                    z0_g[fc_d] * np.sin(th_g[fc_d]), q_g[fc_d])
                bw_zc_d = vtx_rw.weights(
                    pt_g[bc_d], np.degrees(th_g[bc_d]),
                    z0_g[bc_d] * np.sin(th_g[bc_d]), q_g[bc_d])
            fill_tag_and_probe(hists, nf_d, nb_d,
                q_g[fc_d], np.degrees(th_g[fc_d]), p_g[fc_d], pt_g[fc_d], dpp_g[fc_d],
                q_g[bc_d], np.degrees(th_g[bc_d]), p_g[bc_d], pt_g[bc_d], dpp_g[bc_d],
                fpx=px_g[fc_d], fpy=py_g[fc_d], fpz=pz_g[fc_d],
                bpx=px_g[bc_d], bpy=py_g[bc_d], bpz=pz_g[bc_d],
                is_mc=args.is_mc,
                gen_q=gen_q_fs, gen_theta=gen_th_fs, gen_phi=gen_phi_fs,
                gen_topo=gen_topo,
                fw_zc=fw_zc_d, bw_zc=bw_zc_d,
                suffix='_dpp0p5')

        if args.is_mc and has_gen:
            gpx = np.asarray(data["gen_px"][ievt], dtype=np.float64)
            gpy = np.asarray(data["gen_py"][ievt], dtype=np.float64)
            gpz = np.asarray(data["gen_pz"][ievt], dtype=np.float64)
            gpdg = np.asarray(data["GenPart_pdgId"][ievt], dtype=np.int32)
            gst  = np.asarray(data["GenPart_status"][ievt], dtype=np.int32)
            for k in range(len(gpdg)):
                if abs(gpdg[k]) == 15 and gst[k] == 4:
                    gpt = math.sqrt(gpx[k]**2 + gpy[k]**2)
                    gt = math.degrees(math.atan2(gpt, gpz[k]))
                    gsign = "pos" if gpdg[k]==-15 else "neg"
                    hists[f'gen_tau_theta_{gsign}'].Fill(gt)
                    hists[f'gen_tau_theta_{gsign}_cos'].Fill(_cos_th(gt))

    # ---- Summary ----
    print(f"\n=== Results ===")
    print(f"Events: {nevt}  Passed: {n_pass}  Eff: {n_pass/nevt*100:.2f}%" if nevt else "")
    print(f"Z-pole cut (|Ecm - 91.25| > 1 GeV) removed: {n_zpole_cut}")

    print(f"\nCutflow:")
    cf = hists['cutflow']
    for i in range(1, cf.GetNbinsX()+1):
        c = int(cf.GetBinContent(i))
        if c: print(f"  {cf.GetXaxis().GetBinLabel(i):>12}: {c:>8}")

    # Per-charge efficiency and mis-ID from 1+1 events
    print(f"\n1+1 Tag-and-Probe (pT thresholds on probe):")
    for pcut in PT_CUTS:
        t = f"pt{pcut:.1f}".replace('.', 'p')
        n_tag_pos = hists[f'tag_pos_{t}'].Integral()
        n_tag_neg = hists[f'tag_neg_{t}'].Integral()
        # OS: correct charge
        n_os_pos = hists[f'probe_pos_tag_neg_{t}'].Integral()  # tag=-, probe=+ (correct)
        n_os_neg = hists[f'probe_neg_tag_pos_{t}'].Integral()  # tag=+, probe=- (correct)
        # SS: wrong charge (mis-ID)
        n_ss_pos = hists[f'probe_pos_tag_pos_{t}'].Integral()  # tag=+, probe=+ (wrong)
        n_ss_neg = hists[f'probe_neg_tag_neg_{t}'].Integral()  # tag=-, probe=- (wrong)

        f_pos = n_ss_pos / n_tag_pos if n_tag_pos > 0 else 0  # mis-ID: found + when expecting -
        f_neg = n_ss_neg / n_tag_neg if n_tag_neg > 0 else 0  # mis-ID: found - when expecting +
        print(f"  pT>{pcut:4.1f}: tag+={n_tag_pos:.0f} tag-={n_tag_neg:.0f} | "
              f"OS+={n_os_pos:.0f} OS-={n_os_neg:.0f} | "
              f"SS+={n_ss_pos:.0f} SS-={n_ss_neg:.0f} | "
              f"misID(+)={f_pos:.4f} misID(-)={f_neg:.4f}")

    print(f"\n1+1 Tag-and-Probe (|p| thresholds on probe, baseline pT>0.4):")
    for pcut in PMAG_CUTS:
        t = f"pmag{pcut:.1f}".replace('.', 'p')
        n_tag_pos = hists[f'tag_pos_{t}'].Integral()
        n_tag_neg = hists[f'tag_neg_{t}'].Integral()
        n_os_pos = hists[f'probe_pos_tag_neg_{t}'].Integral()
        n_os_neg = hists[f'probe_neg_tag_pos_{t}'].Integral()
        n_ss_pos = hists[f'probe_pos_tag_pos_{t}'].Integral()
        n_ss_neg = hists[f'probe_neg_tag_neg_{t}'].Integral()

        f_pos = n_ss_pos / n_tag_pos if n_tag_pos > 0 else 0
        f_neg = n_ss_neg / n_tag_neg if n_tag_neg > 0 else 0
        print(f"  |p|>{pcut:4.1f}: tag+={n_tag_pos:.0f} tag-={n_tag_neg:.0f} | "
              f"OS+={n_os_pos:.0f} OS-={n_os_neg:.0f} | "
              f"SS+={n_ss_pos:.0f} SS-={n_ss_neg:.0f} | "
              f"misID(+)={f_pos:.4f} misID(-)={f_neg:.4f}")

    # Write
    fout = ROOT.TFile(args.outfile, 'RECREATE')
    for h in hists.values(): h.Write()
    fout.Close()
    print(f"\nOutput: {args.outfile}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
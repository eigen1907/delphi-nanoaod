#!/usr/bin/env python3
"""
Thrust analysis on **raw DelphiNanoAOD** ttree files — full RECO + GEN.

Reads the raw nanoaod `t` tree (Part_*/Trac_*/Event_* and, for MC, SimPart_*)
directly and reproduces EVERYTHING analysis_thrust.py produces, by reusing its
exact systematics / event-selection / response / moments machinery (IMPORTED,
not copied — see analysis_thrust.process_reco_level_arrays / process_gen_arrays):

  Data:  counter, counter_weighted, reco/<syst>, moments/reco
  MC:    + gen/ (V0-aware SimPart leaves), response/<syst>,
         moments/{gen_full,gen_passReco}, and genPart/ (parallel GenPart-based
         gen, histograms only — see "Two gen definitions" below)

This exists because analysis_thrust.py reads a pre-flattened skim (tree `t` with
flat px/py/pz/d0/z0 branches), which the raw DelphiNanoAOD files do NOT contain.
Here we build those per-event reco/gen arrays from the raw collections and hand
them to the shared cores, so the physics is identical bit-for-bit.

Two reader specifics
--------------------
* RECO final state = Part with Part_lock == 0 (LVLOCK standard selection).
  px,py,pz,energy <- Part_fourMomentum_fCoordinates_{fX,fY,fZ,fT}; mass from
  E^2 - p^2; charge <- Part_charge. Impact parameters d0,z0 are scattered into
  the per-Part arrays from Trac_impParToVertexRPhi / Trac_impParToVertexZ via
  Trac_partIdx (neutrals keep d0=z0=0). Because real d0/z0 ARE available, the
  charged z0*sin(theta) cut is ACTIVE here (unlike the EDM4hep reader).
* GEN final state = V0-aware SimPart leaf recovery (select_leaves): descend
  through a V0 only when its daughters energy-balance, else keep the V0 as a
  single neutral leaf. This is the CORRECT gen definition; it deliberately does
  NOT use analysis_thrust.py's flat-skim tgenBefore->tgen energy fallback.

Two gen definitions (the "decay V0 vs not decay V0" study)
----------------------------------------------------------
For MC we fill TWO independent gen-histogram sets:
  * gen/      <- SimPart, V0-aware select_leaves. DELSIM lets V0s decay in the
                tracker, so this gen sees the V0 DAUGHTERS ("V0 decayed").
                Keeps the primary (generator-matched) neutrinos (~0.22/evt).
  * genPart/  <- GenPart (JETSET truth), status==1 final state, neutrinos kept
                (to match gen/). With V0 decays disabled in JETSET the V0s
                survive as status-1 particles, so this gen sees the V0s WHOLE
                ("V0 not decayed"). build_genpart_arrays; histograms only (no
                moments, no response matrix). Charge comes from the PDG code.
Both sets keep neutrinos and share identical binning/booking, so they differ
only in V0 treatment -- genPart/ vs gen/ isolates the effect of V0 decay on
the gen-level thrust.

Usage
-----
  Data: python analysis_thrust_nanoaod.py 'data/*.root' out.root
  MC:   python analysis_thrust_nanoaod.py 'mc/*.root'  out.root --is-mc
        (optional: --systematic nominal, --max-events N)
"""

import sys
import glob
import argparse

import numpy as np
import uproot
import awkward as ak
import ROOT  # noqa: F401

from analysis_thrust import (
    SYSTEMATICS,
    create_counter_histogram,
    create_counter_weighted_histogram,
    create_gen_histograms,
    create_reco_histograms,
    create_response_matrices,
    process_reco_level_arrays,
    process_gen_arrays,
)
from thrust_moments import (
    create_all_moment_histograms,
    write_moment_histograms,
)

# DELPHI V0 codes (DELSIM 89-15 Appendix C):
#   61 = K0_S, 81 = Lambda, 67 = Sigma+, 69 = Sigma-, 70 = Xi0, 71 = Xi-
V0_DELPHI_CODES = {61, 81, 67, 69, 70, 71}

# Heavy-flavor (charm/bottom) hadrons + Omega-. In samples where DELSIM tracks
# them -- notably the OFFICIAL kk2f4146_qqpy -- these appear in SimPart as
# separate particles with displaced decay vertices (~1.2/event, 100% decaying).
# They decay promptly (ctau ~ 0.1-0.5 mm, effectively at the IP), so the
# detector measures their DAUGHTERS, exactly like V0s -- so select_leaves must
# descend them too. Without this, each D/B is kept as one "stable" leaf and its
# daughters are orphaned, undercounting the gen final state (kk2f leaf was ~3
# particles below GenPart status==1). Samples that decay b/c at generator level
# (Pythia8, the user-generated kk2f_ISRon/off) carry NONE of these codes in
# SimPart, so this set is a no-op there.
# Codes inferred from reconstructed mass -- VERIFY against DELSIM 89-15 App. C:
#   44=D0  45=D+-  46=Ds   82=Lambda_c  83/84=Xi_c
#   125=B0 126=B+- 127=Bs  265/266=b-baryons   94=Omega-
HEAVY_FLAVOR_DELPHI_CODES = {44, 45, 46, 82, 83, 84, 94, 125, 126, 127, 265, 266}

# Particles whose decays select_leaves descends (the detector measures the
# daughters, not the parent): strange V0s + prompt heavy-flavor.
DESCEND_DELPHI_CODES = V0_DELPHI_CODES | HEAVY_FLAVOR_DELPHI_CODES

E_TOLERANCE = 0.01  # GeV; descend through a decay only if its immediate
                    # daughters energy-balance within this tolerance

# EDM4hep-style four-momentum suffixes shared by Part_ and SimPart_.
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

# --- GenPart (LUJETS / JETSET truth) -------------------------------------
# Second, parallel gen definition. Unlike Part_/SimPart_ (XYZTVectorF field
# "fourMomentum"), GenPart's four-momentum field is named "vector".
_GPX = "_vector_fCoordinates_fX"
_GPY = "_vector_fCoordinates_fY"
_GPZ = "_vector_fCoordinates_fZ"
_GE = "_vector_fCoordinates_fT"
GENPART_BRANCHES = [
    "GenPart_status", "GenPart_pdgId",
    f"GenPart{_GPX}", f"GenPart{_GPY}", f"GenPart{_GPZ}", f"GenPart{_GE}",
]
# GenPart has NO charge branch -> derive charge from the PDG code.
# JETSET status==1 = final-state stable. With V0 decays disabled in JETSET the
# V0s (K0_S, Lambda, ...) survive AS status-1 particles, so this gen is the
# "V0 NOT decayed" counterpart to the SimPart V0-aware gen ("V0 decayed").
# Neutrinos (PDG 12/14/16) are KEPT here: the SimPart gen also keeps its
# primary (generator-matched) neutrinos -- select_leaves picks them up as
# vertex-0 leaves (~0.22/evt) -- so keeping them on this side too leaves the V0
# decay treatment as the only intended difference between gen/ and genPart/.
GENPART_STABLE_STATUS = 1

_PDG_DB = ROOT.TDatabasePDG.Instance()
_CHARGE_CACHE = {}


def pdg_charge(pid):
    """Signed charge in units of e for a PDG code (cached). Unknown -> 0.0.

    TDatabasePDG.Charge() returns charge in units of |e|/3, so divide by 3.
    """
    pid = int(pid)
    q = _CHARGE_CACHE.get(pid)
    if q is None:
        p = _PDG_DB.GetParticle(pid)
        q = (p.Charge() / 3.0) if p else 0.0
        _CHARGE_CACHE[pid] = q
    return q


# ---------------------------------------------------------------------------
# V0-aware gen leaf selection (unchanged; the correct gen definition)
# ---------------------------------------------------------------------------
def select_leaves(ovtx, dvtx, pdg, E):
    nP = len(ovtx)
    vtx_outgoing = {}
    for i in range(nP):
        if ovtx[i] != -1:
            vtx_outgoing.setdefault(int(ovtx[i]), []).append(i)

    abs_pdg = np.abs(pdg)
    leaf_mask = np.zeros(nP, dtype=bool)
    n_v0_clean = 0
    n_v0_shower = 0
    n_v0_undecayed = 0

    def handle(i):
        nonlocal n_v0_clean, n_v0_shower, n_v0_undecayed
        if abs_pdg[i] not in DESCEND_DELPHI_CODES:
            leaf_mask[i] = True
            return
        if dvtx[i] == -1:
            leaf_mask[i] = True
            n_v0_undecayed += 1
            return
        daughters = vtx_outgoing.get(int(dvtx[i]), [])
        if not daughters:
            leaf_mask[i] = True
            n_v0_undecayed += 1
            return
        if abs(sum(E[j] for j in daughters) - E[i]) < E_TOLERANCE:
            n_v0_clean += 1
            for j in daughters:
                handle(j)
        else:
            leaf_mask[i] = True
            n_v0_shower += 1

    for i in vtx_outgoing.get(0, []):
        handle(i)

    return leaf_mask, n_v0_clean, n_v0_shower, n_v0_undecayed


# ---------------------------------------------------------------------------
# V0-decay convention: re-decay the strange V0s that select_leaves could NOT
# descend. Those are kept as single leaves because DELSIM either left them
# undecayed (escaped the tracking volume) or they interacted in material so
# their daughters fail the energy-balance gate. To report the gen final state in
# the Pythia V0-decay convention (modern QCD particle level), drop each such kept
# V0 leaf and splice in a fresh Pythia decay of its frozen 4-momentum. Cleanly
# DESCENDED V0s are left untouched -- their DELSIM daughters stay, which keeps the
# gen<->reco correlation for the response. Gate = the existing select_leaves
# energy balance. Default ON; set REDECAY_KEPT_V0 = False for the pure
# SimPart-leaf gen (A/B checks, legacy reproduction).
# ---------------------------------------------------------------------------
REDECAY_KEPT_V0 = True

_V0_ARR = np.array(sorted(V0_DELPHI_CODES))
# DELPHI internal code -> PDG id. K0S (61) is self-conjugate; hyperon sign comes
# from the signed DELPHI code (e.g. -81 = anti-Lambda -> -3122).
_DELPHI_TO_PDG = {61: 310, 81: 3122, 67: 3222, 69: 3112, 70: 3322, 71: 3312}
_v0_decayer = None


def _get_v0_decayer():
    """Lazily build a process-level-off Pythia8 used only to decay frozen V0s.
    Fixed seed for reproducibility. pythia8 is provided by the LCG view the
    analyzer already sources (LCG_108/109). Imported lazily so this module still
    imports where the re-decay is never triggered."""
    global _v0_decayer
    if _v0_decayer is None:
        import pythia8
        p = pythia8.Pythia("", False)
        for s in ("ProcessLevel:all = off", "Print:quiet = on",
                  "Next:numberShowEvent = 0", "Next:numberCount = 0",
                  "Random:setSeed = on", "Random:seed = 12345"):
            p.readString(s)
        for pid in _DELPHI_TO_PDG.values():
            p.readString(f"{pid}:mayDecay = on")
        p.init()
        _v0_decayer = p
    return _v0_decayer


def _redecay_v0s(codes, Px, Py, Pz, E4):
    """Decay frozen V0s (DELPHI codes + 4-momenta) at the origin with Pythia's
    decay tables. Returns (px,py,pz,e,q,pdg) of ALL final daughters (charged +
    neutral; neutral modes cascade to photons, matching DELSIM's storage). pdg
    is the standard PDG id of each daughter (for downstream species tagging)."""
    dec = _get_v0_decayer()
    ev = dec.event
    ev.reset()
    for c, x, y, z, e in zip(codes, Px, Py, Pz, E4):
        ac = abs(int(c))
        pdg = _DELPHI_TO_PDG[ac]
        if ac != 61 and c < 0:
            pdg = -pdg
        m2 = e * e - (x * x + y * y + z * z)
        m = np.sqrt(m2) if m2 > 0.0 else 0.0
        # append(id, status, mother1, mother2, px, py, pz, e, m)
        ev.append(int(pdg), 91, 0, 0, float(x), float(y), float(z), float(e), float(m))
    dec.moreDecays()
    px, py, pz, en, q, dpdg = [], [], [], [], [], []
    for k in range(ev.size()):
        pa = ev[k]
        if pa.isFinal():
            px.append(pa.px()); py.append(pa.py()); pz.append(pa.pz())
            en.append(pa.e()); q.append(pa.charge()); dpdg.append(pa.id())
    return (np.asarray(px, dtype=np.float64), np.asarray(py, dtype=np.float64),
            np.asarray(pz, dtype=np.float64), np.asarray(en, dtype=np.float64),
            np.asarray(q, dtype=np.float64), np.asarray(dpdg, dtype=np.int64))


def build_gen_leaf_redecayed(o, d, pdg, charge, Px, Py, Pz, E4, return_codes=False):
    """Shared gen final-state builder for the EEC, thrust and fbcec analyzers.

    Runs select_leaves, then (when REDECAY_KEPT_V0) drops every strange-V0 leaf
    that could not be descended and splices in its Pythia re-decay daughters, so
    the gen final state is in the V0-decay convention. Returns (px,py,pz,m,q)
    [GeV] over all final particles (charged + neutral), or None if no leaves.

    return_codes=True (used by the fbcec reader for species tagging): also
    returns (codes, redecayed) — the per-particle id and a boolean mask flagging
    the Pythia-re-decayed daughters. codes are DELPHI internal ids for kept
    leaves and standard PDG ids for the re-decayed daughters, so they must be
    interpreted with the mask (species_codes convention 'delphi' vs 'pdg').
    """
    leaf, _, _, _ = select_leaves(o, d, pdg, E4)
    if not leaf.any():
        return None
    if REDECAY_KEPT_V0:
        kept_v0 = leaf & np.isin(np.abs(pdg), _V0_ARR)
    else:
        kept_v0 = np.zeros(len(leaf), dtype=bool)
    keep = leaf & ~kept_v0
    px, py, pz, en = Px[keep], Py[keep], Pz[keep], E4[keep]
    q = charge[keep]
    codes = np.asarray(pdg[keep], dtype=np.int64)
    redec = np.zeros(len(codes), dtype=bool)
    if kept_v0.any():
        dpx, dpy, dpz, den, dq, dpdg = _redecay_v0s(
            pdg[kept_v0], Px[kept_v0], Py[kept_v0], Pz[kept_v0], E4[kept_v0])
        px = np.concatenate([px, dpx]); py = np.concatenate([py, dpy])
        pz = np.concatenate([pz, dpz]); en = np.concatenate([en, den])
        q = np.concatenate([q, dq])
        codes = np.concatenate([codes, dpdg])
        redec = np.concatenate([redec, np.ones(len(dpx), dtype=bool)])
    m = np.sqrt(np.clip(en * en - (px * px + py * py + pz * pz), 0.0, None))
    if return_codes:
        return (px, py, pz, m, q, codes, redec)
    return (px, py, pz, m, q)


# ---------------------------------------------------------------------------
# Per-event readers: raw nanoaod collections -> per-event physics arrays
# ---------------------------------------------------------------------------
def build_reco_arrays(arr, iEv):
    """RECO final state from Part with Part_lock == 0.

    Returns (px, py, pz, m, q, pt, th, d0, z0) [GeV / cm], or None if the event
    has no Part entries / no unlocked particles.
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

    # Scatter Trac impact parameters into per-Part arrays via Trac_partIdx so
    # the lock filter applies uniformly. Neutrals (no track) keep d0=z0=0.
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
    return (px[keep], py[keep], pz[keep], m[keep], q[keep],
            pt[keep], th[keep], d0[keep], z0[keep])


def build_gen_arrays(arr, iEv):
    """GEN final state via V0-aware SimPart leaf recovery, in the Pythia V0-decay
    convention (kept-undecayed / interacted V0s are re-decayed; see
    build_gen_leaf_redecayed).

    Returns (px, py, pz, m, q, pt, th) [GeV], or None if no leaves.
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

    res = build_gen_leaf_redecayed(o, d, pdg, charge, Px, Py, Pz, E4)
    if res is None:
        return None
    px, py, pz, m, q = res
    pt = np.sqrt(px * px + py * py)
    th = np.arctan2(pt, pz)
    return (px, py, pz, m, q, pt, th)


def build_genpart_arrays(arr, iEv):
    """GEN final state from GenPart (JETSET truth), parallel to build_gen_arrays.

    Selection = status==1 (final-state stable), neutrinos included (the SimPart
    gen also keeps its primary neutrinos). No V0 leaf recovery: whatever JETSET
    left as status-1 is the leaf set, so with V0 decays disabled in JETSET the
    V0s appear here undecayed. Charge is derived from the PDG code (no
    GenPart_charge branch). Returns (px,py,pz,m,q,pt,th) [GeV], or None if no
    leaves.
    """
    status = ak.to_numpy(arr["GenPart_status"][iEv]).astype(np.int64)
    if len(status) == 0:
        return None
    pdg = ak.to_numpy(arr["GenPart_pdgId"][iEv]).astype(np.int64)

    keep = (status == GENPART_STABLE_STATUS)
    if not keep.any():
        return None

    px = ak.to_numpy(arr[f"GenPart{_GPX}"][iEv]).astype(np.float64)[keep]
    py = ak.to_numpy(arr[f"GenPart{_GPY}"][iEv]).astype(np.float64)[keep]
    pz = ak.to_numpy(arr[f"GenPart{_GPZ}"][iEv]).astype(np.float64)[keep]
    en = ak.to_numpy(arr[f"GenPart{_GE}"][iEv]).astype(np.float64)[keep]
    q = np.array([pdg_charge(p) for p in pdg[keep]], dtype=np.float64)

    pt = np.sqrt(px * px + py * py)
    th = np.arctan2(pt, pz)
    m = np.sqrt(np.clip(en * en - (px * px + py * py + pz * pz), 0.0, None))
    return (px, py, pz, m, q, pt, th)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("infile", help="raw DelphiNanoAOD file or glob")
    ap.add_argument("outfile")
    ap.add_argument("--is-mc", action="store_true",
                    help="fill gen-level (V0-aware) histograms, response matrices and gen moments")
    ap.add_argument("--systematic", default="all",
                    help='run one systematic or "all" (default: all)')
    ap.add_argument("--max-events", type=int, default=-1)
    args = ap.parse_args()

    if args.systematic == "all":
        systematics = SYSTEMATICS
    elif args.systematic in SYSTEMATICS:
        systematics = {args.systematic: SYSTEMATICS[args.systematic]}
    else:
        print(f"ERROR: unknown systematic '{args.systematic}'", file=sys.stderr)
        print(f"Available: {', '.join(SYSTEMATICS.keys())}", file=sys.stderr)
        return 1

    files = sorted(glob.glob(args.infile)) if any(c in args.infile for c in "*?[") else [args.infile]
    if not files:
        print(f"No files match: {args.infile}", file=sys.stderr)
        return 1
    print(f"Reading {len(files)} file(s); is_mc={args.is_mc}; "
          f"systematics: {', '.join(systematics.keys())}")

    # --- Histograms (mirror analysis_thrust.py) ---
    counter = create_counter_histogram()
    counter_weighted = create_counter_weighted_histogram()
    gen_hists = create_gen_histograms() if args.is_mc else None
    # Parallel GenPart-based gen set (distinct ROOT names; no moments).
    gen_hists_genpart = create_gen_histograms(suffix='_genPart') if args.is_mc else None
    reco_hists = {s: create_reco_histograms(s) for s in systematics}
    moment_hists = create_all_moment_histograms(systematics)
    response_matrices = {}
    if args.is_mc:
        response_matrices = {s: create_response_matrices(s) for s in systematics}

    read_branches = list(RECO_BRANCHES) + (list(GEN_BRANCHES) if args.is_mc else [])

    n_done = 0
    stop = False
    for k, fp in enumerate(files):
        if stop:
            break
        try:
            t = uproot.open(fp)["t"]
        except Exception as exc:
            print(f"  skip {fp}: {exc}")
            continue
        n_entries = t.num_entries
        if n_entries == 0:
            continue
        # GenPart is optional (absent on older MC); read it only when present.
        have_genpart = args.is_mc and ("GenPart_status" in t.keys())
        branches = read_branches + (GENPART_BRANCHES if have_genpart else [])
        arr = t.arrays(branches)

        for iEv in range(n_entries):
            if args.max_events > 0 and n_done >= args.max_events:
                stop = True
                break

            # Count every processed event (bin 1), like analysis_thrust.py main.
            counter.Fill(0.5)
            n_done += 1

            E = float(arr["Event_cmEnergy"][iEv])
            # Off-peak energy-scan rejection (no-op for 94c pole). The shared
            # cores also guard on E, but skip gen filling here for consistency.
            if abs(E - 91.25) > 1:
                continue

            # --- GEN (MC only): once-per-event "before" fill, then per-syst
            #     selected-gen thrust for the response matrices ---
            thrust_gen_dict = {}
            if args.is_mc:
                gen = build_gen_arrays(arr, iEv)
                if gen is not None:
                    gpx, gpy, gpz, gm, gq, gpt, gth = gen
                    process_gen_arrays(
                        gpx, gpy, gpz, gm, gq, gpt, gth, E, gen_hists,
                        charged_p_min=0.4, neutral_e_min=0.5,
                        moment_hists=moment_hists,
                    )
                    for syst_name, cfg in systematics.items():
                        thrust_gen_dict[syst_name] = process_gen_arrays(
                            gpx, gpy, gpz, gm, gq, gpt, gth, E, None,
                            charged_p_min=cfg.get("charged_p_min", 0.4),
                            neutral_e_min=cfg.get("neutral_e_min", 0.5),
                        )

            # --- GEN via GenPart (JETSET status==1, V0 left undecayed):
            #     a second gen-histogram set only — no moments, no response. ---
            if have_genpart:
                genp = build_genpart_arrays(arr, iEv)
                if genp is not None:
                    gp_px, gp_py, gp_pz, gp_m, gp_q, gp_pt, gp_th = genp
                    process_gen_arrays(
                        gp_px, gp_py, gp_pz, gp_m, gp_q, gp_pt, gp_th, E,
                        gen_hists_genpart,
                        charged_p_min=0.4, neutral_e_min=0.5,
                        moment_hists=None,
                    )

            # --- RECO (always): all systematics, event selection, moments,
            #     and (MC) response matrices ---
            reco = build_reco_arrays(arr, iEv)
            if reco is None:
                continue
            rpx, rpy, rpz, rm, rq, rpt, rth, rd0, rz0 = reco
            process_reco_level_arrays(
                rpx, rpy, rpz, rm, rq, rpt, rth, rd0, rz0, E,
                systematics, reco_hists, response_matrices,
                thrust_gen_dict=thrust_gen_dict, is_mc=args.is_mc,
                counter=counter, counter_weighted=counter_weighted,
                moment_hists=moment_hists,
            )

        print(f"  [{k + 1}/{len(files)}] {fp.split('/')[-1]}  -> {n_done} events processed")

    print(f"Total: {int(counter.GetBinContent(1))} events, "
          f"{int(counter.GetBinContent(2))} pass reco selection")

    # --- Write output (same structure as analysis_thrust.py) ---
    fout = ROOT.TFile(args.outfile, "RECREATE")
    fout.cd()
    counter.Write()
    counter_weighted.Write()
    if gen_hists:
        fout.mkdir("gen"); fout.cd("gen")
        for h in gen_hists.values():
            h.Write()
        fout.cd()
    if gen_hists_genpart:
        fout.mkdir("genPart"); fout.cd("genPart")
        for h in gen_hists_genpart.values():
            h.Write()
        fout.cd()
    fout.mkdir("reco"); fout.cd("reco")
    for hists in reco_hists.values():
        for h in hists.values():
            h.Write()
    fout.cd()
    if response_matrices:
        fout.mkdir("response"); fout.cd("response")
        for mats in response_matrices.values():
            for mh in mats.values():
                mh.Write()
        fout.cd()
    write_moment_histograms(fout, moment_hists, top_dir="moments")
    fout.Close()
    print(f"Wrote {args.outfile}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

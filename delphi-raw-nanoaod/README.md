# delphi-raw-nanoaod

A PHDST-level reader that converts a DELPHI shortDST (`.sdst`), full-DST
(`.fadana`), long-DST (`.al`), or DETRAW raw-DST into a flat ROOT RNTuple.
It is a **sibling** of the SKELANA-based
`delphi-nanoaod` tool in the same repository: they share the `delphi-analysis`
static library, but `delphi-raw-nanoaod` subclasses `phdst::Analysis` instead
of `skelana::Analysis` — so none of SKELANA's aggregation runs, and we walk the
ZEBRA banks directly via `LPHPA` + the `LQ`/`IQ`/`Q` macros.

The point of bypassing SKELANA is to get at **cluster-level and hit-level
information** that SKELANA collapses. For example: SKELANA reports the total
HPC shower energy and a 10-bit layer-hit pattern per track; this reader
surfaces the per-layer energies (EmLayer_*), the conversion-vertex locations
(PhotonConv_* lives on the sibling `feature/photon-collection` branch), the
VD silicon hits with their back-pointers to tracks (VdAssocHit_*), the full
set of per-track detector elements on full-DST input (TrackElement_*), and
the B-field, primary vertex covariance, beamspot dimensions, and the raw MTPC
dE/dx fields that a particle-flow reconstructor needs as inputs.

This README describes what's in the RNTuple, how to build, and how to run.
Design notes live in `docs/PHDST_RAW_NANOAOD_PLAN.md`; the raw-DST workflow is
documented in `docs/RAWDST_WORKFLOW.md`.

## Quickstart

```sh
# Inside the cmssw/el9 singularity image with /cvmfs mounted
source /cvmfs/delphi.cern.ch/setup.sh
source /cvmfs/sft.cern.ch/lcg/views/LCG_107/x86_64-el9-gcc13-opt/setup.sh
cd build
cmake ..
make -j4 delphi-raw-nanoaod

./delphi-raw-nanoaod/delphi-raw-nanoaod \
    --pdlinput simana.sdst    --output raw_sdst.root
./delphi-raw-nanoaod/delphi-raw-nanoaod \
    --pdlinput simana.fadana  --output raw_fadana.root
```

Both inputs produce the same RNTuple schema; which collections are populated
depends on the input — see the **shortDST ↔ fullDST** section below.

Quick-look ROOT macro (14 validation histograms):

```sh
root -l -q -b 'scripts/make_plots.C("raw_sdst.root", "plots_dir")'
```

## RNTuple schema

All fields live on a single RNTuple named `Events`. Collections are flat
(one row per `(event, object)`) with back-pointers between them.

### Event-level scalars

| field | source | meaning |
|---|---|---|
| `Event_experimentNumber`, `Event_runNumber`, `Event_fileSequenceNumber`, `Event_eventNumber`, `Event_date`, `Event_time`, `Event_fillNumber` | `/PHCIII/` | event header |
| `Event_bFieldTesla`, `Event_bFieldGevCm` | `BPILOT` | solenoid B (≈ 1.231 T) and the derived GeV/cm factor. `1/R [1/cm] = Event_bFieldGevCm / pT [GeV]` |
| `Event_recordType` | `PHRTY()` | PHDST record type, e.g. `RAW`, `TAN`, `DST`; needed for raw-DST files because raw banks and reconstructed objects are separate records |
| `Event_beamSpot{X,Y,Z}`, `Event_beamSpotSigma{X,Y,Z}`, `Event_beamSpotErrorFlag` | `LQ(LDTOP-25)` | beamspot centre (cm) and Gaussian widths (cm). errorFlag = 0 if the bank is present, -1 if missing |

### Generic raw-DST banks (`RawBank_*`, `RawWord_*`)

For DETRAW raw-DST, `delphi-raw-nanoaod` now stores the same RAW/Rxxx ZEBRA
bank tree printed by `delphi-raw-bank-lister`. `RawBank_*` gives the bank name,
top-store code, parent/link, depth, `nlinks`, `ndata`, and payload range.
`RawWord_*` stores a bounded payload copy as integer and float views.

Controls:

```sh
--no-raw-bank-scan
--raw-bank-all
--raw-bank-max-depth 12
--raw-bank-max-links 128
--raw-bank-max-words 64   # -1 = all words, 0 = metadata only
```

### Charged tracks (`TracRaw_*`)

One row per charged PA-track (`NINT(Q(LMAIN+8)) ≠ 0`), perigee + fit-quality.
Back-pointers from all other per-track collections use the index in this table.

| field | source | meaning |
|---|---|---|
| `nTracRaw` | count | |
| `TracRaw_paIdx` | PV-PA index | 0-based index of the source PA bank |
| `TracRaw_impactRPhi`, `impactZ`, `theta`, `phi`, `invR` | `QTRAC(4..8)` | perigee parameters |
| `TracRaw_weightMatrix[15]` | `QTRAC(9..23)` | 5×5 symmetric weight matrix |
| `TracRaw_chi2NoVD`, `chi2VD`, `ndfNoVD`, `ndfVD`, `chi2VDHits` | `QMAIN` / `IQ(LMAIN+...)` | fit χ² / ndof split by VD-in / VD-out |
| `TracRaw_trackLength`, `firstPointR`, `firstPointZ` | `QMAIN` | track length (cm) and first measured point |
| `TracRaw_detectorsUsed` | `IQ(LPA+2)` | PSCTRA-documented detector-hit bitmask (see docstring caveat) |
| `TracRaw_charge` | `NINT(Q(LMAIN+8))` | +1 / 0 / -1 (DELPHI charge code decoded) |

### Track elements (`TrackElement_*`) — **fullDST only**

Per-detector 3-D track elements: one row per `PA.TETP` / `TEID` / `TEOD` /
`TEFA` / `TEFB` that the track intersects. **Only populated on `.fadana`
input** — shortDST strips these banks.

| field | meaning |
|---|---|
| `TrackElement_tracRawIdx` | index into `TracRaw_*` |
| `TrackElement_subDetector` | 0 = TPC, 1 = ID, 2 = OD, 3 = FCA, 4 = FCB |
| `TrackElement_dataDescriptor` | bitmask: coord-system + measurement code |
| `TrackElement_coord1..3` | X/Y/Z or R/RPhi/Z at reference point (cm) |
| `TrackElement_theta`, `phi` | direction at the reference point |
| `TrackElement_invPOrPt` | 1/P or 1/Pt (cm⁻¹) |

### VD hits (`VdAssocHit_*`, `VdUnassocHit_*`) — **shortDST only**

Per-hit silicon-strip measurements from the `MVDH` bank at `LQ(LDTOP-21)`.
**Only populated on `.sdst` input** — fullDST does not carry `MVDH`.

Associated hits link back to a `TracRaw` row via `VdAssocHit_tracRawIdx`;
unassociated hits are a flat per-event list.

Fields per hit: `module` (int, sign of Z), `localX` / `R` / `RPhi` (cm),
`signalToNoise` (0 if the shortDST uses the 4-word-per-hit packing).

### TPC per-track summary (`MtpcRaw_*`)

From `PA.MTPC(7)`. Present on both fadana and sdst.

| field | meaning |
|---|---|
| `MtpcRaw_dEdx80Max`, `dEdx80Sigma` | 80%-truncated max-amp dE/dx + sigma |
| `MtpcRaw_dEdx65Max`, `dEdx65Sigma` | 65%-truncated variant |
| `MtpcRaw_dEdx80Integrated` | 80%-trunc integrated amplitude |
| `MtpcRaw_packedPadsSectors` | `nPads + 100*sectorIn + 10000*sectorOut` |
| `MtpcRaw_packedWiresHits` | `nWires + 1000*nCloseClusterHits + 1000000*method` |
| `MtpcRaw_zFitChi2` | χ² of the Z-direction fit |

### Calorimetry

#### EM showers + per-layer: `EmShower_*`, `EmLayer_*`
From `PA.EMCA` / `PA.EMNC`. `EmShower_detector` ∈ {9 HPC, 26 FEMC}. `EmLayer_*`
unpacks the 10-bit HPC layer hit pattern into per-layer raw energies;
**sum(layer energies) is NOT guaranteed to equal EmShower_energy** — the
latter is a calibrated cluster energy with shower-containment corrections,
the former are raw per-layer deposits. They agree to <1 % on high-E clean
EM showers and can differ by factors of a few for soft laterally-spread ones.

#### HAD showers + per-layer-hit: `HadShower_*`, `HadHit_*`
From `PA.HCAL` / `PA.HCNC`. HadHit carries `layer`, `nTowers`, `energy`
per active hit (HAC has 4 layers in DELPHI).

#### STIC: `Stic_*`
Forward small-angle EM. One row per track with a `PA.SSTC` extra-module.

### Photon conversions, tier-2 photon ID, UTER, V0
Exposed on the sibling branch `feature/photon-collection` (the one that also
adds the SKELANA-side `Photon_*` collection). Not in this branch directly
since the two feature branches are intended to stand alone.

### Per-track lepton-ID raw: `MuidRaw_*`, `ElidRaw_*`
`PA.MUID` / `PA.ELID`. Tag, loose-refit χ², hit pattern for muons; tag,
γ-conversion tag, refit 3-momentum for electrons.

### MC truth particle list (`GenPart_*`, `Event_isMC`)

Filled on MC events by calling SKELANA's `PSHMC` fortran routine, which reads
the simulation banks at `LQ(LDTOP-28)/(LDTOP-29)` (shortDST) or the fullDST
equivalents and populates the `PSCLUJ` LUJETS-like event record. No other
SKELANA logic runs — we treat `PSHMC` as a standalone bank-unpacking call.
Real-data events come out with `Event_isMC=0` and `nGenPart=0`.

Fields mirror the ones the SKELANA-based `delphi-nanoaod` writes in its own
`GenPart_*` collection, so downstream analysis code can be schema-agnostic
across the two writers.

| field | source | meaning |
|---|---|---|
| `Event_isMC` | `pscluj.np` | 1 if the simulation structure was unpacked this event, 0 otherwise |
| `nGenPart` | `NP` | number of LUJETS particles in the record |
| `GenPart_status` | `KP(i,1)` | JETSET status code (1 = final-state stable, 2 = decayed/unstable, 21 = documentation) |
| `GenPart_pdgId` | `KP(i,2)` | PDG particle code |
| `GenPart_parentIdx` | `KP(i,3)-1` | parent index, -1 if none |
| `GenPart_firstChildIdx`, `GenPart_lastChildIdx` | `KP(i,4)-1`, `KP(i,5)-1` | daughter index range |
| `GenPart_fourMomentum` | `PP(i,1..4)` | (px, py, pz, E) in GeV |
| `GenPart_mass` | `PP(i,5)` | generator mass (GeV) |
| `GenPart_vertex` | `VP(i,1..3)` | production vertex (**mm** — LUJETS convention, *not* cm like the reco `Vtx_position`) |
| `GenPart_productionTime`, `GenPart_properLifetime` | `VP(i,4)`, `VP(i,5)` | generator-level time / lifetime |

For reco↔truth matching the back-pointer currently lives on the analysis
side: match `TracRaw_*` / `EmShower_*` / etc. to `GenPart_*` by proximity in
(θ, φ, 1/p). Direct per-hit truth back-pointers (e.g. `VdAssocHit_genPartIdx`)
would require the simulation-time hit banks and are not included in this
pass — see the "MC truth" section of `docs/PHDST_RAW_NANOAOD_PLAN.md` for the
Tier-B plan.

### Vertices (`Vtx_*`)

From the PV bank chain at `LQ(LDTOP-1)`. Status bits: 1 = dummy, 2 = secondary,
3 = sec-hadronic, 4 = sim.

| field | meaning |
|---|---|
| `Vtx_position` | (X, Y, Z) in cm |
| `Vtx_chi2`, `Vtx_ndf` | fit quality |
| `Vtx_nOutgoing` | vertex multiplicity |
| `Vtx_massCode` | origin-particle mass code |
| `Vtx_statusBits` | see above |
| `Vtx_errXX / XY / YY / XZ / YZ / ZZ` | symmetric 3×3 error matrix |

## shortDST ↔ fullDST

| info source → | `simana.sdst` (shortDST) | `simana.fadana` (fullDST) |
|---|:---:|:---:|
| `Event_*`, `Vtx_*`, `TracRaw_*` | ✓ | ✓ |
| `EmShower_*` + `EmLayer_*` | ✓ | ✓ |
| `HadShower_*` + `HadHit_*` | ✓ | ✓ |
| `Stic_*`, `MuidRaw_*`, `ElidRaw_*`, `MtpcRaw_*` | ✓ | ✓ |
| `VdAssocHit_*`, `VdUnassocHit_*` (MVDH bank) | **✓** | — |
| `TrackElement_*` (per-detector 3-D hits) | — | **✓** |
| `GenPart_*` (MC truth) | ✓ (MC only) | ✓ (MC only) |

For a proper hit-level refit run the reader on **both** files and merge the
two ROOT outputs by `(Event_runNumber, Event_eventNumber)`.

`Delphi-Sim-Pipeline` branch `feature/cmssw-el9-base` preserves both files
in the output directory so the merge is always possible.

## Build notes

Against the CVMFS software stack (LCG_107 + ROOT 6.34 + DELPHI release) in
a `cmssw/el9` singularity image, the target compiles and links cleanly.
See `.github/workflows/build.yml` for the exact recipe — that's what PR CI
uses.

The executable links to libphdst / libdstana / libpxdst / libunfied etc.
plus the CERNLIB set from DELPHI CVMFS, and at runtime the DELPHI binaries
need libgfortran-5, libXm, libXp, libquadmath from the host — the
Delphi-Sim-Pipeline's `container/run_singularity.sh` bind-mounts host
`/lib64` at `/host_lib64` for exactly this reason.

## Scripts

| script | what |
|---|---|
| `scripts/inspect_tracking.C` | aggregate tracking statistics: χ²/ndf, VD hits per track, VD \|R\| histogram, TPC dE/dx, straight-line RPhi residual demo |
| `scripts/make_plots.C` | 14 physics-validation PNGs covering all major collections |
| `scripts/end_to_end_smoke.sh` | run a full Pythia8 → DELSIM → delphi-raw-nanoaod chain locally (cmssw/el9 singularity + `Delphi-Sim-Pipeline/container/run_singularity.sh`), on both `.sdst` and `.fadana`, and dump a per-collection entry count sanity report |
| `scripts/raw_to_rawdst_nanoaod.sh` | run raw open-data → DETRAW raw-DST → bank listing → raw nanoAOD, or inspect an existing raw-DST |

The two `.C` files are vanilla ROOT C++ macros — run with
`root -l -q -b '<name>("input.root", ...)'`. The smoke test is a bash
driver; see the header comment in the script for flags and prereqs.

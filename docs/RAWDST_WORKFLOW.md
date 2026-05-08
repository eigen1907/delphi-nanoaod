# raw -> raw-DST -> raw nanoAOD workflow

This branch changes `delphi-raw-nanoaod` from a shortDST/fullDST-only reader
into a reader that can also consume DELANA raw-DST output. The motivation is:

- raw open-data exists on EOS while centrally available full-DST is not always
  present for the samples we want;
- DETRAW raw-DST keeps detector-near RAW/Rxxx banks that are useful as
  lower-level ML inputs.

## Input layers

| Layer | Location / format | How this workflow uses it |
|---|---|---|
| raw open-data | `/eos/opendata/delphi/raw-data/.../*.sl` | Run DELANA with DETRAW enabled to produce a raw-DST file. |
| reconstructed collision data | `/eos/opendata/delphi/collision-data/.../*.al` | Already a long-DST-like input; can still be read directly by `delphi-raw-nanoaod`, but it is not the new raw-DST path. |
| raw-DST | DELANA output with `INPOUT 'R' 'RTD'` and `DETRAW ...` | Contains separate `RAW`, `TAN`, and `DST` records for the same run/event. |
| raw nanoAOD | ROOT RNTuple `Events` | Stores generic raw-bank metadata/payload plus the existing PHDST object-level collections. |

## What is accessible

`delphi-raw-bank-lister` walks the PHDST-visible ZEBRA tops:

- `LRTOP` -> RAW records
- `LDTOP` -> DST records
- `LTTOP` -> TANAGRA records
- `LITOP` / `LRTINT` -> intermediate/raw-intermediate records when present

On the CVMFS Event Server raw-DST smoke sample
`evserv/tests/EvtServer.20080530.211504/R84078_E10815_et_al.dst`, the lister
sees these RAW/Rxxx-like banks:

```text
RAW RAWD RAWS RAWV REMF RFCA RFCB RHAC RHOF RHP0 RHP1 RID RLUM
RMUB RMUF RMUS ROD RRB0 RRB1 RRF0 RRF1 RSTC RTOF RTP RTP0 RTP1 RVD
```

It also sees `RIFC` on an intermediate/raw-like record. Shallow RAW records show
the expected detector raw banks such as `RTP`, `RVD`, `RID`, `RTP0`, and `RTP1`.

The raw nanoAOD now exposes this generic bank layer:

| Field group | Meaning |
|---|---|
| `Event_recordType` | PHDST record type, e.g. `RAW`, `TAN`, `DST`, `CPT`, `BOF`. |
| `RawBank_*` | One row per scanned RAW/Rxxx bank: name, top-store code, ZEBRA address, tree depth, parent/link, `nlinks`, `ndata`, and payload range. |
| `RawWord_*` | Bounded payload words per raw bank, stored as both integer (`RawWord_i`) and float (`RawWord_f`) views. |

Top-store codes in `RawBank_top` are:

| code | top |
|---:|---|
| 0 | `LRTOP` |
| 1 | `LDTOP` |
| 2 | `LTTOP` |
| 3 | `LITOP` |
| 4 | `LRTINT` |

The existing object-level collections are still filled when the current record
has the matching DST banks:

- `Event_*`, `Vtx_*`
- `TracRaw_*`, `TrackElement_*`, `VdAssocHit_*`, `VdUnassocHit_*`, `MtpcRaw_*`
- `EmShower_*`, `EmLayer_*`, `HadShower_*`, `HadHit_*`, `Stic_*`
- `MuidRaw_*`, `ElidRaw_*`
- `GenPart_*` for MC records with simulation banks

Important operational detail: raw-DST is multi-record. The raw detector banks
live on `Event_recordType == "RAW"` entries; reconstructed PA/DST objects live
on `Event_recordType == "DST"` entries. Downstream code should join or group by
`(Event_runNumber, Event_eventNumber)` and keep the record type as an input
axis.

## Workflow script

Build first:

```bash
source setup.sh
cmake --build build --target delphi-raw-nanoaod delphi-raw-bank-lister -j4
```

Run from a raw open-data file:

```bash
./delphi-raw-nanoaod/scripts/raw_to_rawdst_nanoaod.sh \
    --raw /eos/opendata/delphi/raw-data/y94/<VID>/<VID>.<SEQ>.sl \
    --run 46004 \
    --out /tmp/rawdst_run46004 \
    --events 100 \
    --raw-bank-max-words 64
```

Run from an existing raw-DST:

```bash
./delphi-raw-nanoaod/scripts/raw_to_rawdst_nanoaod.sh \
    --raw-dst /path/to/rawdst.dst \
    --out /tmp/rawdst_check \
    --events 100
```

The script writes:

- `<tag>.rawdst` when starting from raw input;
- `<tag>.banks.log` from `delphi-raw-bank-lister`;
- `<tag>.raw_nanoaod.root` from `delphi-raw-nanoaod`;
- `<tag>.reader.log` with `nRawBank` / `nRawWord` progress lines.

For raw-DST creation, the script chooses a DELANA version/title card from the
run number, patches the title to:

```text
INPOUT    'R'     'RTD'      20      21        'LD'    'LD'
STRNAM   'DST'
ALLOUT   TRUE
DETRAW   9 11 16 23
```

Use `--title`, `--delana-bin`, or `--detraw` when a run period needs a more
specific steering card.

## Smoke result

Verified on:

```text
/cvmfs/delphi.cern.ch/releases/rhel-9-x86_64/latest/evserv/tests/EvtServer.20080530.211504/R84078_E10815_et_al.dst
```

The writer produced raw-bank entries on RAW records:

```text
run=84078 evt=10815 Event_recordType=RAW nRawBank=27 nRawWord=336
run=88588 evt=2459  Event_recordType=RAW nRawBank=27 nRawWord=336
```

and a DST record for the same `run/event` with reconstructed content:

```text
run=84078 evt=10815 Event_recordType=DST nTrac=1 nRawBank=1 nRawWord=16
```

This confirms the branch can preserve raw-bank access and reconstructed PHDST
objects in the same output schema.

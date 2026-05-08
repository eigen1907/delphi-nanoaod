#!/bin/bash
# Convert DELPHI raw open-data into a DETRAW raw-DST and then into raw nanoAOD.
#
# The important difference from the normal full-DST flow is the DELANA title
# patch below: INPOUT writes RAW + TANAGRA + DST output and DETRAW asks DELANA
# to keep selected detector raw banks in the DST record. delphi-raw-nanoaod
# then stores those RAW/Rxxx banks in RawBank_* / RawWord_*.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"

DELPHI_RELEASE_DIR=${DELPHI_RELEASE_DIR:-/cvmfs/delphi.cern.ch/releases/rhel-9-x86_64/latest}
EVSERV_HOME=${EVSERV_HOME:-$DELPHI_RELEASE_DIR/evserv}
RAW_BIN=${RAW_BIN:-$REPO_ROOT/build/delphi-raw-nanoaod/delphi-raw-nanoaod}
BANK_LISTER=${BANK_LISTER:-$REPO_ROOT/build/delphi-raw-nanoaod/delphi-raw-bank-lister}

RAW=
RAW_DST=
ROOT_OUT=
RUN=
TITLE=
DELANA_BIN=
OUT=${OUT:-/tmp/rawdst_nanoaod}
TAG=
EVENTS=${EVENTS:-100}
LISTER_EVENTS=${LISTER_EVENTS:-3}
RAW_BANK_MAX_WORDS=${RAW_BANK_MAX_WORDS:-64}
DETRAW=${DETRAW:-"9 11 16 23"}
SKIP_RAWDST=0
SKIP_NANOAOD=0

usage() {
    sed -n '2,70p' "$0"
    cat <<USAGE

Usage:
  raw_to_rawdst_nanoaod.sh --raw RAW.sl --run RUN [options]
  raw_to_rawdst_nanoaod.sh --raw-dst existing.dst [options]

Options:
  --raw PATH              Raw open-data file, e.g. /eos/opendata/delphi/raw-data/y94/...
  --run N                 Run number used to choose DELANA version/title for raw input
  --raw-dst PATH          Existing raw-DST to inspect or output raw-DST path
  --root PATH             Output raw nanoAOD ROOT path
  --out DIR               Work/output directory (default: /tmp/rawdst_nanoaod)
  --tag NAME              Output tag
  --title PATH            Override DELANA title card
  --delana-bin PATH       Override DELANA executable
  --events N              Max events for delphi-raw-nanoaod (default: 100)
  --lister-events N       Max events for delphi-raw-bank-lister (default: 3)
  --detraw "LIST"         DETRAW detector list (default: "9 11 16 23")
  --raw-bank-max-words N  RawWord payload words per bank, -1 means all (default: 64)
  --skip-rawdst           Skip DELANA and use --raw-dst as input
  --skip-nanoaod          Only create/list raw-DST
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --raw) RAW=$2; shift 2;;
        --run) RUN=$2; shift 2;;
        --raw-dst) RAW_DST=$2; shift 2;;
        --root) ROOT_OUT=$2; shift 2;;
        --out) OUT=$2; shift 2;;
        --tag) TAG=$2; shift 2;;
        --title) TITLE=$2; shift 2;;
        --delana-bin) DELANA_BIN=$2; shift 2;;
        --events) EVENTS=$2; shift 2;;
        --lister-events) LISTER_EVENTS=$2; shift 2;;
        --detraw) DETRAW=$2; shift 2;;
        --raw-bank-max-words) RAW_BANK_MAX_WORDS=$2; shift 2;;
        --skip-rawdst) SKIP_RAWDST=1; shift;;
        --skip-nanoaod) SKIP_NANOAOD=1; shift;;
        -h|--help) usage; exit 0;;
        *) echo "Unknown flag: $1" >&2; usage; exit 2;;
    esac
done

[[ -n "$RAW" || -n "$RAW_DST" ]] || { echo "ERROR: pass --raw or --raw-dst" >&2; exit 2; }
[[ -n "$RAW" || "$SKIP_RAWDST" -eq 1 || -f "$RAW_DST" ]] || {
    echo "ERROR: --raw-dst is being used as input, but it does not exist: $RAW_DST" >&2
    exit 2
}
[[ -z "$RAW" ]] && SKIP_RAWDST=1
if [[ "$SKIP_RAWDST" -eq 0 && -z "$RUN" && ( -z "$TITLE" || -z "$DELANA_BIN" ) ]]; then
    echo "ERROR: --run is required for raw input unless --title/--delana-bin are both supplied" >&2
    exit 2
fi

choose_delana() {
    local run=$1 version title exe
    if   (( run <= 20000 )); then version=v90e_rd; title=cards43_90e.tit;     exe=delana43_rd.exe
    elif (( run <= 28900 )); then version=v91f_rd; title=cards43_91f.tit;     exe=delana43_rd.exe
    elif (( run <= 36780 )); then version=v92e_rd; title=cards43_92e.tit;     exe=delana43_rd.exe
    elif (( run <= 43100 )); then version=v93d_rd; title=cards43_93d.tit;     exe=delana43_rd.exe
    elif (( run <= 55750 )); then version=v94c_rd; title=cards43_94c.tit;     exe=delana43_rd.exe
    elif (( run <= 64634 )); then version=v95d_rd; title=cards43_95d.tit;     exe=delana43_rd.exe
    elif (( run <= 67699 )); then version=v96g_rd; title=cards45_96g_z0.tit;  exe=delana45_rd.exe
    elif (( run <= 69585 )); then version=v96g_rd; title=cards45_96g_161.tit; exe=delana45_rd.exe
    elif (( run <= 70871 )); then version=v96g_rd; title=cards45_96g_z0.tit;  exe=delana45_rd.exe
    elif (( run <= 71473 )); then version=v96g_rd; title=cards45_96g_172.tit; exe=delana45_rd.exe
    elif (( run <= 71911 )); then version=v96g_rd; title=cards45_96g_z0.tit;  exe=delana45_rd.exe
    elif (( run <= 75334 )); then version=v97g_rd; title=cards45_97g_z0.tit;  exe=delana45_rd.exe
    elif (( run <= 78539 )); then version=v97g_rd; title=cards45_97g_184.tit; exe=delana45_rd.exe
    elif (( run <= 79182 )); then version=v97g_rd; title=cards45_97g_130.tit; exe=delana45_rd.exe
    elif (( run <= 80978 )); then version=v97g_rd; title=cards45_97g_184.tit; exe=delana45_rd.exe
    elif (( run <= 83222 )); then version=v98e_rd; title=cards45_98e_z0.tit;  exe=delana45_rd.exe
    elif (( run <= 89752 )); then version=v98e_rd; title=cards45_98e_188.tit; exe=delana45_rd.exe
    elif (( run <= 101799 )); then version=v99e_rd; title=cards45_99e_z0.tit; exe=delana45_rd.exe
    elif (( run <= 108360 )); then version=v99e_rd; title=cards45_99e_200.tit; exe=delana45_rd.exe
    elif (( run <= 115456 )); then version=va0e_rd; title=cards45_00e_z0.tit; exe=delana45_rd.exe
    else version=va0u_rd; title=cards45_00u_he.tit; exe=delana45_rd.exe
    fi
    TITLE=${TITLE:-$EVSERV_HOME/farm/$title}
    DELANA_BIN=${DELANA_BIN:-$DELPHI_RELEASE_DIR/simana/$version/bin/$exe}
}

patch_title() {
    local input=$1 output=$2
    awk -v detraw="$DETRAW" '
        BEGIN { wrote_detraw = 0 }
        /^[[:space:]]*INPOUT/ {
            print "INPOUT    '\''R'\''     '\''RTD'\''      20      21        '\''LD'\''    '\''LD'\''"
            next
        }
        /^[[:space:]]*STRNAM/ {
            print "STRNAM   '\''DST'\''"
            print "ALLOUT  TRUE"
            next
        }
        /^[[:space:]]*ALLOUT/ { next }
        /^[[:space:]]*LUMRAW/ {
            print "LUMRAW    FALSE"
            if (detraw != "" && wrote_detraw == 0) {
                print "DETRAW    " detraw
                wrote_detraw = 1
            }
            next
        }
        {
            gsub(/FARM[[:space:]]+TRUE/, "FARM    FALSE")
            gsub(/DBFARM[[:space:]]+TRUE/, "DBFARM    FALSE")
            gsub(/DATEND[[:space:]]+991231/, "DATEND  9999999")
            if ($0 ~ /^[[:space:]]*STPREQ/) next
            if ($0 ~ /^[[:space:]]*LTPCAL/) sub(/^[[:space:]]*/, "C-- ")
            if ($0 ~ /^[[:space:]]*DBVIRT/) sub(/^[[:space:]]*/, "C-- ")
            if ($0 ~ /^[[:space:]]*BUNCHT/) sub(/^[[:space:]]*/, "C-- ")
            print
        }
        END {
            if (detraw != "" && wrote_detraw == 0) {
                print "DETRAW    " detraw
            }
        }
    ' "$input" > "$output"
}

mkdir -p "$OUT"
TAG=${TAG:-$(basename "${RAW:-$RAW_DST}" | sed 's/\.[^.]*$//')_$(date +%Y%m%d_%H%M%S)}
WORK="$OUT/work_$TAG"
mkdir -p "$WORK"

RAW_DST=${RAW_DST:-$OUT/${TAG}.rawdst}
ROOT_OUT=${ROOT_OUT:-$OUT/${TAG}.raw_nanoaod.root}
LISTER_LOG="$OUT/${TAG}.banks.log"
READER_LOG="$OUT/${TAG}.reader.log"
DELANA_LOG="$OUT/${TAG}.delana.log"

# setup.sh wires DELPHI CVMFS + ROOT. It is harmless if the caller already
# sourced it.
# shellcheck disable=SC1091
set +u
source "$REPO_ROOT/setup.sh" >/dev/null 2>&1
set -u
if ! command -v pdl2pdl >/dev/null 2>&1; then
    export PATH="/cvmfs/delphi.cern.ch/scripts:$DELPHI_RELEASE_DIR/scripts:$PATH"
fi

if [[ "$SKIP_RAWDST" -eq 0 ]]; then
    [[ -f "$RAW" ]] || { echo "ERROR: raw input does not exist: $RAW" >&2; exit 1; }
    [[ -n "$TITLE" && -n "$DELANA_BIN" ]] || choose_delana "$RUN"
    [[ -r "$TITLE" ]] || { echo "ERROR: cannot read title card: $TITLE" >&2; exit 1; }
    [[ -x "$DELANA_BIN" ]] || { echo "ERROR: cannot execute DELANA binary: $DELANA_BIN" >&2; exit 1; }

    echo "=== raw -> raw-DST ==="
    echo "  raw        = $RAW"
    echo "  run        = $RUN"
    echo "  title      = $TITLE"
    echo "  delana     = $DELANA_BIN"
    echo "  detraw     = $DETRAW"
    echo "  raw-dst    = $RAW_DST"

    (
        cd "$WORK"
        patch_title "$TITLE" delana.rawdst.title
        ln -sf delana.rawdst.title fort.29
        ln -sf "$RAW" raw.input
        export dinp01=raw.input
        export dout01=delana.rawdst
        if command -v ddbass >/dev/null 2>&1 && [[ -d "${DELPHI_DDB:-}" ]]; then
            mkdir -p delphiddb
            for f in DBcalb.dat DBgeom.dat DBlepm.dat DBmisc.dat DBrunt.dat DBscon.dat DBsysf.dat; do
                ln -sf "$DELPHI_DDB/$f" "delphiddb/$f"
            done
            ddbass "$WORK/delphiddb" >/dev/null 2>&1 || true
        fi
        "$DELANA_BIN" | tee "$DELANA_LOG"
        [[ -f delana.rawdst ]] || { echo "ERROR: DELANA did not produce delana.rawdst" >&2; exit 1; }
        mv delana.rawdst "$RAW_DST"
        cp delana.rawdst.title "$OUT/${TAG}.delana.title"
    )
fi

echo "=== raw-DST bank listing ==="
"$BANK_LISTER" --pdlinput "$RAW_DST" --max-events "$LISTER_EVENTS" --max-depth 12 --max-links 128 \
    > "$LISTER_LOG" 2>&1 || { tail -80 "$LISTER_LOG"; exit 1; }
if grep -q "%PHDST-E" "$LISTER_LOG"; then
    tail -80 "$LISTER_LOG"
    exit 1
fi
echo "  bank log   = $LISTER_LOG"

if [[ "$SKIP_NANOAOD" -eq 0 ]]; then
    [[ -x "$RAW_BIN" ]] || { echo "ERROR: missing raw nanoAOD binary: $RAW_BIN" >&2; exit 1; }
    echo "=== raw-DST -> raw nanoAOD ==="
    "$RAW_BIN" --pdlinput "$RAW_DST" --output "$ROOT_OUT" --max-events "$EVENTS" \
        --raw-bank-max-depth 12 --raw-bank-max-links 128 --raw-bank-max-words "$RAW_BANK_MAX_WORDS" \
        > "$READER_LOG" 2>&1 || { tail -80 "$READER_LOG"; exit 1; }
    if grep -q "%PHDST-E" "$READER_LOG" || [[ ! -s "$ROOT_OUT" ]]; then
        tail -80 "$READER_LOG"
        exit 1
    fi
    echo "  root       = $ROOT_OUT"
    echo "  reader log = $READER_LOG"
fi

echo "=== done ==="
echo "  raw-dst    = $RAW_DST"
echo "  bank log   = $LISTER_LOG"
[[ "$SKIP_NANOAOD" -eq 1 ]] || echo "  root       = $ROOT_OUT"

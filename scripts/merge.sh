#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-/eos/home-j/joshin/micromamba/envs/delphi-analysis/bin/python}
FLORIAN_DIR=${1:-"${PROJECT_ROOT}/output/florian"}
MERGE_SCRIPT="${PROJECT_ROOT}/python/merge.py"

count=0
while IFS= read -r -d '' nano; do
  job_dir=$(dirname "${nano}")
  raw_sdst="${job_dir}/nanoaod_raw_sdst.root"
  raw_fadana="${job_dir}/nanoaod_raw_fadana.root"
  output="${job_dir}/nanoaod_merged.root"

  if [[ ! -f "${raw_sdst}" || ! -f "${raw_fadana}" ]]; then
    echo "[skip] missing raw inputs in ${job_dir}"
    continue
  fi

  echo "[merge] ${job_dir}"
  "${PYTHON}" "${MERGE_SCRIPT}" \
    --sdst "${nano}" \
    --raw-sdst "${raw_sdst}" \
    --raw-fadana "${raw_fadana}" \
    --output "${output}"
  count=$((count + 1))
done < <(find "${FLORIAN_DIR}" -type f -name nanoaod.root -print0 | sort -z)

echo "[done] wrote ${count} nanoaod_merged.root file(s) under ${FLORIAN_DIR}"

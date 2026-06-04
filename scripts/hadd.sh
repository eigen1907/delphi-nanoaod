#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-/eos/home-j/joshin/micromamba/envs/delphi-analysis/bin/python}
FLORIAN_DIR=${1:-"${PROJECT_ROOT}/output/florian"}
HADD_SCRIPT="${PROJECT_ROOT}/python/hadd.py"

count=0
while IFS= read -r -d '' sample_dir; do
  
  IFS=$'\n' read -r -d '' -a inputs < <(find "${sample_dir}" -type f -path "*/job_*/nanoaod_merged.root" | sort) || true

  if (( ${#inputs[@]} == 0 )); then
    echo "[skip] no nanoaod_merged.root files under ${sample_dir}"
    continue
  fi

  sample_name=$(basename "${sample_dir}")
  sample_name=${sample_name##*_}
  output="${FLORIAN_DIR}/${sample_name}.root"

  echo "[hadd] ${output} (${#inputs[@]} files)"
  "${PYTHON}" "${HADD_SCRIPT}" "${output}" "${inputs[@]}"
  count=$((count + 1))
done < <(find "${FLORIAN_DIR}" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)

echo "[done] wrote ${count} RNTuple file(s) under ${FLORIAN_DIR}"
#!/usr/bin/env bash
set -euo pipefail

MAX_EVENTS=${MAX_EVENTS:-1000}

SAMPLES=(
  "sh_kora_e091_r94_2l_c2" # tautau
  "sh_dssh_b94_2l_c2"      # D0  -> K pi
  "sh_bbsd_b94_2l_c2"      # phi -> K K
  "lo_dymu_r94_1l_b3"      # mumu
  "short94_c2"             # data sample
)

mkdir -p output/sanity-check/{nanoaod,raw-nanoaod,merged}

for sample in "${SAMPLES[@]}"; do
  echo "[INFO] Processing ${sample}"

  nano_file="output/sanity-check/nanoaod/${sample}.root"
  raw_file="output/sanity-check/raw-nanoaod/${sample}.root"
  merged_file="output/sanity-check/merged/${sample}.root"

  ./build/delphi-nanoaod/delphi-nanoaod \
    --nickname "${sample}" \
    --mc \
    --config config/delphi-nanoaod.yaml \
    --output "${nano_file}" \
    -m "${MAX_EVENTS}"

  ./build/delphi-raw-nanoaod/delphi-raw-nanoaod \
    --nickname "${sample}" \
    --output "${raw_file}" \
    -m "${MAX_EVENTS}"

  python3 python/merge_raw_nanoaod.py \
    "${nano_file}" \
    "${raw_file}" \
    "${merged_file}" \
    --strict

  echo "[INFO] Done: ${sample}"
done

echo "[INFO] All samples completed."
#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"

cd "${repo_root}"

data_root="${repo_root}/data"

prepare_split() {
  local name="$1"
  local script="$2"
  local out_dir="${data_root}/${name}"

  if [[ -f "${out_dir}/train.parquet" ]]; then
    echo "Found ${out_dir}/train.parquet; skipping ${name}."
    return 0
  fi

  mkdir -p "${out_dir}"
  python3 "examples/data_preprocess/${script}" --local_save_dir "${out_dir}"
}

prepare_split math competition_math.py
prepare_split amc23 amc23.py
prepare_split aime2024 aime2024.py
prepare_split aime2025 aime2025.py
prepare_split aime2026 aime2026.py

echo "Prepared NFPO forward-trace data under ${data_root}"

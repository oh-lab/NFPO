#!/usr/bin/env bash
set -euo pipefail

OVERLAY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_ROOT="${1:-}"

if [[ -z "${TARGET_ROOT}" ]]; then
  echo "Usage: $0 /path/to/verl-checkout" >&2
  exit 1
fi

if [[ ! -d "${TARGET_ROOT}" ]]; then
  echo "Target root does not exist: ${TARGET_ROOT}" >&2
  exit 1
fi

while IFS= read -r relpath; do
  [[ -z "${relpath}" ]] && continue
  mkdir -p "${TARGET_ROOT}/$(dirname "${relpath}")"
  cp "${OVERLAY_DIR}/${relpath}" "${TARGET_ROOT}/${relpath}"
done < "${OVERLAY_DIR}/MANIFEST.txt"

echo "Applied NFPO forward-trace overlay to ${TARGET_ROOT}"

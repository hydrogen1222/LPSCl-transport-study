#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash 03_scripts/run_md_batch.sh <smoke|core|contrast>" >&2
  exit 1
fi

GROUP="$1"

case "$GROUP" in
  smoke)
    STRUCTURES=(
      bulk_ordered
      gb_Sigma3_t3
    )
    ;;
  core)
    STRUCTURES=(
      bulk_ordered
      bulk_Li_vac_c1_s1
      gb_Sigma3_t3
      gb_Sigma3_t3_Li_vac_c1_s1
      gb_Sigma3_t3_Li_vac_c2_s2
    )
    ;;
  contrast)
    STRUCTURES=(
      gb_Sigma3_t1
      gb_Sigma3_t1_Li_vac_c1_s1
      gb_Sigma3_t2_Li_vac_c1_s2
    )
    ;;
  *)
    echo "Unknown batch group: $GROUP" >&2
    exit 1
    ;;
esac

echo "[INFO] Batch group: $GROUP"
printf '[INFO] Structures: %s\n' "${STRUCTURES[*]}"

for sid in "${STRUCTURES[@]}"; do
  echo
  echo "===================================================================="
  echo "[INFO] Running MD for: $sid"
  echo "===================================================================="
  bash "$(cd "$(dirname "$0")" && pwd)/run_md_single.sh" "$sid"
done

echo "[INFO] Batch group '$GROUP' completed."

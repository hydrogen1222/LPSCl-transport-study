#!/usr/bin/env bash
set -euo pipefail

BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BUNDLE_ROOT"

TEMPERATURES="${TEMPERATURES:-600 700 800}"
STEPS="${STEPS:-20000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-100}"
STRUCTURES="${STRUCTURES:-bulk_ordered gb_Sigma3_t3 gb_Sigma3_t3_Li_vac_c1_s1}"

echo "[INFO] bundle_root      = $BUNDLE_ROOT"
echo "[INFO] temperatures_K  = $TEMPERATURES"
echo "[INFO] steps           = $STEPS"
echo "[INFO] save_interval   = $SAVE_INTERVAL"
echo "[INFO] structures      = $STRUCTURES"

for structure_id in $STRUCTURES; do
  for temp in $TEMPERATURES; do
    echo
    echo "[RUN] structure=$structure_id temp=${temp}K steps=$STEPS"
    MD_TEMP="$temp" \
    MD_STEPS="$STEPS" \
    MD_SAVE_INTERVAL="$SAVE_INTERVAL" \
    bash 03_scripts/run_md_single.sh "$structure_id"
  done
done

echo
echo "[DONE] conductivity batch finished"

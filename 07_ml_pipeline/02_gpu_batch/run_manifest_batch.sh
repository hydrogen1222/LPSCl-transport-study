#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MANIFEST_PATH="${MANIFEST_PATH:-$SCRIPT_DIR/next_md_labeling_manifest.csv}"
BUNDLE_ROOT="${BUNDLE_ROOT:-$PROJECT_ROOT/06_cloud_vm_gpu_bundle}"
INPUT_MODE="${INPUT_MODE:-auto}"

if [[ ! -f "$MANIFEST_PATH" ]]; then
  echo "Manifest not found: $MANIFEST_PATH" >&2
  exit 1
fi

if [[ ! -f "$BUNDLE_ROOT/03_scripts/run_md_single.sh" ]]; then
  echo "run_md_single.sh not found under bundle root: $BUNDLE_ROOT" >&2
  exit 1
fi

echo "[INFO] project_root   = $PROJECT_ROOT"
echo "[INFO] manifest_path  = $MANIFEST_PATH"
echo "[INFO] bundle_root    = $BUNDLE_ROOT"
echo "[INFO] input_mode     = $INPUT_MODE"

tail -n +2 "$MANIFEST_PATH" | while IFS=',' read -r priority structure_id structure_class translation_state vacancy_count target_temperature_K target_md_steps save_interval_steps replicate_id recommended_reason; do
  echo
  echo "[RUN] priority=$priority structure=$structure_id temp=${target_temperature_K}K steps=$target_md_steps replicate=$replicate_id"
  MD_TEMP="$target_temperature_K" \
  MD_STEPS="$target_md_steps" \
  MD_SAVE_INTERVAL="$save_interval_steps" \
  RUN_NAME="md_${target_temperature_K}K_${target_md_steps}steps_${replicate_id}" \
  BUNDLE_ROOT="$BUNDLE_ROOT" \
  bash "$BUNDLE_ROOT/03_scripts/run_md_single.sh" "$structure_id" "$INPUT_MODE"
done

echo
echo "[DONE] manifest batch finished"

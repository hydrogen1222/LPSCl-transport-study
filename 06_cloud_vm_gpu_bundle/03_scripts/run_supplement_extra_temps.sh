#!/usr/bin/env bash
# ==============================================================================
# run_supplement_extra_temps.sh — 全部 20 结构补跑 500K 和 900K
#
# 目的: 将 Arrhenius 拟合从 3 点扩展到 5 点 (500/600/700/800/900 K)
#       使拟合更稳健，尤其改善那 3 个非单调 bulk vacancy 结构。
#
# 输出路径格式 (与现有数据不冲突):
#   04_runs/md/{sid}/md_500K_20000steps/md_500K_20000steps/
#   04_runs/md/{sid}/md_900K_20000steps/md_900K_20000steps/
#
# 用法:
#   bash 03_scripts/run_supplement_extra_temps.sh
# ==============================================================================
set -euo pipefail

BUNDLE_ROOT="${BUNDLE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
INPUT_MODE="auto"
STEPS=20000
SAVE_INTERVAL=100
TEMPERATURES="500 900"  # 只跑新的两个温度
DEVICE="${UMA_DEVICE:-cuda}"
LOG_FILE="${LOG_FILE:-$BUNDLE_ROOT/supplement_extratemps_$(date +%Y%m%d_%H%M%S).log}"

STRUCTURES=(
  bulk_ordered
  bulk_Li_vac_c1_s1
  bulk_Li_vac_c1_s2
  bulk_Li_vac_c2_s1
  bulk_Li_vac_c2_s2
  gb_Sigma3_t1
  gb_Sigma3_t1_Li_vac_c1_s1
  gb_Sigma3_t1_Li_vac_c1_s2
  gb_Sigma3_t1_Li_vac_c2_s1
  gb_Sigma3_t1_Li_vac_c2_s2
  gb_Sigma3_t2
  gb_Sigma3_t2_Li_vac_c1_s1
  gb_Sigma3_t2_Li_vac_c1_s2
  gb_Sigma3_t2_Li_vac_c2_s1
  gb_Sigma3_t2_Li_vac_c2_s2
  gb_Sigma3_t3
  gb_Sigma3_t3_Li_vac_c1_s1
  gb_Sigma3_t3_Li_vac_c1_s2
  gb_Sigma3_t3_Li_vac_c2_s1
  gb_Sigma3_t3_Li_vac_c2_s2
)

log() {
  local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
  echo "$msg"
  echo "$msg" >> "$LOG_FILE"
}

is_done() {
  local sid="$1" temp="$2"
  local run_name="md_${temp}K_${STEPS}steps"
  local run_dir="$BUNDLE_ROOT/04_runs/md/${sid}/${run_name}"
  local nested_dir="${run_dir}/${run_name}"
  [[ -f "${run_dir}/uma_results.json" ]] || [[ -f "${run_dir}/XDATCAR" ]] \
    || [[ -f "${nested_dir}/uma_results.json" ]] || [[ -f "${nested_dir}/XDATCAR" ]]
}

run_md() {
  local sid="$1" temp="$2" run_name="$3" mode="$4"
  set +e
  MD_TEMP="$temp" \
  MD_STEPS="$STEPS" \
  MD_SAVE_INTERVAL="$SAVE_INTERVAL" \
  RUN_NAME="$run_name" \
  UMA_DEVICE="$DEVICE" \
  UMA_INFERENCE_MODE="$mode" \
  bash "$BUNDLE_ROOT/03_scripts/run_md_single.sh" "$sid" "$INPUT_MODE" 2>&1 | tee -a "$LOG_FILE"
  local real_exit=${PIPESTATUS[0]}
  set -e
  return $real_exit
}

# turbo detection
INFERENCE_MODE="default"
if [[ "$DEVICE" == "cuda" ]]; then
  log "Testing turbo mode..."
  set +e
  MD_TEMP=600 MD_STEPS=5 MD_SAVE_INTERVAL=1 \
  RUN_NAME="smoke_extratemp_test" \
  UMA_DEVICE="$DEVICE" UMA_INFERENCE_MODE="turbo" \
  bash "$BUNDLE_ROOT/03_scripts/run_md_single.sh" "${STRUCTURES[0]}" "$INPUT_MODE" >> "$LOG_FILE" 2>&1
  smoke_exit=${PIPESTATUS[0]}
  set -e
  rm -rf "$BUNDLE_ROOT/04_runs/md/${STRUCTURES[0]}/smoke_extratemp_test" 2>/dev/null || true
  [[ $smoke_exit -eq 0 ]] && INFERENCE_MODE="turbo" && log "[OK] turbo" || log "[WARN] using default"
fi

# Count tasks
total_tasks=0
for sid in "${STRUCTURES[@]}"; do
  for temp in $TEMPERATURES; do
    total_tasks=$((total_tasks + 1))
  done
done

task_index=0; skip_count=0; done_count=0; fail_count=0

log "============================================================"
log " SUPPLEMENT: Extra Temperature Points (500K + 900K)"
log "============================================================"
log "structures  = ${#STRUCTURES[@]}"
log "temperatures = $TEMPERATURES"
log "total_tasks  = $total_tasks"
log "============================================================"

for sid in "${STRUCTURES[@]}"; do
  for temp in $TEMPERATURES; do
    task_index=$((task_index + 1))
    run_name="md_${temp}K_${STEPS}steps"

    if is_done "$sid" "$temp"; then
      skip_count=$((skip_count + 1))
      log "[SKIP] ($task_index/$total_tasks) ${sid} / ${run_name}"
      continue
    fi

    log ""
    log "[RUN]  ($task_index/$total_tasks) ${sid} / ${run_name}"
    start_ts=$(date +%s)
    run_ok=0
    if run_md "$sid" "$temp" "$run_name" "$INFERENCE_MODE"; then
      is_done "$sid" "$temp" && run_ok=1 || log "[WARN] output missing"
    fi
    end_ts=$(date +%s)
    elapsed_min=$(echo "scale=1; ($end_ts - $start_ts) / 60" | bc)

    if [[ $run_ok -eq 1 ]]; then
      done_count=$((done_count + 1)); log "[DONE] ${sid} / ${run_name} — ${elapsed_min} min"
    else
      fail_count=$((fail_count + 1)); log "[FAIL] ${sid} / ${run_name} — ${elapsed_min} min"
    fi
    log "[PROGRESS] done=$done_count skip=$skip_count fail=$fail_count remaining=$((total_tasks - task_index))"
  done
done

log ""
log "============================================================"
log " EXTRA TEMPERATURE BATCH COMPLETE"
log "============================================================"
log "total=$total_tasks done=$done_count skipped=$skip_count failed=$fail_count"

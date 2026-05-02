#!/usr/bin/env bash
# ==============================================================================
# run_supplement_multiseed.sh — 对统计不足的 bulk vacancy 结构补跑多随机种子 MD
#
# 目标: 对 3 个 D(T) 非单调的 bulk vacancy 结构，每温度补跑 seed=2,3 两条
#       独立 MD 轨迹（不同 Maxwell-Boltzmann 初速度），用于做统计平均。
#
# 输出路径格式:
#   04_runs/md/{sid}/md_{T}K_20000steps_seed{N}/md_{T}K_20000steps_seed{N}/
#
# is_done 检查: 同时检查 flat 和 nested 路径，避免重复计算。
# 原有的 md_{T}K_20000steps 视为 seed=1，不重跑。
#
# 用法:
#   bash 03_scripts/run_supplement_multiseed.sh
# ==============================================================================
set -euo pipefail

BUNDLE_ROOT="${BUNDLE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
ENV_DIR="${ENV_DIR:-$BUNDLE_ROOT/.venv}"
INPUT_MODE="auto"
STEPS=20000
SAVE_INTERVAL=100
TEMPERATURES="600 700 800"
DEVICE="${UMA_DEVICE:-cuda}"
LOG_FILE="${LOG_FILE:-$BUNDLE_ROOT/supplement_multiseed_$(date +%Y%m%d_%H%M%S).log}"

# 只跑这 3 个有问题的 bulk vacancy 结构
STRUCTURES=(
  bulk_Li_vac_c1_s1
  bulk_Li_vac_c1_s2
  bulk_Li_vac_c2_s1
)

# 补跑的种子编号 (seed=1 已有，补 2 和 3)
SEEDS=(2 3)

log() {
  local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
  echo "$msg"
  echo "$msg" >> "$LOG_FILE"
}

# 检查是否已完成
is_done_seed() {
  local sid="$1" temp="$2" seed="$3"
  local run_name="md_${temp}K_${STEPS}steps_seed${seed}"
  local run_dir="$BUNDLE_ROOT/04_runs/md/${sid}/${run_name}"
  local nested_dir="${run_dir}/${run_name}"
  [[ -f "${run_dir}/uma_results.json" ]] || [[ -f "${run_dir}/XDATCAR" ]] \
    || [[ -f "${nested_dir}/uma_results.json" ]] || [[ -f "${nested_dir}/XDATCAR" ]]
}

# 运行单条 MD
run_md_seed() {
  local sid="$1" temp="$2" run_name="$3" mode="$4" seed="$5"
  set +e
  MD_TEMP="$temp" \
  MD_STEPS="$STEPS" \
  MD_SAVE_INTERVAL="$SAVE_INTERVAL" \
  RUN_NAME="$run_name" \
  UMA_DEVICE="$DEVICE" \
  UMA_INFERENCE_MODE="$mode" \
  UMA_SEED="$seed" \
  bash "$BUNDLE_ROOT/03_scripts/run_md_single.sh" "$sid" "$INPUT_MODE" 2>&1 | tee -a "$LOG_FILE"
  local real_exit=${PIPESTATUS[0]}
  set -e
  return $real_exit
}

# ---- turbo detection (same as main batch) ----
INFERENCE_MODE="default"
if [[ "$DEVICE" == "cuda" ]]; then
  log "Testing turbo mode..."
  set +e
  MD_TEMP=600 MD_STEPS=5 MD_SAVE_INTERVAL=1 \
  RUN_NAME="smoke_seed_test" \
  UMA_DEVICE="$DEVICE" UMA_INFERENCE_MODE="turbo" \
  bash "$BUNDLE_ROOT/03_scripts/run_md_single.sh" "${STRUCTURES[0]}" "$INPUT_MODE" >> "$LOG_FILE" 2>&1
  smoke_exit=${PIPESTATUS[0]}
  set -e
  rm -rf "$BUNDLE_ROOT/04_runs/md/${STRUCTURES[0]}/smoke_seed_test" 2>/dev/null || true
  if [[ $smoke_exit -eq 0 ]]; then
    INFERENCE_MODE="turbo"
    log "[OK] turbo mode — all runs will use turbo"
  else
    INFERENCE_MODE="default"
    log "[WARN] turbo failed, using default"
  fi
fi

# ---- Main loop ----
total_tasks=$(( ${#STRUCTURES[@]} * 3 * ${#SEEDS[@]} ))
task_index=0
skip_count=0
done_count=0
fail_count=0

log "============================================================"
log " SUPPLEMENT: Multi-Seed MD for Noisy Bulk Vacancy Structures"
log "============================================================"
log "structures = ${STRUCTURES[*]}"
log "seeds      = ${SEEDS[*]}"
log "total_tasks = $total_tasks"
log "============================================================"

for sid in "${STRUCTURES[@]}"; do
  for temp in $TEMPERATURES; do
    for seed in "${SEEDS[@]}"; do
      task_index=$((task_index + 1))
      run_name="md_${temp}K_${STEPS}steps_seed${seed}"

      if is_done_seed "$sid" "$temp" "$seed"; then
        skip_count=$((skip_count + 1))
        log "[SKIP] ($task_index/$total_tasks) ${sid} / ${run_name} — already done"
        continue
      fi

      log ""
      log "===================================================================="
      log "[RUN]  ($task_index/$total_tasks) ${sid} / ${run_name}  [seed=$seed]"
      log "===================================================================="

      start_ts=$(date +%s)
      run_ok=0
      if run_md_seed "$sid" "$temp" "$run_name" "$INFERENCE_MODE" "$seed"; then
        if is_done_seed "$sid" "$temp" "$seed"; then
          run_ok=1
        else
          log "[WARN] completed but output missing"
        fi
      fi
      end_ts=$(date +%s)
      elapsed_min=$(echo "scale=1; ($end_ts - $start_ts) / 60" | bc)

      if [[ $run_ok -eq 1 ]]; then
        done_count=$((done_count + 1))
        log "[DONE] ${sid} / ${run_name} — ${elapsed_min} min"
      else
        fail_count=$((fail_count + 1))
        log "[FAIL] ${sid} / ${run_name} — ${elapsed_min} min"
      fi
      log "[PROGRESS] done=$done_count skip=$skip_count fail=$fail_count remaining=$((total_tasks - task_index))"
    done
  done
done

log ""
log "============================================================"
log " MULTI-SEED BATCH COMPLETE"
log "============================================================"
log "total=$total_tasks done=$done_count skipped=$skip_count failed=$fail_count"
log "============================================================"

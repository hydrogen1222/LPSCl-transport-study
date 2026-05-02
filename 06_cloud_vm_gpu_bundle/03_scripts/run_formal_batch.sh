#!/usr/bin/env bash
# ============================================================
# run_formal_batch.sh  (v2 — 2026-04-14 bugfix)
# 正式 formal MD 批量提交脚本
#
# 修复说明 (v2):
#   旧版 Bug: bash ... | tee 管道中 $? 捕获的是 tee 的退出码 (永远0)，
#             导致 turbo 失败后 exit_code=0，但 is_done 返回 false，
#             每条 MD 都会先跑一次 turbo 再跑一次 default。
#   新版修复: 1) 用 ${PIPESTATUS[0]} 捕获真实 bash 退出码
#            2) 启动时只做一次 5-step smoke test 确定全局 mode，
#               后续所有 run 统一用同一个 mode，不再逐条 fallback。
#
# 用法 (在 screen 中):
#   screen -S lpscl
#   cd ~/LPSCl_UMA_gpu/06_cloud_vm_gpu_bundle
#   bash 03_scripts/run_formal_batch.sh
# ============================================================
set -euo pipefail

BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BUNDLE_ROOT"

# ---- 配置 ----
TEMPERATURES="${TEMPERATURES:-600 700 800}"
STEPS="${STEPS:-20000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-100}"
INPUT_MODE="${INPUT_MODE:-auto}"
DEVICE="${UMA_DEVICE:-cuda}"
LOG_FILE="${LOG_FILE:-$BUNDLE_ROOT/formal_batch_$(date +%Y%m%d_%H%M%S).log}"

# 要跑的结构列表 (v3 — full 20 structures)
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

# ---- 日志函数 ----
log() {
  local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
  echo "$msg"
  echo "$msg" >> "$LOG_FILE"
}

# ---- 检查是否已完成 (v4: 检查双层嵌套路径) ----
# run_uma_md_direct.py 在 output_dir 下再建了一层以 RUN_NAME 命名的子目录，
# 所以实际路径是 md/{sid}/{run_name}/{run_name}/uma_results.json
is_done() {
  local sid="$1" temp="$2"
  local run_name="md_${temp}K_${STEPS}steps"
  local run_dir="$BUNDLE_ROOT/04_runs/md/${sid}/${run_name}"
  local nested_dir="${run_dir}/${run_name}"
  # Check both flat and nested paths
  [[ -f "${run_dir}/uma_results.json" ]] || [[ -f "${run_dir}/XDATCAR" ]] \
    || [[ -f "${nested_dir}/uma_results.json" ]] || [[ -f "${nested_dir}/XDATCAR" ]]
}

# ---- 运行单条 MD，使用 PIPESTATUS[0] 捕获真实退出码 ----
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
  # 关键修复: PIPESTATUS[0] 是管道第一个命令 (bash) 的退出码
  local real_exit=${PIPESTATUS[0]}
  set -e
  return $real_exit
}

# ---- 统计 ----
total_tasks=0
skip_count=0
done_count=0
fail_count=0
for sid in "${STRUCTURES[@]}"; do
  for temp in $TEMPERATURES; do
    total_tasks=$((total_tasks + 1))
  done
done

# ---- 一次性 turbo smoke test（只在启动时测一次） ----
# 不在每条 MD 上做 turbo→default fallback，而是确定全局 mode 后统一使用
INFERENCE_MODE="default"
if [[ "$DEVICE" == "cuda" ]] && [[ "${UMA_INFERENCE_MODE:-turbo}" == "turbo" ]]; then
  log "Testing turbo mode (5-step smoke test)..."
  SMOKE_SID="${STRUCTURES[0]}"

  set +e
  MD_TEMP=600 MD_STEPS=5 MD_SAVE_INTERVAL=1 \
  RUN_NAME="smoke_turbo_test" \
  UMA_DEVICE="$DEVICE" \
  UMA_INFERENCE_MODE="turbo" \
  bash "$BUNDLE_ROOT/03_scripts/run_md_single.sh" "$SMOKE_SID" "$INPUT_MODE" >> "$LOG_FILE" 2>&1
  smoke_exit=${PIPESTATUS[0]}
  set -e

  rm -rf "$BUNDLE_ROOT/04_runs/md/${SMOKE_SID}/smoke_turbo_test" 2>/dev/null || true

  if [[ $smoke_exit -eq 0 ]]; then
    INFERENCE_MODE="turbo"
    log "[OK] turbo mode available — ALL runs will use turbo"
  else
    INFERENCE_MODE="default"
    log "[WARN] turbo smoke test failed (exit=$smoke_exit) — ALL runs will use default"
  fi
fi

# ---- 主循环 ----
log "============================================================"
log " FORMAL MD BATCH RUNNER v2"
log "============================================================"
log "bundle_root     = $BUNDLE_ROOT"
log "structures      = ${#STRUCTURES[@]}"
log "temperatures    = $TEMPERATURES"
log "steps           = $STEPS"
log "save_interval   = $SAVE_INTERVAL"
log "device          = $DEVICE"
log "inference_mode  = $INFERENCE_MODE  (fixed at startup for ALL runs)"
log "total_tasks     = $total_tasks"
log "log_file        = $LOG_FILE"
log "============================================================"

task_index=0
for sid in "${STRUCTURES[@]}"; do
  for temp in $TEMPERATURES; do
    task_index=$((task_index + 1))
    run_name="md_${temp}K_${STEPS}steps"

    if is_done "$sid" "$temp"; then
      skip_count=$((skip_count + 1))
      log "[SKIP] ($task_index/$total_tasks) ${sid} / ${run_name} — already done"
      continue
    fi

    log ""
    log "===================================================================="
    log "[RUN]  ($task_index/$total_tasks) ${sid} / ${run_name}  [mode=$INFERENCE_MODE]"
    log "===================================================================="

    start_ts=$(date +%s)
    run_ok=0

    if run_md "$sid" "$temp" "$run_name" "$INFERENCE_MODE"; then
      if is_done "$sid" "$temp"; then
        run_ok=1
      else
        log "[WARN] script exited 0 but uma_results.json missing — marking as failed"
      fi
    fi

    end_ts=$(date +%s)
    elapsed=$(( end_ts - start_ts ))
    elapsed_min=$(echo "scale=1; $elapsed / 60" | bc)

    if [[ $run_ok -eq 1 ]]; then
      done_count=$((done_count + 1))
      log "[DONE] ${sid} / ${run_name} — ${elapsed_min} min"
    else
      fail_count=$((fail_count + 1))
      log "[FAIL] ${sid} / ${run_name} — ${elapsed_min} min"
    fi

    remaining=$(( total_tasks - task_index ))
    log "[PROGRESS] done=$done_count skip=$skip_count fail=$fail_count remaining=$remaining"
  done
done

log ""
log "============================================================"
log " BATCH COMPLETE"
log "============================================================"
log "total=$total_tasks  done=$done_count  skipped=$skip_count  failed=$fail_count"
log "log saved to: $LOG_FILE"
log "============================================================"

if [[ $fail_count -gt 0 ]]; then
  log "[WARN] $fail_count task(s) failed."
  exit 1
fi

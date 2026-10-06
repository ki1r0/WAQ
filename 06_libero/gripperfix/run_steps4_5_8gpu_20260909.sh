#!/usr/bin/env bash
set -Eeuo pipefail

G="/path/to/storage/ops/cosmos_policy_analysis_20260906/gripperfix_20260909"
TRAIN_SESSION="va2-gripperfix-8gpu-20260909"
TRAIN_PID="810086"
STEP3_LOG="$G/va2_gripperfix_sft_8gpu_20260909.log"
STEP3_MARKER="/path/to/storage/checkpoints/va2_libero10_cosmos_gripperfix_20260909/bf16_bs1_accum2_lr1e-5_25k/checkpoints/checkpoint_step_25000/COMPLETE"
STEP4="$G/A_line2/step4_export.sh"
STEP4_LOG="$G/A_line2/logs/export.log"
STEP5_ORIG="$G/A_line2/step5_features.sh"
STEP5_COPY="$G/A_line2/step5_features_8gpu_20260909.sh"
STEP5_LOG="$G/A_line2/logs/launcher.log"
CHAIN_LOG="$G/A_line2/logs/steps4_5_8gpu_20260909.log"

OUT_EXPORT="/path/to/storage/experiments/robotwin_clean_sft_20260903/export/from_va2_gripperfix_20260909_step25000_diffusers"
OUT_OPEN="/path/to/storage/datasets/libero_10_original_500_gripperfix_20260909/lerobot_v21_10hz_vae128/inference_state_cache/va2gf_step25000_server_prefix_8layer_20260909"
OUT_COSMOS="/path/to/storage/datasets/LIBERO-Cosmos-Policy-lingbotva-20260906-all2000/inference_state_cache/va2gf_step25000_server_prefix_8layer_20260909"
CATALOG_OPEN="$G/A_line2/catalog_open.json"
CATALOG_COSMOS="$G/A_line2/catalog_cosmos.json"
OVERLAY_OPEN="$G/A_line2/overlay_open.json"
OVERLAY_COSMOS="$G/A_line2/overlay_cosmos.json"
SIDECAR_OPEN="$G/A_line2/sidecar_open"
SIDECAR_COSMOS="$G/A_line2/sidecar_cosmos"
FEDERATED="$G/A_line2/federated"

mkdir -p "$(dirname "$CHAIN_LOG")"
exec > >(tee -a "$CHAIN_LOG") 2>&1

echo "[$(date --iso-8601=seconds)] CHAIN_START session=$TRAIN_SESSION train_pid=$TRAIN_PID"
echo "[$(date --iso-8601=seconds)] EXPECT_STEP3_MARKER $STEP3_MARKER"

if [[ -f "$STEP3_MARKER" ]]; then
  echo "[$(date --iso-8601=seconds)] STEP3_MARKER_ALREADY_PRESENT"
else
  pane_pid="$(tmux list-panes -t "$TRAIN_SESSION" -F '#{pane_pid}' 2>/dev/null | head -n 1 || true)"
  if [[ "$pane_pid" != "$TRAIN_PID" ]]; then
    echo "STEP3_ABORT pane_pid=$pane_pid expected=$TRAIN_PID"
    exit 20
  fi
  while [[ ! -f "$STEP3_MARKER" ]]; do
    if [[ ! -r "/proc/$TRAIN_PID/cmdline" ]]; then
      echo "STEP3_ABORT process_gone pid=$TRAIN_PID"
      exit 20
    fi
    cmdline="$(tr '\0' ' ' < "/proc/$TRAIN_PID/cmdline" || true)"
    case "$cmdline" in
      *"/path/to/lingbot-env/bin/deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port=29851 $G/run_va2_gripperfix_sft_8gpu.py"*)
        ;;
      *)
        echo "STEP3_ABORT unexpected_cmdline=$cmdline"
        exit 20
        ;;
    esac
    progress="$(tail -n 1 "$STEP3_LOG" 2>/dev/null || true)"
    echo "[$(date --iso-8601=seconds)] STEP3_WAIT $progress"
    sleep 60
  done
  echo "[$(date --iso-8601=seconds)] STEP3_MARKER_DETECTED"
fi

stat -c "STEP3_MARKER %n %F %s bytes %y" "$STEP3_MARKER"

echo "[$(date --iso-8601=seconds)] STEP4_START command=bash $STEP4"
bash "$STEP4"
if ! grep -qx "EXPORT_OK" "$STEP4_LOG"; then
  echo "STEP4_ABORT EXPORT_OK not found in $STEP4_LOG"
  exit 30
fi
echo "[$(date --iso-8601=seconds)] STEP4_CONFIRMED EXPORT_OK log=$STEP4_LOG"

if [[ ! -f "$STEP5_COPY" ]]; then
  echo "STEP5_ABORT missing independent copy: $STEP5_COPY"
  exit 31
fi
if ! cmp <(sed 's/DEVICES:-0,1,2,3,4,5,6,7/DEVICES:-1,2,3,4,5,6/' "$STEP5_COPY") "$STEP5_ORIG"; then
  echo "STEP5_ABORT copy differs from original beyond DEVICES default"
  exit 31
fi
for p in "$OUT_EXPORT" "$OUT_OPEN" "$OUT_COSMOS" "$CATALOG_OPEN" "$CATALOG_COSMOS" "$OVERLAY_OPEN" "$OVERLAY_COSMOS" "$SIDECAR_OPEN" "$SIDECAR_COSMOS" "$FEDERATED"; do
  if [[ -e "$p" ]]; then
    echo "STEP5_ABORT pre-existing output: $p"
    exit 32
  fi
done

echo "[$(date --iso-8601=seconds)] STEP5_START command=DEVICES=0,1,2,3,4,5,6,7 bash $STEP5_COPY"
DEVICES=0,1,2,3,4,5,6,7 bash "$STEP5_COPY"
if ! grep -q "STEP5_DONE" "$STEP5_LOG"; then
  echo "STEP5_ABORT STEP5_DONE not found in $STEP5_LOG"
  exit 33
fi
echo "[$(date --iso-8601=seconds)] STEP5_CONFIRMED STEP5_DONE log=$STEP5_LOG"
echo "[$(date --iso-8601=seconds)] CHAIN_STOP steps=4,5 only; step6/step7 not started"

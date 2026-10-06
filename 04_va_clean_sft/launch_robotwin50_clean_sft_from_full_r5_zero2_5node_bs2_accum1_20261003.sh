#!/usr/bin/env bash
set -Eeuo pipefail

die() { echo "ERROR: $*" >&2; exit 2; }
readonly REPO_ROOT="/path/to/shared-storage/software/robotwin_runtime/lingbot-va-original-rl-adaptive-cpb"
readonly PYTHON="/path/to/shared-storage/software/robotwin_runtime/lingbot-env/bin/python"
readonly PARENT_RUN_ROOT="/path/to/shared-storage/experiments/robotwin_clean_sft_from_full_r5_20261003_5node400k"
readonly RUN_ROOT="$PARENT_RUN_ROOT/zero2_5node_400k_bs2_accum1_retry5"
readonly RUNTIME_HOME_ROOT="$RUN_ROOT/runtime_home"
readonly CACHE_ROOT="$RUN_ROOT/cache"
readonly WRAPPER_PATH="$RUN_ROOT/train_robotwin50_clean_sft_from_full_r5_zero2_5node_bs2_accum1_20261003.py"
readonly LAUNCHER_PATH="$RUN_ROOT/launch_robotwin50_clean_sft_from_full_r5_zero2_5node_bs2_accum1_20261003.sh"
readonly DEEPSPEED_CONFIG="$RUN_ROOT/configs/deepspeed_zero2_bf16_bs2_accum1.json"
readonly HISTORICAL_CONFIG="/path/to/shared-storage/checkpoints/robotwin50_clean_sft_from_joint1185_20260903/bf16_8gpu_bs1_accum2_lr1e-5_25k/checkpoints/config.json"
readonly DATASET="/path/to/shared-storage/wy/giga/dataset/lingbot-robotwin-clean-and-aug-lerobot/robotwin-clean-and-aug-lerobot/lerobot_robotwin_eef_clean_50"
readonly EMPTY_EMB="/path/to/shared-storage/wy/giga/dataset/empty_emb.pt"
readonly INIT_MODEL_ROOT="$PARENT_RUN_ROOT/init_model_step2826_diffusers_mnt"
readonly OUTPUT_ROOT="/path/to/shared-storage/checkpoints/robotwin50_clean_sft_from_full_r5_20261003_zero2_5node400k/bf16_40gpu_bs2_accum1_lr1e-5_5k_save200_retry5"
readonly LOG_ROOT="/path/to/shared-storage/training_logs/robotwin50_clean_sft_from_full_r5_20261003_zero2_5node400k_bs2_accum1_retry5"
readonly MASTER_ADDR="HOST_ADDRESS"
readonly MASTER_PORT="29861"

node_for_rank() {
  case "$1" in
    0) echo "worker-a" ;;
    1) echo "worker-b" ;;
    2) echo "worker-d" ;;
    3) echo "worker-e" ;;
    4) echo "worker-f" ;;
    *) die "node rank must be 0..4" ;;
  esac
}

preflight_node() {
  local rank="$1"
  local expected_host
  expected_host="$(node_for_rank "$rank")"
  [[ "$(hostname -s)" == "$expected_host" ]] || die \
    "rank $rank expects $expected_host; current host is $(hostname -s)"
  [[ -x "$PYTHON" ]] || die "missing Python: $PYTHON"
  [[ -r "$REPO_ROOT/wan_va/train.py" ]] || die "missing training entrypoint"
  [[ -r "$WRAPPER_PATH" ]] || die "missing wrapper: $WRAPPER_PATH"
  [[ -r "$LAUNCHER_PATH" ]] || die "missing launcher: $LAUNCHER_PATH"
  [[ -r "$DEEPSPEED_CONFIG" ]] || die "missing ZeRO-2 config: $DEEPSPEED_CONFIG"
  [[ -r "$HISTORICAL_CONFIG" ]] || die "missing reference config: $HISTORICAL_CONFIG"
  [[ -d "$DATASET" ]] || die "missing dataset: $DATASET"
  [[ -s "$EMPTY_EMB" ]] || die "missing empty embedding: $EMPTY_EMB"
  [[ -s "$INIT_MODEL_ROOT/transformer/config.json" ]] || die "missing initialized transformer config"
  [[ -e "$INIT_MODEL_ROOT/transformer/diffusion_pytorch_model.safetensors" || \
     -e "$INIT_MODEL_ROOT/transformer/diffusion_pytorch_model.safetensors.index.json" ]] || \
    die "missing initialized transformer weights"
  for component in vae text_encoder tokenizer; do
    [[ -e "$INIT_MODEL_ROOT/$component" ]] || die "missing model component: $component"
  done

  local gpu_count
  gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
  [[ "$gpu_count" -eq 8 ]] || die "$expected_host exposes $gpu_count GPUs, expected 8"
  if ! nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits |
    awk 'NF && $1 + 0 > 1024 {busy=1} END {exit busy}'; then
    die "GPU memory is not free on $expected_host"
  fi
  local compute_pids
  compute_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader |
    awk '/^[[:space:]]*[0-9]+[[:space:]]*$/ {print $1}')"
  [[ -z "$compute_pids" ]] || die "compute processes found on $expected_host: $compute_pids"
  export HOME="$RUNTIME_HOME_ROOT/$expected_host"
  export TRITON_CACHE_DIR="$CACHE_ROOT/triton/$expected_host"
  mkdir -p "$HOME" "$TRITON_CACHE_DIR"
  "$PYTHON" -c 'import torch, deepspeed; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 8'
  echo "PRECHECK_OK host=$expected_host rank=$rank GPUs=8 ZeRO-2 runtime=available"
}

worker() {
  local rank="$1"
  local host
  host="$(node_for_rank "$rank")"
  [[ "$(hostname -s)" == "$host" ]] || die "rank $rank started on unexpected host $(hostname -s)"
  export REPO_ROOT
  export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/wan_va"
  export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
  export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
  export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET
  export NCCL_SOCKET_IFNAME=bond0.45
  export WANDB_MODE=offline WANDB_DIR="$LOG_ROOT"
  export HOME="$RUNTIME_HOME_ROOT/$host"
  export HF_HOME="$CACHE_ROOT/hf_home/$host"
  export HF_DATASETS_CACHE="$CACHE_ROOT/hf_datasets/$host"
  export XDG_CACHE_HOME="$CACHE_ROOT/xdg/$host"
  export WANDB_CACHE_DIR="$CACHE_ROOT/wandb/$host"
  export TRITON_CACHE_DIR="$CACHE_ROOT/triton/$host"
  export ROBOTWIN_DATASET_INIT_WORKERS=8
  mkdir -p "$HOME" "$HF_HOME" "$HF_DATASETS_CACHE" "$XDG_CACHE_HOME" \
    "$WANDB_CACHE_DIR" "$TRITON_CACHE_DIR" "$LOG_ROOT/torchrun/$host" "$OUTPUT_ROOT"
  cd "$REPO_ROOT"
  exec "$PYTHON" -m torch.distributed.run \
    --nnodes=5 --nproc_per_node=8 --node_rank="$rank" \
    --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
    --max_restarts=0 --log_dir="$LOG_ROOT/torchrun/$host" \
    --local-ranks-filter=0 --tee=3 "$WRAPPER_PATH"
}

start_cluster() {
  [[ "$(hostname -s)" == "worker-a" ]] || die "start this launcher on worker-a (rank 0)"
  [[ ! -e "$OUTPUT_ROOT" ]] || die "checkpoint output already exists: $OUTPUT_ROOT"
  [[ ! -e "$LOG_ROOT" ]] || die "log output already exists: $LOG_ROOT"
  [[ ! -e "$RUN_ROOT/configs/1185_reference_config.json" ]] || die "run config snapshot already exists"
  if ss -ltnH "sport = :$MASTER_PORT" | grep -q .; then
    die "rendezvous port $MASTER_PORT is already in use on worker-a"
  fi

  local rank host pid1 pid2 pid3 pid4 pid0
  for rank in 0 1 2 3 4; do
    host="$(node_for_rank "$rank")"
    if [[ "$rank" -eq 0 ]]; then
      preflight_node "$rank"
    else
      ssh -o BatchMode=yes -o ConnectTimeout=10 "$host" \
        "bash '$LAUNCHER_PATH' --preflight $rank"
    fi
  done
  "$PYTHON" "$WRAPPER_PATH" --plan

  mkdir -p "$RUN_ROOT/configs" "$LOG_ROOT" "$OUTPUT_ROOT/run_metadata"
  cp -- "$HISTORICAL_CONFIG" "$RUN_ROOT/configs/1185_reference_config.json"
  "$PYTHON" "$WRAPPER_PATH" --plan > "$OUTPUT_ROOT/run_metadata/training_plan.json"
  cp -- "$DEEPSPEED_CONFIG" "$OUTPUT_ROOT/run_metadata/deepspeed_zero2_bf16_bs2_accum1.json"
  printf '%s\n' \
    "nodes=worker-a,worker-b,worker-d,worker-e,worker-f" \
    "gpus=8 per node; world_size=40" \
    "batch_per_rank=2; gradient_accumulation_steps=1; effective_global_batch=80" \
    "batching=latent-frame-length buckets; no padding; per-rank batch=2" \
    "reference_samples=400000" \
    "planned_optimizer_steps=5000" \
    "save_interval=200; periodic_checkpoints=25" \
    "backend=DeepSpeed ZeRO-2" \
    "init=R5 step 2826 transformer weights only; no optimizer resume" \
    > "$OUTPUT_ROOT/run_metadata/allocation.txt"

  pid1="$(ssh -o BatchMode=yes "$(node_for_rank 1)" \
    "nohup bash '$LAUNCHER_PATH' --worker 1 > '$LOG_ROOT/worker_1_worker-b.log' 2>&1 < /dev/null & echo \$!")"
  [[ "$pid1" =~ ^[0-9]+$ ]] || die "could not start rank 1 on worker-b: $pid1"
  echo "WORKER_STARTED host=worker-b rank=1 pid=$pid1 log=$LOG_ROOT/worker_1_worker-b.log"
  pid2="$(ssh -o BatchMode=yes "$(node_for_rank 2)" \
    "nohup bash '$LAUNCHER_PATH' --worker 2 > '$LOG_ROOT/worker_2_worker-d.log' 2>&1 < /dev/null & echo \$!")"
  [[ "$pid2" =~ ^[0-9]+$ ]] || die "could not start rank 2 on worker-d: $pid2"
  echo "WORKER_STARTED host=worker-d rank=2 pid=$pid2 log=$LOG_ROOT/worker_2_worker-d.log"
  pid3="$(ssh -o BatchMode=yes "$(node_for_rank 3)" \
    "nohup bash '$LAUNCHER_PATH' --worker 3 > '$LOG_ROOT/worker_3_worker-e.log' 2>&1 < /dev/null & echo \$!")"
  [[ "$pid3" =~ ^[0-9]+$ ]] || die "could not start rank 3 on worker-e: $pid3"
  echo "WORKER_STARTED host=worker-e rank=3 pid=$pid3 log=$LOG_ROOT/worker_3_worker-e.log"
  pid4="$(ssh -o BatchMode=yes "$(node_for_rank 4)" \
    "nohup bash '$LAUNCHER_PATH' --worker 4 > '$LOG_ROOT/worker_4_worker-f.log' 2>&1 < /dev/null & echo \$!")"
  [[ "$pid4" =~ ^[0-9]+$ ]] || die "could not start rank 4 on worker-f: $pid4"
  echo "WORKER_STARTED host=worker-f rank=4 pid=$pid4 log=$LOG_ROOT/worker_4_worker-f.log"

  nohup bash "$LAUNCHER_PATH" --worker 0 \
    > "$LOG_ROOT/worker_0_worker-a.log" 2>&1 < /dev/null &
  pid0="$!"
  echo "WORKER_STARTED host=worker-a rank=0 pid=$pid0 log=$LOG_ROOT/worker_0_worker-a.log"

  sleep 15
  kill -0 "$pid0" 2>/dev/null || die "rank 0 launcher exited; inspect $LOG_ROOT/worker_0_worker-a.log"
  ssh -o BatchMode=yes worker-b "kill -0 $pid1" || die "rank 1 launcher exited; inspect $LOG_ROOT/worker_1_worker-b.log"
  ssh -o BatchMode=yes worker-d "kill -0 $pid2" || die "rank 2 launcher exited; inspect $LOG_ROOT/worker_2_worker-d.log"
  ssh -o BatchMode=yes worker-e "kill -0 $pid3" || die "rank 3 launcher exited; inspect $LOG_ROOT/worker_3_worker-e.log"
  ssh -o BatchMode=yes worker-f "kill -0 $pid4" || die "rank 4 launcher exited; inspect $LOG_ROOT/worker_4_worker-f.log"
  echo "ALL_FIVE_NODE_LAUNCHERS_ALIVE; logs=$LOG_ROOT"
}

MODE=start
if [[ "$#" -gt 0 ]]; then MODE="$1"; fi
case "$MODE" in
  --preflight)
    [[ "$#" -eq 2 ]] || die "usage: $0 --preflight NODE_RANK"
    preflight_node "$2"
    ;;
  --worker)
    [[ "$#" -eq 2 ]] || die "usage: $0 --worker NODE_RANK"
    worker "$2"
    ;;
  start)
    [[ "$#" -eq 1 ]] || die "usage: $0 [start]"
    start_cluster
    ;;
  *)
    die "usage: $0 [start] | --preflight NODE_RANK | --worker NODE_RANK"
    ;;
esac

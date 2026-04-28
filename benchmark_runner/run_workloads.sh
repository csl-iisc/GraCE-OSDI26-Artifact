#!/usr/bin/env bash
set -euo pipefail

trap 'echo "[ERROR] $0 failed at line $LINENO" >&2' ERR

usage() {
  cat <<'EOF'
Usage:
  benchmark_runner/run_workloads.sh \
    --variant <vanilla|cgct|cgct-pi|full|cgct-copy-debug|cgct-pi-copy-debug> \
    --graph-mode <no-cg|cg|eager|max-autotune> \
    --tp <1|2|4|8> \
    --workloads <path> \
    --output-dir <path> \
    [--repeat 100] \
    [--profile none|nsys]

This script assumes the correct conda environment is already active.
EOF
}

VARIANT=""
GRAPH_MODE=""
TP_SIZE=""
WORKLOADS=""
OUTPUT_DIR=""
REPEAT=100
BACKEND="inductor"
TARGET="performance"
PROFILE="none"
MASTER_PORT="${MASTER_PORT:-29501}"
USE_NUMACTL="${USE_NUMACTL:-1}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --variant) VARIANT="$2"; shift 2 ;;
    --graph-mode) GRAPH_MODE="$2"; shift 2 ;;
    --tp) TP_SIZE="$2"; shift 2 ;;
    --workloads) WORKLOADS="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --repeat) REPEAT="$2"; shift 2 ;;
    --backend) BACKEND="$2"; shift 2 ;;
    --profile) PROFILE="$2"; shift 2 ;;
    --master-port) MASTER_PORT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$VARIANT" || -z "$GRAPH_MODE" || -z "$TP_SIZE" || -z "$WORKLOADS" || -z "$OUTPUT_DIR" ]]; then
  echo "[ERROR] Missing required argument." >&2
  usage
  exit 1
fi

if [[ ! -f "$WORKLOADS" ]]; then
  echo "[ERROR] Workload file not found: $WORKLOADS" >&2
  exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_ROOT="$SCRIPT_DIR"

if [[ "$PROFILE" == "nsys" ]]; then
  if [[ "$TP_SIZE" -eq 1 ]]; then
    BENCH_SUBDIR="dynamo-separate-eager-report-more-stats-nvtx"
  else
    BENCH_SUBDIR="dynamo-separate-eager-report-more-stats-nccl-support-nvtx"
  fi
else
  if [[ "$TP_SIZE" -eq 1 ]]; then
    BENCH_SUBDIR="dynamo-separate-eager-report-more-stats"
  else
    BENCH_SUBDIR="dynamo-separate-eager-report-more-stats-nccl-support"
  fi
fi

BENCH_DIR="$BENCHMARK_ROOT/my_benchmarks/$BENCH_SUBDIR"

if [[ ! -d "$BENCH_DIR" ]]; then
  echo "[ERROR] Benchmark directory not found: $BENCH_DIR" >&2
  exit 1
fi

if [[ "$PROFILE" == "nsys" ]] && ! command -v nsys >/dev/null 2>&1; then
  echo "[ERROR] nsys not found. Install Nsight Systems or use --profile none." >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/nsys"

ITERATIONS_ARGS=()
if [[ -n "${ITERATIONS:-}" ]]; then
  ITERATIONS_ARGS=(--iterations "$ITERATIONS")
fi

EXTRA_ARGS=()
OUTPUT_LABEL=""

case "$GRAPH_MODE" in
  eager)
    EXTRA_ARGS=(--eager)
    OUTPUT_LABEL="eager"
    ;;
  no-cg)
    EXTRA_ARGS=(--disable-cudagraphs)
    OUTPUT_LABEL="no_cudagraphs"
    ;;
  cg)
    EXTRA_ARGS=()
    OUTPUT_LABEL="with_cudagraphs"
    ;;
  max-autotune)
    EXTRA_ARGS=()
    OUTPUT_LABEL="max_autotune"
    export TORCHINDUCTOR_MAX_AUTOTUNE=1
    ;;
  *)
    echo "[ERROR] Unknown graph mode: $GRAPH_MODE" >&2
    exit 1
    ;;
esac

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  case "$TP_SIZE" in
    1) export CUDA_VISIBLE_DEVICES=0 ;;
    2) export CUDA_VISIBLE_DEVICES=0,1 ;;
    4) export CUDA_VISIBLE_DEVICES=0,1,2,3 ;;
    8) export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ;;
    *) echo "[ERROR] Unsupported TP size: $TP_SIZE" >&2; exit 1 ;;
  esac
fi

{
  echo "variant=$VARIANT"
  echo "graph_mode=$GRAPH_MODE"
  echo "tp_size=$TP_SIZE"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
  echo "workloads=$WORKLOADS"
  echo "benchmark_dir=$BENCH_DIR"
  echo "output_dir=$OUTPUT_DIR"
  echo "repeat=$REPEAT"
  echo "profile=$PROFILE"
  echo "backend=$BACKEND"
  echo "python=$(which python)"
  python - <<'PY'
import torch
print("torch_version=" + str(torch.__version__))
print("cuda_available=" + str(torch.cuda.is_available()))
if torch.cuda.is_available():
    print("cuda_device_count=" + str(torch.cuda.device_count()))
    print("cuda_device_0=" + torch.cuda.get_device_name(0))
PY
} > "$OUTPUT_DIR/metadata.txt"

echo "============================================================"
echo "GraCE workload run"
echo "Variant:              $VARIANT"
echo "Graph mode:           $GRAPH_MODE"
echo "TP size:              $TP_SIZE"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "Workloads:            $WORKLOADS"
echo "Benchmark dir:        $BENCH_DIR"
echo "Output dir:           $OUTPUT_DIR"
echo "Repeat:               $REPEAT"
echo "Profile:              $PROFILE"
echo "============================================================"

cd "$BENCHMARK_ROOT"

while IFS= read -r line; do
  [[ -z "$line" ]] && continue
  [[ "$line" =~ ^[[:space:]]*# ]] && continue

  read -r -a fields <<< "$line"
  if [[ "${#fields[@]}" -eq 4 ]]; then
    acronym="${fields[0]}"
    model="${fields[0]}"
    suite="${fields[1]}"
    mode="${fields[2]}"
    batch_size="${fields[3]}"
  elif [[ "${#fields[@]}" -ge 5 ]]; then
    acronym="${fields[0]}"
    model="${fields[1]}"
    suite="${fields[2]}"
    mode="${fields[3]}"
    batch_size="${fields[4]}"
  else
    echo "[WARN] Skipping malformed line: $line"
    continue
  fi

  case "$suite" in
    Huggingface|huggingface) suite="huggingface" ;;
    TIMM|timm) suite="timm_models" ;;
    TorchBench|torchbench) suite="torchbench" ;;
    *)
      echo "[ERROR] Unknown suite: $suite in line: $line" >&2
      exit 1
      ;;
  esac

  if [[ "$mode" == "inference" ]]; then
    dtype="bfloat16"
    if [[ "$model" == detectron2_* ]]; then
      dtype="amp"
    fi
  elif [[ "$mode" == "training" ]]; then
    dtype="amp"
  else
    echo "[ERROR] Unknown mode: $mode in line: $line" >&2
    exit 1
  fi

  bench_script="$BENCH_DIR/$suite.py"
  if [[ ! -f "$bench_script" ]]; then
    echo "[ERROR] Missing benchmark script: $bench_script" >&2
    exit 1
  fi

  target_flag=(--"$TARGET")
  output_csv="${OUTPUT_DIR}/${BACKEND}_${OUTPUT_LABEL}_${suite}_${dtype}_${mode}_cuda_${TARGET}.csv"

  echo
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
  echo "Acronym:    $acronym"
  echo "Model:      $model"
  echo "Suite:      $suite"
  echo "Mode:       $mode"
  echo "DType:      $dtype"
  echo "Batch size: $batch_size"
  echo "Output:     $output_csv"
  echo "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"

  if [[ "$TP_SIZE" -eq 1 ]]; then
    cmd=(
      python -u "$bench_script"
      "${target_flag[@]}"
      --"$mode"
      --"$dtype"
      --backend "$BACKEND"
      "${EXTRA_ARGS[@]}"
      --only "$model"
      --batch_size "$batch_size"
      --repeat "$REPEAT"
      "${ITERATIONS_ARGS[@]}"
      --output "$output_csv"
    )

    if [[ "$USE_NUMACTL" == "1" ]] && command -v numactl >/dev/null 2>&1; then
      cmd=(numactl --cpunodebind=0 --membind=0 "${cmd[@]}")
    fi
  else
    cmd=(
      env TP_SIZE="$TP_SIZE"
      torchrun
      --standalone
      --nnodes=1
      --nproc_per_node="$TP_SIZE"
      --master_port="$MASTER_PORT"
      "$bench_script"
      "${target_flag[@]}"
      --"$mode"
      --"$dtype"
      --backend "$BACKEND"
      "${EXTRA_ARGS[@]}"
      --only "$model"
      --batch_size "$batch_size"
      --repeat "$REPEAT"
      "${ITERATIONS_ARGS[@]}"
      --output "$output_csv"
    )
  fi

  if [[ "$PROFILE" == "nsys" ]]; then
    nsys_prefix="$OUTPUT_DIR/nsys/${acronym}_${VARIANT}_${GRAPH_MODE}_tp${TP_SIZE}"
    nsys profile \
      -w true \
      -t cuda,nvtx,osrt,cudnn,cublas \
      -s cpu \
      --capture-range=cudaProfilerApi \
      --capture-range-end=stop-shutdown \
      --cudabacktrace=true \
      --cuda-graph-trace node \
      --force-overwrite=true \
      -x true \
      -o "$nsys_prefix" \
      "${cmd[@]}"
  else
    "${cmd[@]}"
  fi

  if [[ "$TP_SIZE" -gt 1 ]]; then
    sleep "${SLEEP_BETWEEN_RUNS:-20}"
  fi
done < "$WORKLOADS"

echo
echo "[OK] Workload run completed."
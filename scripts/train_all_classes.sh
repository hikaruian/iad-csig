#!/usr/bin/env bash
# Train one INP-Former per class by calling the existing train.py.
# Sequential: one class at a time so 2xT4 is not filled with 50 copies of DINOv2.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PER_CLASS_DATA="${PER_CLASS_DATA:-$ROOT/data/per_class}"
SAVE_ROOT="${SAVE_ROOT:-/home/runs/per_class}"
EPOCHS="${EPOCHS:-100}"
GPUS="${GPUS:-2}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
if [[ ! -d "$PER_CLASS_DATA" ]]; then
  echo "Run scripts/make_per_class_links.sh first ($PER_CLASS_DATA missing)" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

mapfile -t CLASSES < <(find "$PER_CLASS_DATA" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
echo "training ${#CLASSES[@]} classes  epochs=$EPOCHS  gpus=$GPUS"

i=0
for cls in "${CLASSES[@]}"; do
  i=$((i + 1))
  train_root="$PER_CLASS_DATA/$cls/Train"
  save_dir="$SAVE_ROOT/$cls"
  ckpt="$save_dir/model.pth"
  if [[ "$SKIP_EXISTING" == "1" && -f "$ckpt" ]]; then
    echo "[$i/${#CLASSES[@]}] skip $cls (found $ckpt)"
    continue
  fi
  if [[ ! -d "$train_root/$cls" ]]; then
    echo "[$i/${#CLASSES[@]}] skip $cls (no $train_root/$cls)"
    continue
  fi
  echo "[$i/${#CLASSES[@]}] TRAIN $cls -> $save_dir"
  mkdir -p "$save_dir"
  if [[ "$GPUS" -gt 1 ]]; then
    torchrun --standalone --nnodes=1 --nproc_per_node="$GPUS" train.py \
      --train-root "$train_root" \
      --save-dir "$save_dir" \
      --encoder dinov2reg_vit_large_14 \
      --image-size 448 \
      --inp-num 6 \
      --epochs "$EPOCHS" \
      --batch-size 4 \
      --grad-accum 2 \
      --num-workers 2 \
      --amp
  else
    python train.py \
      --train-root "$train_root" \
      --save-dir "$save_dir" \
      --encoder dinov2reg_vit_large_14 \
      --image-size 448 \
      --inp-num 6 \
      --epochs "$EPOCHS" \
      --batch-size 4 \
      --grad-accum 2 \
      --num-workers 2 \
      --amp
  fi
done

echo "done. weights under $SAVE_ROOT/<class>/model.pth"


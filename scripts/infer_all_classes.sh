#!/usr/bin/env bash
# Per-class inference via existing infer.py (no refine / no view-gate).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PER_CLASS_DATA="${PER_CLASS_DATA:-$ROOT/data/per_class}"
SAVE_ROOT="${SAVE_ROOT:-/home/runs/per_class}"
OUT_ROOT="${OUT_ROOT:-$ROOT/outputs/per_class}"
SIGMA="${SIGMA:-9.5}"

if [[ ! -d "$PER_CLASS_DATA" ]]; then
  echo "Run scripts/make_per_class_links.sh first" >&2
  exit 1
fi

mapfile -t CLASSES < <(find "$PER_CLASS_DATA" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
echo "infer ${#CLASSES[@]} classes  sigma=$SIGMA"

i=0
ok=0
miss=0
for cls in "${CLASSES[@]}"; do
  i=$((i + 1))
  test_root="$PER_CLASS_DATA/$cls/Test_A"
  ckpt=""
  for cand in "$SAVE_ROOT/$cls/best.pth" "$SAVE_ROOT/$cls/model.pth" "$SAVE_ROOT/$cls/last.pth"; do
    if [[ -f "$cand" ]]; then
      ckpt="$cand"
      break
    fi
  done
  if [[ -z "$ckpt" ]]; then
    echo "[$i/${#CLASSES[@]}] SKIP $cls (no ckpt under $SAVE_ROOT/$cls)"
    miss=$((miss + 1))
    continue
  fi
  if [[ ! -d "$test_root/$cls" ]]; then
    echo "[$i/${#CLASSES[@]}] SKIP $cls (no test dir)"
    miss=$((miss + 1))
    continue
  fi
  echo "[$i/${#CLASSES[@]}] INFER $cls  ckpt=$ckpt"
  python infer.py \
    --test-root "$test_root" \
    --ckpt "$ckpt" \
    --out-dir "$OUT_ROOT/$cls" \
    --zip "$OUT_ROOT/${cls}.zip" \
    --no-refine --no-view-gate --no-fg-gate \
    --sigma "$SIGMA"
  ok=$((ok + 1))
done

echo "inferred $ok classes, skipped $miss"
echo "next: scripts/pack_per_class.sh"


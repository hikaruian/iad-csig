#!/usr/bin/env bash
# Build per-class Train/Test trees via symlinks (no image copies).
# Existing train.py/infer.py stay untouched.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TRAIN_SRC="${TRAIN_SRC:-$ROOT/data/Real-IAD/Train}"
TEST_SRC="${TEST_SRC:-$ROOT/data/Real-IAD/Test_A}"
OUT="${PER_CLASS_DATA:-$ROOT/data/per_class}"

if [[ ! -d "$TRAIN_SRC" ]]; then
  echo "Train not found: $TRAIN_SRC" >&2
  echo "Set TRAIN_SRC=/path/to/Train" >&2
  exit 1
fi

mkdir -p "$OUT"
n=0
for cls_dir in "$TRAIN_SRC"/*/; do
  [[ -d "$cls_dir" ]] || continue
  cls="$(basename "$cls_dir")"
  mkdir -p "$OUT/$cls/Train" "$OUT/$cls/Test_A"
  ln -sfn "$(cd "$TRAIN_SRC/$cls" && pwd)" "$OUT/$cls/Train/$cls"
  if [[ -d "$TEST_SRC/$cls" ]]; then
    ln -sfn "$(cd "$TEST_SRC/$cls" && pwd)" "$OUT/$cls/Test_A/$cls"
  else
    echo "[warn] no Test_A class $cls"
  fi
  n=$((n + 1))
done

echo "linked $n classes under $OUT/<class>/{Train,Test_A}/<class>"
echo "example: $OUT/$(ls "$OUT" | head -1)/Train"


#!/usr/bin/env bash
# B-board inference for the per-class scheme.
# Seen class  -> runs/per_class/<cls>/*.pth
# Unseen class -> UNIFIED_CKPT (the 50-class unified model; INP from the test image)
# Existing train.py / infer.py are not modified.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

TEST_B="${TEST_B:-/Test_B}"
SAVE_ROOT="${SAVE_ROOT:-/home/runs/per_class}"
UNIFIED_CKPT="${UNIFIED_CKPT:-$ROOT/runs/inpformer_l14/model.pth}"
OUT_ROOT="${OUT_ROOT:-$ROOT/outputs/board_b}"
SIGMA="${SIGMA:-9.5}"
STAGE="${STAGE:-$OUT_ROOT/_parts}"

if [[ ! -d "$TEST_B" ]]; then
  echo "Set TEST_B=/path/to/Test_B (not found: $TEST_B)" >&2
  exit 1
fi
if [[ ! -f "$UNIFIED_CKPT" ]]; then
  echo "Set UNIFIED_CKPT to the 50-class unified model.pth (not found: $UNIFIED_CKPT)" >&2
  exit 1
fi

rm -rf "$STAGE"
mkdir -p "$STAGE/seen" "$STAGE/unseen_tree" "$OUT_ROOT"

seen=0
unseen=0
mapfile -t CLASSES < <(find "$TEST_B" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)

echo "Test_B classes: ${#CLASSES[@]}"
echo "unified fallback: $UNIFIED_CKPT"

# --- seen: one infer.py call per class ---
for cls in "${CLASSES[@]}"; do
  ckpt=""
  for cand in "$SAVE_ROOT/$cls/best.pth" "$SAVE_ROOT/$cls/model.pth" "$SAVE_ROOT/$cls/last.pth"; do
    if [[ -f "$cand" ]]; then
      ckpt="$cand"
      break
    fi
  done
  if [[ -n "$ckpt" ]]; then
    seen=$((seen + 1))
    part="$STAGE/seen/$cls"
    mkdir -p "$part/Test_A"
    ln -sfn "$(cd "$TEST_B/$cls" && pwd)" "$part/Test_A/$cls"
    echo "[seen $seen] $cls <- $ckpt"
    python infer.py \
      --test-root "$part/Test_A" \
      --ckpt "$ckpt" \
      --out-dir "$part/out" \
      --zip "$part/out.zip" \
      --no-refine --no-view-gate --no-fg-gate \
      --sigma "$SIGMA"
  else
    unseen=$((unseen + 1))
    mkdir -p "$STAGE/unseen_tree"
    ln -sfn "$(cd "$TEST_B/$cls" && pwd)" "$STAGE/unseen_tree/$cls"
    echo "[unseen $unseen] $cls -> unified"
  fi
done

# --- unseen: ONE infer.py call on the unified 50-class model ---
if [[ "$unseen" -gt 0 ]]; then
  echo "INFER $unseen unseen classes with unified ckpt"
  python infer.py \
    --test-root "$STAGE/unseen_tree" \
    --ckpt "$UNIFIED_CKPT" \
    --out-dir "$STAGE/unseen_out" \
    --zip "$STAGE/unseen.zip" \
    --no-refine --no-view-gate --no-fg-gate \
    --sigma "$SIGMA"
fi

# --- pack ---
PACK="$OUT_ROOT/pack"
rm -rf "$PACK"
mkdir -p "$PACK/predicted_masks"
printf 'group_folder,anomaly_score\n' > "$PACK/submission.csv"

if [[ "$seen" -gt 0 ]]; then
  for d in "$STAGE/seen"/*/out; do
    [[ -f "$d/submission.csv" ]] || continue
    tail -n +2 "$d/submission.csv" >> "$PACK/submission.csv"
    if [[ -d "$d/predicted_masks" ]]; then
      cp -a "$d/predicted_masks"/. "$PACK/predicted_masks/"
    fi
  done
fi
if [[ -f "$STAGE/unseen_out/submission.csv" ]]; then
  tail -n +2 "$STAGE/unseen_out/submission.csv" >> "$PACK/submission.csv"
  if [[ -d "$STAGE/unseen_out/predicted_masks" ]]; then
    cp -a "$STAGE/unseen_out/predicted_masks"/. "$PACK/predicted_masks/"
  fi
fi
sed -i 's/\r$//' "$PACK/submission.csv"

ZIP="$OUT_ROOT/submission_B.zip"
rm -f "$ZIP"
( cd "$PACK" && zip -rq "$ZIP" submission.csv predicted_masks )

echo "seen=$seen  unseen=$unseen"
echo "zip $ZIP"
echo "rows $(tail -n +2 "$PACK/submission.csv" | wc -l)"


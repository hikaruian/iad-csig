#!/usr/bin/env bash
# B-board: seen classes -> their 5-class group model; unseen -> unified 50-class ckpt.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

#TEST_B="${TEST_B:-$ROOT/data/Real-IAD/Test_B}"
TEST_B="${TEST_B:-/Test_B}"
GROUP_DATA="${GROUP_DATA:-$ROOT/data/groups}"
#SAVE_ROOT="${SAVE_ROOT:-/home/runs/groups}"
SAVE_ROOT="${SAVE_ROOT:-$ROOT/runs/groups}"
UNIFIED_CKPT="${UNIFIED_CKPT:-/kaggle/input/models/dickdickgo/best/pytorch/default/1/best.pth}"
OUT="${OUT:-$ROOT/outputs/groups_board_b}"
SIGMA="${SIGMA:-9.5}"
MANIFEST="${GROUP_DATA}/groups.tsv"

if [[ ! -d "$TEST_B" ]]; then
  echo "Set TEST_B=/path/to/Test_B" >&2
  exit 1
fi
if [[ ! -f "$UNIFIED_CKPT" ]]; then
  echo "Set UNIFIED_CKPT=.../model.pth" >&2
  exit 1
fi
if [[ ! -f "$MANIFEST" ]]; then
  echo "missing $MANIFEST — run make_group_links.sh on A-board Train first" >&2
  exit 1
fi

STAGE="$OUT/_parts"
rm -rf "$STAGE" "$OUT/pack"
mkdir -p "$STAGE/unseen_tree" "$OUT"

# class -> gid
declare -A CLS2GID
while IFS=$'\t' read -r gid members; do
  [[ -n "${gid:-}" ]] || continue
  for cls in $members; do
    CLS2GID["$cls"]="$gid"
  done
done < "$MANIFEST"

# which groups actually appear in Test_B (seen)
declare -A NEED_GID
unseen=0
mapfile -t BCLS < <(find "$TEST_B" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
for cls in "${BCLS[@]}"; do
  gid="${CLS2GID[$cls]:-}"
  if [[ -n "$gid" ]]; then
    NEED_GID["$gid"]=1
  else
    unseen=$((unseen + 1))
    ln -sfn "$(cd "$TEST_B/$cls" && pwd)" "$STAGE/unseen_tree/$cls"
    echo "[unseen] $cls"
  fi
done

# seen: one infer per group, but Test_B may only need some classes of that group
for gid in $(printf '%s\n' "${!NEED_GID[@]}" | sort); do
  ckpt=""
  for cand in "$SAVE_ROOT/$gid/best.pth" "$SAVE_ROOT/$gid/model.pth" "$SAVE_ROOT/$gid/last.pth"; do
    [[ -f "$cand" ]] && ckpt="$cand" && break
  done
  if [[ -z "$ckpt" ]]; then
    echo "[warn] no ckpt for $gid, those classes go to unified"
    while IFS=$'\t' read -r g members; do
      [[ "$g" == "$gid" ]] || continue
      for cls in $members; do
        [[ -d "$TEST_B/$cls" ]] || continue
        ln -sfn "$(cd "$TEST_B/$cls" && pwd)" "$STAGE/unseen_tree/$cls"
      done
    done < "$MANIFEST"
    continue
  fi
  tree="$STAGE/seen_$gid"
  mkdir -p "$tree"
  while IFS=$'\t' read -r g members; do
    [[ "$g" == "$gid" ]] || continue
    for cls in $members; do
      [[ -d "$TEST_B/$cls" ]] || continue
      ln -sfn "$(cd "$TEST_B/$cls" && pwd)" "$tree/$cls"
    done
  done < "$MANIFEST"
  echo "[seen] $gid <- $ckpt"
  python infer.py \
    --test-root "$tree" \
    --ckpt "$ckpt" \
    --out-dir "$STAGE/out_$gid" \
    --zip "$STAGE/out_$gid.zip" \
    --no-refine --no-view-gate --no-fg-gate \
    --sigma "$SIGMA"
done

if [[ -n "$(find "$STAGE/unseen_tree" -mindepth 1 -maxdepth 1 -type l 2>/dev/null | head -1)" ]]; then
  echo "INFER unseen/fallback with unified"
  python infer.py \
    --test-root "$STAGE/unseen_tree" \
    --ckpt "$UNIFIED_CKPT" \
    --out-dir "$STAGE/out_unseen" \
    --zip "$STAGE/submit_B.zip" \
    --no-refine --no-view-gate --no-fg-gate \
    --sigma "$SIGMA"
fi

PACK="$OUT/pack"
mkdir -p "$PACK/predicted_masks"
printf 'group_folder,anomaly_score\n' > "$PACK/submission.csv"
for d in "$STAGE"/out_*; do
  [[ -f "$d/submission.csv" ]] || continue
  tail -n +2 "$d/submission.csv" >> "$PACK/submission.csv"
  [[ -d "$d/predicted_masks" ]] && cp -a "$d/predicted_masks"/. "$PACK/predicted_masks/"
done
sed -i 's/\r$//' "$PACK/submission.csv"

ZIP="$OUT/submission_B.zip"
rm -f "$ZIP"
( cd "$PACK" && zip -rq "$ZIP" submission.csv predicted_masks )
echo "zip $ZIP  rows=$(tail -n +2 "$PACK/submission.csv" | wc -l)"


#!/usr/bin/env bash
# Recommendation with fewer training users (added after
# E1 was seen; secondary and exploratory). ID-GNN with 1% and 0.1% of training users, otherwise
# exactly E1's protocol (ft-user-item-purchase-128.yaml: lr 1e-3, 10 epochs, best epoch by
# validation MAP, 3 seeds), user-root final checkpoints, every method. Per-user test AP is saved
# by train_idgnn.py for the paired bootstrap in analyze_results.py.
# Waits for "PROBES-FULL-DONE" so it never shares the GPU or RAM with the other queues.
# Usage (inside the container): bash /workspace/scripts/runs/rec_lowlabel.sh
#   Dry run (prints the commands, writes nothing to logs/):
#   DRY_RUN=1 OUT=/tmp/rl WAIT_FILE=/workspace/logs/chain.log WAIT_MARK="user-root" bash ...
# Parsed as one block so that editing this file cannot affect a run in progress.
{
set -uo pipefail
source /workspace/scripts/runs/common.sh
cd /workspace/third_party/task-vector-estimation
PT=/workspace/logs/main
OUT="${OUT:-/workspace/logs/rec_lowlabel}"
mkdir -p "$OUT"
CFG=/workspace/configs/rel-hm
METHODS="${METHODS:-scratch mae tve1 reljepa reljepa-tve}"
SETTINGS="${SETTINGS:-1pct:0.01 0p1pct:0.001}"        # run-name tag : fraction of training users
WAIT_FILE="${WAIT_FILE:-/workspace/logs/probes_full/summary.log}"
WAIT_MARK="${WAIT_MARK:-PROBES-FULL-DONE}"
DRY_RUN="${DRY_RUN:-0}"
export SAVE_FT_CHECKPOINTS=0

done_already() { grep -q " $1 exit=0" "$OUT/summary.log" 2>/dev/null; }
record() { echo "$(date -u +%FT%TZ) $1 exit=$2" >> "$OUT/summary.log"; }
run() { if [[ $DRY_RUN == 1 ]]; then echo "DRY: $*"; else "$@"; fi; }
ckpt() {  # method -> user-root checkpoint ("" for scratch); same rule as finetune_main.sh
  case "$1" in
    scratch) echo "" ;;
    mae|tve1) echo "$(cat "$PT/logdir_pt-user-$1.txt")/epoch81.pth" ;;
    reljepa|reljepa-tve) echo "$(cat "$PT/logdir_pt-user-$1.txt")/run1-final.pth" ;;
  esac
}

until grep -q "$WAIT_MARK" "$WAIT_FILE" 2>/dev/null; do sleep 300; done
record "start-after-$WAIT_MARK" 0

for setting in $SETTINGS; do
  tag="${setting%%:*}"; frac="${setting#*:}"
  for method in $METHODS; do
    name="ft-rec-$tag-$method"
    done_already "$name" && continue
    c="$(ckpt "$method")"
    if [[ -n "$c" && ! -f "$c" ]]; then record "missing-checkpoint-$name" 1; continue; fi
    ck=(); [[ -n "$c" ]] && ck=(--checkpoint "$c")
    set_commit
    run python src/train_idgnn.py -c "$CFG/ft-user-item-purchase-128.yaml" --user_fraction "$frac" \
      --num_runs 3 --num_workers 3 --gpus '[0]' "${ck[@]}" \
      --name "$name" --note "rec $tag of users $method" > "$OUT/$name.log" 2>&1
    record "$name" $?
  done
done
record "REC-LOWLABEL-DONE" 0
exit $?
}

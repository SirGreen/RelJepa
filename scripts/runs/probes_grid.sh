#!/usr/bin/env bash
# Completes the pre-registered learning-rate grid {1e-3, 5e-3} of the full-data linear probes
#. The first pass (probes_full.sh) ran every method at the
# task config's fixed rate (user-churn 5e-3, item-sales 2e-3): those configs had no {{ lr }} variable,
# and TVE's sweep runner ignores hard-coded values. Missing arms: user-churn 1e-3 (its 5e-3 arm
# exists) and item-sales 1e-3 and 5e-3. Same protocol otherwise (frozen backbone, 3 seeds).
# Waits for "REC-LOWLABEL-DONE" so it never shares the GPU or RAM with the other queues.
# Usage (inside the container): bash /workspace/scripts/runs/probes_grid.sh
# Parsed as one block so that editing this file cannot affect a run in progress.
{
set -uo pipefail
source /workspace/scripts/runs/common.sh
cd /workspace/third_party/task-vector-estimation
PT=/workspace/logs/main
OUT="${OUT:-/workspace/logs/probes_grid}"
mkdir -p "$OUT"
CFG=/workspace/configs/rel-hm
SWEEP=/workspace/configs/sweeps/lr2        # 1 = lr0001.yaml (1e-3), 2 = lr0005.yaml (5e-3)
METHODS="${METHODS:-scratch mae tve1 reljepa reljepa-tve}"
ARMS="${ARMS:-user:user-churn:[1] item:item-sales:[1,2]}"   # root : task : sweep files to run
WAIT_FILE="${WAIT_FILE:-/workspace/logs/rec_lowlabel/summary.log}"
WAIT_MARK="${WAIT_MARK:-REC-LOWLABEL-DONE}"
export SAVE_FT_CHECKPOINTS=0 FREEZE_BACKBONE=1 EVAL_NUM_WORKERS=2   # as probes_full.sh

done_already() { grep -q " $1 exit=0" "$OUT/summary.log" 2>/dev/null; }
record() { echo "$(date -u +%FT%TZ) $1 exit=$2" >> "$OUT/summary.log"; }
ckpt() {  # root method -> checkpoint path ("" for scratch); same rule as probes_full.sh
  local final; final=$([[ $1 == user ]] && echo 81 || echo 41)
  case "$2" in
    scratch) echo "" ;;
    mae|tve1) echo "$(cat "$PT/logdir_pt-$1-$2.txt")/epoch${final}.pth" ;;
    reljepa|reljepa-tve) echo "$(cat "$PT/logdir_pt-$1-$2.txt")/run1-final.pth" ;;
  esac
}

until grep -q "$WAIT_MARK" "$WAIT_FILE" 2>/dev/null; do sleep 300; done
record "start-after-$WAIT_MARK" 0

for arm in $ARMS; do
  root="${arm%%:*}"; rest="${arm#*:}"; task="${rest%%:*}"; only="${rest#*:}"
  cfg="$CFG/probe-$task-128.yaml"
  if ! sweep_ready "$cfg"; then record "config-without-lr-variable-$task" 1; continue; fi
  for method in $METHODS; do
    name="ft-probe-$task-$method"
    done_already "$name-grid" && continue
    c="$(ckpt "$root" "$method")"
    if [[ -n "$c" && ! -f "$c" ]]; then record "missing-checkpoint-$name" 1; continue; fi
    ck=(); [[ -n "$c" ]] && ck=(--checkpoint "$c")
    set_commit
    python src/train_baseline.py -c "$cfg" -s "$SWEEP" --sweep_only_for "$only" \
      --num_runs 3 --num_workers 3 --gpus '[0]' "${ck[@]}" \
      --name "$name" --note "full-data linear probe $method, grid completion" > "$OUT/$name.log" 2>&1
    rc=$?; sweep_failed "$OUT/$name.log" && rc=97
    record "$name-grid" $rc
  done
done
record "PROBES-GRID-DONE" 0
exit $?
}

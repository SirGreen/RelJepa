#!/usr/bin/env bash
# Ablation TVE-1 + empty-window head at the main budget, then the primary probe
# endpoints with the main protocol:
#   customers: pretraining (81 epochs) -> E2 churn linear probe over the grid {1e-3, 5e-3} x 3 seeds
#              (as probes_grid.sh) -> information probes E4/E5 (+ count), own CSV, main ones untouched
#   articles:  pretraining (41 epochs) -> E3 sales linear probe over the same grid
# Runs inside the container. Re-running skips finished steps (summary.log).
# Parsed as one block so that editing this file cannot affect a run in progress.
{
set -uo pipefail
source /workspace/scripts/runs/common.sh
cd /workspace/third_party/task-vector-estimation
PT=/workspace/logs/main
OUT=/workspace/logs/ablation_tveempty
mkdir -p "$OUT"
CFG=/workspace/configs/rel-hm
SWEEP=/workspace/configs/sweeps/lr2        # 1 = lr0001.yaml (1e-3), 2 = lr0005.yaml (5e-3)
M=tve1-empty
export SAVE_FT_CHECKPOINTS=0

record() { echo "$(date -u +%FT%TZ) $1 exit=$2" >> "$OUT/summary.log"; }
done_already() { grep -q " $1 exit=0" "$OUT/summary.log" 2>/dev/null; }

for root in user item; do
  task=$([[ $root == user ]] && echo user-churn || echo item-sales)
  name="pt-$root-$M"
  if ! done_already "$name"; then
    set_commit
    EVAL_NUM_WORKERS=0 RUN_LOGDIR_FILE="$PT/logdir_$name.txt" python src/train_reljepa.py \
      -c "$CFG/$root-reljepa-tveempty-cosine.yaml" --num_runs 1 --num_workers 3 --gpus '[0]' \
      --name "$name" --note "TVE-1+head ablation pretraining" > "$OUT/$name.log" 2>&1
    record "$name" $?
  fi
  ckpt="$(cat "$PT/logdir_$name.txt" 2>/dev/null)/run1-final.pth"
  if [[ ! -f "$ckpt" ]]; then record "missing-checkpoint-$name" 1; continue; fi

  pname="ft-probe-$task-$M"
  if ! done_already "$pname-grid"; then
    cfg="$CFG/probe-$task-128.yaml"
    if ! sweep_ready "$cfg"; then record "config-without-lr-variable-$task" 1; continue; fi
    set_commit
    FREEZE_BACKBONE=1 EVAL_NUM_WORKERS=2 python src/train_baseline.py -c "$cfg" -s "$SWEEP" \
      --sweep_only_for "[1,2]" --num_runs 3 --num_workers 3 --gpus '[0]' --checkpoint "$ckpt" \
      --name "$pname" --note "TVE-1+head full-data linear probe" > "$OUT/$pname.log" 2>&1
    rc=$?; sweep_failed "$OUT/$pname.log" && rc=97
    record "$pname-grid" $rc
  fi

  if [[ $root == user ]] && ! done_already "info-probes"; then
    python /workspace/scripts/probes/extract_embeddings.py --methods "$M" > "$OUT/info_extract.log" 2>&1 \
      && python /workspace/scripts/probes/run_probes.py --methods "$M" \
           --out /workspace/results/raw/info_probes_tveempty.csv > "$OUT/info_run.log" 2>&1
    record "info-probes" $?
  fi
done
record "ABLATION-TVEEMPTY-DONE" 0
exit 0
}

#!/usr/bin/env bash
# After the loss pilot: (1) decision probes on full user-churn validation,
# (2) ID-GNN smoke test, (3) item-root throughput benchmarks. Runs inside the container.
set -uo pipefail
cd /workspace/third_party/task-vector-estimation
OUT=/workspace/logs/post_pilot; mkdir -p "$OUT"
export EVAL_NUM_WORKERS=2 SAVE_FT_CHECKPOINTS=0
for l in smooth_l1 cosine scratch; do
  ck=(); [[ $l != scratch ]] && ck=(--checkpoint "$(cat /workspace/logs/pilot/logdir_pt_$l.txt)/run1-final.pth")
  FREEZE_BACKBONE=1 RUN_LOGDIR_FILE="$OUT/logdir_probe_$l.txt" python src/train_baseline.py \
    -c /workspace/configs/rel-hm/pilot-probe-user-churn-128.yaml --num_runs 1 --num_workers 3 --gpus '[0]' \
    "${ck[@]}" --name "pilot-probe-$l" --note "pilot decision probe" > "$OUT/probe_$l.log" 2>&1
  echo "probe $l exit=$?" >> "$OUT/summary.log"
done
python src/train_idgnn.py -c /workspace/configs/rel-hm/ft-user-item-purchase-128.yaml --user_fraction 0.1 \
  --num_runs 1 --num_workers 3 --gpus '[0]' --fast_dev_run --name smoke-idgnn --note smoke \
  > "$OUT/smoke_idgnn.log" 2>&1
echo "idgnn smoke exit=$?" >> "$OUT/summary.log"
export EVAL_NUM_WORKERS=0
for m in tve:train_tve:tve-item mae:train_mae:mae-item reljepa:train_reljepa:reljepa-item; do
  IFS=: read name script cfg <<< "$m"
  start=$(date +%s)
  python src/$script.py -c /workspace/configs/bench/$cfg.yaml --num_runs 1 --num_workers 3 --gpus '[0]' --debug \
    > "$OUT/bench_item_$name.log" 2>&1
  echo "bench item $name exit=$? seconds=$(( $(date +%s) - start ))" >> "$OUT/summary.log"
done
echo "POST-PILOT DONE" >> "$OUT/summary.log"

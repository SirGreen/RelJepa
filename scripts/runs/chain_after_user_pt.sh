#!/usr/bin/env bash
# Rest of the experiment plan, run in one container after the user-root pretraining queue
# (container main-pt, scripts/runs/pretrain_queue.sh) writes "QUEUE DONE". Order puts the core
# evidence first: user limited-label/probe/recommendation, item root, probes,
# full-data tasks and the reproduction check, then the anti-collapse ablation.
# Parsed as one block so that editing this file cannot affect a run in progress.
{
set -uo pipefail
cd /workspace/third_party/task-vector-estimation
R=/workspace/scripts/runs
LOG=/workspace/logs/chain.log
step() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }

until grep -q "QUEUE DONE" /workspace/logs/main/summary.log 2>/dev/null; do sleep 60; done
step "user-root pretraining finished"

ROOT=user STAGES="limited probe rec_low" bash "$R/finetune_main.sh";  step "user fine-tuning (limited, probe, rec_low) done"
LOSS=cosine ROOTS=item bash "$R/pretrain_queue.sh";                   step "item-root pretraining done"
ROOT=item STAGES="limited probe" bash "$R/finetune_main.sh";           step "item fine-tuning (limited, probe) done"

python /workspace/scripts/probes/make_probe_set.py --n_train_subsample 6000000 > /workspace/logs/probes_make.log 2>&1 \
  && python /workspace/scripts/probes/extract_embeddings.py > /workspace/logs/probes_extract.log 2>&1 \
  && python /workspace/scripts/probes/run_probes.py > /workspace/logs/probes_run.log 2>&1
step "information probes exit=$?"

ROOT=user STAGES="full repro" bash "$R/finetune_main.sh";              step "user full-data + reproduction done"
ROOT=item STAGES="full" bash "$R/finetune_main.sh";                    step "item full-data done"
bash "$R/ablations.sh";                                                step "ablations done"
step "CHAIN DONE"
exit $?
}

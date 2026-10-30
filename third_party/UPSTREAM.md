# Third-party code

## task-vector-estimation (TVE)

- Upstream: https://github.com/quang-truong/task-vector-estimation
- Base commit: `48445da63c0f952581e2d165038bf670a26f4731` (2026-02-22). The upstream `LICENSE` was
  added later in commit `ad435252f9d4e500043fc35121e736a68acd49d1` and is included unchanged.
- License: MIT, Copyright (c) 2025 Quang Truong (`task-vector-estimation/LICENSE`).
- Paper: Truong et al., "A Pre-training Framework for Relational Data with Information-theoretic
  Principles", NeurIPS 2025, doi:10.52202/085713-0277.
- Only the files needed for `rel-hm` are included: `src/`, `scripts/`, `cfg/rel-hm/`. The upstream
  visualization notebooks and the configs of other datasets are omitted.

Files added in this copy:
- `src/models/reljepa.py`, `src/helpers/reljepa_train_helper.py`, `src/train_reljepa.py`: RelJEPA
  pretraining.
- `src/models/idgnn.py`, `src/train_idgnn.py`: ID-GNN recommendation head on TVE's backbone,
  adapted from RelBench v1.1.0 `examples/idgnn_link.py` (MIT License, Copyright (c) 2023 RelBench
  Team, see `LICENSE.relbench`).
- `src/helpers/reljepa_ft_helper.py`: linear probing, per-example predictions and learning-rate
  sweep helpers.
- `src/helpers/metrics_compat.py`: metric compatibility shims.

Files modified in this copy:
- `src/helpers/task_helper.py`: registers the shared pretraining-subsample tasks
  (`user-pt-*`, `item-pt-*`) and patches RelBench's RMSE metric for newer scikit-learn.
- `src/helpers/dataset_helper.py`: optional fewer workers for evaluation loaders (`EVAL_NUM_WORKERS`).
- `src/helpers/single_task_train_helper.py`: keeps the latest predictions for bootstrap intervals.
- `src/train_baseline.py`: linear probing with a frozen backbone (`FREEZE_BACKBONE=1`), saved labels
  and predictions, optional checkpoint saving.
- `src/train_contrastive.py`, `src/train_mae.py`, `src/train_tve.py`, `src/train_tve_contrastive.py`,
  `src/train_tve_mae.py`: record the repository commit (`GIT_COMMIT`).
- `src/utils/io_utils.py`, `src/utils/log_utils.py`: W&B mode from `WANDB_MODE`, the run directory
  written to `RUN_LOGDIR_FILE` for the job queues, and per-seed result records read by
  `scripts/collect_results.py`.

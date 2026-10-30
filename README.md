# Learned and Handcrafted Future Targets in Relational Self-Supervised Learning

Code to reproduce the experiments of the paper *Learned and Handcrafted Future Targets in
Relational Self-Supervised Learning* (M.-K. Huynh, T.-M. Truong). The paper compares three
next-window pretraining signals for relational deep learning on RelBench's H&M database (`rel-hm`):

- TVE's handcrafted next-window statistics;
- an empty-window head;
- RelJEPA, a joint-embedding predictive architecture that predicts learned representations of
  each entity's next-window rows.

All methods share TVE's backbone, sampler, optimizer and pretraining budget.

## Repository layout

| Path | Content |
|---|---|
| `src/reljepa/` | RelJEPA components: target view, loss, EMA, SIGReg, collapse diagnostics, recommendation metrics |
| `third_party/task-vector-estimation/` | Fork of the TVE code (MIT) with RelJEPA, the ID-GNN recommendation head and the fine-tuning changes; see `third_party/UPSTREAM.md` |
| `configs/rel-hm/` | Every pretraining, fine-tuning and probe config, generated from TVE's configs by `scripts/make_configs.sh` |
| `configs/sweeps/` | Learning-rate grids for the fine-tuning and probe sweeps |
| `scripts/runs/` | The experiment queues (pilot, pretraining, fine-tuning, probes, recommendation, ablations) |
| `scripts/data/`, `scripts/setup/`, `scripts/probes/` | Data preparation, shared pretraining subsamples and information probes |
| `scripts/collect_results.py`, `scripts/analyze_results.py`, `scripts/tables/`, `scripts/figures/` | Result collection, paired bootstrap analysis, tables and figures |
| `tests/` | Leakage and target-view tests, objective and metric tests |
| `docker/` | Pinned environment |

## Environment

The experiments ran in Docker with Python 3.10.14, PyTorch 2.1.2 (CUDA 12.1), PyG 2.4.0,
pyg_lib 0.3.1, PyTorch Frame 0.2.3 and RelBench 1.1.0 (full freeze in `docker/pip-freeze.txt`),
on one NVIDIA RTX 3050 (4 GB) with 24 GB of RAM.

```bash
docker build -t reljepa-env:1 docker/
```

`scripts/dk.sh <command>` runs a command inside the container. Its details:
- The repository is mounted at `/workspace`, and the working directory is the TVE root.
- Data and weights live in the Docker volumes `reljepa-data` and `reljepa-weights`.
- W&B runs offline.
- `DK_NAME=<name> scripts/dk.sh ...` starts a detached, named container.

Run all commands below from the repository root in a Bash shell (Git Bash on Windows). The
repository must be a git clone, because each run records its commit.

## Data

1. **TVE's pretraining tables.** Download TVE's released `data.zip` (Google Drive link in
   `third_party/task-vector-estimation/README.md`) to `downloads/data.zip`, then extract the
   `rel-hm` part and register the task names used here:
   ```bash
   scripts/dk.sh python /workspace/scripts/setup/extract_tve_data.py /workspace/downloads/data.zip data --datasets rel-hm
   scripts/dk.sh python /workspace/scripts/setup/register_tasks.py
   ```
2. **RelBench database and text encoder.** The first run downloads the RelBench `rel-hm`
   database and the GloVe text encoder. It needs network access, so pass `HF_HUB_OFFLINE=0` to
   `scripts/dk.sh` for that run, e.g. `HF_HUB_OFFLINE=0 scripts/dk.sh python /workspace/scripts/data/task_stats.py`.
3. **Shared pretraining subsamples.** Every pretraining method reads the same subsample. Each
   draw uses seed 2026 and keeps only windows that end on or before the validation cutoff.
   ```bash
   scripts/dk.sh python /workspace/scripts/data/make_pretrain_subsample.py --root user --n_train 6000000
   scripts/dk.sh python /workspace/scripts/data/make_pretrain_subsample.py --root item --n_train 2000000
   ```
   The loss pilot (step 2 of the next section) used a 3,000,000-row customer subsample; create
   it with `--n_train 3000000` before the pilot, then recreate the 6,000,000-row one.

## Running the experiments

1. **Configs:** `bash scripts/make_configs.sh`. This regenerates `configs/rel-hm/` (identical to the committed files).
2. **Loss pilot** (smooth-L1 vs cosine): run `scripts/dk.sh bash /workspace/scripts/runs/pilot_loss.sh`.
   Then run `scripts/dk.sh bash /workspace/scripts/runs/post_pilot.sh` for the decision probes,
   which include the frozen random-encoder churn probe.
3. **Pretraining, customer root:** `scripts/dk.sh env LOSS=cosine ROOTS=user bash /workspace/scripts/runs/pretrain_queue.sh`.
4. **Downstream evaluation:** `scripts/dk.sh bash /workspace/scripts/runs/chain_after_user_pt.sh`.
   It runs, in order:
   - customer fine-tuning (limited labels, linear probes, recommendation with 10% of customers);
   - article-root pretraining and fine-tuning;
   - the information probes;
   - full-data fine-tuning and the reproduction check;
   - the anti-collapse and loss ablations.
5. **Remaining full-data probes:**
   ```bash
   scripts/dk.sh bash /workspace/scripts/runs/probes_full.sh   # first pass
   scripts/dk.sh bash /workspace/scripts/runs/probes_grid.sh   # completes the learning-rate grid
   ```
6. **Recommendation with 1% and 0.1% of customers:** `scripts/dk.sh bash /workspace/scripts/runs/rec_lowlabel.sh`.
7. **TVE-1 + empty-window head control:** `scripts/dk.sh bash /workspace/scripts/runs/ablation_tveempty.sh`.

The queues skip finished steps when re-run. Logs go to `logs/` and to the run directories under
`third_party/task-vector-estimation/logs/`.

## Analysis, tables and figures

```bash
python scripts/collect_results.py                                   # run directories -> results/raw/finetune.csv
scripts/dk.sh python /workspace/scripts/analyze_results.py          # selection on validation, paired bootstrap
python scripts/probes/bootstrap_probes.py                           # information-probe intervals
python scripts/e2e3_firstpass.py                                    # first-pass sensitivity analysis
python scripts/tables/make_table_primary.py                         # Table 2
python scripts/tables/make_table_node.py                            # entity tasks
python scripts/tables/make_table_rec.py                             # recommendation
python scripts/tables/make_table_tasks.py                           # task statistics
python scripts/tables/make_numbers.py                               # every value quoted in the text
python scripts/figures/plot_decomposition.py                        # Fig. 2
python scripts/figures/plot_collapse.py                             # Fig. 3
```

The scripts that run on the host need Python 3 with `numpy`, `pandas`, `scikit-learn` and
`matplotlib`. They can also run inside the container through `scripts/dk.sh`. Tables and
figures are written to `paper/tables/` and `paper/figures/`.

## Tests

```bash
scripts/dk.sh python -m pytest -q -p no:cacheprovider /workspace/tests
```

The tests check, among other things, that every pretraining target view contains only rows in
`(t, t + Δ]`, that context views contain no row after `t`, and that pretraining windows end on or
before the validation cutoff.

## Third-party code

`third_party/task-vector-estimation/` is a modified copy of
[TVE](https://github.com/quang-truong/task-vector-estimation) (MIT License, Copyright (c) 2025
Quang Truong). `third_party/UPSTREAM.md` lists the pinned upstream commit and the modified files.
The ID-GNN recommendation head is adapted from RelBench's examples (MIT License, see
`third_party/LICENSE.relbench`). The SIGReg regularizer re-implements the method of LeJEPA
(Balestriero and LeCun, 2025).

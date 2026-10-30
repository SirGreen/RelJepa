#!/usr/bin/env bash
# Generate every experiment config from TVE's released configs, so that all methods share TVE's
# backbone, optimiser and schedule. Only the listed fields are changed.
# Usage: scripts/make_configs.sh   (from the repo root; rewrites configs/rel-hm/*)
set -euo pipefail
cd "$(dirname "$0")/.."
T=third_party/task-vector-estimation/cfg/rel-hm
O=configs/rel-hm
mkdir -p "$O"

# --- Pretraining; TVE's 81 x 1000-step budget ---
for root in user item; do
  sed -e "s/task: ${root}-tve-1-hop /task: ${root}-pt-tve-1-hop /" \
      "$T/${root}-tve-1-hop/tve-graphsage-128.yaml" > "$O/${root}-tve-1-hop.yaml"
  sed -e "s/task: ${root}-ssl /task: ${root}-pt-ssl /" \
      "$T/${root}-ssl/mae-graphsage-128.yaml" > "$O/${root}-mae.yaml"
done

# RelJEPA: TVE's pretraining config (same optimiser, schedule, budget, sampler) + a jepa section.
jepa_block() {  # $1 = target types, $2 = loss, $3 = use_ema, $4 = beta_tve, $5 = lambda_sigreg
cat <<EOF

# RelJEPA
jepa:
  delta_days: 7                           # Δ = downstream window (rel-hm: 7 days)
  target_types: $1
  loss: $2                                # ℓ on LayerNorm-ed targets (smooth_l1 | cosine | mse)
  tau_start: 0.996                        # EMA momentum, cosine schedule to 1.0
  use_ema: $3                             # False = shared encoder without stop-gradient
  beta_tve: $4                            # > 0: RelJEPA+TVE hybrid (TVE head on z_ctx)
  lambda_sigreg: ${5:-0.0}                 # > 0: RelJEPA-SIGReg, (1-l) L_JEPA + l SIGReg(z_ctx)
  target_temporal_strategy: last          # target view samples the most recent rows <= t+Δ
  target_num_workers: 2
  diag_every: 100                         # collapse diagnostics every N steps
  collapse_std: 0.001                     # fail fast if mean per-dim std(z_ctx) drops below
  mask_rate: 0.15                         # TVE's input corruption on the context view
EOF
}
for root in user item; do
  if [[ $root == user ]]; then types="[transactions, article]"; else types="[transactions, customer]"; fi
  base="$O/${root}-tve-1-hop.yaml"
  for loss in smooth_l1 cosine; do
    { cat "$base"; jepa_block "$types" "$loss" True 0.0; } > "$O/${root}-reljepa-ema-${loss}.yaml"
    { cat "$base"; jepa_block "$types" "$loss" True 1.0; } > "$O/${root}-reljepa-hybrid-${loss}.yaml"
  done
  # Anti-collapse ablation: shared encoder, no EMA/stop-grad,
  # with SIGReg (lambda 0.02, LeJEPA's example) or without any anti-collapse term; 10 x 1000 steps.
  for loss in smooth_l1 cosine; do
    { cat "$base"; jepa_block "$types" "$loss" False 0.0 0.02; } | sed -e 's/num_epochs: 81 /num_epochs: 10 /' > "$O/${root}-reljepa-sigreg-${loss}.yaml"
    { cat "$base"; jepa_block "$types" "$loss" False 0.0 0.0; } | sed -e 's/num_epochs: 81 /num_epochs: 10 /' > "$O/${root}-reljepa-noac-${loss}.yaml"
    { cat "$base"; jepa_block "$types" "$loss" True 0.0 0.0; } | sed -e 's/num_epochs: 81 /num_epochs: 10 /' > "$O/${root}-reljepa-ema10-${loss}.yaml"
    # Empty-head ablation: EMA-10 without the empty-window BCE term, or with
    # the BCE term only (no latent regression).
    sed -e 's/^  mask_rate: .*/&\n  bce_weight: 0.0                         # no empty-window head/' \
        "$O/${root}-reljepa-ema10-${loss}.yaml" > "$O/${root}-reljepa-ema10-noempty-${loss}.yaml"
    sed -e 's/^  mask_rate: .*/&\n  reg_weight: 0.0                         # empty-window head only/' \
        "$O/${root}-reljepa-ema10-${loss}.yaml" > "$O/${root}-reljepa-ema10-emptyonly-${loss}.yaml"
  done
  # Pilot variant: 8 epochs (8k steps) to choose ℓ on validation AUROC.
  for loss in smooth_l1 cosine; do
    sed -e 's/num_epochs: 81 /num_epochs: 8  /' -e 's/eval_every: 5 /eval_every: 4 /' \
        "$O/${root}-reljepa-ema-${loss}.yaml" > "$O/${root}-reljepa-ema-${loss}-pilot.yaml"
  done
done

# Item-root pretraining budget: item subgraphs are ~2x slower (5 vs 13 steps/s),
# so every item-root method uses 41 epochs with the LR decay moved to epoch 25 (same schedule shape).
for f in "$O"/item-tve-1-hop.yaml "$O"/item-mae.yaml "$O"/item-reljepa-ema-*.yaml "$O"/item-reljepa-hybrid-*.yaml; do
  case "$f" in *-pilot.yaml) continue ;; esac
  sed -i -e 's/num_epochs: 81 /num_epochs: 41 /' -e 's/decay_steps: 50 /decay_steps: 25 /' "$f"
done

# TVE-1 + empty-window head: the RelJEPA+TVE hybrid without the target
# regression, so it differs from the hybrid by the JEPA term only (same budget as the hybrid).
for root in user item; do
  sed -e 's/^  mask_rate: .*/&\n  reg_weight: 0.0                         # TVE head + empty-window head, no target regression/' \
      "$O/${root}-reljepa-hybrid-cosine.yaml" > "$O/${root}-reljepa-tveempty-cosine.yaml"
done

# --- Fine-tuning: TVE's task configs; the limited-label user-churn config exists for top50 only ---
for ch in 128 256; do
  cp "$T/user-churn/graphsage-$ch.yaml" "$O/ft-user-churn-$ch.yaml"
  cp "$T/item-sales/graphsage-$ch.yaml" "$O/ft-item-sales-$ch.yaml"
  # Reduced full-data budget for every method: 10 instead of 20 epochs.
  for t in user-churn item-sales; do
    sed -e 's/num_epochs: 20 /num_epochs: 10 /' "$T/$t/graphsage-$ch.yaml" > "$O/ft-$t-$ch-e10.yaml"
  done
  cp "$T/user-churn-top-spending/graphsage-$ch.yaml" "$O/ft-user-churn-top50-spending-$ch.yaml"
  sed -e 's/task: user-churn-top50-spending/task: user-churn-top100-spending/' \
      "$T/user-churn-top-spending/graphsage-$ch.yaml" > "$O/ft-user-churn-top100-spending-$ch.yaml"
  cp "$T/item-sales-top-spending/graphsage-$ch.yaml" "$O/ft-item-sales-top200-spending-$ch.yaml"
  # Reduced limited-label budget: 20 epochs like the other limited tasks.
  sed -e 's/num_epochs: 50 /num_epochs: 20 /' "$T/item-sales-top-spending/graphsage-$ch.yaml"       > "$O/ft-item-sales-top200-spending-$ch-e20.yaml"
  # Recommendation (ID-GNN, RelBench v1.1.0 example settings: lr 1e-3, most-recent sampling); the
  # training-user fraction is a template variable (1.0 = full data, 0.1 = low-label setting).
  sed -e 's/task: user-churn$/task: user-item-purchase/' \
      -e 's/lr: 0.005 /lr: 0.001 /' \
      -e 's/num_epochs: 20 /num_epochs: 10 /' \
      -e 's/temporal_strategy: "uniform"/temporal_strategy: "last"   /' \
      "$T/user-churn/graphsage-$ch.yaml" > "$O/ft-user-item-purchase-$ch.yaml"
  cat >> "$O/ft-user-item-purchase-$ch.yaml" <<'EOF'

# Recommendation fine-tuning (src/train_idgnn.py)
rec:
  train_user_fraction: {{ user_fraction }}   # 1.0 = all training users; 0.1 = low-label setting
  subsample_seed: 2026
EOF
done
# Pilot criterion: short linear probe on full user-churn (76,556 val rows).
sed -e 's/num_epochs: 20 /num_epochs: 3  /' -e 's/max_steps_per_epoch: 1000 /max_steps_per_epoch: 500  /'     "$T/user-churn/graphsage-128.yaml" > "$O/pilot-probe-user-churn-128.yaml"
# Full-data linear probes: the same short budget as the pilot probe.
# lr and warmup_steps are Jinja variables like in TVE's limited-label configs: TVE's sweep runner (-s)
# only sets values through these variables and silently ignores hard-coded ones.
for t in user-churn item-sales; do
  sed -e 's/num_epochs: 20 /num_epochs: 3  /' -e 's/max_steps_per_epoch: 1000 /max_steps_per_epoch: 500  /' \
      -e 's/^  lr: [0-9.]* /  lr: {{ lr }} /' -e 's/^  warmup_steps: 1$/  warmup_steps: {{ warmup_steps }}/' \
      "$T/$t/graphsage-128.yaml" > "$O/probe-$t-128.yaml"
done
ls "$O"

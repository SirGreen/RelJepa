"""RelJEPA pretraining entry point.

Usage (inside the container, TVE root):
  python src/train_reljepa.py -c /workspace/configs/rel-hm/user-reljepa-ema.yaml --num_runs 1 \
      --num_workers 3 --gpus '[0]' --name <name> --note <note>
"""
import json
import os
import signal

import rootutils

import torch
from torch import optim

import warnings
warnings.filterwarnings("ignore")

import wandb

from relbench.base import EntityTask

rootutils.setup_root('.', indicator='.project-root', pythonpath=True)

from src.helpers.dataset_helper import get_data
from src.helpers.loss_helper import get_loss
from src.helpers.lr_scheduler import get_lr_scheduler, step_scheduler
from src.helpers.model_helper import get_conv_layer, get_hetero_encoder
from src.helpers.task_helper import get_task_metadata
from src.helpers.save_helper import save_backbone_checkpoint
from src.helpers.reljepa_train_helper import (
    CollapseError, evaluate, get_paired_loaders, set_epoch, train_one_epoch,
)
from src.models.reljepa import RelJEPA

from src.utils.io_utils import (
    parse_cfg_args, parse_train_args, load_config, is_main_process, get_device, create_log_dir,
    print_config_tree, synchronize, compute_params, get_world_size, handle_exit, init_process_group,
    destroy_process_group, setup_cfg, setup_wandb, infer_dataset_task, sanity_check,
)
from src.utils.io_utils import rank_zero_print as print
from src.utils.io_utils import rank_zero_pprint as pprint
from src.utils.log_utils import CurvesLogger, print_epoch_metrics
from src.utils.random_utils import set_random_seed
from src.utils.wandb_utils import set_wandb_directories

set_wandb_directories()


def main(cfg, cfg_file, sweep_file):
    assert get_world_size() == 1, "RelJEPA pretraining runs on a single GPU in this project"
    device = get_device(cfg)
    setup_cfg(cfg)
    log_dir = create_log_dir(cfg, cfg_file, sweep_file)
    setup_wandb(cfg, log_dir)
    print_config_tree(cfg, save_to_file=True, output_dir=log_dir)

    data_dict = get_data(cfg, device, print_stype=True)
    task = data_dict["task"]
    task_name = data_dict["task_name"]
    assert issubclass(task.__class__, EntityTask)
    metadata = get_task_metadata(task, cfg.dataset.cls, task_name)
    pprint({task_name: metadata})

    loader_dict, delta = get_paired_loaders(cfg, data_dict)

    # Inverse-frequency weights for empty windows, from the training table:
    # TVE's 1-hop null label marks (v, t) with no fact row in (t, t + Δ], which empties every target type.
    train_df = data_dict["train_table"].df
    p_null = float((train_df["label"] == 1).mean()) if "label" in train_df else None
    p_empty = {k: p_null for k in cfg.jepa.target_types}
    print(f"Empty-window rate in the training table: {p_null}")

    for run in range(cfg.num_runs):
        set_random_seed(cfg.seed + run)
        tve_out = metadata["out_channels"] if cfg.jepa.beta_tve > 0 else None
        model = RelJEPA(
            data=data_dict["data"],
            schema_graph=data_dict["schema_graph"],
            schema_line_graph=data_dict["schema_line_graph"],
            col_stats_dict=data_dict["col_stats_dict"],
            target_types=cfg.jepa.target_types,
            hetero_encoder_cls=get_hetero_encoder(cfg),
            hetero_encoder_kwargs={"out_channels": cfg.model.conv.kwargs.channels, **cfg.model.hetero_encoder.kwargs},
            temporal_encoder_kwargs={"out_channels": cfg.model.conv.kwargs.channels, **cfg.model.temporal_encoder.kwargs},
            conv_cls=get_conv_layer(cfg),
            conv_kwargs={"num_layers": len(cfg.train.num_neighbors), **cfg.model.conv.kwargs},
            shallow_list=cfg.model.shallow_list,
            mask_rate=cfg.jepa.mask_rate,
            tve_out_channels=tve_out,
            use_ema=cfg.jepa.use_ema,
        ).to(device)

        cls = cfg.optimizer.pop("cls")
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = getattr(optim, cls)(trainable, **cfg.optimizer)
        cfg.optimizer.cls = cls
        scheduler = get_lr_scheduler(cfg, optimizer, metadata)
        compute_params(model, print_params=False)

        tve_loss_fn = get_loss(device=device, **cfg.train.loss_fn) if cfg.jepa.beta_tve > 0 else None
        curves_logger = CurvesLogger(task_names=[task_name], metadata_dict={task_name: metadata})
        steps_per_epoch = min(len(loader_dict["train"][0]), cfg.train.max_steps_per_epoch + 1)
        state = {"step": 0, "total_steps": cfg.train.num_epochs * steps_per_epoch}
        diag_path = os.path.join(log_dir, f"run{run+1}-diagnostics.csv")
        status = "done"
        for epoch in range(cfg.train.num_epochs):
            set_epoch(loader_dict["train"], epoch + run * cfg.train.num_epochs)
            try:
                train_losses = train_one_epoch(cfg, model, task, loader_dict["train"], optimizer, tve_loss_fn,
                                               p_empty, delta, device, state, diag_path)
            except CollapseError as err:
                print(f"!! {err}. Stopping this run (fail fast).")
                status = "collapsed"
                break
            val_losses = test_losses = None
            if epoch % cfg.train.eval_every == 0 or epoch == cfg.train.num_epochs - 1:
                val_losses = evaluate(cfg, model, task, loader_dict["val"], tve_loss_fn, p_empty, delta, device)
                test_losses = evaluate(cfg, model, task, loader_dict["test"], tve_loss_fn, p_empty, delta, device)
                save_backbone_checkpoint(model, optimizer, epoch, os.path.join(log_dir, f"epoch{epoch+1}.pth"))
            curves_logger.log_loss(
                train_loss_dict={task_name: train_losses},
                val_loss_dict={task_name: val_losses} if val_losses else None,
                test_loss_dict={task_name: test_losses} if test_losses else None,
            )
            print_epoch_metrics(run, epoch, curves_logger, optimizer)
            step_scheduler(cfg, scheduler, optimizer, None, metadata)

        # Final checkpoints: context encoder f_θ (used for fine-tuning) and EMA target encoder f_ξ.
        save_backbone_checkpoint(model, optimizer, epoch, os.path.join(log_dir, f"run{run+1}-final.pth"))
        if model.target is not None:
            torch.save({"target": model.target.state_dict(), "epoch": epoch},
                       os.path.join(log_dir, f"run{run+1}-final-target.pth"))
        # Full state (incl. predictor) for offline diagnostics.
        torch.save({"model": model.state_dict(), "epoch": epoch}, os.path.join(log_dir, f"run{run+1}-final-full.pth"))
        with open(os.path.join(log_dir, f"run{run+1}-status.json"), "w") as f:
            json.dump({"status": status, "steps": state["step"], "epochs_completed": epoch + 1,
                       "git_commit": os.environ.get("GIT_COMMIT", "unknown"),
                       "wandb_run_id": getattr(wandb.run, "id", None)}, f, indent=1)
        print(f"Run {run+1}: {status} after {state['step']} steps. Logs: {log_dir}")
    wandb.finish()


if __name__ == '__main__':
    signal.signal(signal.SIGINT, handle_exit)
    args, unparsed = parse_cfg_args()
    args, vars = parse_train_args(args, unparsed)
    cfg = load_config(args.config, context=vars)
    cfg.sha = os.environ.get("GIT_COMMIT", "unknown")
    sanity_check(cfg)
    cfg.sweep = False
    cfg.dataset.cls, cfg.dataset.task = infer_dataset_task(cfg, args.config)
    cfg.model.conv.kwargs.num_layers = len(cfg.train.num_neighbors)
    init_process_group(cfg)
    main(cfg, args.config, args.sweep_config)
    destroy_process_group()

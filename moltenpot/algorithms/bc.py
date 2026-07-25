"""
algorithms/bc.py — Behavior Cloning
====================================
Trains a policy via supervised learning on dataset actions.
Loss: CrossEntropy(π(a|s), a_dataset)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from moltenpot.data_utils import make_offline_dataloader, shutdown_dataloader
from moltenpot.model import MoltenpotAgent
from moltenpot.algorithms.eval_utils import evaluate_multi_scenario, shutdown_ray

logger = logging.getLogger(__name__)


def train(cfg: DictConfig) -> None:
    """Behavior Cloning training loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    in_dist:  List[str] = list(cfg.in_dist_scenarios)
    out_dist: List[str] = list(cfg.out_dist_scenarios) if cfg.out_dist_eval else []

    logger.info(
        "BC | device=%s | seed=%d | in_dist=%d scenarios | out_dist=%d scenarios",
        device, cfg.seed, len(in_dist), len(out_dist),
    )

    use_wandb = cfg.wandb.enabled
    if use_wandb:
        import wandb
        run_name = cfg.wandb.run_name or f"bc_{cfg.substrate}_{cfg.train_mode}_s{cfg.seed}"
        wandb.init(
            project=cfg.wandb.project,
            name=run_name,
            group=cfg.wandb.group,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    num_actions = int(cfg.model.num_actions)
    logger.info("num_actions=%d (from cfg.model.num_actions)", num_actions)

    use_scenario_id = bool(cfg.model.get("use_scenario_id", False))
    num_scenarios   = len(in_dist) if use_scenario_id else 0
    if use_scenario_id:
        logger.info(
            "Scenario-ID conditioning ON: one-hot over %d scenarios appended to CNN features.",
            num_scenarios,
        )

    dataloader = make_offline_dataloader(cfg, need_next_obs=False)

    model = MoltenpotAgent(
        num_actions=num_actions,
        fc_units=cfg.model.fc_units,
        gru_hidden=cfg.model.gru_hidden,
        max_agents=cfg.model.max_agents,
        num_scenarios=num_scenarios,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.algorithm.lr)
    criterion = nn.CrossEntropyLoss()

    ckpt_dir = Path(cfg.checkpoints_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    def _iter_forever(loader):
        while True:
            yield from loader

    data_iter = _iter_forever(dataloader)
    model.train()

    init_metrics = evaluate_multi_scenario(
        model, in_dist, out_dist,
        num_eval_workers=cfg.num_eval_workers,
        seed=cfg.seed,
        num_episodes=cfg.eval_episodes,
        use_agent_id=cfg.model.use_agent_id,
        use_scenario_id=use_scenario_id,
    )
    logger.info(
        "Eval | step=0 (init) | in_dist=%.2f",
        init_metrics.get("eval/in_dist/return_mean", float("nan")),
    )
    if use_wandb:
        import wandb
        wandb.log(init_metrics, step=0)

    for global_step in range(cfg.num_updates):
        obs, actions, agent_ids, scenario_ids = next(data_iter)
        B, T = obs.shape[:2]
        obs       = obs.to(device)
        actions   = actions.to(device)
        agent_ids = agent_ids.to(device) if cfg.model.use_agent_id else None
        scenario_ids = scenario_ids.to(device) if use_scenario_id else None

        h_state = model.initial_hidden(batch_size=B).to(device)
        logits, _, _ = model(obs, h_state, agent_ids=agent_ids, scenario_ids=scenario_ids)   # (B, T, num_actions)

        loss  = criterion(logits.reshape(-1, model.num_actions), actions.reshape(-1))
        preds = logits.reshape(-1, model.num_actions).argmax(dim=-1)
        acc   = (preds == actions.reshape(-1)).float().mean().item()

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        log_dict: dict = {}

        if global_step % cfg.log_interval == 0:
            sequences_seen  = (global_step + 1) * cfg.batch_size
            timesteps_seen  = sequences_seen * cfg.seq_len
            logger.info(
                "Step %d | loss=%.4f | acc=%.4f",
                global_step, loss.item(), acc,
            )
            log_dict.update({
                "train/loss":           loss.item(),
                "train/accuracy":       acc,
                "train/sequences_seen": sequences_seen,
                "train/timesteps_seen": timesteps_seen,
            })

        if global_step > 0 and global_step % cfg.eval_interval == 0:
            metrics = evaluate_multi_scenario(
                model, in_dist, out_dist,
                num_eval_workers=cfg.num_eval_workers,
                seed=cfg.seed,
                num_episodes=cfg.eval_episodes,
                use_agent_id=cfg.model.use_agent_id,
                use_scenario_id=use_scenario_id,
            )
            logger.info(
                "Eval | step=%d | in_dist=%.2f",
                global_step, metrics.get("eval/in_dist/return_mean", float("nan")),
            )
            log_dict.update(metrics)

        if log_dict and use_wandb:
            import wandb
            wandb.log(log_dict, step=global_step)

    final_metrics = evaluate_multi_scenario(
        model, in_dist, out_dist,
        num_eval_workers=cfg.num_eval_workers,
        seed=cfg.seed,
        num_episodes=cfg.eval_episodes,
        use_agent_id=cfg.model.use_agent_id,
        use_scenario_id=use_scenario_id,
    )
    logger.info(
        "Eval | step=%d (final) | in_dist=%.2f",
        cfg.num_updates, final_metrics.get("eval/in_dist/return_mean", float("nan")),
    )
    if use_wandb:
        import wandb
        wandb.log(final_metrics, step=cfg.num_updates)

    final_path = ckpt_dir / f"bc_s{cfg.seed}_final.pth"
    torch.save(model.state_dict(), final_path)
    logger.info("Saved final checkpoint: %s", final_path)

    if use_wandb:
        import wandb
        art = wandb.Artifact(
            name=f"bc-{cfg.substrate}-{cfg.train_mode}-s{cfg.seed}",
            type="model",
            description=f"BC final checkpoint | substrate={cfg.substrate} split={cfg.train_mode} seed={cfg.seed}",
        )
        art.add_file(str(final_path))
        wandb.log_artifact(art)
        wandb.finish()   # blocks until upload completes

    # Local disk-space cleanup — the final checkpoint is safely in W&B by now.
    final_path.unlink(missing_ok=True)
    try:
        ckpt_dir.rmdir()
    except OSError:
        pass

    # Free DataLoader workers + Ray deterministically, so a single-process
    # Hydra --multirun doesn't accumulate them run-over-run (see helper docs).
    del data_iter
    shutdown_dataloader(dataloader)
    shutdown_ray()

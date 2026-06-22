"""
algorithms/bcq.py — Batch-Constrained Q-Learning (BCQ-D)
=========================================================
Discrete BCQ constrains the Q-learning target to only consider actions
that are sufficiently likely under the behavioural policy (imitator):

    mask = π(a|s) / max_a π(a|s) > τ
    π_BCQ(a|s) = argmax_{a: mask} Q(s, a)

Reference: Fujimoto et al. (2019) "Off-Policy Deep RL without Exploration."
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from moltenpot.data_utils import MultiScenarioTransitionDataset
from moltenpot.model import MoltenpotAgent
from moltenpot.algorithms.eval_utils import evaluate_multi_scenario

logger = logging.getLogger(__name__)


def _soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_((1 - tau) * tp.data + tau * sp.data)


def train(cfg: DictConfig) -> None:
    """BCQ-D training loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    alg      = cfg.algorithm
    in_dist  = list(cfg.in_dist_scenarios)
    out_dist = list(cfg.out_dist_scenarios) if cfg.out_dist_eval else []

    logger.info(
        "BCQ | device=%s | seed=%d | tau=%.2f | in_dist=%d | out_dist=%d",
        device, cfg.seed, alg.tau, len(in_dist), len(out_dist),
    )

    use_wandb = cfg.wandb.enabled
    if use_wandb:
        import wandb
        run_name = cfg.wandb.run_name or f"bcq_{cfg.substrate}_{cfg.train_mode}_s{cfg.seed}"
        wandb.init(
            project=cfg.wandb.project,
            name=run_name,
            group=cfg.wandb.group,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    num_actions = int(cfg.model.num_actions)
    logger.info("num_actions=%d (from cfg.model.num_actions)", num_actions)

    dataset = MultiScenarioTransitionDataset(cfg.data_root, in_dist, seq_len=cfg.seq_len)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        sampler=dataset.make_sampler(),
        num_workers=2,
        drop_last=True,
    )

    model = MoltenpotAgent(
        num_actions=num_actions,
        fc_units=cfg.model.fc_units,
        gru_hidden=cfg.model.gru_hidden,
        max_agents=cfg.model.max_agents,
    ).to(device)
    target = copy.deepcopy(model).requires_grad_(False).to(device)

    q_optimizer = torch.optim.Adam(
        list(model.q1_head.parameters()) + list(model.q2_head.parameters()) +
        list(model.backbone.parameters()) + list(model.gru.parameters()),
        lr=alg.q_lr,
    )
    imitator_optimizer = torch.optim.Adam(model.actor.parameters(), lr=alg.imitator_lr)

    ckpt_dir = Path(cfg.checkpoints_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    def _iter_forever(loader):
        while True:
            yield from loader

    data_iter = _iter_forever(dataloader)

    init_metrics = evaluate_multi_scenario(
        model, in_dist, out_dist,
        act_type="bcq",
        bcq_tau=alg.tau,
        num_eval_workers=cfg.num_eval_workers,
        seed=cfg.seed,
        num_episodes=cfg.eval_episodes,
        use_agent_id=cfg.model.use_agent_id,
    )
    logger.info(
        "Eval | step=0 (init) | in_dist=%.2f",
        init_metrics.get("eval/in_dist/return_mean", float("nan")),
    )
    if use_wandb:
        import wandb
        wandb.log(init_metrics, step=0)

    for global_step in range(cfg.num_updates):
        obs_full, actions, rewards, dones, agent_ids = next(data_iter)
        B, T = actions.shape
        obs_full  = obs_full.to(device)
        actions   = actions.to(device)
        rewards   = rewards.to(device)
        dones     = dones.to(device)
        agent_ids = agent_ids.to(device) if cfg.model.use_agent_id else None

        obs_t   = obs_full[:, :T]
        obs_tp1 = obs_full[:, 1:]

        # --- Target value ---
        with torch.no_grad():
            q1_tp1, q2_tp1, _, logits_tp1, _ = target.get_q_v(
                obs_tp1, target.initial_hidden(B).to(device), agent_ids=agent_ids
            )
            probs_tp1    = F.softmax(logits_tp1, dim=-1)
            max_prob_tp1 = probs_tp1.max(dim=-1, keepdim=True)[0]
            mask_tp1     = (probs_tp1 / max_prob_tp1) > alg.tau

            q_min_tp1            = torch.min(q1_tp1, q2_tp1)
            q_min_tp1[~mask_tp1] = -1e10
            v_tp1    = q_min_tp1.max(dim=-1)[0]
            q_target = rewards + alg.gamma * (1 - dones) * v_tp1

        # --- Q update ---
        q1_s, q2_s, _, logits, _ = model.get_q_v(
            obs_t, model.initial_hidden(B).to(device), agent_ids=agent_ids
        )
        q1_val = q1_s.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        q2_val = q2_s.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        bellman_loss   = F.mse_loss(q1_val, q_target) + F.mse_loss(q2_val, q_target)
        imitation_loss = -F.log_softmax(logits, dim=-1).gather(
            -1, actions.unsqueeze(-1)
        ).squeeze(-1).mean()

        total_loss = bellman_loss + imitation_loss
        q_optimizer.zero_grad()
        imitator_optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        q_optimizer.step()
        imitator_optimizer.step()

        _soft_update(target, model, alg.tau_soft)

        log_dict: dict = {}

        if global_step % cfg.log_interval == 0:
            sequences_seen = (global_step + 1) * cfg.batch_size
            timesteps_seen = sequences_seen * cfg.seq_len
            logger.info(
                "Step %d | bellman=%.4f | imitation=%.4f",
                global_step, bellman_loss.item(), imitation_loss.item(),
            )
            log_dict.update({
                "train/bellman_loss":    bellman_loss.item(),
                "train/imitation_loss":  imitation_loss.item(),
                "train/sequences_seen": sequences_seen,
                "train/timesteps_seen":  timesteps_seen,
            })

        if global_step > 0 and global_step % cfg.eval_interval == 0:
            metrics = evaluate_multi_scenario(
                model, in_dist, out_dist,
                act_type="bcq",
                bcq_tau=alg.tau,
                num_eval_workers=cfg.num_eval_workers,
                seed=cfg.seed,
                num_episodes=cfg.eval_episodes,
                use_agent_id=cfg.model.use_agent_id,
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
        act_type="bcq",
        bcq_tau=alg.tau,
        num_eval_workers=cfg.num_eval_workers,
        seed=cfg.seed,
        num_episodes=cfg.eval_episodes,
        use_agent_id=cfg.model.use_agent_id,
    )
    logger.info(
        "Eval | step=%d (final) | in_dist=%.2f",
        cfg.num_updates, final_metrics.get("eval/in_dist/return_mean", float("nan")),
    )
    if use_wandb:
        import wandb
        wandb.log(final_metrics, step=cfg.num_updates)

    final_path = ckpt_dir / f"bcq_s{cfg.seed}_final.pth"
    torch.save(model.state_dict(), final_path)
    logger.info("Saved final checkpoint: %s", final_path)

    if use_wandb:
        import wandb
        art = wandb.Artifact(
            name=f"bcq-{cfg.substrate}-{cfg.train_mode}-s{cfg.seed}",
            type="model",
            description=f"BCQ final checkpoint | substrate={cfg.substrate} split={cfg.train_mode} seed={cfg.seed}",
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

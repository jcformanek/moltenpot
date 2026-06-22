"""
algorithms/iql.py — Implicit Q-Learning (IQL)
===============================================
IQL avoids querying out-of-distribution Q-values by using expectile
regression to estimate V(s), keeping all updates fully in-sample.

Three update steps per batch:
  1. V-update: expectile regression  V ← E_τ[Q - V]
  2. Q-update: Bellman backup        Q ← r + γ V(s')
  3. π-update: weighted BC           π ← exp(β * A) · log π(a|s)

Reference: Kostrikov et al. (2021) "Offline RL with Implicit Q-Learning."
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


def _expectile_loss(diff: torch.Tensor, tau: float) -> torch.Tensor:
    weight = torch.where(diff > 0, tau, 1 - tau)
    return (weight * diff.pow(2)).mean()


def train(cfg: DictConfig) -> None:
    """IQL training loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    alg      = cfg.algorithm
    in_dist  = list(cfg.in_dist_scenarios)
    out_dist = list(cfg.out_dist_scenarios) if cfg.out_dist_eval else []

    logger.info(
        "IQL | device=%s | seed=%d | tau=%.2f | beta=%.1f | in_dist=%d | out_dist=%d",
        device, cfg.seed, alg.iql_tau, alg.beta, len(in_dist), len(out_dist),
    )

    use_wandb = cfg.wandb.enabled
    if use_wandb:
        import wandb
        run_name = cfg.wandb.run_name or f"iql_{cfg.substrate}_{cfg.train_mode}_s{cfg.seed}"
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

    v_optimizer     = torch.optim.Adam(model.critic.parameters(), lr=alg.v_lr)
    q_optimizer     = torch.optim.Adam(
        list(model.q1_head.parameters()) + list(model.q2_head.parameters()) +
        list(model.backbone.parameters()) + list(model.gru.parameters()),
        lr=alg.q_lr,
    )
    actor_optimizer = torch.optim.Adam(model.actor.parameters(), lr=alg.actor_lr)

    ckpt_dir = Path(cfg.checkpoints_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    def _iter_forever(loader):
        while True:
            yield from loader

    data_iter = _iter_forever(dataloader)

    init_metrics = evaluate_multi_scenario(
        model, in_dist, out_dist,
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

        # --- 1. V-update (expectile regression) ---
        with torch.no_grad():
            q1_t, q2_t, _, _, _ = target.get_q_v(
                obs_t, target.initial_hidden(B).to(device), agent_ids=agent_ids
            )
            q1_val = q1_t.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
            q2_val = q2_t.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
            q_dataset = torch.min(q1_val, q2_val)

        _, _, v_t, _, _ = model.get_q_v(obs_t, model.initial_hidden(B).to(device), agent_ids=agent_ids)
        v_t     = v_t.squeeze(-1)
        v_loss  = _expectile_loss(q_dataset - v_t, alg.iql_tau)

        v_optimizer.zero_grad()
        v_loss.backward()
        v_optimizer.step()

        # --- 2. Q-update (Bellman backup via target V) ---
        with torch.no_grad():
            _, _, v_tp1, _, _ = target.get_q_v(
                obs_tp1, target.initial_hidden(B).to(device), agent_ids=agent_ids
            )
            v_tp1    = v_tp1.squeeze(-1)
            q_target = rewards + alg.gamma * (1 - dones) * v_tp1

        q1_s, q2_s, _, _, _ = model.get_q_v(obs_t, model.initial_hidden(B).to(device), agent_ids=agent_ids)
        q1_s_val = q1_s.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        q2_s_val = q2_s.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        q_loss   = F.mse_loss(q1_s_val, q_target) + F.mse_loss(q2_s_val, q_target)

        q_optimizer.zero_grad()
        q_loss.backward()
        q_optimizer.step()

        _soft_update(target, model, alg.tau)

        # --- 3. Actor update (weighted BC) ---
        with torch.no_grad():
            adv     = q_dataset - v_t
            exp_adv = torch.exp(alg.beta * adv).clamp(max=100.0)

        _, _, _, logits, _ = model.get_q_v(obs_t, model.initial_hidden(B).to(device), agent_ids=agent_ids)
        log_probs  = F.log_softmax(logits, dim=-1)
        action_lp  = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        actor_loss = -(exp_adv * action_lp).mean()

        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        log_dict: dict = {}

        if global_step % cfg.log_interval == 0:
            sequences_seen = (global_step + 1) * cfg.batch_size
            timesteps_seen = sequences_seen * cfg.seq_len
            logger.info(
                "Step %d | v=%.4f | q=%.4f | pi=%.4f",
                global_step, v_loss.item(), q_loss.item(), actor_loss.item(),
            )
            log_dict.update({
                "train/v_loss":          v_loss.item(),
                "train/q_loss":          q_loss.item(),
                "train/actor_loss":      actor_loss.item(),
                "train/adv_mean":        adv.mean().item(),
                "train/sequences_seen": sequences_seen,
                "train/timesteps_seen":  timesteps_seen,
            })

        if global_step > 0 and global_step % cfg.eval_interval == 0:
            metrics = evaluate_multi_scenario(
                model, in_dist, out_dist,
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

    final_path = ckpt_dir / f"iql_s{cfg.seed}_final.pth"
    torch.save(model.state_dict(), final_path)
    logger.info("Saved final checkpoint: %s", final_path)

    if use_wandb:
        import wandb
        art = wandb.Artifact(
            name=f"iql-{cfg.substrate}-{cfg.train_mode}-s{cfg.seed}",
            type="model",
            description=f"IQL final checkpoint | substrate={cfg.substrate} split={cfg.train_mode} seed={cfg.seed}",
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

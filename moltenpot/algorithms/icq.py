"""
algorithms/icq.py — Independent Implicit Constraint Q-learning (ICQ)
====================================================================
Independent version of MAICQ (Yang et al. 2021, "Believe What You See:
Implicit Constraint Approach for Offline Multi-Agent RL", https://arxiv.org/abs/2106.03400),
with the QMIX value-decomposition mixer OMITTED so it runs as a fully
independent per-agent algorithm — appropriate for our mixed-motive setting
where CTDE value factorisation does not apply.

Reference implementation (with the mixer we drop):
  og_marl/baselines/tf2_systems/offline/maicq.py

ICQ avoids querying out-of-distribution actions entirely: it never maxes over
actions. Two updates per batch (shared trunk trained by both, single optimizer):

  1. Critic (within-sequence SARSA, no OOD actions):
       target_t = r_t + γ (1 - d_t) · ζ · Q_target(s_{t+1}, a_{t+1})
     where ζ is the ICQ implicit-constraint importance weight
       ζ = B · softmax_batch(Q_target(s,a) / β_target)
     (β_target large ⇒ ζ ≈ 1, i.e. near-standard SARSA). Loss: 0.5 · (target − Q(s_t,a_t))².

  2. Actor (advantage-weighted regression):
       A(s,a) = Q(s,a) − Σ_a π(a|s) Q(s,a)
       w      = B · softmax_batch(A / β_adv)          (stop-grad weights)
       loss   = − mean( w · log π(a|s) )

Target network: hard copy every ``target_update_period`` steps.
Discrete-action, recurrent; requires seq_len ≥ 2 (needs s_t, s_{t+1} in-sequence).
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

from moltenpot.data_utils import make_offline_dataloader, shutdown_dataloader
from moltenpot.model import MoltenpotAgent
from moltenpot.algorithms.eval_utils import evaluate_multi_scenario, shutdown_ray

logger = logging.getLogger(__name__)


def train(cfg: DictConfig) -> None:
    """Independent ICQ training loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    alg      = cfg.algorithm
    in_dist  = list(cfg.in_dist_scenarios)
    out_dist = list(cfg.out_dist_scenarios) if cfg.out_dist_eval else []

    logger.info(
        "ICQ | device=%s | seed=%d | lr=%.1e | gamma=%.2f | beta_tgt=%.1f | beta_adv=%.3f | "
        "in_dist=%d | out_dist=%d",
        device, cfg.seed, alg.lr, alg.gamma, alg.icq_target_q_taken_beta,
        alg.icq_advantages_beta, len(in_dist), len(out_dist),
    )

    use_wandb = cfg.wandb.enabled
    if use_wandb:
        import wandb
        run_name = cfg.wandb.run_name or f"icq_{cfg.substrate}_{cfg.train_mode}_s{cfg.seed}"
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

    # eval_at_end: run NO eval during training (skip the step-0 and interval
    # evals), keeping only the final end-of-training eval. See other algorithms.
    eval_at_end = bool(cfg.get("eval_at_end", False))
    train_eval_workers = 0 if eval_at_end else cfg.num_eval_workers

    dataloader = make_offline_dataloader(cfg, need_next_obs=True)

    model = MoltenpotAgent(
        num_actions=num_actions,
        fc_units=cfg.model.fc_units,
        gru_hidden=cfg.model.gru_hidden,
        max_agents=cfg.model.max_agents,
        num_scenarios=num_scenarios,
    ).to(device)
    target = copy.deepcopy(model).requires_grad_(False).to(device)

    # Single optimizer over all parameters (matches og-marl MAICQ): the critic
    # loss trains the shared trunk + Q head, the actor loss trains the shared
    # trunk + policy head; unused heads (q2, V) simply receive no gradient.
    optimizer = torch.optim.Adam(model.parameters(), lr=alg.lr)

    beta_tgt = float(alg.icq_target_q_taken_beta)
    beta_adv = float(alg.icq_advantages_beta)
    gamma    = float(alg.gamma)
    tgt_period = int(alg.target_update_period)

    ckpt_dir = Path(cfg.checkpoints_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    def _iter_forever(loader):
        while True:
            yield from loader

    data_iter = _iter_forever(dataloader)
    model.train()

    init_metrics = evaluate_multi_scenario(
        model, in_dist, out_dist,
        num_eval_workers=train_eval_workers,
        seed=cfg.seed,
        num_episodes=cfg.eval_episodes,
        use_agent_id=cfg.model.use_agent_id, use_scenario_id=use_scenario_id,
    )
    logger.info(
        "Eval | step=0 (init) | in_dist=%.2f",
        init_metrics.get("eval/in_dist/return_mean", float("nan")),
    )
    if use_wandb:
        import wandb
        wandb.log(init_metrics, step=0)

    for global_step in range(cfg.num_updates):
        obs_full, actions, rewards, dones, agent_ids, scenario_ids = next(data_iter)
        B, T = actions.shape
        if T < 2:
            raise ValueError(f"ICQ needs seq_len >= 2 (within-sequence SARSA); got T={T}.")
        obs_full  = obs_full.to(device)
        actions   = actions.to(device)
        rewards   = rewards.to(device)
        dones     = dones.to(device)
        agent_ids = agent_ids.to(device) if cfg.model.use_agent_id else None
        scenario_ids = scenario_ids.to(device) if use_scenario_id else None

        obs_t = obs_full[:, :T]   # s_0 .. s_{T-1}, aligned with actions/rewards/dones

        # --- Critic target: within-sequence SARSA with ICQ importance weighting ---
        with torch.no_grad():
            tq1, _, _, _, _ = target.get_q_v(
                obs_t, target.initial_hidden(B).to(device),
                agent_ids=agent_ids, scenario_ids=scenario_ids,
            )
            target_q_taken = tq1.gather(-1, actions.unsqueeze(-1)).squeeze(-1)   # (B, T)
            # ICQ implicit constraint: reweight the target across the batch by a
            # softmax over Q (β large -> ~uniform -> ~standard SARSA target).
            adv_Q = F.softmax(target_q_taken / beta_tgt, dim=0)                  # over batch
            target_q_taken = target_q_taken.shape[0] * adv_Q * target_q_taken
            targets = rewards[:, :-1] + gamma * (1.0 - dones[:, :-1]) * target_q_taken[:, 1:]

        # --- Online forward (Q + policy) ---
        q1, _, _, logits, _ = model.get_q_v(
            obs_t, model.initial_hidden(B).to(device),
            agent_ids=agent_ids, scenario_ids=scenario_ids,
        )
        q_taken = q1.gather(-1, actions.unsqueeze(-1)).squeeze(-1)               # (B, T)

        # 1. Critic (MSE to the ICQ SARSA target)
        td_error = targets - q_taken[:, :-1]
        q_loss   = 0.5 * td_error.pow(2).mean()

        # 2. Actor (advantage-weighted regression; weights are stop-grad)
        with torch.no_grad():
            q_vals = q1.detach()
            probs  = F.softmax(logits, dim=-1).detach()
            action_values = q_vals.gather(-1, actions.unsqueeze(-1)).squeeze(-1)  # (B, T)
            baseline = (probs * q_vals).sum(dim=-1)                               # (B, T)
            adv = action_values - baseline
            w = F.softmax(adv / beta_adv, dim=0)                                  # over batch
            w = w.shape[0] * w                                                    # scale by B
        log_probs = F.log_softmax(logits, dim=-1)
        log_pi    = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)       # (B, T)
        actor_loss = -(w * log_pi).mean()

        loss = q_loss + actor_loss
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()

        # Hard target update (matches og-marl: periodic copy, no soft tau)
        if global_step % tgt_period == 0:
            target.load_state_dict(model.state_dict())

        log_dict: dict = {}

        if global_step % cfg.log_interval == 0:
            sequences_seen = (global_step + 1) * cfg.batch_size
            timesteps_seen = sequences_seen * cfg.seq_len
            logger.info(
                "Step %d | q=%.4f | pi=%.4f | adv=%.3f",
                global_step, q_loss.item(), actor_loss.item(), adv.mean().item(),
            )
            log_dict.update({
                "train/q_loss":         q_loss.item(),
                "train/actor_loss":     actor_loss.item(),
                "train/adv_mean":       adv.mean().item(),
                "train/sequences_seen": sequences_seen,
                "train/timesteps_seen": timesteps_seen,
            })

        if (not eval_at_end) and global_step > 0 and global_step % cfg.eval_interval == 0:
            metrics = evaluate_multi_scenario(
                model, in_dist, out_dist,
                num_eval_workers=cfg.num_eval_workers,
                seed=cfg.seed,
                num_episodes=cfg.eval_episodes,
                use_agent_id=cfg.model.use_agent_id, use_scenario_id=use_scenario_id,
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
        use_agent_id=cfg.model.use_agent_id, use_scenario_id=use_scenario_id,
    )
    logger.info(
        "Eval | step=%d (final) | in_dist=%.2f",
        cfg.num_updates, final_metrics.get("eval/in_dist/return_mean", float("nan")),
    )
    if use_wandb:
        import wandb
        wandb.log(final_metrics, step=cfg.num_updates)

    final_path = ckpt_dir / f"icq_s{cfg.seed}_final.pth"
    torch.save(model.state_dict(), final_path)
    logger.info("Saved final checkpoint: %s", final_path)

    if use_wandb:
        import wandb
        art = wandb.Artifact(
            name=f"icq-{cfg.substrate}-{cfg.train_mode}-s{cfg.seed}",
            type="model",
            description=f"ICQ final checkpoint | substrate={cfg.substrate} split={cfg.train_mode} seed={cfg.seed}",
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

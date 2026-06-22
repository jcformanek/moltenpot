"""
workers.py — Ray Distributed Workers
=====================================
Implements three Ray remote actors:

1. RolloutWorker  — CPU actor.  Owns one MeltingPot environment and a
                    CPU copy of MoltenpotAgent.  Produces trajectory
                    fragments of length T for all focal agents.

2. PPOLearner     — GPU (or CPU) actor.  Owns the authoritative model,
                    consumes fragments from a Ray Queue, computes
                    GAE + PPO loss, and exposes updated weights.
                    Flattens the agent dimension into the batch dimension
                    so each agent's rollout is treated independently.

3. HDF5Writer     — CPU actor.  Receives fragments and streams them
                    to a structured HDF5 file with LZF compression.

Fragment data contract (all ndarray, host memory)
--------------------------------------------------
    obs        : (T, A, 3, 88, 88)  float32  [0, 1]   A = num_focal
    actions    : (T, A)             int32
    rewards    : (T, A)             float32
    dones      : (T,)               bool
    log_probs  : (T, A)             float32
    values     : (T, A)             float32
    h_states   : (T+1, A, H)       float32
    num_focal     : int
    scenario_name : str
    episode_id    : int
    worker_id     : int
    step_offset   : int
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

import numpy as np
import ray
import torch
import torch.nn as nn
from torch.distributions import Categorical

from moltenpot.wrappers import MeltingPotShimmy
from moltenpot.model import MoltenpotAgent

logger = logging.getLogger(__name__)

try:
    from absl import logging as absl_logging
    absl_logging.set_verbosity(absl_logging.ERROR)
except ImportError:
    pass

HDF5_CHUNK_ROWS = 100


# ===========================================================================
# 1.  RolloutWorker
# ===========================================================================
@ray.remote(num_cpus=1)
class RolloutWorker:
    """
    Collects trajectory fragments of length T steps from one MeltingPot
    scenario, controlling all focal agents with a shared policy.

    Parameters
    ----------
    worker_id     : int
    scenario_name : str
    cfg           : DictConfig
    """

    def __init__(self, worker_id: int, scenario_name: str, cfg: Any) -> None:
        self.worker_id     = worker_id
        self.scenario_name = scenario_name
        self.cfg           = cfg
        self.T             = cfg.rollout_len
        self.global_step   = 0

        max_steps = int(getattr(cfg, "max_steps", 1000))
        self.env = MeltingPotShimmy(scenario_name, seed=worker_id, max_steps=max_steps)
        self.num_focal = self.env.num_focal

        self.model = MoltenpotAgent(
            num_actions=cfg.model.num_actions,
            fc_units=cfg.model.fc_units,
            gru_hidden=cfg.model.gru_hidden,
            max_agents=getattr(cfg.model, "max_agents", 8),
        )
        self.model.eval()

        self._obs      = torch.from_numpy(self.env.reset()).float()
        self._h_state  = self.model.initial_hidden(batch_size=self.num_focal)
        _use_agent_id  = getattr(cfg.model, "use_agent_id", True)
        self._agent_ids = torch.arange(self.num_focal, dtype=torch.long) if _use_agent_id else None
        self._episode_id = 0
        self._ep_return  = np.zeros(self.num_focal, dtype=np.float32)
        self._ep_length  = 0

        logger.info(
            "RolloutWorker %d ready  scenario=%s  num_focal=%d",
            worker_id, scenario_name, self.num_focal,
        )

    def collect_fragment(self) -> Dict[str, Any]:
        """Roll-out T steps and return a fragment dict."""
        T = self.T
        A = self.num_focal
        H = self.cfg.model.gru_hidden

        obs_buf  = np.zeros((T, A, 3, 88, 88), dtype=np.float32)
        act_buf  = np.zeros((T, A),             dtype=np.int32)
        rew_buf  = np.zeros((T, A),             dtype=np.float32)
        done_buf = np.zeros((T,),               dtype=bool)
        logp_buf = np.zeros((T, A),             dtype=np.float32)
        val_buf  = np.zeros((T, A),             dtype=np.float32)
        h_buf    = np.zeros((T + 1, A, H),      dtype=np.float32)

        h = self._h_state.detach()
        h_buf[0] = h[0].numpy()
        step_offset = self.global_step

        ep_returns, ep_agent_returns, ep_lengths = [], [], []

        for t in range(T):
            actions, log_probs, values, h = self.model.act_batch(self._obs, h, agent_ids=self._agent_ids)

            obs_np, rewards, done, info = self.env.step(actions.tolist())

            obs_buf[t]   = self._obs.numpy()
            act_buf[t]   = actions
            rew_buf[t]   = rewards
            done_buf[t]  = done
            logp_buf[t]  = log_probs
            val_buf[t]   = values
            h_buf[t + 1] = h[0].detach().numpy()

            self.global_step   += 1
            self._ep_return    += rewards
            self._ep_length    += 1

            if done:
                ep_returns.append(float(self._ep_return.mean()))
                ep_agent_returns.append(self._ep_return.copy())  # (A,) per-agent
                ep_lengths.append(self._ep_length)
                self._episode_id += 1
                obs_np = self.env.reset()
                h = self.model.initial_hidden(batch_size=A)
                self._ep_return = np.zeros(A, dtype=np.float32)
                self._ep_length = 0

            self._obs = torch.from_numpy(obs_np).float()

        self._h_state = h

        return {
            "obs":              obs_buf,
            "actions":          act_buf,
            "rewards":          rew_buf,
            "dones":            done_buf,
            "log_probs":        logp_buf,
            "values":           val_buf,
            "h_states":         h_buf,
            "num_focal":        A,
            "ep_returns":       ep_returns,
            "ep_agent_returns": ep_agent_returns,
            "ep_lengths":       ep_lengths,
            "scenario_name":    self.scenario_name,
            "episode_id":       self._episode_id,
            "worker_id":        self.worker_id,
            "step_offset":      step_offset,
        }

    def set_weights(self, weights: dict) -> None:
        self.model.set_weights(weights)

    def record_rollout(self, video_path: str, fps: int = 10) -> None:
        """Execute one complete rollout and save it to a video file."""
        A = self.num_focal
        self.env.start_recording()
        obs_np = self.env.reset()
        h = self.model.initial_hidden(batch_size=A)
        agent_ids = torch.arange(A, dtype=torch.long)
        done = False
        total_reward = np.zeros(A, dtype=np.float32)
        steps = 0

        while not done and steps < 2000:
            obs_t = torch.from_numpy(obs_np).float()
            with torch.no_grad():
                actions, _, _, h = self.model.act_batch(obs_t, h, agent_ids=agent_ids)
            obs_np, rewards, done, _ = self.env.step(actions.tolist())
            total_reward += rewards
            steps += 1

        self.env.stop_recording(video_path, fps=fps)
        logger.info(
            "Recorded rollout: %s (steps=%d, mean_reward=%.1f)",
            video_path, steps, total_reward.mean(),
        )


# ===========================================================================
# 2.  PPOLearner
# ===========================================================================
@ray.remote(num_gpus=1)
class PPOLearner:
    """
    Central learner that consumes fragments and applies PPO updates.

    Flattens the agent dimension into the batch dimension so each
    agent's rollout is treated as an independent sample.
    """

    def __init__(self, fragment_queue, cfg: Any) -> None:
        from collections import defaultdict, deque
        self.queue       = fragment_queue
        self.cfg         = cfg
        self.num_workers = cfg.learner_batch or cfg.num_actors
        self.use_wandb   = cfg.wandb.enabled
        self.device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._reward_deque = deque(maxlen=100)
        self._length_deque = deque(maxlen=100)
        # keyed by (scenario_name, agent_idx) → rolling mean of per-agent returns
        self._agent_return_deques = defaultdict(lambda: deque(maxlen=100))
        self._use_agent_id = getattr(cfg.model, "use_agent_id", True)

        if self.use_wandb:
            import wandb
            from omegaconf import OmegaConf
            run = wandb.init(
                project=cfg.wandb.project,
                name=cfg.wandb.run_name,
                config=OmegaConf.to_container(cfg, resolve=True),
            )
            logger.info("W&B run: %s", run.get_url())

        self.model = MoltenpotAgent(
            num_actions=cfg.model.num_actions,
            fc_units=cfg.model.fc_units,
            gru_hidden=cfg.model.gru_hidden,
            max_agents=getattr(cfg.model, "max_agents", 8),
        ).to(self.device)
        self.model.train()
        self.optim = torch.optim.Adam(self.model.parameters(), lr=cfg.ppo.lr)
        self.update_count = 0

        logger.info("PPOLearner online  device=%s", self.device)

    def get_weights(self) -> dict:
        return self.model.get_weights()

    def log_video(self, video_path: str, label: str, global_step: int) -> None:
        """Upload a local video file to the active W&B run."""
        if not self.use_wandb:
            return
        import wandb
        # Use update_count as step to keep the axis consistent with metric logs.
        # global_step is attached as metadata via a separate scalar so it can be
        # cross-referenced in the W&B UI without breaking step monotonicity.
        wandb.log(
            {label: wandb.Video(video_path, fps=4, format="mp4"),
             f"{label}/env_step": global_step},
            step=self.update_count,
        )
        logger.info("W&B video logged: %s at update %d (env step %d)", label, self.update_count, global_step)

    def step(self, global_steps: int = 0) -> Dict[str, float]:
        """Drain num_workers fragments, run one PPO update, return metrics."""
        fragments: List[Dict] = []
        for _ in range(self.num_workers):
            frag = self.queue.get(block=True, timeout=120)
            fragments.append(frag)
            self._reward_deque.extend(frag.get("ep_returns", []))
            self._length_deque.extend(frag.get("ep_lengths", []))
            scenario = frag["scenario_name"]
            for ep_agent_ret in frag.get("ep_agent_returns", []):
                for i, ret in enumerate(ep_agent_ret):
                    self._agent_return_deques[(scenario, i)].append(float(ret))

        metrics = self._ppo_update(fragments)

        if self._reward_deque:
            metrics["episode_return_mean"] = float(np.mean(self._reward_deque))
            metrics["episode_return_max"]  = float(np.max(self._reward_deque))
            metrics["episode_return_min"]  = float(np.min(self._reward_deque))
            metrics["episode_length_mean"] = float(np.mean(self._length_deque))

        for (scenario, agent_idx), dq in self._agent_return_deques.items():
            if dq:
                metrics[f"{scenario}/agent_{agent_idx}/return_mean"] = float(np.mean(dq))

        metrics["env_steps"] = global_steps

        if self.use_wandb:
            import wandb
            wandb.log(metrics, step=self.update_count)

        return metrics

    def _ppo_update(self, fragments: List[Dict]) -> Dict[str, float]:
        """Flatten agent dim into batch, compute GAE, and run PPO epochs."""
        all_obs, all_actions, all_rewards = [], [], []
        all_dones, all_logprobs, all_values, all_h0, all_agent_ids = [], [], [], [], []

        for frag in fragments:
            A = frag["num_focal"]
            for a in range(A):
                all_obs.append(frag["obs"][:, a])
                all_actions.append(frag["actions"][:, a])
                all_rewards.append(frag["rewards"][:, a])
                all_dones.append(frag["dones"])
                all_logprobs.append(frag["log_probs"][:, a])
                all_values.append(frag["values"][:, a])
                all_h0.append(frag["h_states"][0, a])
                all_agent_ids.append(a)

        N = len(all_obs)
        obs      = np.stack(all_obs)
        actions  = np.stack(all_actions)
        rewards  = np.stack(all_rewards)
        dones    = np.stack(all_dones)
        log_probs_old = np.stack(all_logprobs)
        values_old    = np.stack(all_values)
        h_states = np.stack(all_h0)

        advantages_list, returns_list = [], []
        for i in range(N):
            adv, ret = self._compute_gae(rewards[i], values_old[i], dones[i])
            advantages_list.append(adv)
            returns_list.append(ret)

        advantages = np.stack(advantages_list)
        returns    = np.stack(returns_list)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs_t      = torch.from_numpy(obs).float().to(self.device)
        act_t      = torch.from_numpy(actions).long().to(self.device)
        adv_t      = torch.from_numpy(advantages).float().to(self.device)
        ret_t      = torch.from_numpy(returns).float().to(self.device)
        logp_old   = torch.from_numpy(log_probs_old).float().to(self.device)
        h_init     = torch.from_numpy(h_states).float().unsqueeze(0).to(self.device)
        agent_ids_t = torch.tensor(all_agent_ids, dtype=torch.long).to(self.device) if self._use_agent_id else None

        num_mb = self.cfg.ppo.num_minibatches
        mb_n   = max(1, N // num_mb)
        metrics: Dict[str, float] = {}

        for epoch in range(self.cfg.ppo.epochs):
            perm = torch.randperm(N)
            ep_loss, ep_pol, ep_val, ep_ent = [], [], [], []

            for start in range(0, N, mb_n):
                idx = perm[start:start + mb_n].tolist()

                logits, values, _ = self.model(obs_t[idx], h_init[:, idx, :],
                                               agent_ids=agent_ids_t[idx] if agent_ids_t is not None else None)
                dist     = Categorical(logits=logits)
                logp     = dist.log_prob(act_t[idx])
                entropy  = dist.entropy().mean()

                ratio       = torch.exp(logp - logp_old[idx])
                surr1       = ratio * adv_t[idx]
                surr2       = torch.clamp(ratio,
                                          1 - self.cfg.ppo.clip_eps,
                                          1 + self.cfg.ppo.clip_eps) * adv_t[idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss  = 0.5 * (values.squeeze(-1) - ret_t[idx]).pow(2).mean()
                loss        = (policy_loss
                               + self.cfg.ppo.value_coef * value_loss
                               - self.cfg.ppo.entropy_coef * entropy)

                self.optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.ppo.max_grad_norm)
                self.optim.step()

                ep_loss.append(float(loss.detach()))
                ep_pol.append(float(policy_loss.detach()))
                ep_val.append(float(value_loss.detach()))
                ep_ent.append(float(entropy.detach()))

            metrics = {
                "loss_total":   float(np.mean(ep_loss)),
                "loss_policy":  float(np.mean(ep_pol)),
                "loss_value":   float(np.mean(ep_val)),
                "entropy":      float(np.mean(ep_ent)),
                "epoch":        epoch,
                "num_rollouts": N,
            }

        del obs_t, act_t, adv_t, ret_t, logp_old, h_init

        self.update_count += 1
        metrics["update"] = self.update_count
        return metrics

    def _compute_gae(
        self,
        rewards:    np.ndarray,
        values:     np.ndarray,
        dones:      np.ndarray,
        next_value: float = 0.0,
    ):
        """Compute Generalised Advantage Estimation (GAE-λ)."""
        T          = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        last_gae   = 0.0

        for t in reversed(range(T)):
            next_val     = next_value if t == T - 1 else values[t + 1]
            next_nondone = 1.0 - float(dones[t])
            delta        = rewards[t] + self.cfg.ppo.gamma * next_val * next_nondone - values[t]
            last_gae     = delta + self.cfg.ppo.gamma * self.cfg.ppo.gae_lambda * next_nondone * last_gae
            advantages[t] = last_gae

        return advantages, advantages + values


# ===========================================================================
# 3.  HDF5Writer
# ===========================================================================
@ray.remote(num_cpus=1)
class HDF5Writer:
    """
    Asynchronous disk writer that streams trajectory fragments to HDF5.

    File layout
    -----------
    /<scenario_name>/
        ep_0/
            obs       (T, A, 3, 88, 88) uint8   LZF compressed
            actions   (T, A)             int32
            rewards   (T, A)             float32
            dones     (T,)               bool
            h_states  (T+1, A, H)        float32
        ep_1/ ...
    """

    def __init__(self, output_path: str) -> None:
        import h5py
        from pathlib import Path

        p = Path(output_path)
        if p.exists():
            stem, suffix = p.stem, p.suffix
            i = 1
            while True:
                candidate = p.with_name(f"{stem}_{i}{suffix}")
                if not candidate.exists():
                    p = candidate
                    break
                i += 1
            logger.warning(
                "HDF5Writer: '%s' already exists — writing to '%s' to avoid corruption.",
                output_path, p,
            )

        self.path = str(p)
        os.makedirs(p.parent, exist_ok=True)
        self._file       = h5py.File(self.path, "w")
        self._ep_ctrs: Dict[str, int] = {}
        self._frag_count = 0
        logger.info("HDF5Writer ready  path=%s", self.path)

    def write(self, fragment: Dict[str, Any]) -> None:
        """Persist one fragment to disk (obs stored as uint8 for space savings)."""
        scenario   = fragment["scenario_name"]
        episode_id = fragment["episode_id"]
        key        = f"{scenario}/ep_{episode_id}"
        grp        = self._file.require_group(key)

        obs_uint8 = (fragment["obs"] * 255).clip(0, 255).astype(np.uint8)

        def _ds(name: str, data: np.ndarray) -> None:
            if name in grp:
                ds = grp[name]
                old_len = ds.shape[0]
                new_len = old_len + data.shape[0]
                ds.resize(new_len, axis=0)
                ds[old_len:new_len] = data
            else:
                maxshape = (None,) + data.shape[1:]
                chunks   = (HDF5_CHUNK_ROWS,) + data.shape[1:]
                grp.create_dataset(name, data=data, maxshape=maxshape,
                                   chunks=chunks, compression="lzf")

        _ds("obs",      obs_uint8)
        _ds("actions",  fragment["actions"])
        _ds("rewards",  fragment["rewards"])
        _ds("dones",    fragment["dones"])
        _ds("h_states", fragment["h_states"])

        self._file.flush()
        self._frag_count += 1

        if self._frag_count % 50 == 0:
            logger.info("HDF5Writer: %d fragments written  file=%s",
                        self._frag_count, self.path)

    def close(self) -> None:
        self._file.flush()
        self._file.close()
        logger.info("HDF5Writer closed  total_fragments=%d  path=%s",
                    self._frag_count, self.path)


# ===========================================================================
# 4.  DataRecorder
# ===========================================================================
@ray.remote(num_cpus=1)
class DataRecorder:
    """
    Dedicated recording actor that builds the offline dataset.

    Completely decoupled from the PPO training workers — it never
    contributes fragments to the learner queue.  The driver polls it at
    uniform step intervals; each poll triggers one full episode recorded
    with the latest policy weights fetched directly from the learner.

    Parameters
    ----------
    scenario_name : str
        Scenario to record from.
    cfg : DictConfig
        Hydra configuration.
    learner : ray.actor.ActorHandle
        PPOLearner actor — weights are pulled on every ``record_episode`` call.
    writer : ray.actor.ActorHandle
        HDF5Writer actor — completed episodes are pushed here.
    """

    def __init__(
        self,
        scenario_name: str,
        cfg: Any,
        learner,
        writer,
    ) -> None:
        self.scenario_name  = scenario_name
        self.cfg            = cfg
        self.learner        = learner
        self.writer         = writer
        self._episode_id    = 0

        max_steps = int(getattr(cfg, "max_steps", 1000))
        self.env = MeltingPotShimmy(scenario_name, seed=999, max_steps=max_steps)
        self.num_focal = self.env.num_focal

        self.model = MoltenpotAgent(
            num_actions=cfg.model.num_actions,
            fc_units=cfg.model.fc_units,
            gru_hidden=cfg.model.gru_hidden,
            max_agents=getattr(cfg.model, "max_agents", 8),
        )
        self.model.eval()

        logger.info("DataRecorder ready  scenario=%s  num_focal=%d",
                    scenario_name, self.num_focal)

    def record_episode(self, global_step: int) -> None:
        """
        Fetch the latest weights from the learner, run one full episode,
        and stream it to the HDF5Writer.

        Called fire-and-forget by the driver — returns immediately once the
        write is dispatched.

        Parameters
        ----------
        global_step : int
            Current training step — stored in the episode for provenance.
        """
        # Pull latest policy weights from the learner
        weights = ray.get(self.learner.get_weights.remote())
        self.model.set_weights(weights)

        A = self.num_focal

        obs_list:  list = []
        act_list:  list = []
        rew_list:  list = []
        done_list: list = []
        h_list:    list = []

        obs_np    = self.env.reset()
        h         = self.model.initial_hidden(batch_size=A)
        agent_ids = torch.arange(A, dtype=torch.long) if getattr(self.cfg.model, "use_agent_id", True) else None
        h_list.append(h[0].numpy().copy())

        done = False
        while not done:
            obs_t = torch.from_numpy(obs_np).float()
            with torch.no_grad():
                actions, _, _, h = self.model.act_batch(obs_t, h, agent_ids=agent_ids)

            obs_list.append(obs_np.copy())
            act_list.append(actions.copy())
            h_list.append(h[0].detach().numpy().copy())

            obs_np, rewards, done, _ = self.env.step(actions.tolist())
            rew_list.append(rewards.copy())
            done_list.append(done)

        T = len(act_list)
        episode = {
            "obs":           np.stack(obs_list),                 # (T, A, 3, 88, 88)
            "actions":       np.stack(act_list),                 # (T, A)
            "rewards":       np.stack(rew_list),                 # (T, A)
            "dones":         np.array(done_list, dtype=bool),    # (T,)
            "h_states":      np.stack(h_list),                   # (T+1, A, H)
            "scenario_name": self.scenario_name,
            "episode_id":    self._episode_id,
            "global_step":   global_step,
        }

        self.writer.write.remote(episode)

        mean_return = np.stack(rew_list).sum(axis=0).mean()
        logger.info(
            "DataRecorder: ep %d at step %d  T=%d  return=%.1f",
            self._episode_id, global_step, T, mean_return,
        )
        self._episode_id += 1

    def record_video(self, video_path: str, global_step: int) -> str:
        """
        Record one full episode as a video using the current policy weights.

        Fetches the latest weights from the learner, runs a complete episode
        with video capture enabled, and saves the result to ``video_path``.

        Parameters
        ----------
        video_path : str
            Destination path for the .mp4 file.
        global_step : int
            Current training step — used for logging only.

        Returns
        -------
        str
            The resolved ``video_path`` (pass to ``PPOLearner.log_video``).
        """
        weights = ray.get(self.learner.get_weights.remote())
        self.model.set_weights(weights)

        A = self.num_focal
        self.env.start_recording()
        obs_np    = self.env.reset()
        h         = self.model.initial_hidden(batch_size=A)
        agent_ids = torch.arange(A, dtype=torch.long) if getattr(self.cfg.model, "use_agent_id", True) else None
        done      = False
        total_reward = np.zeros(A, dtype=np.float32)
        steps = 0

        while not done:
            obs_t = torch.from_numpy(obs_np).float()
            with torch.no_grad():
                actions, _, _, h = self.model.act_batch(obs_t, h, agent_ids=agent_ids)
            obs_np, rewards, done, _ = self.env.step(actions.tolist())
            total_reward += rewards
            steps += 1

        self.env.stop_recording(video_path, fps=int(getattr(self.cfg.recording, "fps", 3)))
        logger.info(
            "Video recorded: %s  steps=%d  mean_reward=%.2f  global_step=%d",
            video_path, steps, total_reward.mean(), global_step,
        )
        return video_path

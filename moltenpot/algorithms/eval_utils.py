"""
algorithms/eval_utils.py — Parallel multi-scenario evaluation via Ray
======================================================================
Provides ``evaluate_multi_scenario``, used by all offline RL algorithms to
evaluate a trained policy across in-distribution and out-of-distribution
scenarios in parallel across Ray CPU workers.

Each scenario is dispatched as an independent Ray remote task.  A sliding
window of ``num_eval_workers`` tasks runs concurrently, so no more than that
many envs are alive at once.

Metrics logged
--------------
eval/in_dist/<scenario>/return_mean      — per in-dist scenario
eval/in_dist/return_mean                 — aggregate mean across in-dist scenarios
eval/in_dist/return_std                  — std across scenarios (social robustness)
eval/in_dist/return_min                  — worst-case scenario return
eval/in_dist/return_max                  — best-case scenario return
eval/in_dist/return_iqr                  — IQR across scenarios
eval/in_dist/<scenario>/focal_gini       — Gini coefficient over per-agent mean returns
eval/in_dist/focal_gini                  — aggregate mean focal Gini
eval/in_dist/<scenario>/background_return_mean — mean per-capita background return
eval/in_dist/background_return_mean            — aggregate mean, background_return_{std,min,max,iqr} across scenarios
eval/in_dist/<scenario>/background_gini        — Gini coefficient over background agent mean returns
eval/in_dist/background_gini                   — aggregate mean background Gini
eval/out_dist/...                        — same keys for out-dist scenarios (if enabled)
eval/generalization_gap                  — in_dist/return_mean - out_dist/return_mean
eval/generalization_gap_norm             — gap normalised by |in_dist/return_mean|
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import ray

from moltenpot.model import MoltenpotAgent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gini(returns: np.ndarray) -> float:
    """Gini coefficient over a 1-D array of per-agent mean returns."""
    x = np.sort(returns.ravel().astype(np.float64))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * x).sum()) / (n * x.sum()) - (n + 1) / n)


# ---------------------------------------------------------------------------
# Ray remote worker
# ---------------------------------------------------------------------------

@ray.remote(num_cpus=1)
def _eval_scenario_worker(
    weights:       dict,
    model_cfg:     dict,   # {num_actions, fc_units, gru_hidden, max_agents}
    scenario:      str,
    act_type:      str,    # "standard" | "bcq" | "cql"
    bcq_tau:       float,
    seed:          int,
    num_episodes:  int,
    max_steps:     int,
    use_agent_id:  bool = True,
) -> dict:
    """
    Evaluate one scenario in an isolated Ray worker process.

    Returns a dict with keys: return_mean, focal_gini, and optionally background_return_mean, background_gini.
    """
    import torch
    import torch.nn.functional as F
    import numpy as np
    from moltenpot.model import MoltenpotAgent
    from moltenpot.wrappers import MeltingPotShimmy

    model = MoltenpotAgent(**model_cfg)
    model.set_weights(weights)
    model.eval()

    env = MeltingPotShimmy(scenario, seed=seed, max_steps=max_steps)
    A   = env.num_focal
    agent_ids = torch.arange(A, dtype=torch.long) if use_agent_id else None

    # The model may have more action heads than the substrate exposes (e.g. an
    # 8-head model evaluated on coins where the env only has 7 actions). Sampled
    # actions outside the valid range crash the env; clamp during eval.
    try:
        spec = env._env.action_spec()
        per_agent = spec[0] if hasattr(spec, "__iter__") else spec
        n_valid_actions = int(per_agent.num_values)
    except Exception:
        n_valid_actions = env.action_space_n

    if act_type == "bcq":
        def _act(obs_t: torch.Tensor, h_state: torch.Tensor) -> Tuple:
            q1, q2, _, logits, h_state = model.get_q_v(obs_t.unsqueeze(1), h_state, agent_ids=agent_ids)
            q_min    = torch.min(q1, q2).squeeze(1)
            probs    = F.softmax(logits.squeeze(1), dim=-1)
            max_prob = probs.max(dim=-1, keepdim=True)[0]
            mask     = (probs / max_prob) > bcq_tau
            q_min[~mask] = -1e10
            return q_min.argmax(dim=-1).cpu().numpy(), h_state
    elif act_type == "cql":
        # Discrete CQL: greedy w.r.t. min(Q1, Q2). The actor head is unused.
        def _act(obs_t: torch.Tensor, h_state: torch.Tensor) -> Tuple:
            q1, q2, _, _, h_state = model.get_q_v(obs_t.unsqueeze(1), h_state, agent_ids=agent_ids)
            q_min = torch.min(q1, q2).squeeze(1)
            return q_min.argmax(dim=-1).cpu().numpy(), h_state
    else:
        def _act(obs_t: torch.Tensor, h_state: torch.Tensor) -> Tuple:
            actions, _, _, h_state = model.act_batch(obs_t, h_state, agent_ids=agent_ids)
            return actions, h_state

    ep_returns    = []
    ep_bg_returns = []

    for _ in range(num_episodes):
        obs_np       = env.reset()
        done         = False
        h_state      = model.initial_hidden(batch_size=A)
        ep_return    = np.zeros(A, dtype=np.float32)
        ep_bg_return = None

        while not done:
            obs_t = torch.from_numpy(obs_np).float()
            with torch.no_grad():
                actions_np, h_state = _act(obs_t, h_state)
            actions_np = np.minimum(actions_np, n_valid_actions - 1)
            obs_np, rewards, done, info = env.step(actions_np.tolist())
            ep_return += rewards
            if "background_rewards" in info:
                if ep_bg_return is None:
                    ep_bg_return = np.zeros(len(info["background_rewards"]), dtype=np.float32)
                ep_bg_return += info["background_rewards"]

        ep_returns.append(ep_return.copy())
        if ep_bg_return is not None:
            ep_bg_returns.append(ep_bg_return.copy())

    env.close()
    # ep_returns_mat: (num_episodes, A)
    ep_returns_mat = np.array(ep_returns)
    result = {
        "return_mean": float(ep_returns_mat.mean()),
        "focal_gini":  float(_gini(ep_returns_mat.mean(axis=0))),
    }
    if ep_bg_returns:
        ep_bg_returns_mat = np.array(ep_bg_returns)   # (num_episodes, num_background)
        result["background_return_mean"] = float(ep_bg_returns_mat.mean())
        result["background_gini"]        = float(_gini(ep_bg_returns_mat.mean(axis=0)))
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_multi_scenario(
    model:               MoltenpotAgent,
    in_dist_scenarios:   List[str],
    out_dist_scenarios:  Optional[List[str]] = None,
    *,
    act_type:            str   = "standard",
    bcq_tau:             float = 0.3,
    num_eval_workers:    int   = 4,
    seed:                int   = 42,
    num_episodes:        int   = 2,
    max_steps:           int   = 1000,
    use_agent_id:        bool  = True,
) -> Dict[str, float]:
    """
    Evaluate a policy across multiple scenarios in parallel using Ray workers.

    Parameters
    ----------
    model : MoltenpotAgent
        The policy to evaluate.
    in_dist_scenarios : list of str
        Scenarios the model was trained on.
    out_dist_scenarios : list of str, optional
        Held-out scenarios for zero-shot generalisation evaluation.
    act_type : {"standard", "bcq", "cql"}
        Action selection strategy.  BCQ uses Q-masked greedy; all others
        use the standard stochastic actor.
    bcq_tau : float
        BCQ action-constraint threshold (only used when act_type="bcq").
    num_eval_workers : int
        Maximum number of Ray workers to run concurrently.  Each worker
        occupies 1 CPU.
    seed : int
        Environment seed.
    num_episodes : int
        Episodes to average per scenario.
    max_steps : int
        Maximum steps per episode.

    Returns
    -------
    dict
        Flat metrics dict suitable for ``wandb.log``.
    """
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    weights   = model.get_weights()
    model_cfg = {
        "num_actions": model.num_actions,
        "fc_units":    model.fc_units,
        "gru_hidden":  model.gru_hidden,
        "max_agents":  model.max_agents,
    }

    # Build the full list of (scenario, prefix) to evaluate
    all_tasks: List[Tuple[str, str]] = [
        (s, "eval/in_dist") for s in in_dist_scenarios
    ]
    if out_dist_scenarios:
        all_tasks += [(s, "eval/out_dist") for s in out_dist_scenarios]

    total = len(all_tasks)
    logger.info(
        "Parallel eval: %d scenarios (%d in-dist, %d out-dist), %d workers",
        total, len(in_dist_scenarios), len(out_dist_scenarios or []), num_eval_workers,
    )

    # Sliding-window dispatch: keep num_eval_workers tasks in flight at a time
    pending: Dict[ray.ObjectRef, Tuple[str, str]] = {}
    queue   = list(all_tasks)
    results: Dict[str, float] = {}

    def _submit(scenario: str, prefix: str) -> None:
        ref = _eval_scenario_worker.remote(
            weights, model_cfg, scenario, act_type, bcq_tau,
            seed, num_episodes, max_steps, use_agent_id,
        )
        pending[ref] = (scenario, prefix)

    # Fill the initial window
    for _ in range(min(num_eval_workers, len(queue))):
        _submit(*queue.pop(0))

    while pending:
        ready, _ = ray.wait(list(pending.keys()), num_returns=1)
        ref = ready[0]
        scenario, prefix = pending.pop(ref)
        worker_result = ray.get(ref)

        base_key = f"{prefix}/{scenario}"
        results[f"{base_key}/return_mean"] = worker_result["return_mean"]
        results[f"{base_key}/focal_gini"]  = worker_result["focal_gini"]
        for bg_metric in ("background_return_mean", "background_gini"):
            if bg_metric in worker_result:
                results[f"{base_key}/{bg_metric}"] = worker_result[bg_metric]
        logger.info(
            "  %-40s  return_mean=%.2f  focal_gini=%.3f",
            base_key, worker_result["return_mean"], worker_result["focal_gini"],
        )

        # Submit the next task to keep the window full
        if queue:
            _submit(*queue.pop(0))

    # Aggregate per split
    for split in ("eval/in_dist", "eval/out_dist"):
        for metric in ("return_mean", "focal_gini", "background_return_mean", "background_gini"):
            vals = [v for k, v in results.items()
                    if k.startswith(f"{split}/") and k.endswith(f"/{metric}")]
            if not vals:
                continue
            results[f"{split}/{metric}"] = float(np.mean(vals))
            if metric == "return_mean":
                results[f"{split}/return_std"] = float(np.std(vals))
                results[f"{split}/return_min"] = float(np.min(vals))
                results[f"{split}/return_max"] = float(np.max(vals))
                results[f"{split}/return_iqr"] = float(np.percentile(vals, 75) - np.percentile(vals, 25))
            elif metric == "background_return_mean":
                results[f"{split}/background_return_std"] = float(np.std(vals))
                results[f"{split}/background_return_min"] = float(np.min(vals))
                results[f"{split}/background_return_max"] = float(np.max(vals))
                results[f"{split}/background_return_iqr"] = float(np.percentile(vals, 75) - np.percentile(vals, 25))

    # Generalization gap (in-dist minus out-dist)
    in_mean  = results.get("eval/in_dist/return_mean")
    out_mean = results.get("eval/out_dist/return_mean")
    if in_mean is not None and out_mean is not None:
        results["eval/generalization_gap"]      = float(in_mean - out_mean)
        results["eval/generalization_gap_norm"] = float((in_mean - out_mean) / (abs(in_mean) + 1e-8))

    return results

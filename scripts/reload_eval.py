"""
scripts/reload_eval.py — Post-training reload-eval from W&B checkpoints.
=======================================================================
For each run in a W&B group, download its final model checkpoint, evaluate it
across every scenario of a substrate (N episodes per scenario), and write the
per-scenario **mean focal-agent episode return** to a JSON file per run.

This is the post-training eval the seq_len ablation runs were designed for: they
train with in-loop eval disabled (num_eval_workers=0), then get scored uniformly
here — 64 episodes on all clean_up scenarios — so different seq_len runs are
compared on identical footing.

Usage
-----
# seq_len ablation (seq_len=1 and =4 share this group)
python scripts/reload_eval.py \
    --group clean_up_ALL_seqlen_ablation \
    --num-episodes 64 \
    --out-dir results/seqlen_eval

# Later, the seq_len=128 baseline (same command, different group) for comparison:
python scripts/reload_eval.py \
    --group clean_up_ALL \
    --num-episodes 64 \
    --out-dir results/setting2_baseline_eval

Notes
-----
* Scenario set defaults to the union of in_dist + out_dist clean_up scenarios in
  configs/train_offline.yaml (all 18). Override with --scenarios for other subs.
* A fixed --eval-seed is used for every run so the env randomness is identical
  across runs (64 episodes then average out the rest).
* Runs trained with scenario-ID conditioning (model.use_scenario_id=true) are
  skipped — this uniform cross-scenario eval does not apply to them.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("reload_eval")

_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "configs", "train_offline.yaml",
)

# Action-selection strategy each algorithm uses at evaluation (see the algos'
# evaluate_multi_scenario calls). BC and IQL sample the actor (standard); BCQ
# uses Q-masked greedy with tau; CQL uses greedy min(Q1,Q2).
_ACT_TYPE = {"bc": "standard", "iql": "standard", "bcq": "bcq", "cql": "cql"}


def _algo_name(cfg: dict) -> str | None:
    """Algorithm name, robust to the wandb+Hydra sweep-param clash where the
    sweep's ``algorithm=bc`` overwrites the nested ``algorithm`` dict with a
    bare string."""
    a = cfg.get("algorithm")
    if isinstance(a, str):
        return a
    if isinstance(a, dict):
        return a.get("name")
    return None


def _algo_field(cfg: dict, key: str, default):
    """Read a nested algorithm hyper-parameter, or fall back to ``default`` when
    the nested dict was clobbered by the sweep param (string). The seq_len
    ablation did not sweep these, so the training default is the right value."""
    a = cfg.get("algorithm")
    if isinstance(a, dict):
        return a.get(key, default)
    return default


def _default_scenarios() -> tuple[list[str], list[str]]:
    """Return (in_dist, out_dist) clean_up scenarios from the base config."""
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(_CONFIG_PATH)
    in_dist = list(cfg.in_dist_scenarios)
    out_dist = list(cfg.out_dist_scenarios)
    return in_dist, out_dist


def _find_checkpoint(run, download_root: str) -> str | None:
    """Download the run's model artifact and return the local .pth path."""
    for art in run.logged_artifacts():
        if art.type != "model":
            continue
        d = art.download(root=os.path.join(download_root, art.name.replace(":", "_")))
        pths = glob.glob(os.path.join(d, "**", "*.pth"), recursive=True)
        if pths:
            return pths[0]
    return None


def _build_model(model_cfg: dict, num_scenarios: int = 0):
    """Instantiate MoltenpotAgent matching a run's model config. num_scenarios>0
    is required to load scenario-ID-conditioned checkpoints (their GRU input is
    fc_units + max_agents + num_scenarios)."""
    from moltenpot.model import MoltenpotAgent
    model = MoltenpotAgent(
        num_actions=int(model_cfg["num_actions"]),
        fc_units=int(model_cfg.get("fc_units", 256)),
        gru_hidden=int(model_cfg.get("gru_hidden", 256)),
        max_agents=int(model_cfg.get("max_agents", 8)),
        num_scenarios=int(num_scenarios),
    )
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", default=None, help="W&B group to evaluate")
    ap.add_argument("--sweep", default=None,
                    help="W&B sweep path (entity/project/sweep_id) to evaluate instead of a group")
    ap.add_argument("--algos", default=None,
                    help="Comma-separated algorithms to keep (e.g. 'iql'); default all")
    ap.add_argument("--in-dist-only", action="store_true",
                    help="Evaluate only the in-distribution scenarios (Setting-2 metric)")
    ap.add_argument("--entity", default=os.environ.get("WANDB_ENTITY", "moltenpot"))
    ap.add_argument("--project", default="moltenpot-offline-all")
    ap.add_argument("--num-episodes", type=int, default=64)
    ap.add_argument("--eval-seed", type=int, default=42,
                    help="Fixed env seed applied to every run (fair comparison)")
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--num-eval-workers", type=int, default=8,
                    help="Ray CPU workers (scenarios run in parallel)")
    ap.add_argument("--scenarios", nargs="+", default=None,
                    help="Explicit scenario list (default: clean_up in+out dist)")
    ap.add_argument("--out-dir", default="results/reload_eval")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-evaluate runs whose JSON already exists")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only evaluate the first N eligible runs (smoke test)")
    args = ap.parse_args()

    import wandb
    from moltenpot.algorithms.eval_utils import evaluate_multi_scenario, shutdown_ray

    if args.scenarios:
        scenarios = list(args.scenarios)
        in_dist_set, out_dist_set = set(scenarios), set()
    else:
        in_dist, out_dist = _default_scenarios()
        scenarios = in_dist + out_dist
        in_dist_set, out_dist_set = set(in_dist), set(out_dist)

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_root = os.path.join(args.out_dir, "_checkpoints")

    algos_filter = set(a.strip() for a in args.algos.split(",")) if args.algos else None

    api = wandb.Api()
    if args.sweep:
        sw = api.sweep(args.sweep)
        runs = list(sw.runs)
        run_path = args.sweep.rsplit("/", 1)[0]  # entity/project
        src_desc = f"sweep {args.sweep}"
    else:
        if not args.group:
            ap.error("provide either --group or --sweep")
        run_path = f"{args.entity}/{args.project}"
        runs = list(api.runs(run_path, filters={"group": args.group}))
        src_desc = f"group '{args.group}' ({run_path})"
    logger.info("Found %d run(s) in %s. Episodes/scenario=%d.%s",
                len(runs), src_desc, args.num_episodes,
                f" Filtering algos={sorted(algos_filter)}." if algos_filter else "")

    summary: dict = {}
    done, skipped, failed = 0, 0, 0

    for run in runs:
        if args.limit is not None and done >= args.limit:
            logger.info("Reached --limit=%d evaluated runs; stopping.", args.limit)
            break
        cfg = run.config
        algo = _algo_name(cfg)
        model_cfg = cfg.get("model") or {}
        seq_len = cfg.get("seq_len")
        seed = cfg.get("seed")
        use_sid = bool(model_cfg.get("use_scenario_id", False))
        run_in_dist = list(cfg.get("in_dist_scenarios") or in_dist)

        # Per-run eval scope + conditioning:
        #  - scenario-ID runs: eval the run's in-dist scenarios in TRAINING order
        #    (so the fed scenario one-hot index matches training), num_scenarios
        #    from the config, use_scenario_id=True.
        #  - otherwise: unconditioned eval on the requested scenario set.
        if use_sid:
            eval_scenarios, num_scen = run_in_dist, len(run_in_dist)
            e_in, e_out = set(run_in_dist), set()
        elif args.in_dist_only:
            eval_scenarios, num_scen = run_in_dist, 0
            e_in, e_out = set(run_in_dist), set()
        else:
            eval_scenarios, num_scen = scenarios, 0
            e_in, e_out = in_dist_set, out_dist_set

        cond = "_scenarioID" if use_sid else ""
        tag = f"{algo}_seqlen{seq_len}{cond}_seed{seed}_{run.id}"
        out_path = os.path.join(args.out_dir, f"{tag}.json")

        if algos_filter and algo not in algos_filter:
            skipped += 1
            continue
        if algo not in _ACT_TYPE:
            logger.warning("[skip] %s: unknown algorithm '%s'", run.id, algo)
            skipped += 1
            continue
        if run.state not in ("finished",):
            logger.warning("[skip] %s: run state=%s (not finished).", run.id, run.state)
            skipped += 1
            continue
        if os.path.exists(out_path) and not args.overwrite:
            logger.info("[have] %s already evaluated -> %s", run.id, out_path)
            with open(out_path) as f:
                summary[tag] = json.load(f).get("scenario_return_mean", {})
            done += 1
            continue

        try:
            ckpt = _find_checkpoint(run, ckpt_root)
            if ckpt is None:
                logger.warning("[skip] %s: no model artifact found.", run.id)
                skipped += 1
                continue

            import torch
            model = _build_model(model_cfg, num_scenarios=num_scen)
            state = torch.load(ckpt, map_location="cpu")
            model.load_state_dict(state)
            model.eval()

            act_type = _ACT_TYPE[algo]
            bcq_tau = float(_algo_field(cfg, "tau", 0.3))

            logger.info("[eval] %s | algo=%s seq_len=%s seed=%s | act_type=%s | "
                        "scenario_id=%s (%d scenarios)", run.id, algo, seq_len, seed,
                        act_type, use_sid, len(eval_scenarios))
            metrics = evaluate_multi_scenario(
                model, eval_scenarios, [],
                act_type=act_type,
                bcq_tau=bcq_tau,
                num_eval_workers=args.num_eval_workers,
                seed=args.eval_seed,
                num_episodes=args.num_episodes,
                max_steps=args.max_steps,
                use_agent_id=bool(model_cfg.get("use_agent_id", True)),
                use_scenario_id=use_sid,
            )

            per_scenario = {
                s: metrics.get(f"eval/in_dist/{s}/return_mean") for s in eval_scenarios
            }
            def _avg(names):
                vals = [per_scenario[s] for s in names if per_scenario.get(s) is not None]
                return (sum(vals) / len(vals)) if vals else None

            record = {
                "run_id": run.id,
                "run_name": run.name,
                "wandb_path": f"{run_path}/{run.id}",
                "group": run.group,
                "algorithm": algo,
                "substrate": cfg.get("substrate", "clean_up"),
                "train_mode": cfg.get("train_mode"),
                "seq_len": seq_len,
                "batch_size": cfg.get("batch_size"),
                "seed": seed,
                "use_scenario_id": use_sid,
                "num_scenarios": num_scen,
                "eval": {
                    "num_episodes": args.num_episodes,
                    "eval_seed": args.eval_seed,
                    "max_steps": args.max_steps,
                    "act_type": act_type,
                    "bcq_tau": bcq_tau if algo == "bcq" else None,
                    "use_scenario_id": use_sid,
                },
                "scenario_return_mean": per_scenario,
                "aggregates": {
                    "in_dist_mean": _avg(sorted(e_in)),
                    "out_dist_mean": _avg(sorted(e_out)) if e_out else None,
                    "all_mean": _avg(eval_scenarios) if e_out else None,
                },
            }
            with open(out_path, "w") as f:
                json.dump(record, f, indent=2)
            summary[tag] = per_scenario
            logger.info("[done] %s -> %s | in_dist_mean=%.3f", run.id, out_path,
                        record["aggregates"]["in_dist_mean"] or float("nan"))
            done += 1
        except Exception as exc:  # keep going on a single bad run
            logger.exception("[fail] %s: %s", run.id, exc)
            failed += 1

    # Combined index for the later comparison script.
    with open(os.path.join(args.out_dir, "all_runs.json"), "w") as f:
        json.dump(summary, f, indent=2)

    shutdown_ray()
    logger.info("Reload-eval complete: %d evaluated, %d skipped, %d failed. "
                "Per-run JSON + all_runs.json in %s", done, skipped, failed, args.out_dir)


if __name__ == "__main__":
    main()

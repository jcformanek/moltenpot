"""
scripts/export_tier2_final_eval.py — Export the tier2 (Setting-2) final-eval runs
to the same per-run JSON schema as scripts/reload_eval.py.

The seq_len=128 baseline was final-evaluated separately and stored in the W&B
project ``moltenpot-tier2-final-eval``. Each run's *summary* holds
``final_eval/<scenario>/return_mean`` (+ per_episode_returns, return_sem) over
the substrate's in-distribution scenarios, plus ``final_eval/substrate_mean_return``.

This pulls those into our JSON so seq_len=128 slots into the same results dir and
comparison table as the seq_len={1,4} reload-eval. NOTE: the baseline evaluated
in-dist scenarios only, so out_dist_mean / all_mean are null for these runs — the
apples-to-apples comparison is the in-dist mean.

Usage
-----
python scripts/export_tier2_final_eval.py \
    --group clean_up_tier2_final_eval --seq-len 128 \
    --out-dir results/seqlen_eval
"""

from __future__ import annotations

import argparse
import json
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("export_tier2")


def _algo_name(cfg: dict):
    a = cfg.get("algorithm")
    if isinstance(a, str):
        return a
    if isinstance(a, dict):
        return a.get("name")
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", default=os.environ.get("WANDB_ENTITY", "moltenpot"))
    ap.add_argument("--project", default="moltenpot-tier2-final-eval")
    ap.add_argument("--group", required=True)
    ap.add_argument("--seq-len", type=int, default=128,
                    help="seq_len to record (config doesn't log it for these runs)")
    ap.add_argument("--out-dir", default="results/seqlen_eval")
    args = ap.parse_args()

    import wandb
    api = wandb.Api(timeout=45)
    run_path = f"{args.entity}/{args.project}"
    runs = list(api.runs(run_path, filters={"group": args.group}))
    os.makedirs(args.out_dir, exist_ok=True)
    logger.info("Found %d run(s) in %s / group '%s'.", len(runs), run_path, args.group)

    done = 0
    for run in runs:
        cfg = run.config
        algo = _algo_name(cfg)
        seed = cfg.get("seed")
        skeys = list(run.summary.keys())

        # per-scenario mean focal return (exclude the substrate_mean_return aggregate,
        # which ends in 'mean_return', not '/return_mean')
        scen_means = {}
        n_eps = None
        for k in skeys:
            if k.startswith("final_eval/") and k.endswith("/return_mean"):
                scen = k[len("final_eval/"):-len("/return_mean")]
                scen_means[scen] = run.summary.get(k)
            if n_eps is None and k.endswith("/per_episode_returns"):
                v = run.summary.get(k)
                if isinstance(v, (list, tuple)):
                    n_eps = len(v)

        sub_mean = run.summary.get("final_eval/substrate_mean_return")
        if sub_mean is None and scen_means:
            vals = [v for v in scen_means.values() if v is not None]
            sub_mean = sum(vals) / len(vals) if vals else None

        record = {
            "run_id": run.id,
            "run_name": run.name,
            "wandb_path": f"{run_path}/{run.id}",
            "group": args.group,
            "algorithm": algo,
            "substrate": cfg.get("substrate"),
            "train_mode": cfg.get("train_mode") or "ALL",
            "seq_len": args.seq_len,
            "batch_size": cfg.get("batch_size"),
            "seed": seed,
            "eval": {
                "num_episodes": n_eps,
                "source": f"wandb:{args.project}",
                "note": "in-dist scenarios only (baseline final-eval)",
            },
            "scenario_return_mean": scen_means,
            "aggregates": {
                "in_dist_mean": sub_mean,
                "out_dist_mean": None,   # baseline did not evaluate OOD scenarios
                "all_mean": None,        # ... so no all-18 aggregate exists
            },
        }
        tag = f"{algo}_seqlen{args.seq_len}_seed{seed}_{run.id}"
        out_path = os.path.join(args.out_dir, f"{tag}.json")
        with open(out_path, "w") as f:
            json.dump(record, f, indent=2)
        logger.info("[done] %s -> %s | scenarios=%d eps=%s in_dist_mean=%.3f",
                    run.id, out_path, len(scen_means), n_eps,
                    sub_mean if sub_mean is not None else float("nan"))
        done += 1

    logger.info("Exported %d baseline run(s) to %s", done, args.out_dir)


if __name__ == "__main__":
    main()

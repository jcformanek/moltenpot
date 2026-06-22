"""
scripts/train_offline.py — Offline RL Training Entry Point
===========================================================
Unified launcher for all offline RL algorithms: BC, BCQ, IQL, CQL.
The algorithm is selected via the Hydra config group ``algorithm``.

Usage
-----
# IQL across all Clean Up in-dist scenarios
python scripts/train_offline.py \\
    algorithm=iql \\
    data_root=data/clean_up \\
    substrate=clean_up

# Hyperparameter sweep (Hydra multirun)
python scripts/train_offline.py --multirun \\
    algorithm=bc,iql,cql \\
    data_root=data/clean_up \\
    substrate=clean_up

# Multi-seed sweep
python scripts/train_offline.py --multirun \\
    seed=1,2,3,4,5 \\
    algorithm=iql \\
    data_root=data/clean_up \\
    substrate=clean_up

# Load a predefined experiment config (one per benchmark split)
python scripts/train_offline.py +experiment=benchmark_coins_C1 \\
    algorithm=iql

# Reproduce a whole evaluation setting via a W&B sweep
wandb sweep sweeps/coins/setting1.yaml
"""

from __future__ import annotations

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
from omegaconf import DictConfig, OmegaConf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_offline")


@hydra.main(version_base=None, config_path="../configs", config_name="train_offline")
def main(cfg: DictConfig) -> None:
    from moltenpot.algorithms import ALGORITHMS

    algorithm_name = cfg.algorithm.name
    if algorithm_name not in ALGORITHMS:
        raise ValueError(
            f"Unknown algorithm '{algorithm_name}'. "
            f"Available: {list(ALGORITHMS.keys())}"
        )

    from moltenpot import hub
    hub.configure(repo_id=cfg.hub.repo_id, auto_download=cfg.hub.auto_download)

    logger.info("=" * 60)
    logger.info("  Moltenpot Offline RL Training")
    logger.info("  algorithm  : %s", algorithm_name)
    logger.info("  substrate  : %s", cfg.substrate)
    logger.info("  data_root  : %s", cfg.data_root)
    logger.info("  in_dist    : %s", list(cfg.in_dist_scenarios))
    logger.info("  out_dist   : %s", list(cfg.out_dist_scenarios) if cfg.out_dist_eval else "disabled")
    logger.info("  seed       : %d", cfg.seed)
    logger.info("  num_updates: %d", cfg.num_updates)
    logger.info("=" * 60)
    logger.info("Full config:\n%s", OmegaConf.to_yaml(cfg))

    train_fn = ALGORITHMS[algorithm_name]
    train_fn(cfg)


if __name__ == "__main__":
    main()

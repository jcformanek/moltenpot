"""
scripts/train_ppo.py — PPO Online Training + Data Collection
=============================================================
Orchestrates the distributed Actor-Learner-Writer pipeline.

Usage
-----
# Smoke test (no GPU, no data writing)
python scripts/train_ppo.py num_actors=2 total_steps=20000 no_gpu=true nowrite=true

# Full data collection run
python scripts/train_ppo.py \\
    scenarios=clean_up_0,clean_up_1 \\
    num_actors=8 \\
    total_steps=10000000 \\
    output=data/clean_up_mixed.hdf5 \\
    wandb.run_name=ppo_clean_up_mixed

# Hyperparameter sweep (Hydra multirun)
python scripts/train_ppo.py --multirun \\
    ppo.lr=1e-4,3e-4,1e-3 \\
    ppo.entropy_coef=0.005,0.01,0.05 \\
    scenarios=clean_up_0 \\
    total_steps=2000000 \\
    nowrite=true

Architecture
------------
    RolloutWorker × N  ──fragment──▶  ray.Queue
    PPOLearner         ◀── queue (blocking pop, batch update)
         └──weights──▶ broadcast to workers
    HDF5Writer         ◀── fire-and-forget fragment writes
"""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import List

# Make moltenpot importable when running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
import ray
import torch
from omegaconf import DictConfig
from ray.util.queue import Queue

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_ppo")

try:
    from absl import logging as absl_logging
    absl_logging.set_verbosity(absl_logging.ERROR)
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Ray initialisation
# ---------------------------------------------------------------------------
def _init_ray(cfg: DictConfig) -> None:
    ray_kwargs = {}
    if cfg.ray.address:
        ray_kwargs["address"] = cfg.ray.address
    else:
        if cfg.ray.num_cpus is not None:
            ray_kwargs["num_cpus"] = cfg.ray.num_cpus
        if cfg.ray.num_gpus is not None:
            ray_kwargs["num_gpus"] = cfg.ray.num_gpus

    if not ray.is_initialized():
        ray.init(**ray_kwargs, ignore_reinit_error=True)

    res = ray.available_resources()
    logger.info("Ray ready — CPUs=%.0f  GPUs=%.0f", res.get("CPU", 0), res.get("GPU", 0))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _scenario_for(worker_idx: int, scenarios: List[str]) -> str:
    return scenarios[worker_idx % len(scenarios)]


def _spawn_workers(num_actors: int, scenarios: List[str], cfg: DictConfig):
    from moltenpot.workers import RolloutWorker
    actors = []
    for i in range(num_actors):
        scenario = _scenario_for(i, scenarios)
        actor = RolloutWorker.remote(worker_id=i, scenario_name=scenario, cfg=cfg)
        actors.append(actor)
        logger.info("  Worker %2d → %s", i, scenario)
    return actors


def _spawn_learner(fragment_queue, cfg: DictConfig):
    from moltenpot.workers import PPOLearner
    use_gpu = not cfg.no_gpu and torch.cuda.is_available()
    opts = {"num_gpus": 1 if use_gpu else 0}
    return PPOLearner.options(**opts).remote(fragment_queue=fragment_queue, cfg=cfg)


def _spawn_writer(output_path: str):
    from moltenpot.workers import HDF5Writer
    return HDF5Writer.remote(output_path=output_path)


def _spawn_recorder(scenario: str, cfg: DictConfig, learner, writer):
    from moltenpot.workers import DataRecorder
    return DataRecorder.remote(
        scenario_name=scenario,
        cfg=cfg,
        learner=learner,
        writer=writer,
    )


def _broadcast_weights(learner, actors) -> None:
    weights = ray.get(learner.get_weights.remote())
    ray.get([a.set_weights.remote(weights) for a in actors])


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
@hydra.main(version_base=None, config_path="../configs", config_name="train_ppo")
def main(cfg: DictConfig) -> None:
    scenarios     = [s.strip() for s in cfg.scenarios.split(",")]
    learner_batch = cfg.learner_batch or cfg.num_actors
    use_gpu       = not cfg.no_gpu and torch.cuda.is_available()

    logger.info("=" * 60)
    logger.info("  Moltenpot PPO Training")
    logger.info("  scenarios  : %s", scenarios)
    logger.info("  num_actors : %d", cfg.num_actors)
    logger.info("  rollout_len: %d", cfg.rollout_len)
    logger.info("  total_steps: %d", cfg.total_steps)
    logger.info("  use_gpu    : %s", use_gpu)
    logger.info("  output     : %s", cfg.output)
    logger.info("=" * 60)

    _init_ray(cfg)

    fragment_queue = Queue(maxsize=learner_batch * 4)
    actors  = _spawn_workers(cfg.num_actors, scenarios, cfg)
    learner = _spawn_learner(fragment_queue, cfg)

    writer   = None
    recorder = None
    if not cfg.nowrite:
        logger.info("Spawning HDF5Writer → %s", cfg.output)
        writer = _spawn_writer(cfg.output)
        logger.info("Spawning DataRecorder → %s", scenarios[0])
        recorder = _spawn_recorder(scenarios[0], cfg, learner, writer)

    time.sleep(2.0)
    _broadcast_weights(learner, actors)

    global_steps       = 0
    update_count       = 0
    start_wall         = time.time()
    rounds_per_update  = max(1, learner_batch // cfg.num_actors)

    # Uniform recording schedule: poll the DataRecorder once every
    # (total_steps / target_episodes) training steps.
    rec_episode_count = 0
    if recorder is not None:
        record_interval  = max(1, cfg.total_steps // cfg.target_episodes)
        next_record_step = record_interval
        logger.info(
            "Dataset recording: %d episodes, one every %d steps → ~%d timesteps",
            cfg.target_episodes, record_interval,
            cfg.target_episodes * cfg.max_steps,
        )
    else:
        record_interval  = None
        next_record_step = float("inf")

    # W&B video checkpoints: record one episode at 5%, 50%, and end of training.
    _video_checkpoints: dict = {}
    if recorder is not None:
        _video_checkpoints = {
            int(cfg.total_steps * 0.05): "video/5pct",
            int(cfg.total_steps * 0.50): "video/50pct",
        }
    _logged_videos: set = set()

    logger.info(
        "PPO: %d round(s) × %d actors = %d fragments per update.",
        rounds_per_update, cfg.num_actors, rounds_per_update * cfg.num_actors,
    )

    while global_steps < cfg.total_steps:
        for _ in range(rounds_per_update):
            frag_futures = [a.collect_fragment.remote() for a in actors]
            pending = list(frag_futures)
            while pending:
                ready, pending = ray.wait(pending, num_returns=1, timeout=120)
                if not ready:
                    logger.warning("Actor timed out waiting for fragment!")
                    continue
                frag = ray.get(ready[0])
                fragment_queue.put(frag)
                global_steps += cfg.rollout_len

                # Poll the DataRecorder if we've crossed the next checkpoint.
                # Uses while to handle cases where rollout_len > record_interval.
                while global_steps >= next_record_step and recorder is not None:
                    recorder.record_episode.remote(global_step=global_steps)
                    rec_episode_count += 1
                    next_record_step  += record_interval
                    logger.info(
                        "DataRecorder polled: ep %d/%d at step %d",
                        rec_episode_count, cfg.target_episodes, global_steps,
                    )

                # W&B video checkpoints (5% and 50% of training).
                for threshold, label in _video_checkpoints.items():
                    if global_steps >= threshold and threshold not in _logged_videos:
                        _logged_videos.add(threshold)
                        vpath = os.path.join(
                            os.path.abspath(cfg.recording.directory),
                            f"wandb_{label.replace('/', '_')}_{global_steps:08d}.mp4",
                        )
                        logger.info("Recording W&B video checkpoint: %s at step %d", label, global_steps)
                        video_path = ray.get(recorder.record_video.remote(vpath, global_steps))
                        learner.log_video.remote(video_path, label, global_steps)

        metrics = ray.get(learner.step.remote(global_steps))
        update_count += 1
        _broadcast_weights(learner, actors)

        if update_count % cfg.log_interval == 0:
            elapsed = time.time() - start_wall
            fps = global_steps / max(elapsed, 1)
            logger.info(
                "Update %5d | Steps %8d / %d | FPS %6.0f | "
                "loss=%.4f  policy=%.4f  value=%.4f  entropy=%.4f | "
                "RetMean: %.1f",
                update_count, global_steps, cfg.total_steps, fps,
                metrics.get("loss_total",  float("nan")),
                metrics.get("loss_policy", float("nan")),
                metrics.get("loss_value",  float("nan")),
                metrics.get("entropy",     float("nan")),
                metrics.get("episode_return_mean", float("nan")),
            )

        if cfg.recording.enabled and update_count % cfg.recording.interval == 0:
            rec_dir    = os.path.abspath(cfg.recording.directory)
            os.makedirs(rec_dir, exist_ok=True)
            video_path = os.path.join(rec_dir, f"rollout_{update_count:05d}.mp4")
            actors[0].record_rollout.remote(video_path, fps=cfg.recording.fps)

    # Flush and close HDF5 writer before doing anything else post-training.
    if writer is not None:
        ray.get(writer.close.remote())

    # Record end-of-training video and upload to W&B.
    if recorder is not None:
        vpath = os.path.join(
            os.path.abspath(cfg.recording.directory),
            f"wandb_video_end_{global_steps:08d}.mp4",
        )
        logger.info("Recording end-of-training W&B video at step %d", global_steps)
        video_path = ray.get(recorder.record_video.remote(vpath, global_steps))
        ray.get(learner.log_video.remote(video_path, "video/end", global_steps))

    elapsed_total = time.time() - start_wall
    logger.info(
        "Done in %.1f s — %d updates — %d total steps",
        elapsed_total, update_count, global_steps,
    )
    ray.shutdown()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted. Shutting down Ray …")
        if ray.is_initialized():
            ray.shutdown()
        sys.exit(0)

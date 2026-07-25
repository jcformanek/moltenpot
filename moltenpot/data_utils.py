"""
data_utils.py — Dataset loading utilities for offline RL training.
==================================================================

Provides PyTorch Dataset classes that stream trajectory sequences directly
from the HDF5 files produced by the data collection pipeline, without
loading the full dataset into memory.

Dataset classes
---------------
MultiScenarioDataset            — (obs, actions) for BC, across N scenario files
MultiScenarioTransitionDataset  — (obs, actions, rewards, dones) for BCQ/IQL/CQL

Both sample uniformly across scenarios in expectation, regardless of the
number of episodes or focal agents per scenario.  A single scenario is just
the one-element case.  Pass ``dataset.make_sampler()`` as the ``sampler``
argument to DataLoader.

HDF5 schema expected
--------------------
/<scenario_name>/
    ep_0/
        obs       (T, A, 3, 88, 88)  uint8   LZF-compressed
        actions   (T, A)             int32
        rewards   (T, A)             float32
        dones     (T,)               bool
        h_states  (T+1, A, H)        float32
    ep_1/
        ...
"""

from __future__ import annotations

import logging
import os
import random
from typing import List, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset, Sampler, get_worker_info

from moltenpot.hub import ensure_datasets

logger = logging.getLogger(__name__)


class UniformPerScenarioSampler(Sampler):
    """Sample indices *with replacement* by picking a scenario uniformly, then a
    sample uniformly within that scenario.

    This reproduces the exact distribution of
    ``WeightedRandomSampler(weights=1/n_scenario)`` — every scenario equally
    likely, uniform within — i.e. ``P(i) = 1 / (num_scenarios * n_scenario)`` —
    but WITHOUT building a ``torch.multinomial`` over the whole index.
    ``torch.multinomial`` caps the number of categories at ``2**24`` and fails
    for small ``seq_len`` (where the index has tens of millions of windows).
    Draws happen in small vectorised chunks, so memory stays flat regardless of
    index size.
    """

    def __init__(self, scenario_ranges, num_samples, generator=None):
        # scenario_ranges: list of (start, end) half-open spans into the flat index
        self._los  = torch.tensor([lo for lo, _ in scenario_ranges], dtype=torch.long)
        self._his  = torch.tensor([hi for _, hi in scenario_ranges], dtype=torch.long)
        self._num_samples = int(num_samples)
        self._generator = generator

    def __len__(self) -> int:
        return self._num_samples

    def __iter__(self):
        S = self._los.numel()
        los, his = self._los, self._his
        spans = his - los
        remaining = self._num_samples
        CHUNK = 65536
        while remaining > 0:
            n = min(CHUNK, remaining)
            scen = torch.randint(0, S, (n,), generator=self._generator)
            span = spans[scen]
            # floor(uniform[0,1) * span); clamp guards the rare rand→1.0 rounding.
            off = (torch.rand(n, generator=self._generator) * span.to(torch.float64)).to(torch.long)
            off = torch.minimum(off, span - 1)
            yield from (los[scen] + off).tolist()
            remaining -= n


def _build_index(
    data_root: str,
    scenario_names: List[str],
    seq_len: int,
    need_next_obs: bool,
) -> Tuple[list, list]:
    """
    Build a flat sample index across all scenarios.

    Each entry is ``(hdf5_path, scenario_key, ep_name, agent_idx, start_t)``.
    Returns ``(samples, scenario_ranges)`` where ``scenario_ranges`` gives the
    ``(start, end)`` half-open span of each scenario in the flat index; the
    sampler uses these to draw every scenario with equal probability regardless
    of episode/agent count (see UniformPerScenarioSampler).

    Parameters
    ----------
    need_next_obs : bool
        If True, sequences require T+1 observations (transition datasets).
        The effective episode window is shorter by one step.
    """
    # Batch-check all scenarios and offer to download any that are missing.
    ensure_datasets(data_root, scenario_names)

    samples: list = []
    scenario_ranges: list = []   # (start, end) half-open span per non-empty scenario

    for scenario in scenario_names:
        hdf5_path = os.path.join(data_root, scenario, "dataset.hdf5")

        scenario_samples: list = []
        step = seq_len
        min_len = seq_len + (1 if need_next_obs else 0)

        with h5py.File(hdf5_path, "r") as f:
            if scenario not in f:
                raise ValueError(
                    f"Scenario key '{scenario}' not found inside {hdf5_path}"
                )
            grp = f[scenario]
            for ep_name in grp.keys():
                ep = grp[ep_name]
                T = ep["obs"].shape[0]
                A = ep["obs"].shape[1]
                for start_t in range(0, T - min_len + 1, step):
                    for a in range(A):
                        scenario_samples.append(
                            (hdf5_path, scenario, ep_name, a, start_t)
                        )

        n = len(scenario_samples)
        if n == 0:
            logger.warning("No valid sequences for scenario '%s', skipping.", scenario)
            continue

        # Record this scenario's contiguous span in the flat index. Uniform
        # sampling of a scenario + uniform within ⟺ equal probability per
        # scenario (see UniformPerScenarioSampler). No per-sample weights list
        # is built — that would be tens of millions of entries at small seq_len.
        start = len(samples)
        samples.extend(scenario_samples)
        scenario_ranges.append((start, len(samples)))

        logger.info(
            "MultiScenarioDataset: %4d sequences (seq_len=%d) from %s",
            n, seq_len, scenario,
        )

    if not samples:
        raise RuntimeError("No samples found across all provided scenarios.")

    return samples, scenario_ranges


class MultiScenarioDataset(Dataset):
    """
    Streams ``(obs, actions)`` sequences from multiple HDF5 scenario files.

    Scenarios with more focal agents or more episodes are *not* over-represented.
    Use ``dataset.make_sampler()`` as the ``sampler`` argument to DataLoader
    to enforce uniform scenario sampling.

    Parameters
    ----------
    data_root : str
        Root directory containing one sub-directory per scenario, each
        holding a ``dataset.hdf5`` file.
    scenario_names : list of str
        Scenario keys to load (e.g. ``["clean_up_0", "clean_up_2"]``).
    seq_len : int
        Length of each sampled sequence.
    """

    def __init__(
        self,
        data_root: str,
        scenario_names: List[str],
        seq_len: int = 128,
    ) -> None:
        self.seq_len = seq_len
        # Canonical scenario index = position in the provided scenario_names list
        # (i.e. cfg.in_dist_scenarios order). Emitted per sample so the model can
        # append a one-hot scenario ID; kept stable even if a scenario is skipped.
        self._scenario_to_idx = {name: i for i, name in enumerate(scenario_names)}
        self._samples, self._scenario_ranges = _build_index(
            data_root, scenario_names, seq_len, need_next_obs=False
        )
        logger.info(
            "MultiScenarioDataset ready: %d total sequences across %d scenarios",
            len(self._samples), len(scenario_names),
        )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hdf5_path, scenario, ep_name, agent_idx, start_t = self._samples[idx]
        end_t = start_t + self.seq_len

        with h5py.File(hdf5_path, "r") as f:
            grp = f[scenario][ep_name]
            obs_uint8 = grp["obs"][start_t:end_t, agent_idx]       # (T, 3, 88, 88)
            actions   = grp["actions"][start_t:end_t, agent_idx]   # (T,)

        obs = torch.from_numpy(obs_uint8).float() / 255.0
        act = torch.from_numpy(actions).long()
        aid = torch.tensor(agent_idx, dtype=torch.long)
        sid = torch.tensor(self._scenario_to_idx[scenario], dtype=torch.long)
        return obs, act, aid, sid

    def make_sampler(self) -> Sampler:
        """Return a sampler giving equal probability per scenario (uniform within).

        Uses UniformPerScenarioSampler rather than WeightedRandomSampler so the
        (huge, at small seq_len) index doesn't hit torch.multinomial's 2**24
        category cap. Same sampling distribution as before.
        """
        return UniformPerScenarioSampler(
            self._scenario_ranges, num_samples=len(self._samples)
        )


def _build_block_index(
    data_root: str,
    scenario_names: List[str],
    seq_len: int,
    need_next_obs: bool,
    block_len: int,
) -> Tuple[list, list, list, list]:
    """Index at *block* granularity for the block-reading loader.

    A block is ``block_len`` consecutive timesteps of one (episode, agent),
    tiled (non-overlapping) and aligned to the HDF5 obs chunk boundary so that
    reading a block costs ~one LZF decompress yet yields ``block_len // seq_len``
    training windows. Far fewer entries than the per-window index (~block_len×
    smaller), so it is cheap and fork-safe.

    Returns (scen_paths, scen_names, blocks, scenario_ranges) where
    ``blocks[i] = (scen_idx, ep_name, agent_idx, block_start)`` and
    ``scenario_ranges`` gives each scenario's (start, end) span into ``blocks``
    (for uniform-per-scenario sampling, matching the map-style loader).
    """
    ensure_datasets(data_root, scenario_names)
    scen_paths: list = []
    scen_names: list = []
    blocks: list = []
    scenario_ranges: list = []
    # A block must hold the read span: block_len obs (+1 more when need_next_obs).
    read_span = block_len + (1 if need_next_obs else 0)

    for scenario in scenario_names:
        hdf5_path = os.path.join(data_root, scenario, "dataset.hdf5")
        s_idx = len(scen_paths)
        start = len(blocks)
        with h5py.File(hdf5_path, "r") as f:
            if scenario not in f:
                raise ValueError(f"Scenario key '{scenario}' not found inside {hdf5_path}")
            grp = f[scenario]
            for ep_name in grp.keys():
                ep = grp[ep_name]
                T = ep["obs"].shape[0]
                A = ep["obs"].shape[1]
                # Tiled, chunk-aligned block starts that leave room for the read span.
                for bstart in range(0, T - read_span + 1, block_len):
                    for a in range(A):
                        blocks.append((s_idx, ep_name, a, bstart))
        if len(blocks) == start:
            logger.warning("No valid blocks for scenario '%s' (seq_len=%d, block_len=%d), skipping.",
                           scenario, seq_len, block_len)
            continue
        scen_paths.append(hdf5_path)
        scen_names.append(scenario)
        scenario_ranges.append((start, len(blocks)))
        logger.info("BlockShuffleDataset: %6d blocks from %s", len(blocks) - start, scenario)

    if not blocks:
        raise RuntimeError("No blocks found across all provided scenarios.")
    return scen_paths, scen_names, blocks, scenario_ranges


class BlockShuffleDataset(IterableDataset):
    """Opt-in block-reading loader that amortises HDF5 chunk decompression.

    The obs are stored in LZF-compressed chunks of ~100 timesteps, so a
    per-sample read decompresses a whole chunk to use ``seq_len`` steps — a wall
    at small seq_len (batch_size chunk-decompresses per batch). This loader reads
    one chunk-aligned block per decompress, extracts ALL ``block_len // seq_len``
    windows from it, and pushes them through a shuffle buffer so emitted batches
    are ~i.i.d. (matching the map-style loader) rather than temporally correlated.

    Emits the SAME per-sample tuples as the map-style datasets, so it is a drop-in
    replacement (``sid`` is the canonical scenario index for scenario-ID conditioning):
      need_next_obs=False -> (obs[T,3,88,88] float, act[T] long, aid long, sid long)   [BC]
      need_next_obs=True  -> (obs[T+1,...] , act[T], rew[T], done[T], aid, sid)         [BCQ/IQL/CQL]

    Sampling: scenario uniform -> block uniform within scenario -> all windows in
    block. With regular episode lengths this matches the map-style per-scenario
    uniform distribution. Each DataLoader worker seeds its own RNG (independent
    streams, sampling with replacement), so workers never duplicate data.
    """

    def __init__(
        self,
        data_root: str,
        scenario_names: List[str],
        seq_len: int = 1,
        need_next_obs: bool = True,
        block_len: int = 100,
        shuffle_buffer: int = 4096,
        seed: int = 0,
    ) -> None:
        if seq_len > block_len:
            raise ValueError(f"block_len ({block_len}) must be >= seq_len ({seq_len}).")
        self.seq_len = int(seq_len)
        self.need_next_obs = bool(need_next_obs)
        self.block_len = int(block_len)
        self.shuffle_buffer = int(shuffle_buffer)
        self.seed = int(seed)
        # Canonical scenario index = position in the provided scenario_names list
        # (cfg.in_dist_scenarios order), emitted per window for scenario-ID
        # conditioning. Keyed by name so it stays correct if a scenario is skipped.
        self._scenario_to_idx = {name: i for i, name in enumerate(scenario_names)}
        (self._scen_paths, self._scen_names,
         self._blocks, self._scenario_ranges) = _build_block_index(
            data_root, scenario_names, seq_len, need_next_obs, block_len
        )
        # windows extracted per block (drops a trailing partial window if any)
        self._wins_per_block = self.block_len // self.seq_len
        logger.info(
            "BlockShuffleDataset ready: %d blocks across %d scenarios "
            "(%d windows/block, buffer=%d)",
            len(self._blocks), len(self._scen_paths),
            self._wins_per_block, self.shuffle_buffer,
        )

    def _read_block_windows(self, files: dict, block) -> list:
        """Read one block (≈one decompress) and return its window tuples (numpy)."""
        s_idx, ep_name, agent, bstart = block
        path = self._scen_paths[s_idx]
        f = files.get(path)
        if f is None:
            f = h5py.File(path, "r")
            files[path] = f
        grp = f[self._scen_names[s_idx]][ep_name]
        scen_id = self._scenario_to_idx[self._scen_names[s_idx]]   # constant over the block
        sl = self.seq_len
        n = self._wins_per_block
        span = self.block_len + (1 if self.need_next_obs else 0)
        obs_blk = grp["obs"][bstart:bstart + span, agent]          # (span, 3,88,88) uint8
        act_blk = grp["actions"][bstart:bstart + self.block_len, agent]
        if self.need_next_obs:
            rew_blk = grp["rewards"][bstart:bstart + self.block_len, agent]
            don_blk = grp["dones"][bstart:bstart + self.block_len]
        out = []
        for w in range(n):
            o = w * sl
            if self.need_next_obs:
                out.append((obs_blk[o:o + sl + 1].copy(),
                            act_blk[o:o + sl].copy(),
                            rew_blk[o:o + sl].copy(),
                            don_blk[o:o + sl].copy(),
                            agent, scen_id))
            else:
                out.append((obs_blk[o:o + sl].copy(),
                            act_blk[o:o + sl].copy(),
                            agent, scen_id))
        return out

    def _to_tensors(self, item):
        """Convert a stored numpy window to the exact map-style tensor tuple."""
        if self.need_next_obs:
            obs_u8, act, rew, don, aid, sid = item
            return (
                torch.from_numpy(obs_u8).float() / 255.0,
                torch.from_numpy(act).long(),
                torch.from_numpy(rew).float(),
                torch.from_numpy(don).float(),
                torch.tensor(aid, dtype=torch.long),
                torch.tensor(sid, dtype=torch.long),
            )
        obs_u8, act, aid, sid = item
        return (
            torch.from_numpy(obs_u8).float() / 255.0,
            torch.from_numpy(act).long(),
            torch.tensor(aid, dtype=torch.long),
            torch.tensor(sid, dtype=torch.long),
        )

    def __iter__(self):
        worker = get_worker_info()
        wid = worker.id if worker is not None else 0
        rng = random.Random(self.seed * 1_000_003 + wid)
        S = len(self._scenario_ranges)
        files: dict = {}

        def source():
            while True:
                s = rng.randrange(S)
                lo, hi = self._scenario_ranges[s]
                b = rng.randrange(lo, hi)
                for win in self._read_block_windows(files, self._blocks[b]):
                    yield win

        src = source()
        # Warm the shuffle buffer, then emit-with-replacement (tf.data style):
        # each emitted window is replaced by a fresh one, giving continuous mixing.
        buf = [next(src) for _ in range(self.shuffle_buffer)]
        try:
            while True:
                i = rng.randrange(self.shuffle_buffer)
                out = buf[i]
                buf[i] = next(src)
                yield self._to_tensors(out)
        finally:
            for f in files.values():
                try:
                    f.close()
                except Exception:
                    pass


def make_offline_dataloader(cfg, need_next_obs: bool) -> DataLoader:
    """Build the training DataLoader, honouring the opt-in block-reading loader.

    cfg.block_read=False (default) → original map-style dataset + per-scenario
    sampler (num_workers=2), unchanged. cfg.block_read=True → BlockShuffleDataset
    (no sampler; IterableDataset) for fast small-seq_len loading.
    """
    scenarios = list(cfg.in_dist_scenarios)
    if bool(cfg.get("block_read", False)):
        ds = BlockShuffleDataset(
            cfg.data_root, scenarios,
            seq_len=cfg.seq_len, need_next_obs=need_next_obs,
            block_len=int(cfg.get("block_len", 100)),
            shuffle_buffer=int(cfg.get("block_shuffle_buffer", 4096)),
            seed=int(cfg.seed),
        )
        nw = int(cfg.get("block_workers", 4))
        return DataLoader(
            ds, batch_size=cfg.batch_size, num_workers=nw,
            drop_last=True, persistent_workers=(nw > 0),
        )
    Cls = MultiScenarioTransitionDataset if need_next_obs else MultiScenarioDataset
    ds = Cls(cfg.data_root, scenarios, seq_len=cfg.seq_len)
    return DataLoader(
        ds, batch_size=cfg.batch_size, sampler=ds.make_sampler(),
        num_workers=2, drop_last=True,
    )


def shutdown_dataloader(dataloader) -> None:
    """Deterministically terminate a DataLoader's worker processes.

    With ``persistent_workers=True`` over the infinite ``BlockShuffleDataset``,
    the ``block_workers`` daemon workers (each holding a full shuffle buffer of
    ``block_shuffle_buffer`` windows) stay alive until GC happens to collect the
    loader's iterator cycle. Under a subprocess-per-run launch (``wandb agent``)
    the OS reaps them at process exit, but under a single-process Hydra
    ``--multirun`` they pile up run-over-run and leak RAM. Call at the end of
    ``train()`` to free them immediately. No-op for ``num_workers=0`` or a loader
    whose workers are already gone.
    """
    if dataloader is None:
        return
    it = getattr(dataloader, "_iterator", None)
    if it is not None:
        shutdown = getattr(it, "_shutdown_workers", None)
        if shutdown is not None:
            try:
                shutdown()
            except Exception:  # best-effort; process teardown will reap regardless
                pass
        dataloader._iterator = None


class MultiScenarioTransitionDataset(Dataset):
    """
    Streams ``(obs, actions, rewards, dones)`` sequences from multiple HDF5
    scenario files, with T+1 observations for TD-style offline RL algorithms.

    Use ``dataset.make_sampler()`` as the ``sampler`` argument to DataLoader.

    Parameters
    ----------
    data_root : str
        Root directory containing one sub-directory per scenario.
    scenario_names : list of str
        Scenario keys to load.
    seq_len : int
        Length T of each sampled sequence (window reads T+1 observations).
    """

    def __init__(
        self,
        data_root: str,
        scenario_names: List[str],
        seq_len: int = 128,
    ) -> None:
        self.seq_len = seq_len
        # Canonical scenario index = position in the provided scenario_names list
        # (i.e. cfg.in_dist_scenarios order); emitted per sample for scenario-ID
        # conditioning (see MultiScenarioDataset).
        self._scenario_to_idx = {name: i for i, name in enumerate(scenario_names)}
        self._samples, self._scenario_ranges = _build_index(
            data_root, scenario_names, seq_len, need_next_obs=True
        )
        logger.info(
            "MultiScenarioTransitionDataset ready: %d total sequences across %d scenarios",
            len(self._samples), len(scenario_names),
        )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hdf5_path, scenario, ep_name, agent_idx, start_t = self._samples[idx]
        end_t = start_t + self.seq_len + 1  # +1 for next_obs

        with h5py.File(hdf5_path, "r") as f:
            grp = f[scenario][ep_name]
            obs_raw = grp["obs"][start_t:end_t, agent_idx]             # (T+1, 3, 88, 88)
            actions  = grp["actions"][start_t:end_t - 1, agent_idx]    # (T,)
            rewards  = grp["rewards"][start_t:end_t - 1, agent_idx]    # (T,)
            dones    = grp["dones"][start_t:end_t - 1]                  # (T,)

        obs = torch.from_numpy(obs_raw).float() / 255.0
        act = torch.from_numpy(actions).long()
        rew = torch.from_numpy(rewards).float()
        don = torch.from_numpy(dones).float()
        aid = torch.tensor(agent_idx, dtype=torch.long)
        sid = torch.tensor(self._scenario_to_idx[scenario], dtype=torch.long)
        return obs, act, rew, don, aid, sid

    def make_sampler(self) -> Sampler:
        """Return a sampler giving equal probability per scenario (uniform within).

        Uses UniformPerScenarioSampler rather than WeightedRandomSampler so the
        (huge, at small seq_len) index doesn't hit torch.multinomial's 2**24
        category cap. Same sampling distribution as before.
        """
        return UniformPerScenarioSampler(
            self._scenario_ranges, num_samples=len(self._samples)
        )

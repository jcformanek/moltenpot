"""
tests/test_wrappers.py
======================
Unit tests for MeltingPotShimmy.  MeltingPot / dmlab2d are mocked so
these tests run in any clean Python environment.
"""

import sys
import os
import types

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Bootstrap mocks BEFORE importing wrappers so the import-time
# `from meltingpot import factory` doesn't fail.
# ---------------------------------------------------------------------------
# Build a minimal fake meltingpot package tree
_mp_pkg         = types.ModuleType("meltingpot")
_mp_factory     = types.ModuleType("meltingpot.factory")
_mp_pkg.factory = _mp_factory
sys.modules.setdefault("meltingpot",         _mp_pkg)
sys.modules.setdefault("meltingpot.factory", _mp_factory)

# Build a minimal fake dm_env package
_dm_env = types.ModuleType("dm_env")

class _StepType:
    FIRST = 0
    MID   = 1
    LAST  = 2

_dm_env.StepType = _StepType
sys.modules.setdefault("dm_env", _dm_env)

# Now import the module under test
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from moltenpot.wrappers import MeltingPotShimmy, OBS_HEIGHT, OBS_WIDTH, OBS_CHANNELS


# ---------------------------------------------------------------------------
# Mock environment factory
# ---------------------------------------------------------------------------
DEFAULT_NUM_FOCAL = 3


class _FakeTimestep:
    """Mimics a MeltingPot timestep (tuple-style observations for focal players)."""

    def __init__(self, num_focal, step_type=_StepType.MID, fill_value=128):
        # Observations: tuple of dicts, one per focal player
        self.observation = tuple(
            {"RGB": np.full((OBS_HEIGHT, OBS_WIDTH, OBS_CHANNELS), fill_value, dtype=np.uint8)}
            for _ in range(num_focal)
        )
        # Rewards: tuple of floats, one per focal player
        self.reward = tuple(0.5 for _ in range(num_focal))
        self.step_type = step_type


class _FakeActionSpec:
    """Minimal action spec placeholder."""
    pass


class _FakeEnv:
    """Minimal fake underlying environment supporting multiple focal players."""

    def __init__(self, num_focal=DEFAULT_NUM_FOCAL):
        self._num_focal = num_focal
        self._call_count = 0

    def reset(self):
        return _FakeTimestep(self._num_focal, step_type=_StepType.FIRST)

    def step(self, action_list):
        assert isinstance(action_list, list), "action_list must be a list"
        assert len(action_list) == self._num_focal, (
            f"Expected {self._num_focal} actions, got {len(action_list)}"
        )
        self._call_count += 1
        # Signal LAST on every 10th step
        stype = _StepType.LAST if self._call_count % 10 == 0 else _StepType.MID
        return _FakeTimestep(self._num_focal, step_type=stype, fill_value=self._call_count % 255)

    def action_spec(self):
        """Return a tuple of specs, one per focal player."""
        return tuple(_FakeActionSpec() for _ in range(self._num_focal))

    def close(self):
        pass


def _make_shim(monkeypatch, num_focal=DEFAULT_NUM_FOCAL):
    """Return a MeltingPotShimmy backed by _FakeEnv."""
    monkeypatch.setattr(
        MeltingPotShimmy, "_build_env",
        staticmethod(lambda scenario, seed: _FakeEnv(num_focal=num_focal)),
    )
    return MeltingPotShimmy("clean_up_0", seed=42)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestReset:
    def test_returns_float32(self, monkeypatch):
        shim = _make_shim(monkeypatch)
        obs = shim.reset()
        assert obs.dtype == np.float32, f"Expected float32, got {obs.dtype}"

    def test_shape_multi_agent(self, monkeypatch):
        shim = _make_shim(monkeypatch, num_focal=3)
        obs = shim.reset()
        assert obs.shape == (3, OBS_CHANNELS, OBS_HEIGHT, OBS_WIDTH), \
            f"Expected (3,3,88,88), got {obs.shape}"

    def test_shape_single_agent(self, monkeypatch):
        shim = _make_shim(monkeypatch, num_focal=1)
        obs = shim.reset()
        assert obs.shape == (1, OBS_CHANNELS, OBS_HEIGHT, OBS_WIDTH)

    def test_values_in_unit_interval(self, monkeypatch):
        shim = _make_shim(monkeypatch)
        obs = shim.reset()
        assert obs.min() >= 0.0
        assert obs.max() <= 1.0

    def test_normalisation_correct(self, monkeypatch):
        """128 uint8 → ~0.502 float32."""
        shim = _make_shim(monkeypatch)
        obs = shim.reset()
        expected = 128.0 / 255.0
        assert abs(obs.mean() - expected) < 1e-4

    def test_num_focal_attribute(self, monkeypatch):
        shim = _make_shim(monkeypatch, num_focal=5)
        assert shim.num_focal == 5


class TestStep:
    def test_returns_four_tuple(self, monkeypatch):
        shim = _make_shim(monkeypatch)
        shim.reset()
        result = shim.step([0] * shim.num_focal)
        assert len(result) == 4

    def test_obs_shape_after_step(self, monkeypatch):
        shim = _make_shim(monkeypatch, num_focal=3)
        shim.reset()
        obs, _, _, _ = shim.step([7, 0, 1])
        assert obs.shape == (3, OBS_CHANNELS, OBS_HEIGHT, OBS_WIDTH)

    def test_rewards_shape(self, monkeypatch):
        shim = _make_shim(monkeypatch, num_focal=3)
        shim.reset()
        _, rewards, _, _ = shim.step([0, 0, 0])
        assert isinstance(rewards, np.ndarray)
        assert rewards.shape == (3,)
        assert rewards.dtype == np.float32

    def test_done_is_bool(self, monkeypatch):
        shim = _make_shim(monkeypatch)
        shim.reset()
        _, _, done, _ = shim.step([0] * shim.num_focal)
        assert isinstance(done, bool)

    def test_info_contains_num_focal(self, monkeypatch):
        shim = _make_shim(monkeypatch, num_focal=4)
        shim.reset()
        _, _, _, info = shim.step([0] * 4)
        assert info["num_focal"] == 4

    def test_wrong_action_count_raises(self, monkeypatch):
        shim = _make_shim(monkeypatch, num_focal=3)
        shim.reset()
        with pytest.raises(AssertionError):
            shim.step([0, 0])  # too few actions

    def test_done_triggers_auto_reset(self, monkeypatch):
        """step() auto-resets on the next call after an episode ends."""
        shim = _make_shim(monkeypatch)
        shim.reset()
        done = False
        for _ in range(15):
            _, _, done, _ = shim.step([0] * shim.num_focal)
            if done:
                break
        assert done, "Episode should have ended by step 10"
        # The next step should work without explicit reset()
        obs, _, _, _ = shim.step([0] * shim.num_focal)
        assert obs.shape == (shim.num_focal, OBS_CHANNELS, OBS_HEIGHT, OBS_WIDTH)

    def test_episode_id_increments(self, monkeypatch):
        shim = _make_shim(monkeypatch)
        shim.reset()
        ep_ids = set()
        for _ in range(25):
            _, _, done, info = shim.step([0] * shim.num_focal)
            ep_ids.add(info["episode_id"])
        assert len(ep_ids) >= 2, "episode_id should increment across episodes"


class TestClose:
    def test_close_does_not_raise(self, monkeypatch):
        shim = _make_shim(monkeypatch)
        shim.reset()
        shim.close()   # must not raise


class TestVariableAgentCounts:
    """Test with different numbers of focal agents."""

    @pytest.mark.parametrize("num_focal", [1, 2, 3, 5, 7])
    def test_obs_shape_varies(self, monkeypatch, num_focal):
        shim = _make_shim(monkeypatch, num_focal=num_focal)
        obs = shim.reset()
        assert obs.shape == (num_focal, OBS_CHANNELS, OBS_HEIGHT, OBS_WIDTH)

    @pytest.mark.parametrize("num_focal", [1, 2, 3, 5, 7])
    def test_step_reward_shape_varies(self, monkeypatch, num_focal):
        shim = _make_shim(monkeypatch, num_focal=num_focal)
        shim.reset()
        _, rewards, _, _ = shim.step([0] * num_focal)
        assert rewards.shape == (num_focal,)

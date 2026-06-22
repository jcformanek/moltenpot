"""
wrappers.py — MeltingPotShimmy
==============================
Thin wrapper that adapts the MeltingPot 2.0 scenario API to a simple
Gym-style interface for **all focal agents**.

Background bot policies are handled entirely by the scenario engine
— we never touch nor observe their actions from Python.

Observation returned:
    np.ndarray, shape (num_focal, 3, 88, 88), dtype float32, range [0, 1]
    (Channel-first format expected by the PyTorch CNN)

Action accepted:
    List[int], length num_focal, one discrete action per focal agent.
    The action-space size depends on the substrate (e.g. 7 for Coins, 8 for
    Coop Mining / Commons Harvest, 9 for Clean Up, 11 for Allelopathic
    Harvest). The common movement actions (NOOP, FORWARD, BACKWARD, STRAFE,
    TURN) are shared; higher indices are substrate-specific interactions
    (e.g. ZAP / CLEAN in Clean Up). Set ``model.num_actions`` to match.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

# MeltingPot transitively imports TensorFlow via dmlab2d. TF's default is to
# grab the entire visible GPU on first init, which collides with PyTorch (used
# by the trainer) and prevents running multiple processes per GPU. The env var
# below tells TF to grow memory on demand instead. Must be set before TF is
# imported, hence here at module-import time.
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import cv2  # noqa: E402  (must follow the env-var set above)
import numpy as np  # noqa: E402

logger = logging.getLogger(__name__)

# Suppress absl warnings (Melting Pot can be very noisy with observation tensors)
try:
    from absl import logging as absl_logging
    absl_logging.set_verbosity(absl_logging.ERROR)
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OBS_HEIGHT: int = 88
OBS_WIDTH: int = 88
OBS_CHANNELS: int = 3          # RGB
NUM_ACTIONS: int = 9
RGB_KEY: str = "RGB"           # key MeltingPot uses for visual observations
WORLD_RGB_KEY: str = "WORLD.RGB" # key for global environment view


class MeltingPotShimmy:
    """
    Shim between MeltingPot and a multi-agent Gym-style API.

    Controls all focal players in the scenario.  Background bots are
    handled internally by the MeltingPot scenario.

    Parameters
    ----------
    scenario_name : str
        One of the ``clean_up_*`` scenario names.
    seed : int, optional
        Environment seed (forwarded to the underlying substrate).
    max_steps : int
        Maximum steps per episode before forced termination.

    Example
    -------
    >>> env = MeltingPotShimmy("clean_up_0")
    >>> obs = env.reset()                      # (num_focal, 3, 88, 88) float32
    >>> obs, rewards, done, info = env.step([7, 0, 1])
    >>> env.close()
    """

    action_space_n: int = NUM_ACTIONS
    scenario_name: str

    def __init__(self, scenario_name: str, seed: Optional[int] = None, max_steps: int = 1000) -> None:
        self.scenario_name = scenario_name
        self._seed = seed
        self._env = self._build_env(scenario_name, seed)
        self._episode_id: int = 0
        self._max_steps = max_steps
        self._done: bool = True   # triggers automatic reset on first step()
        self._record_buffer: List[np.ndarray] = []
        self._is_recording: bool = False

        self.num_focal: int = len(self._env.action_spec())
        self.observation_space_shape: Tuple[int, ...] = (
            self.num_focal, OBS_CHANNELS, OBS_HEIGHT, OBS_WIDTH,
        )

        self._latest_bg_rewards: Optional[np.ndarray] = None
        try:
            self._env.observables().background.timestep.subscribe(
                on_next=self._on_background_timestep
            )
            logger.debug("Subscribed to background timestep observables")
        except Exception as exc:
            logger.debug("Background observables not available: %s", exc)

        logger.debug(
            "MeltingPotShimmy initialised  scenario=%s  seed=%s  num_focal=%d",
            scenario_name, seed, self.num_focal,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> np.ndarray:
        """
        Reset the environment and return the first observation.

        Returns
        -------
        obs : np.ndarray
            Shape ``(num_focal, 3, 88, 88)``, dtype ``float32``, values in ``[0, 1]``.
        """
        timestep = self._env.reset()
        self._step_count = 0
        self._done = False
        return self._extract_all_focal_obs(timestep)

    def step(self, actions: List[int]) -> Tuple[np.ndarray, np.ndarray, bool, Dict[str, Any]]:
        """
        Advance the environment by one step.

        Parameters
        ----------
        actions : List[int]
            One discrete action per focal agent, each ∈ ``[0, NUM_ACTIONS)``.
            Length must equal ``self.num_focal``.

        Returns
        -------
        obs     : np.ndarray   (num_focal, 3, 88, 88) float32
        rewards : np.ndarray   (num_focal,) float32
        done    : bool         (shared — episode ends for all players at once)
        info    : dict
        """
        if self._done:
            logger.debug("Auto-resetting environment (previous episode ended).")
            self.reset()

        assert len(actions) == self.num_focal, (
            f"Expected {self.num_focal} actions, got {len(actions)}"
        )

        action_list = [int(a) for a in actions]
        # The background.timestep observable subscription fires synchronously
        # inside _env.step(), so _latest_bg_rewards is already updated by the
        # time we construct info below.
        timestep = self._env.step(action_list)
        self._step_count += 1

        if self._is_recording:
            raw_rgb = self._get_raw_rgb(timestep)
            self._record_buffer.append(raw_rgb)

        obs     = self._extract_all_focal_obs(timestep)
        rewards = self._extract_rewards(timestep)
        done    = self._is_done(timestep) or (self._step_count >= self._max_steps)

        info: Dict[str, Any] = {
            "episode_id": self._episode_id,
            "step": self._step_count,
            "scenario": self.scenario_name,
            "num_focal": self.num_focal,
        }
        if self._latest_bg_rewards is not None:
            info["background_rewards"] = self._latest_bg_rewards.copy()

        if done:
            self._done = True
            self._episode_id += 1

        return obs, rewards, done, info

    def close(self) -> None:
        """Release the subprocess / dmlab2d resources."""
        try:
            self._env.close()
        except Exception as exc:
            logger.warning("Error while closing environment: %s", exc)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _on_background_timestep(self, ts) -> None:
        """Callback fired synchronously by the background observable during _env.step()."""
        reward = getattr(ts, 'reward', None)
        if reward is None:
            logger.warning("Background timestep has no reward field — background metrics will be unavailable")
            return
        self._latest_bg_rewards = np.array(reward, dtype=np.float32)

    @staticmethod
    def _build_env(scenario_name: str, seed: Optional[int]):
        """Instantiate the MeltingPot scenario."""
        try:
            from meltingpot import scenario  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "MeltingPot is not installed. "
                "Please follow the instructions in README.md."
            ) from exc

        return scenario.build(scenario_name, substrate_transform=lambda x: x)

    @staticmethod
    def _extract_all_focal_obs(timestep) -> np.ndarray:
        """
        Pull all focal players' RGB frames out of a MeltingPot timestep
        and convert to a ``(num_focal, 3, 88, 88)`` float32 array in [0, 1].
        """
        observations = timestep.observation

        if isinstance(observations, (list, tuple)):
            frames = []
            for player_obs in observations:
                rgb = np.asarray(player_obs[RGB_KEY], dtype=np.uint8)
                frames.append(np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0)
            return np.stack(frames, axis=0)

        elif isinstance(observations, dict):
            frames = []
            for key in sorted(observations.keys()):
                player_obs = observations[key]
                rgb = np.asarray(player_obs[RGB_KEY], dtype=np.uint8)
                frames.append(np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0)
            return np.stack(frames, axis=0)
        else:
            raise TypeError(f"Unexpected observation type: {type(observations)}")

    @staticmethod
    def _extract_rewards(timestep) -> np.ndarray:
        """Extract rewards for all focal players as a float32 array."""
        reward = timestep.reward
        if isinstance(reward, (list, tuple)):
            return np.array(reward, dtype=np.float32)
        elif isinstance(reward, np.ndarray):
            return reward.astype(np.float32)
        elif isinstance(reward, dict):
            return np.array(list(reward.values()), dtype=np.float32)
        else:
            return np.array([float(reward)], dtype=np.float32)

    @staticmethod
    def _is_done(timestep) -> bool:
        """Return True if the episode has ended."""
        import dm_env  # type: ignore[import]

        if hasattr(timestep, "step_type"):
            st = timestep.step_type
            if isinstance(st, dict):
                return any(v == dm_env.StepType.LAST for v in st.values())
            return st == dm_env.StepType.LAST
        return False

    # ------------------------------------------------------------------
    # Recording support
    # ------------------------------------------------------------------
    def start_recording(self) -> None:
        """Start buffering RGB frames."""
        self._record_buffer = []
        self._is_recording = True

    def stop_recording(self, video_path: str, fps: int = 10) -> None:
        """Save buffered frames to a video file using OpenCV."""
        self._is_recording = False
        if not self._record_buffer:
            logger.warning("No frames to save for %s", video_path)
            return

        try:
            import os
            os.makedirs(os.path.dirname(video_path), exist_ok=True)

            height, width, _ = self._record_buffer[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

            for frame in self._record_buffer:
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(bgr_frame)

            out.release()
            logger.info("Video saved to %s (%d frames, %dx%d)",
                        video_path, len(self._record_buffer), width, height)
        except Exception as e:
            logger.error("Failed to save video %s: %s", video_path, e)
        finally:
            self._record_buffer = []

    def _get_raw_rgb(self, timestep) -> np.ndarray:
        """Extract RGB frame for recording. Prefers WORLD.RGB if available."""
        observations = timestep.observation

        if isinstance(observations, (list, tuple)):
            for player_obs in observations:
                if hasattr(player_obs, "keys") and WORLD_RGB_KEY in player_obs:
                    return np.asarray(player_obs[WORLD_RGB_KEY], dtype=np.uint8)
            obs_to_check = observations[0]
        else:
            obs_to_check = observations

        if hasattr(obs_to_check, "keys"):
            if WORLD_RGB_KEY in obs_to_check:
                return np.asarray(obs_to_check[WORLD_RGB_KEY], dtype=np.uint8)
            if RGB_KEY in obs_to_check:
                return np.asarray(obs_to_check[RGB_KEY], dtype=np.uint8)

        raise ValueError("Could not find suitable RGB observation for recording.")

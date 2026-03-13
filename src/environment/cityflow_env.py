"""
CityFlow-based single-intersection Gymnasium environment.

CityFlow is a high-performance traffic simulator designed for RL research.
When ``cityflow`` is not installed the environment still imports cleanly
and raises ``ImportError`` only when ``reset()`` is called.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np

from src.environment.state_representation import (
    StateConfig,
    StateEncoder,
    TrafficState,
)

try:
    import cityflow  # type: ignore

    _CITYFLOW_AVAILABLE = True
except ImportError:
    _CITYFLOW_AVAILABLE = False

DURATION_BUCKETS = [10, 20, 30, 45]


class CityFlowEnvironment(gym.Env):
    """
    OpenAI Gymnasium wrapper for a CityFlow single-intersection simulation.

    Observation
    -----------
    Flat normalised vector produced by :class:`StateEncoder`.

    Action
    ------
    ``phase_id * len(DURATION_BUCKETS) + bucket_id``

    Reward
    ------
    Same weighted combination as :class:`SumoEnvironment`.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config_file: str,
        min_green: int = 5,
        max_green: int = 90,
        yellow_time: int = 3,
        max_steps: int = 3600,
        reward_weights: Optional[Dict[str, float]] = None,
        state_config: Optional[StateConfig] = None,
        seed: int = 42,
    ):
        super().__init__()

        self.config_file = config_file
        self.min_green = min_green
        self.max_green = max_green
        self.yellow_time = yellow_time
        self.max_steps = max_steps
        self.seed_val = seed

        self.reward_weights = reward_weights or {
            "pressure": 0.40,
            "queue": 0.25,
            "throughput": 0.20,
            "wait": 0.10,
            "pedestrian": 0.05,
        }

        self.state_cfg = state_config or StateConfig()
        self.encoder = StateEncoder(self.state_cfg)

        # Parse number of phases from config
        self._num_phases = self._parse_num_phases(config_file)
        num_actions = self._num_phases * len(DURATION_BUCKETS)
        self.action_space = gym.spaces.Discrete(num_actions)
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.state_cfg.flat_dim,),
            dtype=np.float32,
        )

        self._eng: Optional[Any] = None
        self._step_count = 0
        self._current_phase = 0
        self._phase_duration = 0.0
        self._prev_throughput = 0
        self._episode_metrics: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        if not _CITYFLOW_AVAILABLE:
            raise ImportError(
                "CityFlow is not installed.  Install it with:\n"
                "  pip install cityflow\n"
                "(requires Linux / WSL2)"
            )

        self._eng = cityflow.Engine(self.config_file, thread_num=1)
        self._step_count = 0
        self._current_phase = 0
        self._phase_duration = 0.0
        self._prev_throughput = 0
        self._episode_metrics = {
            "total_wait": 0.0,
            "total_throughput": 0,
            "total_reward": 0.0,
            "safety_overrides": 0,
        }

        obs = self.encoder.encode(self._get_state())
        return obs, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        assert self._eng is not None, "Call reset() before step()."

        phase_id, bucket_id = divmod(action, len(DURATION_BUCKETS))
        requested_duration = DURATION_BUCKETS[bucket_id]

        # Safety: enforce minimum green time
        if self._phase_duration < self.min_green:
            phase_id = self._current_phase
            self._episode_metrics["safety_overrides"] += 1

        duration = int(np.clip(requested_duration, self.min_green, self.max_green))

        # Yellow phase on phase change
        if phase_id != self._current_phase:
            self._run_steps(self.yellow_time)

        self._set_phase(phase_id)
        state_before = self._get_state()

        self._run_steps(duration)
        state_after = self._get_state()

        reward = self._compute_reward(state_before, state_after)
        self._episode_metrics["total_reward"] += reward
        self._episode_metrics["total_wait"] += float(np.mean(state_after.waiting_times))

        cur_throughput = self._eng.get_vehicle_count()
        self._episode_metrics["total_throughput"] = cur_throughput

        truncated = self._step_count >= self.max_steps
        obs = self.encoder.encode(state_after)
        return obs, reward, False, truncated, dict(self._episode_metrics)

    def close(self):
        self._eng = None

    def render(self):
        pass

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_steps(self, n: int):
        for _ in range(n):
            self._eng.next_step()
            self._step_count += 1
            self._phase_duration += 1.0
            if self._step_count >= self.max_steps:
                break

    def _set_phase(self, phase_id: int):
        inter_ids = list(self._eng.get_intersection_ids())
        if inter_ids:
            self._eng.set_tl_phase(inter_ids[0], phase_id)
        self._current_phase = phase_id
        self._phase_duration = 0.0

    def _get_state(self) -> TrafficState:
        lane_vehicle_count = self._eng.get_lane_vehicle_count()
        lane_waiting_count = self._eng.get_lane_waiting_vehicle_count()

        counts = np.array(list(lane_vehicle_count.values()), dtype=np.float32)
        queues = np.array(list(lane_waiting_count.values()), dtype=np.float32)
        speeds = np.zeros_like(counts)   # CityFlow per-lane speed not always available
        waits = queues * 5.0             # rough proxy: each halted veh ~5s wait

        sim_time = float(self._step_count)
        time_of_day = (sim_time % 86400) / 86400.0

        return TrafficState(
            queue_lengths=queues,
            vehicle_speeds=speeds,
            waiting_times=waits,
            vehicle_counts=counts,
            current_phase=self._current_phase,
            num_phases=self._num_phases,
            phase_duration=self._phase_duration,
            time_of_day=time_of_day,
            day_of_week=0,
            pedestrian_waiting=np.zeros(4, dtype=bool),
            pedestrian_wait_times=np.zeros(4, dtype=np.float32),
            neighbor_embeddings=np.zeros(
                (self.state_cfg.max_neighbors, self.state_cfg.embed_dim),
                dtype=np.float32,
            ),
        )

    def _compute_reward(
        self, state_before: TrafficState, state_after: TrafficState
    ) -> float:
        w = self.reward_weights

        pressure = float(np.sum(state_after.queue_lengths))
        r_pressure = -abs(pressure) / max(self.state_cfg.queue_length_max, 1)

        r_queue = float(
            np.sum(state_before.queue_lengths) - np.sum(state_after.queue_lengths)
        ) / max(self.state_cfg.queue_length_max, 1)

        arrived = max(0, self._eng.get_vehicle_count() - self._prev_throughput)
        r_throughput = float(arrived) / 10.0

        wait_thresh = 120.0
        r_wait = -float(
            np.sum(np.maximum(0, state_after.waiting_times - wait_thresh))
        ) / wait_thresh

        r_ped = 0.0

        return float(
            w.get("pressure", 0.4) * r_pressure
            + w.get("queue", 0.25) * r_queue
            + w.get("throughput", 0.2) * r_throughput
            + w.get("wait", 0.1) * r_wait
            + w.get("pedestrian", 0.05) * r_ped
        )

    @staticmethod
    def _parse_num_phases(config_file: str) -> int:
        """Extract the number of signal phases from a CityFlow roadnet JSON."""
        try:
            with open(config_file) as f:
                cfg = json.load(f)
            roadnet_file = cfg.get("roadnetFile", "")
            base = Path(config_file).parent
            with open(base / roadnet_file) as f:
                roadnet = json.load(f)
            for inter in roadnet.get("intersections", []):
                phases = inter.get("trafficLight", {}).get("lightphases", [])
                if phases:
                    return len(phases)
        except Exception:
            pass
        return 8   # default

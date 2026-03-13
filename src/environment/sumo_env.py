"""
SUMO-based single-intersection Gymnasium environment.

When SUMO/TraCI is not installed the environment still imports cleanly and
raises ``ImportError`` only when ``reset()`` is called.  This lets the rest
of the codebase (agents, tests) import the module unconditionally.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np

from src.environment.state_representation import (
    StateConfig,
    StateEncoder,
    TrafficState,
)

# ---------------------------------------------------------------------------
# Lazy SUMO / TraCI import
# ---------------------------------------------------------------------------
try:
    import sumolib  # type: ignore  # noqa: F401
    import traci  # type: ignore

    _SUMO_AVAILABLE = True
except ImportError:
    _SUMO_AVAILABLE = False


# ---------------------------------------------------------------------------
# Phase definitions for a standard 4-way intersection (NEMA-style, 8 phases)
# ---------------------------------------------------------------------------
PHASE_DEFINITIONS = {
    0: {"name": "NS_GREEN", "duration": 30, "movements": ["N_straight", "S_straight"]},
    1: {"name": "NS_LEFT", "duration": 20, "movements": ["N_left", "S_left"]},
    2: {"name": "EW_GREEN", "duration": 30, "movements": ["E_straight", "W_straight"]},
    3: {"name": "EW_LEFT", "duration": 20, "movements": ["E_left", "W_left"]},
    4: {"name": "ALL_RED_1", "duration": 3, "movements": []},
    5: {"name": "ALL_RED_2", "duration": 3, "movements": []},
    6: {"name": "PED_NS", "duration": 20, "movements": ["ped_N", "ped_S"]},
    7: {"name": "PED_EW", "duration": 20, "movements": ["ped_E", "ped_W"]},
}

DURATION_BUCKETS = [10, 20, 30, 45]  # seconds


class SumoEnvironment(gym.Env):
    """
    OpenAI Gymnasium wrapper for a SUMO single-intersection simulation.

    Observation
    -----------
    Flat normalised vector produced by :class:`StateEncoder`.

    Action
    ------
    Discrete integer encoding ``phase_id * len(DURATION_BUCKETS) + bucket_id``.
    Total = ``num_phases × len(duration_buckets)``.

    Reward
    ------
    Weighted combination of pressure, queue reduction, throughput, wait
    penalty, and pedestrian penalty (see ``configs/default.yaml``).
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        net_file: str,
        route_file: str,
        out_csv_name: Optional[str] = None,
        use_gui: bool = False,
        min_green: int = 5,
        max_green: int = 90,
        yellow_time: int = 3,
        all_red_time: int = 2,
        max_steps: int = 3600,
        reward_weights: Optional[Dict[str, float]] = None,
        state_config: Optional[StateConfig] = None,
        seed: int = 42,
    ):
        super().__init__()

        self.net_file = net_file
        self.route_file = route_file
        self.out_csv_name = out_csv_name
        self.use_gui = use_gui
        self.min_green = min_green
        self.max_green = max_green
        self.yellow_time = yellow_time
        self.all_red_time = all_red_time
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

        # Gym spaces
        num_actions = len(PHASE_DEFINITIONS) * len(DURATION_BUCKETS)
        self.action_space = gym.spaces.Discrete(num_actions)
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.state_cfg.flat_dim,),
            dtype=np.float32,
        )

        # Internal state
        self._step_count = 0
        self._current_phase = 0
        self._phase_duration = 0.0
        self._sumo_started = False
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

        if self._sumo_started:
            traci.close()

        self._start_sumo()
        self._step_count = 0
        self._current_phase = 0
        self._phase_duration = 0.0
        self._episode_metrics = {
            "total_wait": 0.0,
            "total_throughput": 0,
            "total_reward": 0.0,
            "safety_overrides": 0,
        }

        obs = self.encoder.encode(self._get_state())
        return obs, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        phase_id, bucket_id = divmod(action, len(DURATION_BUCKETS))
        requested_duration = DURATION_BUCKETS[bucket_id]

        # Safety constraints
        if self._phase_duration < self.min_green:
            # Stay in current phase
            phase_id = self._current_phase
            self._episode_metrics["safety_overrides"] += 1

        duration = np.clip(requested_duration, self.min_green, self.max_green)

        # Execute yellow + all-red transition if phase is changing
        if phase_id != self._current_phase:
            self._run_yellow_phase()
            self._run_all_red_phase()

        # Set new phase
        self._set_phase(phase_id)
        state_before = self._get_state()

        # Advance simulation for *duration* steps
        for _ in range(int(duration)):
            traci.simulationStep()
            self._step_count += 1
            self._phase_duration += 1.0
            if self._step_count >= self.max_steps:
                break

        state_after = self._get_state()
        reward = self._compute_reward(state_before, state_after)

        self._episode_metrics["total_reward"] += reward
        self._episode_metrics["total_wait"] += float(np.mean(state_after.waiting_times))
        self._episode_metrics["total_throughput"] += int(traci.simulation.getArrivedNumber())

        terminated = False
        truncated = self._step_count >= self.max_steps
        obs = self.encoder.encode(state_after)
        info = dict(self._episode_metrics)

        return obs, reward, terminated, truncated, info

    def close(self):
        if self._sumo_started:
            traci.close()
            self._sumo_started = False

    def render(self):
        pass  # GUI is controlled by SUMO when use_gui=True

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _start_sumo(self):
        if not _SUMO_AVAILABLE:
            raise ImportError(
                "SUMO/TraCI is not installed.  Install it with:\n"
                "  pip install traci sumolib\n"
                "and ensure SUMO binaries are on your PATH."
            )
        sumo_binary = "sumo-gui" if self.use_gui else "sumo"
        sumo_cmd = [
            sumo_binary,
            "-n",
            self.net_file,
            "-r",
            self.route_file,
            "--no-step-log",
            "true",
            "--waiting-time-memory",
            "10000",
            "--time-to-teleport",
            "-1",
            "--seed",
            str(self.seed_val),
        ]
        if self.out_csv_name:
            sumo_cmd += ["--statistic-output", self.out_csv_name]

        traci.start(sumo_cmd)
        self._sumo_started = True

    def _set_phase(self, phase_id: int):
        tl_ids = traci.trafficlight.getIDList()
        if tl_ids:
            traci.trafficlight.setPhase(tl_ids[0], phase_id)
        self._current_phase = phase_id
        self._phase_duration = 0.0

    def _run_yellow_phase(self):
        for _ in range(self.yellow_time):
            traci.simulationStep()
            self._step_count += 1

    def _run_all_red_phase(self):
        for _ in range(self.all_red_time):
            traci.simulationStep()
            self._step_count += 1

    def _get_state(self) -> TrafficState:
        """Collect per-lane and intersection features from TraCI."""
        lane_ids = traci.lane.getIDList()
        # Filter to only incoming lanes (those ending at the controlled TL)
        incoming = [lid for lid in lane_ids if not lid.startswith(":")]

        queue_lengths = np.array(
            [traci.lane.getLastStepHaltingNumber(lid) for lid in incoming],
            dtype=np.float32,
        )
        vehicle_speeds = np.array(
            [traci.lane.getLastStepMeanSpeed(lid) * 3.6 for lid in incoming],
            dtype=np.float32,
        )
        waiting_times = np.array(
            [traci.lane.getWaitingTime(lid) for lid in incoming],
            dtype=np.float32,
        )
        vehicle_counts = np.array(
            [traci.lane.getLastStepVehicleNumber(lid) for lid in incoming],
            dtype=np.float32,
        )

        sim_time = traci.simulation.getTime()
        time_of_day = (sim_time % 86400) / 86400.0
        # Use a fixed day-of-week for simulation (can be overridden via options)
        day_of_week = 0

        return TrafficState(
            queue_lengths=queue_lengths,
            vehicle_speeds=vehicle_speeds,
            waiting_times=waiting_times,
            vehicle_counts=vehicle_counts,
            current_phase=self._current_phase,
            num_phases=len(PHASE_DEFINITIONS),
            phase_duration=self._phase_duration,
            time_of_day=time_of_day,
            day_of_week=day_of_week,
            pedestrian_waiting=np.zeros(4, dtype=bool),
            pedestrian_wait_times=np.zeros(4, dtype=np.float32),
            neighbor_embeddings=np.zeros(
                (self.state_cfg.max_neighbors, self.state_cfg.embed_dim),
                dtype=np.float32,
            ),
        )

    def _compute_reward(self, state_before: TrafficState, state_after: TrafficState) -> float:
        w = self.reward_weights

        # Pressure: imbalance between upstream and downstream queues
        pressure = float(np.sum(state_after.queue_lengths))
        r_pressure = -abs(pressure) / max(self.state_cfg.queue_length_max, 1)

        # Queue reduction
        r_queue = float(
            np.sum(state_before.queue_lengths) - np.sum(state_after.queue_lengths)
        ) / max(self.state_cfg.queue_length_max, 1)

        # Throughput (vehicles that left the simulation in this interval)
        r_throughput = float(traci.simulation.getArrivedNumber()) / 10.0

        # Excessive waiting penalty
        wait_thresh = 120.0
        r_wait = (
            -float(np.sum(np.maximum(0, state_after.waiting_times - wait_thresh))) / wait_thresh
        )

        # Pedestrian penalty
        ped_thresh = 45.0
        r_ped = -float(np.sum(state_after.pedestrian_wait_times > ped_thresh))

        reward = (
            w.get("pressure", 0.4) * r_pressure
            + w.get("queue", 0.25) * r_queue
            + w.get("throughput", 0.2) * r_throughput
            + w.get("wait", 0.1) * r_wait
            + w.get("pedestrian", 0.05) * r_ped
        )
        return float(reward)

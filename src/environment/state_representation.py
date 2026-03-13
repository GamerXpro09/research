"""State representation utilities for FedNS-Traffic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class TrafficState:
    """
    Unified traffic state observed by one intersection agent.

    All per-lane arrays are ordered as:
        [N_straight, N_left, N_right, S_straight, S_left, S_right,
         E_straight, E_left, E_right, W_straight, W_left, W_right]
    (padded with zeros when fewer lanes are present).
    """

    # Per-lane features
    queue_lengths: np.ndarray  # vehicles waiting  (int, per lane)
    vehicle_speeds: np.ndarray  # avg speed km/h    (float, per lane)
    waiting_times: np.ndarray  # avg wait seconds  (float, per lane)
    vehicle_counts: np.ndarray  # total in det. zone (int, per lane)

    # Intersection-level features
    current_phase: int  # phase index
    num_phases: int  # total number of phases
    phase_duration: float  # seconds in current phase
    time_of_day: float  # normalised [0, 1]
    day_of_week: int  # 0 = Monday … 6 = Sunday

    # Pedestrian features
    pedestrian_waiting: np.ndarray  # bool per crosswalk
    pedestrian_wait_times: np.ndarray  # seconds per crosswalk

    # Coordination (neighbour embeddings received via MQTT)
    neighbor_embeddings: np.ndarray  # shape (K, embed_dim)

    # Optional extras
    emergency_vehicle_detected: bool = False
    emergency_vehicle_direction: Optional[str] = None
    neighbor_influence: float = 0.0


@dataclass
class StateConfig:
    """Hyper-parameters controlling the state representation."""

    num_lanes: int = 12
    num_phases: int = 8
    embed_dim: int = 16
    max_neighbors: int = 4
    queue_length_max: float = 50.0
    speed_max: float = 60.0
    wait_time_max: float = 300.0

    @property
    def flat_dim(self) -> int:
        """Dimension of the flattened, normalised observation vector."""
        per_lane = 4  # queue, speed, wait, count
        phase_enc = self.num_phases  # one-hot
        time_feats = 1 + 7  # time_of_day + day_of_week (one-hot)
        phase_dur = 1
        ped = 2 * 4  # wait bool + wait time for up to 4 crosswalks
        neighbors = self.max_neighbors * self.embed_dim
        return self.num_lanes * per_lane + phase_enc + time_feats + phase_dur + ped + neighbors


class StateEncoder:
    """
    Encodes a :class:`TrafficState` into a flat, normalised numpy array
    suitable for neural-network input.
    """

    def __init__(self, config: StateConfig):
        self.cfg = config

    def encode(self, state: TrafficState) -> np.ndarray:
        """Return a 1-D float32 array of dimension ``config.flat_dim``."""
        parts: List[np.ndarray] = []

        # Per-lane features (normalised)
        parts.append(self._pad(state.queue_lengths / self.cfg.queue_length_max, self.cfg.num_lanes))
        parts.append(self._pad(state.vehicle_speeds / self.cfg.speed_max, self.cfg.num_lanes))
        parts.append(self._pad(state.waiting_times / self.cfg.wait_time_max, self.cfg.num_lanes))
        parts.append(
            self._pad(state.vehicle_counts / self.cfg.queue_length_max, self.cfg.num_lanes)
        )

        # Phase one-hot
        phase_oh = np.zeros(self.cfg.num_phases, dtype=np.float32)
        phase_oh[min(state.current_phase, self.cfg.num_phases - 1)] = 1.0
        parts.append(phase_oh)

        # Phase duration (normalised)
        parts.append(np.array([state.phase_duration / 90.0], dtype=np.float32))

        # Time of day
        parts.append(np.array([state.time_of_day], dtype=np.float32))

        # Day of week one-hot
        dow_oh = np.zeros(7, dtype=np.float32)
        dow_oh[state.day_of_week % 7] = 1.0
        parts.append(dow_oh)

        # Pedestrian features (up to 4 crosswalks)
        ped_bool = self._pad(state.pedestrian_waiting.astype(np.float32), 4)
        ped_wait = self._pad(state.pedestrian_wait_times / self.cfg.wait_time_max, 4)
        parts.append(ped_bool)
        parts.append(ped_wait)

        # Neighbour embeddings (padded / trimmed to max_neighbors)
        nb_emb = self._pad_2d(state.neighbor_embeddings, self.cfg.max_neighbors)
        parts.append(nb_emb.ravel())

        return np.concatenate(parts).astype(np.float32)

    @staticmethod
    def _pad(arr: np.ndarray, target: int) -> np.ndarray:
        """1-D zero-pad or truncate *arr* to *target* length."""
        arr = np.asarray(arr, dtype=np.float32).ravel()
        if len(arr) >= target:
            return arr[:target]
        return np.pad(arr, (0, target - len(arr)))

    @staticmethod
    def _pad_2d(arr: np.ndarray, target_rows: int) -> np.ndarray:
        """2-D zero-pad or truncate *arr* to (*target_rows*, cols)."""
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        rows, cols = arr.shape
        if rows >= target_rows:
            return arr[:target_rows]
        pad = np.zeros((target_rows - rows, cols), dtype=np.float32)
        return np.vstack([arr, pad])

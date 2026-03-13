"""
Traffic signal control baselines for comparison with learned agents.

Fixed-Timing Baseline
---------------------
Cycles through signal phases with fixed durations determined by a pre-set
cycle length and split ratios. Represents classical signal timing (Webster).

Max-Pressure Baseline
---------------------
At each decision point, activates the phase with the highest "pressure"
(upstream queue minus downstream queue) following Varaiya (2013).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from src.environment.state_representation import TrafficState


class FixedTimingBaseline:
    """
    Fixed-cycle traffic signal controller (deterministic, no learning).

    Parameters
    ----------
    num_phases : int
        Total number of signal phases at the intersection.
    cycle_length : int
        Total cycle length in seconds.
    phase_splits : list[float] | None
        Fraction of the cycle allocated to each phase.
        Must sum to 1.0.  If *None*, equal splits are used.
    yellow_time : int
        Yellow phase duration in seconds (not counted against splits).
    min_green : int
        Minimum green time per phase in seconds.
    """

    def __init__(
        self,
        num_phases: int = 4,
        cycle_length: int = 90,
        phase_splits: Optional[List[float]] = None,
        yellow_time: int = 3,
        min_green: int = 5,
    ):
        self.num_phases = num_phases
        self.cycle_length = cycle_length
        self.yellow_time = yellow_time
        self.min_green = min_green

        if phase_splits is None:
            phase_splits = [1.0 / num_phases] * num_phases
        assert abs(sum(phase_splits) - 1.0) < 1e-6, "phase_splits must sum to 1.0"
        assert len(phase_splits) == num_phases

        self.phase_durations: List[int] = [
            max(min_green, round(cycle_length * s)) for s in phase_splits
        ]

        self._step = 0
        self._current_phase = 0
        self._time_in_phase = 0

    def select_action(self, state: TrafficState) -> int:
        """Return the current phase; advance schedule by one step."""
        action = self._current_phase
        self._time_in_phase += 1
        if self._time_in_phase >= self.phase_durations[self._current_phase]:
            self._current_phase = (self._current_phase + 1) % self.num_phases
            self._time_in_phase = 0
        self._step += 1
        return action

    def reset(self):
        self._step = 0
        self._current_phase = 0
        self._time_in_phase = 0


class MaxPressureBaseline:
    """
    Max-Pressure controller (Varaiya, 2013).

    At each decision point it activates the phase that maximises
    Σ |queue_upstream − queue_downstream| for movements in that phase.

    Parameters
    ----------
    phase_movements : dict[int, list[int]]
        Mapping from phase index to list of *lane indices* served.
        Upstream lanes should appear first; the next lane in the road
        direction is treated as the downstream lane.
    num_phases : int
    min_green : int
    max_green : int
    """

    def __init__(
        self,
        phase_movements: Optional[Dict[int, List[int]]] = None,
        num_phases: int = 4,
        min_green: int = 5,
        max_green: int = 90,
    ):
        self.num_phases = num_phases
        self.min_green = min_green
        self.max_green = max_green

        # Default: assign equal groups of lanes to phases
        if phase_movements is None:
            phase_movements = {i: [] for i in range(num_phases)}
        self.phase_movements = phase_movements

        self._current_phase = 0
        self._time_in_phase = 0

    def select_action(self, state: TrafficState) -> int:
        """Select the max-pressure phase, respecting min-green constraint."""
        if self._time_in_phase < self.min_green:
            # Must stay in current phase
            self._time_in_phase += 1
            return self._current_phase

        if self._time_in_phase >= self.max_green:
            # Force a phase change
            pressures = self._compute_pressures(state)
            pressures[self._current_phase] = -np.inf  # exclude current
            best_phase = int(np.argmax(pressures))
        else:
            pressures = self._compute_pressures(state)
            best_phase = int(np.argmax(pressures))

        if best_phase != self._current_phase:
            self._current_phase = best_phase
            self._time_in_phase = 0
        else:
            self._time_in_phase += 1

        return self._current_phase

    def reset(self):
        self._current_phase = 0
        self._time_in_phase = 0

    def _compute_pressures(self, state: TrafficState) -> np.ndarray:
        """Compute pressure for each phase."""
        pressures = np.zeros(self.num_phases, dtype=np.float32)
        n = len(state.queue_lengths)

        for phase_id, lane_indices in self.phase_movements.items():
            total = 0.0
            for lane_idx in lane_indices:
                if lane_idx < n:
                    # Pressure = queue on upstream lane
                    # (downstream implicitly zero when no queue info available)
                    total += float(state.queue_lengths[lane_idx])
            pressures[phase_id] = total

        # Fallback: if no movement mapping, use total queue per group of lanes
        if all(len(v) == 0 for v in self.phase_movements.values()):
            lanes_per_phase = max(1, n // self.num_phases)
            for phase_id in range(self.num_phases):
                start = phase_id * lanes_per_phase
                end = min(start + lanes_per_phase, n)
                pressures[phase_id] = float(np.sum(state.queue_lengths[start:end]))

        return pressures

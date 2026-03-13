"""
Unit tests for FedNS-Traffic environment components (Phase 1).

These tests run without SUMO or CityFlow installed by exercising the
state-representation and environment logic with mock/synthetic data.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.environment.state_representation import (
    StateConfig,
    StateEncoder,
    TrafficState,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_state(
    num_lanes: int = 4,
    num_phases: int = 8,
    phase: int = 0,
    phase_duration: float = 10.0,
) -> TrafficState:
    return TrafficState(
        queue_lengths=np.random.randint(0, 20, size=num_lanes).astype(float),
        vehicle_speeds=np.random.uniform(0, 50, size=num_lanes),
        waiting_times=np.random.uniform(0, 120, size=num_lanes),
        vehicle_counts=np.random.randint(0, 30, size=num_lanes).astype(float),
        current_phase=phase,
        num_phases=num_phases,
        phase_duration=phase_duration,
        time_of_day=0.5,
        day_of_week=1,
        pedestrian_waiting=np.zeros(4, dtype=bool),
        pedestrian_wait_times=np.zeros(4, dtype=np.float32),
        neighbor_embeddings=np.zeros((4, 16), dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# StateConfig
# ---------------------------------------------------------------------------


class TestStateConfig:
    def test_flat_dim_default(self):
        cfg = StateConfig()
        # 12 lanes × 4 features + 8 phases + 1 phase_dur + 1 time_of_day
        # + 7 day_of_week + 4 ped_bool + 4 ped_wait + 4×16 neighbors = 121
        expected = 12 * 4 + 8 + 1 + 1 + 7 + 4 + 4 + 4 * 16
        assert cfg.flat_dim == expected

    def test_flat_dim_custom(self):
        cfg = StateConfig(num_lanes=8, num_phases=4, max_neighbors=2, embed_dim=8)
        expected = 8 * 4 + 4 + 1 + 1 + 7 + 4 + 4 + 2 * 8
        assert cfg.flat_dim == expected

    def test_flat_dim_positive(self):
        cfg = StateConfig()
        assert cfg.flat_dim > 0


# ---------------------------------------------------------------------------
# StateEncoder
# ---------------------------------------------------------------------------


class TestStateEncoder:
    def test_output_shape(self):
        cfg = StateConfig()
        encoder = StateEncoder(cfg)
        state = make_state()
        obs = encoder.encode(state)
        assert obs.shape == (cfg.flat_dim,)

    def test_output_dtype(self):
        encoder = StateEncoder(StateConfig())
        obs = encoder.encode(make_state())
        assert obs.dtype == np.float32

    def test_output_range(self):
        """Most encoded values should be in [0, 1] when inputs are in range."""
        cfg = StateConfig()
        encoder = StateEncoder(cfg)
        state = TrafficState(
            queue_lengths=np.array([10.0, 5.0, 0.0, 20.0]),
            vehicle_speeds=np.array([30.0, 40.0, 0.0, 50.0]),
            waiting_times=np.array([60.0, 30.0, 0.0, 90.0]),
            vehicle_counts=np.array([8.0, 4.0, 0.0, 12.0]),
            current_phase=2,
            num_phases=8,
            phase_duration=15.0,
            time_of_day=0.75,
            day_of_week=3,
            pedestrian_waiting=np.zeros(4, dtype=bool),
            pedestrian_wait_times=np.zeros(4, dtype=np.float32),
            neighbor_embeddings=np.zeros((4, 16), dtype=np.float32),
        )
        obs = encoder.encode(state)
        # Per-lane features and phase-duration should be clipped to [0, 1]
        assert np.all(obs >= -0.1), "Unexpected large negative values"

    def test_phase_one_hot(self):
        cfg = StateConfig(num_phases=4)
        encoder = StateEncoder(cfg)
        for phase in range(4):
            state = make_state(num_phases=4, phase=phase)
            obs = encoder.encode(state)
            # One-hot section starts after 4-lane × 4-feature = 16 floats
            # (with num_lanes=12 default, but here state has 4 lanes)
            # We only check the phase via round-trip: encode then check argmax
            # in the phase slice
            phase_start = cfg.num_lanes * 4  # 12 * 4 = 48
            phase_slice = obs[phase_start : phase_start + cfg.num_phases]
            assert int(np.argmax(phase_slice)) == phase

    def test_padding_short_lanes(self):
        """State with fewer lanes than StateConfig.num_lanes should be padded."""
        cfg = StateConfig(num_lanes=12)
        encoder = StateEncoder(cfg)
        state = make_state(num_lanes=3)  # fewer lanes
        obs = encoder.encode(state)
        assert obs.shape == (cfg.flat_dim,)

    def test_truncation_many_lanes(self):
        """State with more lanes than StateConfig.num_lanes should be truncated."""
        cfg = StateConfig(num_lanes=4)
        encoder = StateEncoder(cfg)
        state = make_state(num_lanes=20)
        obs = encoder.encode(state)
        assert obs.shape == (cfg.flat_dim,)

    def test_reproducibility(self):
        """Same state → same encoding."""
        encoder = StateEncoder(StateConfig())
        state = make_state(num_lanes=4)
        obs1 = encoder.encode(state)
        obs2 = encoder.encode(state)
        np.testing.assert_array_equal(obs1, obs2)

    def test_neighbor_embedding_zero(self):
        """Zero neighbour embeddings should encode cleanly."""
        cfg = StateConfig(max_neighbors=4, embed_dim=16)
        encoder = StateEncoder(cfg)
        state = make_state()
        obs = encoder.encode(state)
        assert np.isfinite(obs).all()


# ---------------------------------------------------------------------------
# Environment reward logic (stateless helper)
# ---------------------------------------------------------------------------


class TestRewardComputation:
    """
    Test reward computation in isolation (no SUMO/TraCI required).
    Instantiates a minimal reward helper to avoid SUMO dependency.
    """

    def _compute_reward(self, before: TrafficState, after: TrafficState) -> float:
        w = {"pressure": 0.40, "queue": 0.25, "throughput": 0.20, "wait": 0.10, "pedestrian": 0.05}
        q_max = 50.0
        pressure = float(np.sum(after.queue_lengths))
        r_pressure = -abs(pressure) / q_max
        r_queue = float(np.sum(before.queue_lengths) - np.sum(after.queue_lengths)) / q_max
        r_throughput = 0.0
        wait_thresh = 120.0
        r_wait = -float(np.sum(np.maximum(0, after.waiting_times - wait_thresh))) / wait_thresh
        r_ped = 0.0
        return (
            w["pressure"] * r_pressure
            + w["queue"] * r_queue
            + w["throughput"] * r_throughput
            + w["wait"] * r_wait
            + w["pedestrian"] * r_ped
        )

    def test_reward_decreasing_queues(self):
        """Reward should be higher when queues decrease."""
        before = make_state(num_lanes=4)
        before.queue_lengths = np.array([10.0, 10.0, 10.0, 10.0])
        after_good = make_state(num_lanes=4)
        after_good.queue_lengths = np.array([5.0, 5.0, 5.0, 5.0])
        after_bad = make_state(num_lanes=4)
        after_bad.queue_lengths = np.array([15.0, 15.0, 15.0, 15.0])

        r_good = self._compute_reward(before, after_good)
        r_bad = self._compute_reward(before, after_bad)
        assert r_good > r_bad

    def test_reward_is_finite(self):
        before = make_state()
        after = make_state()
        reward = self._compute_reward(before, after)
        assert np.isfinite(reward)


# ---------------------------------------------------------------------------
# SumoEnvironment import / instantiation guard
# ---------------------------------------------------------------------------


class TestSumoEnvironmentImport:
    def test_import_without_sumo(self):
        """SumoEnvironment should be importable even without SUMO installed."""
        from src.environment.sumo_env import SumoEnvironment  # noqa: F401

    def test_reset_raises_without_sumo(self, monkeypatch):
        """_start_sumo() should raise ImportError when SUMO is not available."""
        import src.environment.sumo_env as sumo_mod
        from src.environment.sumo_env import SumoEnvironment

        monkeypatch.setattr(sumo_mod, "_SUMO_AVAILABLE", False)
        env = SumoEnvironment.__new__(SumoEnvironment)
        with pytest.raises(ImportError, match="SUMO"):
            env._start_sumo()


# ---------------------------------------------------------------------------
# CityFlowEnvironment import guard
# ---------------------------------------------------------------------------


class TestCityFlowEnvironmentImport:
    def test_import_without_cityflow(self):
        """CityFlowEnvironment should be importable even without CityFlow."""
        from src.environment.cityflow_env import CityFlowEnvironment  # noqa: F401

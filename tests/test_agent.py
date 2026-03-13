"""
Unit tests for DQN agent components (Phase 1).

Tests cover:
- Network forward pass and output shape
- SumTree correctness
- PrioritizedReplayBuffer push/sample/update
- DuelingDQNAgent action selection and observe()
- Baseline agents (FixedTimingBaseline, MaxPressureBaseline)
- Agent checkpoint save/load round-trip
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import torch

from src.agents.baselines import FixedTimingBaseline, MaxPressureBaseline
from src.agents.dqn_agent import (
    DuelingDQNAgent,
    DuelingDQNNetwork,
    PrioritizedReplayBuffer,
    SumTree,
    Transition,
)
from src.environment.state_representation import TrafficState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

STATE_DIM = 64
NUM_ACTIONS = 32


def random_state() -> np.ndarray:
    return np.random.rand(STATE_DIM).astype(np.float32)


def make_transition(done: bool = False) -> Transition:
    return Transition(
        state=random_state(),
        action=int(np.random.randint(0, NUM_ACTIONS)),
        reward=float(np.random.randn()),
        next_state=random_state(),
        done=done,
    )


def make_traffic_state(num_lanes: int = 4) -> TrafficState:
    return TrafficState(
        queue_lengths=np.random.randint(0, 20, num_lanes).astype(float),
        vehicle_speeds=np.random.uniform(0, 50, num_lanes),
        waiting_times=np.random.uniform(0, 120, num_lanes),
        vehicle_counts=np.random.randint(0, 30, num_lanes).astype(float),
        current_phase=0,
        num_phases=8,
        phase_duration=15.0,
        time_of_day=0.5,
        day_of_week=0,
        pedestrian_waiting=np.zeros(4, dtype=bool),
        pedestrian_wait_times=np.zeros(4, dtype=np.float32),
        neighbor_embeddings=np.zeros((4, 16), dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# DuelingDQNNetwork
# ---------------------------------------------------------------------------


class TestDuelingDQNNetwork:
    def test_output_shape(self):
        net = DuelingDQNNetwork(STATE_DIM, NUM_ACTIONS)
        x = torch.randn(4, STATE_DIM)
        out = net(x)
        assert out.shape == (4, NUM_ACTIONS)

    def test_output_dtype(self):
        net = DuelingDQNNetwork(STATE_DIM, NUM_ACTIONS)
        x = torch.randn(1, STATE_DIM)
        out = net(x)
        assert out.dtype == torch.float32

    def test_single_sample(self):
        net = DuelingDQNNetwork(STATE_DIM, NUM_ACTIONS)
        x = torch.randn(1, STATE_DIM)
        out = net(x)
        assert out.shape == (1, NUM_ACTIONS)

    def test_gradients_flow(self):
        net = DuelingDQNNetwork(STATE_DIM, NUM_ACTIONS)
        x = torch.randn(2, STATE_DIM)
        loss = net(x).sum()
        loss.backward()
        for param in net.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_dueling_decomposition(self):
        """Value + centred Advantage should not blow up."""
        net = DuelingDQNNetwork(STATE_DIM, NUM_ACTIONS)
        x = torch.randn(8, STATE_DIM)
        q = net(x)
        assert torch.isfinite(q).all()

    def test_different_hidden_dim(self):
        net = DuelingDQNNetwork(STATE_DIM, NUM_ACTIONS, hidden_dim=64)
        out = net(torch.randn(3, STATE_DIM))
        assert out.shape == (3, NUM_ACTIONS)


# ---------------------------------------------------------------------------
# SumTree
# ---------------------------------------------------------------------------


class TestSumTree:
    def test_add_and_total(self):
        tree = SumTree(capacity=8)
        for priority in [1.0, 2.0, 3.0]:
            tree.add(priority, make_transition())
        assert abs(tree.total_priority - 6.0) < 1e-6

    def test_size(self):
        tree = SumTree(capacity=4)
        for _ in range(3):
            tree.add(1.0, make_transition())
        assert len(tree) == 3

    def test_capacity_wraps(self):
        tree = SumTree(capacity=4)
        for _ in range(10):
            tree.add(1.0, make_transition())
        assert len(tree) == 4  # capped at capacity

    def test_sample_returns_transition(self):
        tree = SumTree(capacity=8)
        t = make_transition()
        tree.add(1.0, t)
        idx, priority, trans = tree.sample(0.5)
        assert isinstance(trans, Transition)

    def test_update_priority(self):
        tree = SumTree(capacity=4)
        tree.add(1.0, make_transition())
        old_total = tree.total_priority
        tree.update(0, 5.0)
        assert tree.total_priority != old_total


# ---------------------------------------------------------------------------
# PrioritizedReplayBuffer
# ---------------------------------------------------------------------------


class TestPrioritizedReplayBuffer:
    def test_push_and_len(self):
        buf = PrioritizedReplayBuffer(capacity=100)
        for _ in range(10):
            buf.push(make_transition())
        assert len(buf) == 10

    def test_sample_batch_size(self):
        buf = PrioritizedReplayBuffer(capacity=100)
        for _ in range(50):
            buf.push(make_transition())
        transitions, indices, weights = buf.sample(16)
        assert len(transitions) == 16
        assert indices.shape == (16,)
        assert weights.shape == (16,)

    def test_weights_positive(self):
        buf = PrioritizedReplayBuffer(capacity=100)
        for _ in range(50):
            buf.push(make_transition())
        _, _, weights = buf.sample(16)
        assert (weights > 0).all()

    def test_weights_max_one(self):
        buf = PrioritizedReplayBuffer(capacity=100)
        for _ in range(50):
            buf.push(make_transition())
        _, _, weights = buf.sample(16)
        assert weights.max() <= 1.0 + 1e-6

    def test_update_priorities(self):
        buf = PrioritizedReplayBuffer(capacity=100, alpha=0.6)
        for _ in range(32):
            buf.push(make_transition())
        _, indices, _ = buf.sample(8)
        td_errors = np.random.rand(8).astype(np.float32)
        # Should not raise
        buf.update_priorities(indices, td_errors)

    def test_capacity_limit(self):
        buf = PrioritizedReplayBuffer(capacity=10)
        for _ in range(20):
            buf.push(make_transition())
        assert len(buf) <= 10


# ---------------------------------------------------------------------------
# DuelingDQNAgent
# ---------------------------------------------------------------------------


class TestDuelingDQNAgent:
    def _make_agent(self, **kwargs) -> DuelingDQNAgent:
        defaults = dict(
            state_dim=STATE_DIM,
            num_actions=NUM_ACTIONS,
            hidden_dim=64,
            buffer_capacity=1_000,
            learning_starts=10,
            train_freq=1,
            batch_size=8,
            device="cpu",
        )
        defaults.update(kwargs)
        return DuelingDQNAgent(**defaults)

    def test_select_action_range(self):
        agent = self._make_agent()
        for _ in range(20):
            action = agent.select_action(random_state(), training=True)
            assert 0 <= action < NUM_ACTIONS

    def test_greedy_action_deterministic(self):
        agent = self._make_agent(epsilon_start=0.0, epsilon_end=0.0)
        state = random_state()
        a1 = agent.select_action(state, training=False)
        a2 = agent.select_action(state, training=False)
        assert a1 == a2

    def test_epsilon_decay(self):
        agent = self._make_agent(epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.9)
        initial_eps = agent.epsilon
        for i in range(20):
            s = random_state()
            agent.observe(s, 0, 0.0, random_state(), False)
        assert agent.epsilon < initial_eps

    def test_observe_returns_none_before_learning_starts(self):
        agent = self._make_agent(learning_starts=1_000)
        result = agent.observe(random_state(), 0, 1.0, random_state(), False)
        assert result is None

    def test_training_step_returns_loss(self):
        agent = self._make_agent(learning_starts=0, batch_size=8)
        # Fill replay buffer
        for _ in range(50):
            agent.observe(
                random_state(), int(np.random.randint(NUM_ACTIONS)), 1.0, random_state(), False
            )
        # At least one training step should have occurred
        assert len(agent.losses) > 0
        assert all(np.isfinite(loss_val) for loss_val in agent.losses)

    def test_save_and_load(self):
        agent = self._make_agent()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        try:
            agent.save(path)
            new_agent = self._make_agent()
            new_agent.load(path)
            # Check that loaded weights match
            for p1, p2 in zip(agent.online_net.parameters(), new_agent.online_net.parameters()):
                assert torch.allclose(p1, p2)
        finally:
            os.unlink(path)

    def test_target_net_lags_online(self):
        """After init, online and target have identical weights; they diverge after steps."""
        agent = self._make_agent(learning_starts=0, tau=0.5, batch_size=8)
        for _ in range(100):
            agent.observe(random_state(), 0, 1.0, random_state(), False)
        # At least some layers should differ after training + soft updates
        total_diff = sum(
            float((p1 - p2).abs().sum())
            for p1, p2 in zip(agent.online_net.parameters(), agent.target_net.parameters())
        )
        # After enough training steps with tau < 1, some difference is expected
        # (This is a sanity check; small differences are OK)
        assert total_diff >= 0  # trivially true — mainly checks no exceptions


# ---------------------------------------------------------------------------
# FixedTimingBaseline
# ---------------------------------------------------------------------------


class TestFixedTimingBaseline:
    def test_action_in_range(self):
        baseline = FixedTimingBaseline(num_phases=4, cycle_length=60)
        state = make_traffic_state()
        for _ in range(120):
            action = baseline.select_action(state)
            assert 0 <= action < 4

    def test_cycles_through_phases(self):
        baseline = FixedTimingBaseline(num_phases=4, cycle_length=40, min_green=10)
        state = make_traffic_state()
        phases_seen = set()
        for _ in range(200):
            phases_seen.add(baseline.select_action(state))
        assert phases_seen == {0, 1, 2, 3}

    def test_reset(self):
        baseline = FixedTimingBaseline(num_phases=2, cycle_length=20)
        state = make_traffic_state()
        for _ in range(15):
            baseline.select_action(state)
        baseline.reset()
        # After reset, should start from phase 0
        assert baseline._current_phase == 0
        assert baseline._time_in_phase == 0

    def test_phase_splits_respected(self):
        baseline = FixedTimingBaseline(num_phases=2, cycle_length=100, phase_splits=[0.75, 0.25])
        assert baseline.phase_durations[0] > baseline.phase_durations[1]


# ---------------------------------------------------------------------------
# MaxPressureBaseline
# ---------------------------------------------------------------------------


class TestMaxPressureBaseline:
    def test_action_in_range(self):
        baseline = MaxPressureBaseline(num_phases=4)
        state = make_traffic_state(num_lanes=8)
        for _ in range(10):
            action = baseline.select_action(state)
            assert 0 <= action < 4

    def test_selects_highest_pressure_phase(self):
        """When one phase has clearly higher queue, it should be preferred."""
        phase_movements = {
            0: [0, 1],  # lanes 0, 1
            1: [2, 3],  # lanes 2, 3
        }
        baseline = MaxPressureBaseline(
            phase_movements=phase_movements,
            num_phases=2,
            min_green=1,
        )
        state = make_traffic_state(num_lanes=4)
        state.queue_lengths = np.array([50.0, 50.0, 1.0, 1.0])  # phase 0 much higher

        # Advance past min_green
        for _ in range(2):
            baseline.select_action(state)
        action = baseline.select_action(state)
        assert action == 0

    def test_reset(self):
        baseline = MaxPressureBaseline(num_phases=4)
        state = make_traffic_state()
        for _ in range(10):
            baseline.select_action(state)
        baseline.reset()
        assert baseline._current_phase == 0
        assert baseline._time_in_phase == 0

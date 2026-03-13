"""
Dueling DQN agent with Prioritized Experience Replay (PER).

Architecture
------------
  Input → Feature Encoder → (split) → Value stream  → V(s)
                                    → Advantage stream → A(s,a)
  Q(s,a) = V(s) + A(s,a) − mean_a A(s,a)

References
----------
Wang et al. (2016). "Dueling Network Architectures for Deep Reinforcement Learning."
Schaul et al. (2016). "Prioritized Experience Replay."
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ---------------------------------------------------------------------------
# Neural network
# ---------------------------------------------------------------------------


class DuelingDQNNetwork(nn.Module):
    """
    Dueling DQN network.

    Parameters
    ----------
    state_dim : int
        Dimensionality of the (flat) input observation.
    num_actions : int
        Number of discrete actions.
    hidden_dim : int
        Width of fully-connected hidden layers.
    """

    def __init__(self, state_dim: int, num_actions: int, hidden_dim: int = 128):
        super().__init__()

        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Value stream  →  scalar V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Advantage stream  →  A(s, a)  for each action
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_layer(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        # Dueling combination: Q = V + (A − mean(A))
        q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        return q_values


# ---------------------------------------------------------------------------
# Prioritised Experience Replay buffer
# ---------------------------------------------------------------------------


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class SumTree:
    """
    Binary sum-tree for O(log n) priority updates and sampling.
    Stores *capacity* leaf priorities; internal nodes store sum of subtree.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._tree = np.zeros(2 * capacity, dtype=np.float32)
        self._data: List[Optional[Transition]] = [None] * capacity
        self._write_ptr = 0
        self._size = 0

    # ---- public API ----

    def add(self, priority: float, transition: Transition):
        idx = self._write_ptr + self.capacity
        self._data[self._write_ptr] = transition
        self._update(idx, priority)
        self._write_ptr = (self._write_ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def update(self, idx: int, priority: float):
        """Update priority for leaf *idx* (index in [0, capacity))."""
        self._update(idx + self.capacity, priority)

    def sample(self, value: float) -> Tuple[int, float, Transition]:
        """
        Retrieve the leaf whose cumulative priority range contains *value*.
        Returns (leaf_idx, priority, transition).
        """
        idx = self._retrieve(1, value)
        leaf_idx = idx - self.capacity
        return leaf_idx, self._tree[idx], self._data[leaf_idx]  # type: ignore[return-value]

    @property
    def total_priority(self) -> float:
        return float(self._tree[1])

    def __len__(self) -> int:
        return self._size

    # ---- private ----

    def _update(self, idx: int, priority: float):
        change = priority - self._tree[idx]
        self._tree[idx] = priority
        while idx > 1:
            idx //= 2
            self._tree[idx] += change

    def _retrieve(self, idx: int, value: float) -> int:
        left = 2 * idx
        if left >= len(self._tree):
            return idx
        if value <= self._tree[left]:
            return self._retrieve(left, value)
        return self._retrieve(left + 1, value - self._tree[left])


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer (PER).

    Parameters
    ----------
    capacity : int
    alpha : float
        Priority exponent (0 = uniform, 1 = full prioritisation).
    beta_start : float
        Initial importance-sampling (IS) weight exponent.
    beta_frames : int
        Number of env frames over which beta is annealed to 1.0.
    epsilon : float
        Small constant added to priorities to avoid zero probability.
    """

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100_000,
        epsilon: float = 1e-6,
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.epsilon = epsilon
        self._tree = SumTree(capacity)
        self._frame = 0
        self._max_priority: float = 1.0

    def push(self, transition: Transition):
        self._tree.add(self._max_priority**self.alpha, transition)

    def sample(self, batch_size: int) -> Tuple[List[Transition], np.ndarray, np.ndarray]:
        """
        Sample a batch.

        Returns
        -------
        transitions : list of Transition
        indices : np.ndarray of shape (batch_size,) — leaf indices for updates
        weights : np.ndarray of shape (batch_size,) — IS weights
        """
        self._frame += 1
        beta = min(
            1.0,
            self.beta_start + (1.0 - self.beta_start) * self._frame / self.beta_frames,
        )

        segment = self._tree.total_priority / batch_size
        transitions, indices, priorities = [], [], []

        for i in range(batch_size):
            lo, hi = segment * i, segment * (i + 1)
            value = random.uniform(lo, hi)
            idx, priority, trans = self._tree.sample(value)
            transitions.append(trans)
            indices.append(idx)
            priorities.append(priority)

        # Importance-sampling weights
        probs = np.array(priorities) / self._tree.total_priority
        probs = np.maximum(probs, 1e-10)
        min_prob = np.min(probs)
        weights = (probs / min_prob) ** (-beta)
        weights = weights / weights.max()

        return transitions, np.array(indices, dtype=np.int64), weights.astype(np.float32)

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        priorities = (np.abs(td_errors) + self.epsilon) ** self.alpha
        for idx, prio in zip(indices, priorities):
            self._tree.update(int(idx), float(prio))
            self._max_priority = max(self._max_priority, float(prio))

    def __len__(self) -> int:
        return len(self._tree)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class DuelingDQNAgent:
    """
    Dueling DQN agent with PER and soft target-network updates.

    Parameters
    ----------
    state_dim : int
    num_actions : int
    hidden_dim : int
    lr : float
    gamma : float
        Discount factor.
    tau : float
        Soft-update coefficient for target network.
    batch_size : int
    buffer_capacity : int
    per_alpha, per_beta_start, per_beta_frames, per_epsilon : float
        PER hyper-parameters.
    epsilon_start, epsilon_end, epsilon_decay : float
        ε-greedy exploration schedule.
    learning_starts : int
        Steps before training begins.
    train_freq : int
        Train every *train_freq* env steps.
    target_update_freq : int
        Hard-copy target network every *target_update_freq* steps.
    device : str
        ``"auto"``, ``"cpu"``, or ``"cuda"``.
    """

    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        hidden_dim: int = 128,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        batch_size: int = 64,
        buffer_capacity: int = 100_000,
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        per_beta_frames: int = 100_000,
        per_epsilon: float = 1e-6,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        learning_starts: int = 1_000,
        train_freq: int = 4,
        target_update_freq: int = 1_000,
        device: str = "auto",
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.state_dim = state_dim
        self.num_actions = num_actions
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.target_update_freq = target_update_freq

        # Networks
        self.online_net = DuelingDQNNetwork(state_dim, num_actions, hidden_dim).to(self.device)
        self.target_net = DuelingDQNNetwork(state_dim, num_actions, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)

        # Replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=buffer_capacity,
            alpha=per_alpha,
            beta_start=per_beta_start,
            beta_frames=per_beta_frames,
            epsilon=per_epsilon,
        )

        # Exploration
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Counters
        self._step = 0
        self.losses: List[float] = []

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """ε-greedy action selection."""
        if training and random.random() < self.epsilon:
            return int(random.randrange(self.num_actions))
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.online_net(state_t)
            return int(q_values.argmax(dim=1).item())

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def observe(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> Optional[float]:
        """
        Store a transition and (possibly) perform a training step.

        Returns
        -------
        loss : float | None
            Training loss if a gradient step was taken, else ``None``.
        """
        self.replay_buffer.push(
            Transition(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
            )
        )
        self._step += 1

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        if self._step < self.learning_starts:
            return None
        if len(self.replay_buffer) < self.batch_size:
            return None
        if self._step % self.train_freq != 0:
            return None

        loss = self._train_step()

        # Soft target update
        self._soft_update()

        return loss

    def _train_step(self) -> float:
        transitions, indices, weights = self.replay_buffer.sample(self.batch_size)

        states = torch.FloatTensor(np.array([t.state for t in transitions])).to(self.device)
        actions = torch.LongTensor(np.array([t.action for t in transitions])).to(self.device)
        rewards = torch.FloatTensor(np.array([t.reward for t in transitions])).to(self.device)
        next_states = torch.FloatTensor(np.array([t.next_state for t in transitions])).to(
            self.device
        )
        dones = torch.FloatTensor(np.array([float(t.done) for t in transitions])).to(self.device)
        weights_t = torch.FloatTensor(weights).to(self.device)

        # Double DQN target
        with torch.no_grad():
            next_actions = self.online_net(next_states).argmax(dim=1)
            next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            targets = rewards + self.gamma * next_q * (1.0 - dones)

        current_q = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        td_errors = (targets - current_q).detach().cpu().numpy()

        # Huber loss with IS weights
        loss = (weights_t * F.huber_loss(current_q, targets, reduction="none")).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Update priorities
        self.replay_buffer.update_priorities(indices, td_errors)

        loss_val = float(loss.item())
        self.losses.append(loss_val)
        return loss_val

    def _soft_update(self):
        for p_online, p_target in zip(self.online_net.parameters(), self.target_net.parameters()):
            p_target.data.copy_(self.tau * p_online.data + (1.0 - self.tau) * p_target.data)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str):
        torch.save(
            {
                "online_net": self.online_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "step": self._step,
            },
            path,
        )

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(ckpt["online_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon = ckpt["epsilon"]
        self._step = ckpt["step"]

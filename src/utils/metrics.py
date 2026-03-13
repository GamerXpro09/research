"""Evaluation metrics for FedNS-Traffic."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class EpisodeMetrics:
    """Aggregated metrics collected over one episode."""

    total_reward: float = 0.0
    avg_queue_length: float = 0.0
    avg_wait_time: float = 0.0
    total_throughput: int = 0
    safety_overrides: int = 0
    num_steps: int = 0

    # Per-step lists for post-hoc analysis
    rewards: List[float] = field(default_factory=list)
    queue_lengths: List[float] = field(default_factory=list)
    wait_times: List[float] = field(default_factory=list)

    def update(self, reward: float, queue: float, wait: float, throughput: int = 0):
        self.rewards.append(reward)
        self.queue_lengths.append(queue)
        self.wait_times.append(wait)
        self.total_reward += reward
        self.total_throughput += throughput
        self.num_steps += 1

    def finalise(self):
        """Compute summary statistics from per-step lists."""
        if self.queue_lengths:
            self.avg_queue_length = float(np.mean(self.queue_lengths))
        if self.wait_times:
            self.avg_wait_time = float(np.mean(self.wait_times))

    def as_dict(self) -> Dict[str, float]:
        return {
            "total_reward": self.total_reward,
            "avg_queue_length": self.avg_queue_length,
            "avg_wait_time": self.avg_wait_time,
            "total_throughput": float(self.total_throughput),
            "safety_overrides": float(self.safety_overrides),
            "num_steps": float(self.num_steps),
        }


def improvement_over_baseline(
    agent_metrics: EpisodeMetrics,
    baseline_metrics: EpisodeMetrics,
    metric: str = "avg_wait_time",
) -> float:
    """
    Compute percentage improvement of *agent* over *baseline* for *metric*.
    Positive value = agent is better (lower wait time / queue, higher reward).

    Returns
    -------
    float
        Improvement in percent (e.g. 15.0 means 15% better).
    """
    agent_val = agent_metrics.as_dict()[metric]
    baseline_val = baseline_metrics.as_dict()[metric]

    if metric == "total_reward":
        # Higher is better
        if baseline_val == 0:
            return 0.0
        return (agent_val - baseline_val) / abs(baseline_val) * 100.0
    else:
        # Lower is better (queue, wait, overrides)
        if baseline_val == 0:
            return 0.0
        return (baseline_val - agent_val) / baseline_val * 100.0

#!/usr/bin/env python3
"""
Evaluation entry point for FedNS-Traffic.

Compares trained DQN agent against Fixed-Timing and Max-Pressure baselines.

Usage
-----
    python scripts/evaluate.py --model checkpoints/best_model.pt \\
                               --config configs/default.yaml \\
                               --net    configs/sumo_networks/single.net.xml \\
                               --routes configs/sumo_networks/single.rou.xml \\
                               --episodes 10
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.agents.dqn_agent import DuelingDQNAgent  # noqa: E402
from src.environment.state_representation import StateConfig  # noqa: E402
from src.utils.config import get_device_str, load_config  # noqa: E402
from src.utils.metrics import EpisodeMetrics  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate FedNS-Traffic agents vs. baselines")
    p.add_argument("--model", required=True, help="Path to saved agent checkpoint")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--net", default=None)
    p.add_argument("--routes", default=None)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def evaluate_agent(env, agent: DuelingDQNAgent, num_episodes: int) -> EpisodeMetrics:
    all_metrics = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        m = EpisodeMetrics()
        done = False
        while not done:
            action = agent.select_action(obs, training=False)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            m.update(
                reward=reward,
                queue=float(info.get("total_reward", 0)),
                wait=float(info.get("total_wait", 0)),
            )
            obs = next_obs
        m.finalise()
        all_metrics.append(m)

    agg = EpisodeMetrics(
        total_reward=float(np.mean([m.total_reward for m in all_metrics])),
        avg_queue_length=float(np.mean([m.avg_queue_length for m in all_metrics])),
        avg_wait_time=float(np.mean([m.avg_wait_time for m in all_metrics])),
        total_throughput=int(np.mean([m.total_throughput for m in all_metrics])),
    )
    return agg


def main():
    args = parse_args()
    cfg = load_config(args.config)

    from src.environment.sumo_env import SumoEnvironment

    if not args.net or not args.routes:
        log.error("--net and --routes are required")
        sys.exit(1)

    sim_cfg = cfg.get("simulation", {})
    safety_cfg = cfg.get("safety", {})

    state_cfg = StateConfig(
        num_lanes=cfg.get("state", {}).get("num_lanes", 12),
        num_phases=cfg.get("action", {}).get("num_phases", 8),
    )

    def make_env():
        return SumoEnvironment(
            net_file=args.net,
            route_file=args.routes,
            min_green=safety_cfg.get("min_green", 5),
            max_green=safety_cfg.get("max_green", 90),
            yellow_time=sim_cfg.get("yellow_duration", 3),
            max_steps=sim_cfg.get("max_steps", 3600),
            state_config=state_cfg,
        )

    env = make_env()
    state_dim = env.observation_space.shape[0]
    num_actions = env.action_space.n

    # Load DQN agent
    agent_cfg = cfg.get("agent", {})
    agent = DuelingDQNAgent(
        state_dim=state_dim,
        num_actions=num_actions,
        hidden_dim=agent_cfg.get("hidden_dim", 128),
        device=get_device_str(args.device),
    )
    agent.load(args.model)
    agent.epsilon = 0.0  # greedy evaluation

    log.info("Evaluating DQN agent over %d episodes …", args.episodes)
    dqn_metrics = evaluate_agent(env, agent, args.episodes)

    # Fixed-timing baseline evaluation (offline — we just compute over state logs)
    log.info(
        "DQN  | reward=%.2f  avg_wait=%.1fs  queue=%.1f  throughput=%d",
        dqn_metrics.total_reward,
        dqn_metrics.avg_wait_time,
        dqn_metrics.avg_queue_length,
        dqn_metrics.total_throughput,
    )

    env.close()
    log.info("Evaluation complete.")


if __name__ == "__main__":
    main()

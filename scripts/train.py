#!/usr/bin/env python3
"""
Training entry point for FedNS-Traffic Phase 1 (single-intersection DQN).

Usage
-----
    python scripts/train.py --config configs/default.yaml \\
                            --net    configs/sumo_networks/single.net.xml \\
                            --routes configs/sumo_networks/single.rou.xml \\
                            --episodes 500
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path when run as a script
ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.agents.dqn_agent import DuelingDQNAgent
from src.environment.state_representation import StateConfig
from src.utils.config import load_config, merge_configs, get_device_str
from src.utils.metrics import EpisodeMetrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train single-intersection Dueling DQN")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--net",    default=None, help="SUMO .net.xml file")
    p.add_argument("--routes", default=None, help="SUMO .rou.xml file")
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--device",   default=None, choices=["auto", "cpu", "cuda"])
    p.add_argument("--simulator", default=None, choices=["sumo", "cityflow"])
    p.add_argument("--gui", action="store_true", help="Launch SUMO GUI")
    return p.parse_args()


def build_env(cfg: dict, net: str | None, routes: str | None, use_gui: bool):
    """Instantiate the configured simulator environment."""
    sim_cfg = cfg.get("simulation", {})
    simulator = sim_cfg.get("simulator", "sumo")
    state_cfg = StateConfig(**{
        k: sim_cfg.get(k, v)
        for k, v in {
            "num_lanes":        cfg.get("state", {}).get("num_lanes", 12),
            "num_phases":       cfg.get("action", {}).get("num_phases", 8),
            "embed_dim":        cfg.get("state", {}).get("embed_dim", 16),
            "max_neighbors":    cfg.get("state", {}).get("max_neighbors", 4),
            "queue_length_max": cfg.get("state", {}).get("queue_length_max", 50.0),
            "speed_max":        cfg.get("state", {}).get("speed_max", 60.0),
            "wait_time_max":    cfg.get("state", {}).get("wait_time_max", 300.0),
        }.items()
    })

    safety = cfg.get("safety", {})
    reward_cfg = cfg.get("reward", {})
    reward_weights = {
        "pressure":   reward_cfg.get("w_pressure",  0.40),
        "queue":      reward_cfg.get("w_queue",      0.25),
        "throughput": reward_cfg.get("w_throughput", 0.20),
        "wait":       reward_cfg.get("w_wait",       0.10),
        "pedestrian": reward_cfg.get("w_pedestrian", 0.05),
    }

    if simulator == "sumo":
        from src.environment.sumo_env import SumoEnvironment
        if not net or not routes:
            raise ValueError("--net and --routes are required for SUMO simulator")
        env = SumoEnvironment(
            net_file=net,
            route_file=routes,
            use_gui=use_gui,
            min_green=safety.get("min_green", 5),
            max_green=safety.get("max_green", 90),
            yellow_time=sim_cfg.get("yellow_duration", 3),
            all_red_time=sim_cfg.get("all_red_duration", 2),
            max_steps=sim_cfg.get("max_steps", 3600),
            reward_weights=reward_weights,
            state_config=state_cfg,
        )
    else:
        raise NotImplementedError(
            f"Simulator '{simulator}' requires a CityFlow config file. "
            "Use CityFlowEnvironment directly."
        )

    return env, state_cfg


def build_agent(cfg: dict, state_dim: int, num_actions: int) -> DuelingDQNAgent:
    agent_cfg = cfg.get("agent", {})
    training_cfg = cfg.get("training", {})
    return DuelingDQNAgent(
        state_dim=state_dim,
        num_actions=num_actions,
        hidden_dim=agent_cfg.get("hidden_dim", 128),
        lr=float(agent_cfg.get("lr", 3e-4)),
        gamma=float(agent_cfg.get("gamma", 0.99)),
        tau=float(agent_cfg.get("tau", 0.005)),
        batch_size=int(agent_cfg.get("batch_size", 64)),
        buffer_capacity=int(agent_cfg.get("buffer_capacity", 100_000)),
        per_alpha=float(agent_cfg.get("per_alpha", 0.6)),
        per_beta_start=float(agent_cfg.get("per_beta_start", 0.4)),
        per_beta_frames=int(agent_cfg.get("per_beta_frames", 100_000)),
        per_epsilon=float(agent_cfg.get("per_epsilon", 1e-6)),
        epsilon_start=float(agent_cfg.get("epsilon_start", 1.0)),
        epsilon_end=float(agent_cfg.get("epsilon_end", 0.05)),
        epsilon_decay=float(agent_cfg.get("epsilon_decay", 0.995)),
        learning_starts=int(agent_cfg.get("learning_starts", 1_000)),
        train_freq=int(agent_cfg.get("train_freq", 4)),
        target_update_freq=int(agent_cfg.get("target_update_freq", 1_000)),
        device=get_device_str(training_cfg.get("device", "auto")),
    )


def run_episode(env, agent: DuelingDQNAgent, training: bool = True) -> EpisodeMetrics:
    obs, _ = env.reset()
    metrics = EpisodeMetrics()
    done = False

    while not done:
        action = agent.select_action(obs, training=training)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if training:
            agent.observe(obs, action, reward, next_obs, done)

        metrics.update(
            reward=reward,
            queue=float(info.get("total_reward", 0)),
            wait=float(info.get("total_wait", 0)),
        )
        obs = next_obs

    metrics.finalise()
    return metrics


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # Apply CLI overrides
    overrides: dict = {}
    if args.episodes:
        overrides.setdefault("training", {})["total_episodes"] = args.episodes
    if args.device:
        overrides.setdefault("training", {})["device"] = args.device
    if args.simulator:
        overrides.setdefault("simulation", {})["simulator"] = args.simulator
    cfg = merge_configs(cfg, overrides)

    training_cfg = cfg.get("training", {})
    total_episodes = int(training_cfg.get("total_episodes", 500))
    eval_freq = int(training_cfg.get("eval_freq", 50))
    save_freq = int(training_cfg.get("save_freq", 100))
    log_dir = training_cfg.get("log_dir", "logs/")
    ckpt_dir = training_cfg.get("checkpoint_dir", "checkpoints/")

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    env, state_cfg = build_env(cfg, args.net, args.routes, args.gui)
    num_actions = env.action_space.n
    state_dim = env.observation_space.shape[0]

    agent = build_agent(cfg, state_dim, num_actions)
    log.info(
        "Training Dueling DQN  state_dim=%d  num_actions=%d  device=%s",
        state_dim, num_actions, agent.device,
    )

    best_reward = -np.inf

    for ep in range(1, total_episodes + 1):
        metrics = run_episode(env, agent, training=True)
        log.info(
            "Episode %4d/%d  reward=%7.2f  ε=%.3f  avg_wait=%.1fs",
            ep, total_episodes,
            metrics.total_reward,
            agent.epsilon,
            metrics.avg_wait_time,
        )

        if ep % eval_freq == 0:
            eval_metrics = run_episode(env, agent, training=False)
            log.info(
                "  [EVAL] reward=%7.2f  avg_queue=%.1f  avg_wait=%.1fs  throughput=%d",
                eval_metrics.total_reward,
                eval_metrics.avg_queue_length,
                eval_metrics.avg_wait_time,
                eval_metrics.total_throughput,
            )
            if eval_metrics.total_reward > best_reward:
                best_reward = eval_metrics.total_reward
                agent.save(str(Path(ckpt_dir) / "best_model.pt"))
                log.info("  [SAVE] New best model (reward=%.2f)", best_reward)

        if ep % save_freq == 0:
            agent.save(str(Path(ckpt_dir) / f"checkpoint_ep{ep}.pt"))

    env.close()
    log.info("Training complete.  Best eval reward: %.2f", best_reward)


if __name__ == "__main__":
    main()

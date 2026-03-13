"""Tests for utility modules (config, metrics)."""

from __future__ import annotations

import pytest

from src.utils.config import load_config, merge_configs, get_device_str
from src.utils.metrics import EpisodeMetrics, improvement_over_baseline


class TestLoadConfig:
    def test_loads_default(self):
        cfg = load_config()
        assert "simulation" in cfg
        assert "agent" in cfg
        assert "training" in cfg

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("/tmp/nonexistent_config_xyz.yaml")

    def test_reward_weights_present(self):
        cfg = load_config()
        reward = cfg["reward"]
        for key in ["w_pressure", "w_queue", "w_throughput", "w_wait", "w_pedestrian"]:
            assert key in reward


class TestMergeConfigs:
    def test_simple_override(self):
        base = {"a": 1, "b": 2}
        override = {"b": 99}
        merged = merge_configs(base, override)
        assert merged["a"] == 1
        assert merged["b"] == 99

    def test_nested_merge(self):
        base = {"training": {"lr": 0.001, "epochs": 100}}
        override = {"training": {"lr": 0.0001}}
        merged = merge_configs(base, override)
        assert merged["training"]["lr"] == 0.0001
        assert merged["training"]["epochs"] == 100

    def test_new_key(self):
        base = {"a": 1}
        override = {"b": 2}
        merged = merge_configs(base, override)
        assert merged["b"] == 2

    def test_does_not_mutate_base(self):
        base = {"a": 1}
        override = {"a": 99}
        merge_configs(base, override)
        assert base["a"] == 1


class TestGetDeviceStr:
    def test_cpu(self):
        assert get_device_str("cpu") == "cpu"

    def test_auto_returns_string(self):
        result = get_device_str("auto")
        assert result in ("cpu", "cuda")

    def test_explicit_cuda(self):
        assert get_device_str("cuda") == "cuda"


class TestEpisodeMetrics:
    def test_update_and_finalise(self):
        m = EpisodeMetrics()
        m.update(reward=1.0, queue=5.0, wait=30.0)
        m.update(reward=2.0, queue=3.0, wait=20.0)
        m.finalise()
        assert m.total_reward == pytest.approx(3.0)
        assert m.avg_queue_length == pytest.approx(4.0)
        assert m.avg_wait_time == pytest.approx(25.0)
        assert m.num_steps == 2

    def test_as_dict_keys(self):
        m = EpisodeMetrics()
        m.finalise()
        d = m.as_dict()
        for key in ["total_reward", "avg_queue_length", "avg_wait_time",
                    "total_throughput", "safety_overrides", "num_steps"]:
            assert key in d

    def test_empty_finalise(self):
        m = EpisodeMetrics()
        m.finalise()   # Should not raise
        assert m.avg_wait_time == 0.0


class TestImprovementOverBaseline:
    def test_wait_time_improvement(self):
        agent = EpisodeMetrics(avg_wait_time=50.0)
        baseline = EpisodeMetrics(avg_wait_time=100.0)
        pct = improvement_over_baseline(agent, baseline, metric="avg_wait_time")
        assert pct == pytest.approx(50.0)

    def test_reward_improvement(self):
        agent = EpisodeMetrics(total_reward=200.0)
        baseline = EpisodeMetrics(total_reward=100.0)
        pct = improvement_over_baseline(agent, baseline, metric="total_reward")
        assert pct == pytest.approx(100.0)

    def test_zero_baseline(self):
        agent = EpisodeMetrics(avg_wait_time=30.0)
        baseline = EpisodeMetrics(avg_wait_time=0.0)
        pct = improvement_over_baseline(agent, baseline, metric="avg_wait_time")
        assert pct == 0.0

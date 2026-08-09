import pytest

from celldiffa.benchmark.config import expand_environment


def test_environment_default(monkeypatch):
    monkeypatch.delenv("BENCHMARK_TEST_ROOT", raising=False)
    assert expand_environment("${BENCHMARK_TEST_ROOT:-/data/default}/x") == "/data/default/x"


def test_environment_override(monkeypatch):
    monkeypatch.setenv("BENCHMARK_TEST_ROOT", "/scratch/data")
    assert expand_environment("${BENCHMARK_TEST_ROOT:-/data/default}/x") == "/scratch/data/x"


def test_missing_environment_fails(monkeypatch):
    monkeypatch.delenv("BENCHMARK_REQUIRED", raising=False)
    with pytest.raises(ValueError, match="BENCHMARK_REQUIRED"):
        expand_environment("${BENCHMARK_REQUIRED}/x")

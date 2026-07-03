"""Bench harness tests: percentile math and request shaping (no network)."""

from __future__ import annotations

from ripple.bench import _request_params, percentile


def test_percentile_nearest_rank() -> None:
    values = [float(v) for v in range(1, 101)]  # 1..100
    assert percentile(values, 50) == 50.0
    assert percentile(values, 95) == 95.0
    assert percentile(values, 99) == 99.0
    assert percentile(values, 100) == 100.0
    assert percentile([], 95) == 0.0
    assert percentile([7.0], 99) == 7.0


def test_search_scenario_busts_cache_within_and_across_runs() -> None:
    _, params_a = _request_params("search", 1, "x", nonce="run1")
    _, params_b = _request_params("search", 11, "x", nonce="run1")
    assert params_a["q"] != params_b["q"]  # unique within a run
    _, params_c = _request_params("search", 1, "x", nonce="run2")
    assert params_a["q"] != params_c["q"]  # and across runs (no stale Redis hits)


def test_cached_scenario_repeats_query_within_run_only() -> None:
    _, params_a = _request_params("search-cached", 0, "x", nonce="run1")
    _, params_b = _request_params("search-cached", 10, "x", nonce="run1")
    assert params_a["q"] == params_b["q"]  # repeats -> cache hits within the run
    _, params_c = _request_params("search-cached", 0, "x", nonce="run2")
    assert params_a["q"] != params_c["q"]  # fresh miss on a new run


def test_impact_scenario_uses_symbol() -> None:
    path, params = _request_params("impact", 0, "pkg.mod.func")
    assert path == "/impact"
    assert params["symbol"] == "pkg.mod.func"

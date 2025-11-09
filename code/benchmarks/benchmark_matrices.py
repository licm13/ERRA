"""Benchmark scripts comparing legacy and optimised ERRA matrix builders.

This script generates synthetic precipitation and discharge data to measure
performance of the optimised ``_build_design_matrix`` and
``create_xprime_matrix`` implementations against reference Python versions
based on the original loops.  It also verifies that the optimised outputs are
numerically equivalent to the baselines within floating-point tolerances.

运行方式::

    python -m code.benchmarks.benchmark_matrices --n 2000 --k 3 --m 60

"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np

from code.erra import _build_design_matrix as optimised_design
from src.erra.nonlin import create_xprime_matrix as optimised_xprime


@dataclass
class BenchmarkResult:
    name: str
    elapsed: float
    speedup: float


def _legacy_build_design_matrix(
    p: np.ndarray, q: np.ndarray, wt: np.ndarray, m: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, k = p.shape
    cols = k * (m + 1)
    design = np.zeros((n, cols), dtype=float)
    for lag in range(m + 1):
        shifted = np.roll(p, shift=lag, axis=0)
        shifted[:lag, :] = np.nan
        design[:, lag * k : (lag + 1) * k] = shifted

    valid = (~np.isnan(design).any(axis=1)) & (~np.isnan(q))
    design = design[valid]
    response = q[valid]
    weights = wt[valid]

    mean_w = weights.mean()
    if not np.isfinite(mean_w) or mean_w <= 0:
        weights = np.ones_like(weights)
    else:
        weights = weights / mean_w
    return design, response, weights


def _legacy_create_xprime_matrix(
    p: np.ndarray, xknots: np.ndarray, xknot_type: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_timesteps, n_drivers = p.shape

    xknots = np.asarray(xknots, dtype=float)
    if xknots.ndim == 1:
        xknots = np.tile(xknots[:, np.newaxis], (1, n_drivers))

    n_xknots = xknots.shape[0]
    knot_values = np.zeros((n_xknots + 2, n_drivers))

    for i in range(n_drivers):
        p_col = p[:, i]
        p_nonzero = p_col[p_col > 0]
        if len(p_nonzero) == 0:
            raise ValueError("Driver has no positive values")

        if xknot_type == "values":
            kpts = xknots[:, i]
        elif xknot_type == "percentiles":
            kpts = np.percentile(p_nonzero, xknots[:, i])
        elif xknot_type == "cumsum":
            p_sorted = np.sort(p_nonzero)
            cumsum = np.cumsum(p_sorted)
            kpts = []
            for pct in xknots[:, i]:
                target = cumsum[-1] * pct / 100
                idx = np.argmin(np.abs(cumsum - target))
                kpts.append(p_sorted[idx])
            kpts = np.array(kpts)
        elif xknot_type == "sqsum":
            p_sorted = np.sort(p_nonzero)
            cumsum_sq = np.cumsum(p_sorted**2)
            kpts = []
            for pct in xknots[:, i]:
                target = cumsum_sq[-1] * pct / 100
                idx = np.argmin(np.abs(cumsum_sq - target))
                kpts.append(p_sorted[idx])
            kpts = np.array(kpts)
        elif xknot_type == "even":
            n_knots = int(xknots[0, i])
            percentiles = np.linspace(0, 100, n_knots + 2)[1:-1]
            kpts = np.percentile(p_nonzero, percentiles)
        else:
            raise ValueError(f"Unknown xknot_type: {xknot_type}")

        knot_values[:, i] = np.concatenate([[0.0], kpts, [np.max(p_col)]])

    n_segments = n_xknots + 1
    xprime = np.zeros((n_timesteps, n_drivers * n_segments))
    seg_wtd_meanx = np.zeros((n_segments, n_drivers))

    for driver_idx in range(n_drivers):
        p_col = p[:, driver_idx]
        kpts = knot_values[:, driver_idx]
        for seg_idx in range(n_segments):
            k_lower = kpts[seg_idx]
            k_upper = kpts[seg_idx + 1]
            interval_width = k_upper - k_lower
            xp_col = np.maximum(0, np.minimum(p_col - k_lower, interval_width))
            col_idx = driver_idx * n_segments + seg_idx
            xprime[:, col_idx] = xp_col

            in_segment = (p_col > k_lower) & (p_col <= k_upper)
            if np.any(in_segment):
                p_in_seg = p_col[in_segment]
                seg_wtd_meanx[seg_idx, driver_idx] = np.sum(p_in_seg**2) / np.sum(
                    p_in_seg
                )
            else:
                seg_wtd_meanx[seg_idx, driver_idx] = (k_lower + k_upper) / 2

    return xprime, knot_values, seg_wtd_meanx


def _time_function(func: Callable, *args, repeats: int = 5) -> float:
    start = time.perf_counter()
    for _ in range(repeats):
        func(*args)
    end = time.perf_counter()
    return (end - start) / repeats


def benchmark_design(n: int, k: int, m: int, repeats: int) -> BenchmarkResult:
    rng = np.random.default_rng(42)
    p = rng.gamma(shape=2.0, scale=1.0, size=(n, k))
    q = rng.normal(size=n)
    wt = rng.uniform(0.5, 1.5, size=n)

    legacy = _time_function(_legacy_build_design_matrix, p, q, wt, m, repeats=repeats)
    optimised = _time_function(optimised_design, p, q, wt, m, repeats=repeats)

    design_old, resp_old, w_old = _legacy_build_design_matrix(p, q, wt, m)
    design_new, resp_new, w_new = optimised_design(p, q, wt, m)

    assert np.allclose(design_old, design_new)
    assert np.allclose(resp_old, resp_new)
    assert np.allclose(w_old, w_new)

    return BenchmarkResult("design", legacy, legacy / optimised)


def benchmark_xprime(n: int, k: int, n_knots: int, repeats: int) -> BenchmarkResult:
    rng = np.random.default_rng(123)
    p = rng.gamma(shape=2.0, scale=1.5, size=(n, k))
    percentiles = np.linspace(10, 90, n_knots)

    legacy = _time_function(
        _legacy_create_xprime_matrix, p, percentiles, "percentiles", repeats=repeats
    )
    optimised = _time_function(
        optimised_xprime, p, percentiles, "percentiles", repeats=repeats
    )

    legacy_res = _legacy_create_xprime_matrix(p, percentiles, "percentiles")
    optimised_res = optimised_xprime(p, percentiles, "percentiles")

    for a, b in zip(legacy_res, optimised_res):
        assert np.allclose(a, b)

    return BenchmarkResult("xprime", legacy, legacy / optimised)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=2000, help="Number of time steps")
    parser.add_argument("--k", type=int, default=3, help="Number of drivers")
    parser.add_argument("--m", type=int, default=60, help="Maximum lag")
    parser.add_argument(
        "--n-knots", type=int, default=5, help="Number of interior x' knots"
    )
    parser.add_argument(
        "--repeats", type=int, default=3, help="Number of timing repetitions"
    )
    args = parser.parse_args()

    results = [
        benchmark_design(args.n, args.k, args.m, args.repeats),
        benchmark_xprime(args.n, args.k, args.n_knots, args.repeats),
    ]

    for result in results:
        print(
            f"{result.name:7s} : {result.elapsed:.6f}s per run | "
            f"speedup ×{result.speedup:5.2f}"
        )


if __name__ == "__main__":
    main()

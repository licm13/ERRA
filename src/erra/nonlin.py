"""Nonlinear response function analysis.

非线性响应函数分析

This module implements nonlinear impulse response function (NRF) estimation,
allowing ERRA to capture how runoff response varies with precipitation intensity.

本模块实现非线性脉冲响应函数（NRF）估计，
使 ERRA 能够捕捉径流响应如何随降水强度变化。

The key concept is to split precipitation into intensity-based segments using
"xknots" (intensity knots), estimate separate responses for each segment, and
then reconstruct the full nonlinear response function.

关键概念是使用"xknots"（强度节点）将降水分割为基于强度的片段，
为每个片段估计单独的响应，然后重建完整的非线性响应函数。
"""

from __future__ import annotations

from typing import List, Literal, Tuple

import numpy as np

# Constant for minimum precipitation threshold
_MIN_PRECIPITATION_VALUE = 0  # Exclude zero precipitation in knot calculations
_EPSILON_WEIGHT = 1e-10  # Small value to prevent division by zero


def _percentiles_from_sorted(sorted_values: np.ndarray, percentiles: np.ndarray) -> np.ndarray:
    """Interpolate percentile values from a sorted sample."""

    if sorted_values.size == 0:
        raise ValueError("Cannot compute percentiles of an empty array")

    percentiles = np.clip(percentiles, 0.0, 100.0)
    if sorted_values.size == 1:
        return np.full(percentiles.shape, sorted_values[0], dtype=float)

    ranks = percentiles / 100.0 * (sorted_values.size - 1)
    return np.interp(ranks, np.arange(sorted_values.size), sorted_values)


def _values_from_cumulative(
    sorted_values: np.ndarray, cumulative: np.ndarray, percentiles: np.ndarray
) -> np.ndarray:
    """Return values corresponding to percentiles of a cumulative distribution."""

    if cumulative.size == 0:
        raise ValueError("Cannot compute cumulative percentiles of an empty array")

    total = cumulative[-1]
    if total <= 0:
        return np.zeros_like(percentiles, dtype=float)

    targets = np.clip(percentiles, 0.0, 100.0) * total / 100.0
    indices = np.searchsorted(cumulative, targets, side="left")
    indices = np.clip(indices, 0, sorted_values.size - 1)
    return sorted_values[indices]


def create_xprime_matrix(
    p: np.ndarray,
    xknots: np.ndarray,
    xknot_type: Literal[
        "values", "percentiles", "cumsum", "sqsum", "even"
    ] = "percentiles",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create x' (x-prime) matrix for nonlinear analysis.

    为非线性分析创建 x' (x-prime) 矩阵。

    This function implements the core transformation from precipitation (p) to
    incremental precipitation (x') based on intensity knots. This allows the
    regression to estimate how response varies across different precipitation
    intensity ranges.

    此函数实现从降水 (p) 到基于强度节点的增量降水 (x') 的核心转换。
    这允许回归估计响应如何在不同降水强度范围内变化。

    The transformation follows Eq. 43 from Kirchner (2022):
    x'_i(t) = max(0, min(p(t) - k_i, k_{i+1} - k_i))

    where k_i are the knot points and x'_i represents the increment of
    precipitation between knots k_i and k_{i+1}.

    变换遵循 Kirchner (2022) 的方程 43：
    x'_i(t) = max(0, min(p(t) - k_i, k_{i+1} - k_i))

    其中 k_i 是节点，x'_i 表示节点 k_i 和 k_{i+1} 之间的降水增量。

    Parameters / 参数
    ----------
    p : np.ndarray
        Precipitation matrix (n_timesteps × n_drivers)
        降水矩阵（时间步数 × 驱动变量数）
    xknots : np.ndarray
        Knot values or percentiles, shape (n_knots,) or (n_knots, n_drivers)
        节点值或百分位数，形状 (n_knots,) 或 (n_knots, n_drivers)
    xknot_type : str
        How to interpret xknots:
        如何解释 xknots：
        - "values": Direct precipitation values (直接降水值)
        - "percentiles": Percentiles of p distribution (p 分布的百分位数)
        - "cumsum": Percentiles of cumulative sum of p (p 累积和的百分位数)
        - "sqsum": Percentiles of cumulative sum of p² (p² 累积和的百分位数)
        - "even": Evenly spaced knots (均匀间隔节点)

    Returns / 返回
    -------
    xprime : np.ndarray
        Transformed precipitation matrix (n_timesteps × n_drivers × n_knots)
        转换后的降水矩阵（时间步数 × 驱动变量数 × 节点数）
    knot_values : np.ndarray
        Actual knot values used (including min and max)
        实际使用的节点值（包括最小值和最大值）
    segment_weighted_mean_precip : np.ndarray
        Segment-weighted mean precipitation in each interval
        每个区间中段加权平均降水
    """
    n_timesteps, n_drivers = p.shape

    # Convert xknots to array if needed
    xknots = np.asarray(xknots, dtype=float)

    # Ensure xknots is 2D
    if xknots.ndim == 1:
        xknots = np.tile(xknots[:, np.newaxis], (1, n_drivers))
    elif xknots.shape[1] != n_drivers:
        raise ValueError(
            f"xknots has {xknots.shape[1]} columns but p has {n_drivers} drivers"
        )

    n_xknots = xknots.shape[0]

    # Calculate actual knot values based on type
    knot_values = np.zeros((n_xknots + 2, n_drivers))  # +2 for min and max

    sorted_columns: List[np.ndarray] = []
    cumulative_columns: List[np.ndarray] = []
    cumulative_sq_columns: List[np.ndarray] = []
    max_values = np.max(p, axis=0)

    for i in range(n_drivers):
        p_col = p[:, i]
        positive = p_col[p_col > _MIN_PRECIPITATION_VALUE]
        if positive.size == 0:
            raise ValueError(f"Driver {i} has no positive values")

        sorted_col = np.sort(positive)
        sorted_columns.append(sorted_col)
        cumulative_columns.append(np.cumsum(sorted_col))
        cumulative_sq_columns.append(np.cumsum(sorted_col**2))

    for i in range(n_drivers):
        if xknot_type == "values":
            kpts = xknots[:, i]
        elif xknot_type == "percentiles":
            kpts = _percentiles_from_sorted(sorted_columns[i], xknots[:, i])
        elif xknot_type == "cumsum":
            kpts = _values_from_cumulative(
                sorted_columns[i], cumulative_columns[i], xknots[:, i]
            )
        elif xknot_type == "sqsum":
            kpts = _values_from_cumulative(
                sorted_columns[i], cumulative_sq_columns[i], xknots[:, i]
            )
        elif xknot_type == "even":
            n_knots = int(xknots[0, i])
            percentiles = np.linspace(0, 100, n_knots + 2)[1:-1]
            kpts = _percentiles_from_sorted(sorted_columns[i], percentiles)
        else:
            raise ValueError(f"Unknown xknot_type: {xknot_type}")

        max_val = max_values[i]
        kpts = np.clip(kpts, 0.0, max_val)
        kpts = np.maximum.accumulate(kpts)
        knot_values[:, i] = np.concatenate([[0.0], kpts, [max_val]])

    # Create x-prime matrix
    # For each driver and each knot interval, create an x-prime column
    n_segments = n_xknots + 1  # Number of intervals between knots
    lower_bounds = knot_values[:-1].T  # (n_drivers, n_segments)
    upper_bounds = knot_values[1:].T
    interval_width = upper_bounds - lower_bounds

    p_expanded = p[:, :, None]
    xprime = np.clip(
        p_expanded - lower_bounds[None, :, :], 0.0, interval_width[None, :, :]
    )

    # Calculate segment-weighted means using vectorised operations
    in_segment = (p_expanded > lower_bounds[None, :, :]) & (
        p_expanded <= upper_bounds[None, :, :]
    )
    weighted_sum = np.sum(np.where(in_segment, p_expanded**2, 0.0), axis=0)
    weight = np.sum(np.where(in_segment, p_expanded, 0.0), axis=0)

    with np.errstate(invalid="ignore", divide="ignore"):
        seg_wtd_mean = np.divide(
            weighted_sum,
            weight,
            out=np.zeros_like(weighted_sum),
            where=weight > _EPSILON_WEIGHT,
        )

    fallback = 0.5 * (lower_bounds + upper_bounds)
    empty_mask = weight <= _EPSILON_WEIGHT
    seg_wtd_mean[empty_mask] = fallback[empty_mask]

    xprime_matrix = xprime.reshape(n_timesteps, n_drivers * n_segments)
    return xprime_matrix, knot_values, seg_wtd_mean.T


def betaprime_to_nrf(
    betaprime: np.ndarray,
    knot_values: np.ndarray,
    segment_weighted_mean_precip: np.ndarray,
    n_drivers: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert β' (beta-prime) coefficients to NRF (Nonlinear Response Functions).

    将 β'（beta-prime）系数转换为 NRF（非线性响应函数）。

    After regressing q on x', we obtain β' coefficients that represent
    the response per unit increment of x'. To get the actual response
    as a function of precipitation intensity, we need to convert these
    to NRF values at each knot.

    在对 x' 回归 q 后，我们获得 β' 系数，表示每单位 x' 增量的响应。
    为了获得作为降水强度函数的实际响应，我们需要将这些转换为每个节点的 NRF 值。

    The conversion follows:
    NRF(k_i) = Σ_{j=1}^{i} β'_j × (k_j - k_{j-1})

    转换遵循：
    NRF(k_i) = Σ_{j=1}^{i} β'_j × (k_j - k_{j-1})

    Parameters / 参数
    ----------
    betaprime : np.ndarray
        Beta-prime coefficients from regression, shape (m+1, n_drivers × n_segments)
        回归得到的 beta-prime 系数，形状 (m+1, n_drivers × n_segments)
    knot_values : np.ndarray
        Knot values including min and max, shape (n_knots+2, n_drivers)
        包括最小值和最大值的节点值，形状 (n_knots+2, n_drivers)
    segment_weighted_mean_precip : np.ndarray
        Segment-weighted mean precipitation, shape (n_segments, n_drivers)
        段加权平均降水，形状 (n_segments, n_drivers)
    n_drivers : int
        Number of precipitation drivers
        降水驱动变量数量

    Returns / 返回
    -------
    nrf : np.ndarray
        Nonlinear Response Functions at each knot, shape (m+1, n_drivers × n_segments)
        每个节点的非线性响应函数，形状 (m+1, n_drivers × n_segments)
    rrd : np.ndarray
        Runoff Response Distribution (average NRF), shape (m+1, n_drivers)
        径流响应分布（平均 NRF），形状 (m+1, n_drivers)
    """
    m_plus_1 = betaprime.shape[0]
    n_segments = knot_values.shape[0] - 1

    # Initialize NRF matrix
    nrf = np.zeros_like(betaprime)

    # Convert β' to NRF for each driver
    for driver_idx in range(n_drivers):
        kpts = knot_values[:, driver_idx]

        for lag_idx in range(m_plus_1):
            # For this lag, get all β' values for this driver
            bp_start = driver_idx * n_segments
            bp_end = (driver_idx + 1) * n_segments
            bp_lag = betaprime[lag_idx, bp_start:bp_end]

            # Calculate cumulative NRF at each knot
            # NRF[i] = Σ_{j=0}^{i} β'[j] × (k_{j+1} - k_j)
            for seg_idx in range(n_segments):
                cumulative_response = np.sum(
                    bp_lag[: seg_idx + 1] * np.diff(kpts[: seg_idx + 2])
                )
                nrf[lag_idx, bp_start + seg_idx] = cumulative_response

    # Calculate average RRD (weighted by precipitation)
    rrd = np.zeros((m_plus_1, n_drivers))

    for driver_idx in range(n_drivers):
        bp_start = driver_idx * n_segments
        bp_end = (driver_idx + 1) * n_segments

        # Weight by segment-weighted mean precipitation
        weights = segment_weighted_mean_precip[:, driver_idx]
        weights = weights / (np.sum(weights) + _EPSILON_WEIGHT)  # Normalize

        for lag_idx in range(m_plus_1):
            rrd[lag_idx, driver_idx] = np.sum(nrf[lag_idx, bp_start:bp_end] * weights)

    return nrf, rrd


def create_nrf_labels(
    base_labels: List[str],
    knot_values: np.ndarray,
    show_top_knot: bool = False,
) -> List[str]:
    """Create labels for NRF columns.

    为 NRF 列创建标签。

    Parameters / 参数
    ----------
    base_labels : list of str
        Original precipitation driver labels
        原始降水驱动标签
    knot_values : np.ndarray
        Knot values, shape (n_knots+2, n_drivers)
        节点值，形状 (n_knots+2, n_drivers)
    show_top_knot : bool
        Whether to include the top knot in labels
        是否在标签中包含顶部节点

    Returns / 返回
    -------
    labels : list of str
        Labels for each NRF column
        每个 NRF 列的标签
    """
    n_segments = knot_values.shape[0] - 1

    labels = []
    for driver_idx, base_label in enumerate(base_labels):
        kpts = knot_values[:, driver_idx]

        n_segs_to_show = n_segments if show_top_knot else n_segments - 1

        for seg_idx in range(n_segs_to_show):
            k_lower = kpts[seg_idx]
            k_upper = kpts[seg_idx + 1]
            label = f"{base_label}_k{seg_idx}({k_lower:.2f}-{k_upper:.2f})"
            labels.append(label)

    return labels

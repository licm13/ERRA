"""Shared utility functions for the ERRA core implementation.

用于 ERRA 核心实现的通用辅助函数。

The helpers in this module are imported by :mod:`erra.erra_core` and by any
legacy entry points so that there is a single authoritative implementation.

本模块中的辅助函数由 :mod:`erra.erra_core` 以及旧版入口共同使用，
保证实现的一致性。
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def convert_to_numpy_array(
    arr: Sequence[float] | pd.Series | np.ndarray,
) -> np.ndarray:
    """Return *arr* as a one-dimensional ``float`` NumPy array.

    将输入 ``arr`` 转换为一维 ``float`` 类型的 NumPy 数组。
    """

    if isinstance(arr, np.ndarray):
        return np.asarray(arr, dtype=float).ravel()
    if isinstance(arr, pd.Series):
        return arr.to_numpy(dtype=float)
    return np.asarray(list(arr), dtype=float)


def prepare_precipitation_matrix(
    precipitation: Iterable[Sequence[float]]
    | pd.DataFrame
    | pd.Series
    | np.ndarray,
    labels: Optional[Sequence[str]],
) -> Tuple[np.ndarray, Sequence[str]]:
    """Normalise precipitation inputs to a two-dimensional matrix.

    将降水输入标准化为二维矩阵，并返回对应的列名。
    """

    if isinstance(precipitation, pd.DataFrame):
        matrix = precipitation.to_numpy(dtype=float)
        column_labels = list(precipitation.columns)
    elif isinstance(precipitation, pd.Series):
        matrix = precipitation.to_numpy(dtype=float)[:, None]
        column_labels = [precipitation.name or "p1"]
    else:
        array = np.asarray(precipitation, dtype=float)
        if array.ndim == 1:
            matrix = array[:, None]
        elif array.ndim == 2:
            matrix = array
        else:
            raise ValueError("precipitation input must be 1-D or 2-D")
        column_labels = [f"p{i + 1}" for i in range(matrix.shape[1])]

    if labels is not None:
        if len(labels) != matrix.shape[1]:
            raise ValueError("labels must match the number of precipitation columns")
        column_labels = list(labels)

    return matrix, column_labels


def aggregate_time_series(
    precipitation: np.ndarray,
    discharge: np.ndarray,
    weights: Optional[Sequence[float]],
    aggregation_window: int,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Aggregate time series by summing precipitation and averaging discharge.

    将降水求和、流量与权重取平均，实现 ``aggregation_window`` 长度的时间聚合。
    """

    total_steps = precipitation.shape[0]
    trimmed_steps = total_steps - (total_steps % aggregation_window)
    if trimmed_steps == 0:
        raise ValueError("time series shorter than aggregation window")

    precip_trimmed = precipitation[:trimmed_steps]
    discharge_trimmed = discharge[:trimmed_steps]

    precip_aggregated = precip_trimmed.reshape(
        trimmed_steps // aggregation_window,
        aggregation_window,
        precipitation.shape[1],
    ).sum(axis=1)
    discharge_aggregated = discharge_trimmed.reshape(
        trimmed_steps // aggregation_window,
        aggregation_window,
    ).mean(axis=1)

    if weights is None:
        weights_aggregated = None
    else:
        weight_array = convert_to_numpy_array(weights)[:trimmed_steps]
        weights_aggregated = weight_array.reshape(
            trimmed_steps // aggregation_window,
            aggregation_window,
        ).mean(axis=1)

    return precip_aggregated, discharge_aggregated, weights_aggregated


def apply_quantile_filter(
    discharge: np.ndarray,
    quantile: float,
    window: int,
) -> np.ndarray:
    """Remove slow trends from discharge using a running quantile filter.

    使用滑动分位滤波去除流量序列中的缓慢趋势。
    """

    if quantile <= 0.0:
        return discharge
    if quantile >= 1.0:
        raise ValueError("fq must be in [0, 1)")

    series = pd.Series(discharge)
    window = max(3, min(window, len(series)))
    if quantile == 0.5:
        trend = series.rolling(window, center=True, min_periods=1).median()
    else:
        trend = series.rolling(window, center=True, min_periods=1).quantile(quantile)
    return (series - trend).to_numpy(dtype=float)


def build_design_matrix(
    precipitation: np.ndarray,
    discharge: np.ndarray,
    weights: np.ndarray,
    max_lag: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct the regression design matrix for the ERRA solver.

    根据 ``max_lag`` 构建 ERRA 回归所需的设计矩阵，并同步过滤缺测值。
    """

    n_timesteps, n_drivers = precipitation.shape
    n_columns = n_drivers * (max_lag + 1)
    design = np.zeros((n_timesteps, n_columns), dtype=float)

    for lag in range(max_lag + 1):
        shifted = np.roll(precipitation, shift=lag, axis=0)
        shifted[:lag, :] = np.nan
        design[:, lag * n_drivers : (lag + 1) * n_drivers] = shifted

    valid_rows = (~np.isnan(design).any(axis=1)) & (~np.isnan(discharge))
    design = design[valid_rows]
    response = discharge[valid_rows]
    filtered_weights = weights[valid_rows]

    mean_weight = filtered_weights.mean()
    if not np.isfinite(mean_weight) or mean_weight <= 0:
        filtered_weights = np.ones_like(filtered_weights)
    else:
        filtered_weights = filtered_weights / mean_weight

    return design, response, filtered_weights


def create_tikhonov_regularization_matrix(lag_count: int) -> np.ndarray:
    """Return the second-order difference operator used for smoothing.

    返回用于 Tikhonov 平滑的二阶差分算子矩阵。
    """

    if lag_count <= 2:
        return np.eye(lag_count, dtype=float)
    operator = np.zeros((lag_count - 2, lag_count), dtype=float)
    for idx in range(lag_count - 2):
        operator[idx, idx : idx + 3] = np.array([1.0, -2.0, 1.0])
    return operator


def solve_rrd(
    design: np.ndarray,
    response: np.ndarray,
    weights: np.ndarray,
    regularisation: float,
    max_lag: int,
    n_drivers: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Solve the weighted ridge regression for the runoff response distribution.

    求解带权岭回归以获得径流响应分布，并返回系数、标准误、拟合值和残差。
    """

    weight_sqrt = np.sqrt(weights)[:, None]
    design_weighted = design * weight_sqrt
    response_weighted = response * weight_sqrt[:, 0]

    beta_size = (max_lag + 1) * n_drivers
    xtx = design_weighted.T @ design_weighted
    xty = design_weighted.T @ response_weighted

    if regularisation > 0.0:
        operator = create_tikhonov_regularization_matrix(max_lag + 1)
        block_operator = np.kron(np.eye(n_drivers), operator)
        regulariser = regularisation * (block_operator.T @ block_operator)
        xtx = xtx + regulariser

    try:
        coefficients = np.linalg.solve(xtx, xty)
    except np.linalg.LinAlgError as exc:  # pragma: no cover - defensive branch
        raise np.linalg.LinAlgError(
            "Regression matrix is singular; consider increasing nu or reducing m",
        ) from exc

    fitted = design @ coefficients
    residuals = response - fitted

    degrees_of_freedom = max(len(response) - beta_size, 1)
    sigma_squared = float((weights * residuals**2).sum() / degrees_of_freedom)

    try:
        inv_xtx = np.linalg.inv(xtx)
    except np.linalg.LinAlgError:  # pragma: no cover - fallback branch
        inv_xtx = np.linalg.pinv(xtx)
    covariance = inv_xtx * sigma_squared
    stderr = np.sqrt(np.maximum(np.diag(covariance), 0.0))

    coefficients = coefficients.reshape(max_lag + 1, n_drivers)
    stderr = stderr.reshape(max_lag + 1, n_drivers)

    return coefficients, stderr, fitted, residuals


def to_rrd_dataframe(arr: np.ndarray, labels: Sequence[str]) -> pd.DataFrame:
    """Convert a coefficient array to a tidy DataFrame indexed by lag.

    将系数数组转换为以时滞为索引的整洁 ``DataFrame``。
    """

    data = {label: arr[:, idx] for idx, label in enumerate(labels)}
    dataframe = pd.DataFrame(data)
    dataframe.index.name = "lag"
    return dataframe

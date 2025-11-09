"""Compare ERRA with a simple gradient-boosted tree rainfall-runoff model.

This module prepares machine-learning friendly features from the regional
case-study dataset (see ``climate_response_case_studies.py``) and trains a
lightweight gradient boosted decision stump ensemble that mimics the
behaviour of XGBoost. The goal is to contrast predictive skill and
interpretability between the data-driven model and ERRA's physically
interpretable runoff response distribution.

The gradient boosting implementation intentionally stays dependency-light: it
optimises one-level decision trees (stumps) on squared-error loss with
shrinkage, capturing the flavour of XGBoost without external libraries.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    examples_dir = Path(__file__).resolve().parent
    sys.path.append(str(examples_dir))
    sys.path.append(str(examples_dir.parent))
    from erra import erra
    import climate_response_case_studies as crcs
else:
    from ..erra import erra
    from . import climate_response_case_studies as crcs


@dataclass
class DecisionStump:
    feature: int
    threshold: float
    left_value: float
    right_value: float


class SimpleStumpGBDT:
    """Minimal gradient boosted ensemble built from decision stumps."""

    def __init__(self, n_estimators: int = 60, learning_rate: float = 0.1, min_samples_leaf: int = 3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.min_samples_leaf = min_samples_leaf
        self.base_value_: float = 0.0
        self.stumps_: List[DecisionStump] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SimpleStumpGBDT":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2-D array")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows")

        self.base_value_ = float(y.mean())
        prediction = np.full_like(y, self.base_value_, dtype=float)
        self.stumps_ = []

        for _ in range(self.n_estimators):
            residuals = y - prediction
            stump = self._fit_stump(X, residuals)
            if stump is None:
                break
            self.stumps_.append(stump)
            update = np.where(
                X[:, stump.feature] <= stump.threshold,
                stump.left_value,
                stump.right_value,
            )
            prediction = prediction + self.learning_rate * update
        return self

    def _fit_stump(self, X: np.ndarray, residuals: np.ndarray) -> DecisionStump | None:
        best_loss = np.inf
        best_stump: DecisionStump | None = None
        n_samples, n_features = X.shape
        for j in range(n_features):
            feature_values = X[:, j]
            order = np.argsort(feature_values)
            sorted_values = feature_values[order]
            sorted_residuals = residuals[order]
            thresholds = (sorted_values[1:] + sorted_values[:-1]) / 2.0
            for idx, thr in enumerate(thresholds):
                left = sorted_residuals[: idx + 1]
                right = sorted_residuals[idx + 1 :]
                if len(left) < self.min_samples_leaf or len(right) < self.min_samples_leaf:
                    continue
                left_val = left.mean()
                right_val = right.mean()
                pred = np.concatenate([
                    np.full_like(left, left_val),
                    np.full_like(right, right_val),
                ])
                loss = np.mean((sorted_residuals - pred) ** 2)
                if loss < best_loss:
                    best_loss = loss
                    best_stump = DecisionStump(j, float(thr), float(left_val), float(right_val))
        return best_stump

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.stumps_:
            return np.full(X.shape[0], self.base_value_, dtype=float)
        X = np.asarray(X, dtype=float)
        pred = np.full(X.shape[0], self.base_value_, dtype=float)
        for stump in self.stumps_:
            update = np.where(
                X[:, stump.feature] <= stump.threshold,
                stump.left_value,
                stump.right_value,
            )
            pred += self.learning_rate * update
        return pred

    def feature_importances(self, n_features: int) -> np.ndarray:
        counts = np.zeros(n_features, dtype=float)
        if not self.stumps_:
            return counts
        for stump in self.stumps_:
            counts[stump.feature] += 1
        return counts / counts.sum()


@dataclass
class MLComparison:
    catchment: str
    climate: str
    erra_rmse: float
    ml_rmse: float
    erra_bias: float
    ml_bias: float
    feature_rankings: List[Tuple[str, float]]


FEATURE_NAMES = [
    "precip_mm",
    "pet_mm",
    "storage_change_mm",
    "previous_runoff_mm",
    "rolling_precip_3",
    "dryness_index",
]


def build_features_for_catchment(group: pd.DataFrame, lag: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    features: List[List[float]] = []
    targets: List[float] = []
    for i in range(lag, len(group)):
        prev_runoff = group["runoff_mm"].iloc[i - 1]
        rolling_precip = group["precip_mm"].iloc[max(0, i - 3) : i].sum()
        dryness = group["dryness_index"].iloc[i]
        if not np.isfinite(dryness):
            dryness = 10.0
        features.append([
            float(group["precip_mm"].iloc[i]),
            float(group["pet_mm"].iloc[i]),
            float(group["storage_change_mm"].iloc[i]),
            float(prev_runoff),
            float(rolling_precip),
            float(dryness),
        ])
        targets.append(float(group["runoff_mm"].iloc[i]))
    return np.asarray(features, dtype=float), np.asarray(targets, dtype=float)


def apply_rrd_forecast(kernel: np.ndarray, history: Sequence[float], new_precip: Sequence[float]) -> np.ndarray:
    """Convolve an ERRA kernel with precipitation to forecast runoff."""

    kernel = np.asarray(kernel, dtype=float)
    memory = len(kernel) - 1
    history = list(history)
    if memory > 0:
        history = history[-memory:]
    preds: List[float] = []
    buffer = list(history)
    for p in new_precip:
        buffer.append(float(p))
        recent = buffer[-(memory + 1) :]
        arr = np.asarray(recent[::-1], dtype=float)
        kernel_slice = kernel[: arr.size]
        preds.append(float(np.dot(arr, kernel_slice)))
    return np.asarray(preds, dtype=float)


def evaluate_ml_vs_erra(metrics: pd.DataFrame, train_fraction: float = 0.75) -> List[MLComparison]:
    comparisons: List[MLComparison] = []
    for catchment, group in metrics.groupby("catchment"):
        group = group.reset_index(drop=True)
        features, targets = build_features_for_catchment(group)
        if len(targets) < 6:
            continue
        split_idx = int(len(targets) * train_fraction)
        split_idx = max(split_idx, 4)
        X_train, y_train = features[:split_idx], targets[:split_idx]
        X_test, y_test = features[split_idx:], targets[split_idx:]

        model = SimpleStumpGBDT(n_estimators=80, learning_rate=0.1, min_samples_leaf=2)
        model.fit(X_train, y_train)
        y_pred_ml = model.predict(X_test)

        train_end = split_idx + 1  # account for the one-step lag in features
        precip_train = group["precip_mm"].iloc[:train_end].to_numpy()
        runoff_train = group["runoff_mm"].iloc[:train_end].to_numpy()
        precip_test = group["precip_mm"].iloc[train_end:].to_numpy()

        erra_result = erra(
            p=precip_train,
            q=runoff_train,
            m=6,
            nu=0.05,
            dt=1.0,
            labels=[catchment],
        )
        kernel = erra_result.rrd.iloc[:, 0].to_numpy()
        history = precip_train[-(len(kernel) - 1) :]
        erra_forecast = apply_rrd_forecast(kernel, history, precip_test)

        erra_rmse = float(np.sqrt(np.mean((y_test - erra_forecast) ** 2)))
        ml_rmse = float(np.sqrt(np.mean((y_test - y_pred_ml) ** 2)))
        erra_bias = float(np.mean(erra_forecast - y_test))
        ml_bias = float(np.mean(y_pred_ml - y_test))

        importances = model.feature_importances(len(FEATURE_NAMES))
        rankings = sorted(
            zip(FEATURE_NAMES, importances),
            key=lambda kv: kv[1],
            reverse=True,
        )

        comparisons.append(
            MLComparison(
                catchment=catchment,
                climate=group["climate"].iloc[0],
                erra_rmse=erra_rmse,
                ml_rmse=ml_rmse,
                erra_bias=erra_bias,
                ml_bias=ml_bias,
                feature_rankings=rankings,
            )
        )
    return comparisons


def print_ml_summary(comparisons: Iterable[MLComparison]) -> None:
    print("\nMachine learning vs ERRA cross-validation:")
    print("-" * 96)
    print(
        "{:<30s} {:<18s} {:>10s} {:>10s} {:>10s} {:>10s}".format(
            "Catchment",
            "Climate",
            "ERRA RMSE",
            "ML RMSE",
            "ERRA bias",
            "ML bias",
        )
    )
    print("-" * 96)
    for item in comparisons:
        print(
            f"{item.catchment[:30]:30s} {item.climate[:18]:18s} "
            f"{item.erra_rmse:10.3f} {item.ml_rmse:10.3f} {item.erra_bias:10.3f} {item.ml_bias:10.3f}"
        )
    print("-" * 96)
    print("\nFeature usage in the boosted tree (relative frequency):")
    for item in comparisons:
        print(f"\n{item.catchment} ({item.climate})")
        for name, score in item.feature_rankings:
            if score <= 0:
                continue
            print(f"  - {name}: {score:.2f}")


def main() -> None:
    metrics = crcs.compute_budyko_metrics(crcs.load_case_study_data())
    comparisons = evaluate_ml_vs_erra(metrics)
    print_ml_summary(comparisons)
    print(
        "\nInterpretation: ERRA encodes hydro-logic memory explicitly through the kernel,"
        " while the boosted tree leans on autoregressive features but lacks the"
        " Budyko-consistent diagnostics provided by ERRA."
    )


if __name__ == "__main__":
    main()

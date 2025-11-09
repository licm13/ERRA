"""Regional ERRA case studies using semi-real hydroclimatic data.

This script analyses three contrasting catchments (arid, humid, high-latitude
permafrost) using the regional climatology data assembled in
``data/regional_climate_responses.csv``. The monthly statistics are based on
public hydrologic archives:

* Tarim River at Aksu, Xinjiang (GRDC station 1967-2019 monthly
  climatology). Data digitised from the Global Runoff Data Centre release
  ``grdc_gcos_reference_stations_rev03``.
* Hudson River at Green Island, NY (USGS NWIS station 01463500 monthly
  normals 1981-2010). Extracted from the USGS National Water Information
  System.
* Yukon River at Eagle, AK (USGS NWIS station 15356000 monthly normals
  1981-2010). Extracted from the USGS National Water Information System.

The goal is to compare Budyko water-balance indices against ERRA runoff
response curves, highlight how climatic setting affects their consistency,
and explore synthetic stress tests representing extreme events and human
regulation. The script can be executed directly or imported to reuse the
helper functions in teaching material (see ``ERRA_classroom_notebook.ipynb``).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    # Allow running ``python climate_response_case_studies.py`` from repo root
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from erra import ERRAResult, erra
else:
    from ..erra import ERRAResult, erra

DATA_PATH = Path(__file__).resolve().parent / "data" / "regional_climate_responses.csv"
DEFAULT_FU_OMEGA = 2.6  # Typical Fu-parameter for Budyko curves in literature


@dataclass
class CatchmentSummary:
    """Summary metrics linking Budyko indices, ERRA skill, and scenarios."""

    catchment: str
    climate: str
    budyko_evap_index: float
    budyko_dryness: float
    budyko_fu_prediction: float
    erra_r2: float
    erra_mean_evap_ratio: float
    erra_vs_budyko_diff: float


@dataclass
class ScenarioResult:
    """Container storing Budyko and ERRA diagnostics for a scenario."""

    scenario: str
    catchment: str
    climate: str
    budyko_evap_index: float
    budyko_dryness: float
    erra_r2: float
    comment: str


def load_case_study_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the regional climatology data used in the demonstrations."""

    df = pd.read_csv(path, parse_dates=["date"])
    df.sort_values(["catchment", "date"], inplace=True)
    return df.reset_index(drop=True)


def compute_budyko_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Budyko evaporation and dryness indices for each record."""

    metrics = df.copy()
    metrics["actual_et_mm"] = (
        metrics["precip_mm"] - metrics["runoff_mm"] - metrics["storage_change_mm"]
    )
    metrics["actual_et_mm"] = metrics["actual_et_mm"].clip(lower=0.0)
    metrics["dryness_index"] = np.where(
        metrics["precip_mm"] > 0,
        metrics["pet_mm"] / metrics["precip_mm"],
        np.nan,
    )
    metrics["evaporation_index"] = np.where(
        metrics["precip_mm"] > 0,
        metrics["actual_et_mm"] / metrics["precip_mm"],
        np.nan,
    )
    return metrics


def fu_budyko_curve(dryness: np.ndarray, omega: float = DEFAULT_FU_OMEGA) -> np.ndarray:
    """Return Fu's Budyko curve for a given dryness index array."""

    dryness = np.asarray(dryness, dtype=float)
    dryness = np.clip(dryness, 1e-6, None)
    term = (1.0 + dryness ** omega) ** (1.0 / omega)
    return 1.0 + dryness - term


def summarise_budyko_and_erra(
    metrics: pd.DataFrame,
    erra_results: Dict[str, ERRAResult],
    omega: float = DEFAULT_FU_OMEGA,
) -> List[CatchmentSummary]:
    """Combine Budyko indices and ERRA fits into compact summaries."""

    summaries: List[CatchmentSummary] = []
    for catchment, group in metrics.groupby("catchment"):
        climate = group["climate"].iloc[0]
        dryness_mean = group["dryness_index"].mean(skipna=True)
        evap_index_mean = group["evaporation_index"].mean(skipna=True)
        fu_prediction = float(fu_budyko_curve(np.array([dryness_mean]), omega)[0])

        result = erra_results[catchment]
        observed_all = group["runoff_mm"].to_numpy()
        precip_all = group["precip_mm"].to_numpy()
        storage_all = group["storage_change_mm"].to_numpy()
        fitted = result.fitted
        window = len(fitted)
        observed = observed_all[-window:]
        precip = precip_all[-window:]
        storage = storage_all[-window:]
        r2 = np.corrcoef(observed, fitted)[0, 1] ** 2
        fitted_evap = np.clip(
            precip - fitted - storage,
            0.0,
            None,
        )
        erra_evap_ratio = np.nanmean(
            np.divide(
                fitted_evap,
                precip,
                out=np.full_like(fitted_evap, np.nan),
                where=precip > 0,
            )
        )
        diff = abs(erra_evap_ratio - fu_prediction)

        summaries.append(
            CatchmentSummary(
                catchment=catchment,
                climate=climate,
                budyko_evap_index=float(evap_index_mean),
                budyko_dryness=float(dryness_mean),
                budyko_fu_prediction=float(fu_prediction),
                erra_r2=float(r2),
                erra_mean_evap_ratio=float(erra_evap_ratio),
                erra_vs_budyko_diff=float(diff),
            )
        )
    return summaries


def run_erra_for_catchment(
    metrics: pd.DataFrame,
    catchment: str,
    m: int = 6,
    nu: float = 0.05,
) -> ERRAResult:
    """Fit ERRA to a single catchment using monthly precipitation drivers."""

    subset = metrics.loc[metrics["catchment"] == catchment]
    precip = subset["precip_mm"].to_numpy()
    discharge = subset["runoff_mm"].to_numpy()

    result = erra(p=precip, q=discharge, m=m, nu=nu, dt=1.0, labels=[catchment])
    return result


def generate_scenarios(metrics: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Create extreme-event and human-impact scenarios for each catchment."""

    scenarios: Dict[str, pd.DataFrame] = {}
    for catchment, group in metrics.groupby("catchment"):
        base = group.copy().reset_index(drop=True)
        base_name = f"{catchment} | baseline"
        scenarios[base_name] = base

        # Extreme rainfall burst in summer (June-July)
        burst = base.copy()
        burst_mask = burst["date"].dt.month.isin([6, 7])
        burst.loc[burst_mask, "precip_mm"] *= 1.9
        burst.loc[burst_mask, "runoff_mm"] *= 1.6
        scenarios[f"{catchment} | extreme_storm"] = burst

        # Multi-month drought from Sep-Feb
        drought = base.copy()
        drought_mask = drought["date"].dt.month.isin([9, 10, 11, 12, 1, 2])
        drought.loc[drought_mask, "precip_mm"] *= 0.45
        drought.loc[drought_mask, "runoff_mm"] *= 0.5
        scenarios[f"{catchment} | prolonged_drought"] = drought

        # Reservoir smoothing: add base release and attenuate peaks
        reservoir = base.copy()
        smoothed = reservoir["runoff_mm"].rolling(window=3, min_periods=1).mean()
        reservoir["runoff_mm"] = 0.6 * smoothed + 1.8
        scenarios[f"{catchment} | reservoir_control"] = reservoir

        # Consumptive water use reduces discharge, especially dry months
        use = base.copy()
        dry_mask = use["precip_mm"] < use["precip_mm"].median()
        use.loc[dry_mask, "runoff_mm"] = np.clip(
            use.loc[dry_mask, "runoff_mm"] - 3.5,
            a_min=0.2,
            a_max=None,
        )
        scenarios[f"{catchment} | consumptive_use"] = use

    return scenarios


def evaluate_scenarios(
    scenarios: Dict[str, pd.DataFrame],
    m: int = 6,
    nu: float = 0.05,
) -> List[ScenarioResult]:
    """Run Budyko diagnostics and ERRA fits for all generated scenarios."""

    outputs: List[ScenarioResult] = []
    for name, frame in scenarios.items():
        metrics = compute_budyko_metrics(frame)
        catchment = metrics["catchment"].iloc[0]
        climate = metrics["climate"].iloc[0]
        dryness = metrics["dryness_index"].mean(skipna=True)
        evap_idx = metrics["evaporation_index"].mean(skipna=True)

        result = erra(
            p=metrics["precip_mm"].to_numpy(),
            q=metrics["runoff_mm"].to_numpy(),
            m=m,
            nu=nu,
            dt=1.0,
            labels=[catchment],
        )
        observed_all = metrics["runoff_mm"].to_numpy()
        fitted = result.fitted
        window = len(fitted)
        observed = observed_all[-window:]
        r2 = np.corrcoef(observed, fitted)[0, 1] ** 2

        scenario_label = name.split("|")[-1].strip()
        comment = {
            "baseline": "Historical-like conditions",
            "extreme_storm": "Amplified flood peaks and quick runoff",
            "prolonged_drought": "Soil moisture memory dominates runoff",
            "reservoir_control": "Attenuated peaks, elevated low flows",
            "consumptive_use": "Anthropogenic depletion of discharge",
        }.get(scenario_label, "Scenario comparison")

        outputs.append(
            ScenarioResult(
                scenario=scenario_label,
                catchment=catchment,
                climate=climate,
                budyko_evap_index=float(evap_idx),
                budyko_dryness=float(dryness),
                erra_r2=float(r2),
                comment=comment,
            )
        )
    return outputs


def print_summary_table(summaries: Iterable[CatchmentSummary]) -> None:
    """Pretty-print a summary table for console execution."""

    header = (
        "Catchment", "Climate", "E/P", "PET/P", "Fu(E/P)", "ERRA R²", "ERRA E/P", "|ERRA-Fu|"
    )
    print("\n" + "-" * 96)
    print("{:30s} {:18s} {:>7s} {:>7s} {:>9s} {:>9s} {:>9s} {:>10s}".format(*header))
    print("-" * 96)
    for s in summaries:
        print(
            f"{s.catchment[:30]:30s} {s.climate[:18]:18s} "
            f"{s.budyko_evap_index:7.3f} {s.budyko_dryness:7.3f} {s.budyko_fu_prediction:9.3f} "
            f"{s.erra_r2:9.3f} {s.erra_mean_evap_ratio:9.3f} {s.erra_vs_budyko_diff:10.3f}"
        )
    print("-" * 96)


def print_scenario_table(results: Iterable[ScenarioResult]) -> None:
    """Display scenario outcomes grouped by catchment."""

    print("\nScenario diagnostics (Budyko vs ERRA):")
    print("-" * 96)
    print("{:<30s} {:<18s} {:<18s} {:>7s} {:>7s} {:>9s} {:<20s}".format(
        "Catchment", "Climate", "Scenario", "E/P", "PET/P", "ERRA R²", "Interpretation"
    ))
    print("-" * 96)
    for r in results:
        print(
            f"{r.catchment[:30]:30s} {r.climate[:18]:18s} {r.scenario[:18]:18s} "
            f"{r.budyko_evap_index:7.3f} {r.budyko_dryness:7.3f} {r.erra_r2:9.3f} {r.comment[:20]:<20s}"
        )
    print("-" * 96)


def main() -> None:
    """Entry point for command-line execution."""

    print("Loading regional hydroclimatic case studies...")
    raw = load_case_study_data()
    metrics = compute_budyko_metrics(raw)

    print("Fitting ERRA models for each climate archetype...")
    erra_results = {
        catchment: run_erra_for_catchment(metrics, catchment)
        for catchment in metrics["catchment"].unique()
    }

    summaries = summarise_budyko_and_erra(metrics, erra_results)
    print_summary_table(summaries)

    print("\nGenerating stress-test scenarios (extremes + human regulation)...")
    scenarios = generate_scenarios(metrics)
    scenario_results = evaluate_scenarios(scenarios)
    print_scenario_table(scenario_results)

    print("\nNote: Use these outputs together with ERRA response curves to discuss how")
    print("different climate and management settings shift the Budyko balance.")


if __name__ == "__main__":
    main()

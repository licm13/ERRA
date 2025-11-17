# CLAUDE.md - AI Assistant Guide for ERRA Repository

> **Purpose**: Comprehensive guide for AI coding assistants (Claude, Copilot, etc.) to understand and contribute effectively to the ERRA Python package.

**Last Updated**: 2025-11-17
**Repository**: https://github.com/licm13/ERRA
**Package**: `erra-hydrology` (Python 3.9+)

---

## Table of Contents

1. [Quick Orientation](#quick-orientation)
2. [Codebase Architecture](#codebase-architecture)
3. [Code Conventions & Patterns](#code-conventions--patterns)
4. [Development Workflows](#development-workflows)
5. [Testing Guidelines](#testing-guidelines)
6. [Bilingual Documentation](#bilingual-documentation)
7. [Common Tasks](#common-tasks)
8. [File Reference](#file-reference)
9. [Git Workflow](#git-workflow)

---

## Quick Orientation

### What is ERRA?

**Ensemble Rainfall-Runoff Analysis (ERRA)** is a scientific Python package for hydrologic analysis that estimates Runoff Response Distributions (RRDs) - impulse response functions describing how catchments transform precipitation into streamflow.

### Package Structure (5-Minute Overview)

```
ERRA/
├── src/erra/                    # Main Python package (installable)
│   ├── __init__.py              # Public API: erra(), ERRAResult, plot_erra_results()
│   ├── erra_core.py             # Core algorithm (843 lines)
│   ├── utils_core.py            # Shared utilities for core (246 lines)
│   ├── nonlin.py                # Nonlinear Response Functions (345 lines)
│   ├── splitting.py             # Non-stationary splitting (437 lines)
│   └── utils.py                 # Plotting & visualization (500 lines)
│
├── code/examples/               # Example scripts (use installed package)
│   ├── master_demonstration.py  # Comprehensive demo of all features
│   ├── gao2025_*.py             # Scientific case study examples
│   ├── sharif_ameli2025_*.py
│   ├── tu2025_*.py
│   └── data_prep/               # Data fetching scripts (USGS, NOAA)
│
├── tests/                       # Unit tests (pytest)
│   └── test_splitting.py        # Currently minimal coverage
│
├── reference_materials/         # Original R code, papers, theory PDFs
│   ├── R_implementation/        # Original R scripts by James Kirchner
│   ├── papers/                  # Application papers
│   └── theory_pdfs/             # Theory documentation
│
├── pyproject.toml               # Package config, dependencies, tooling
├── README.md                    # User documentation (bilingual)
└── .github/
    └── copilot-instructions.md  # Quick reference (also consult this!)
```

### Core Concepts

- **RRD (Runoff Response Distribution)**: The impulse response function estimated by ridge regression
- **Nonlinear Analysis (xknots)**: Splitting by precipitation intensity to capture non-linear behavior
- **Non-stationary Analysis (split_params)**: Splitting by covariates (e.g., wetness) to capture temporal changes
- **Robust Estimation (IRLS)**: Iteratively Reweighted Least Squares for outlier resistance
- **Broken-stick (nk)**: Compressed lag representation for computational efficiency
- **Tikhonov Regularization (nu)**: Smoothing parameter to handle ill-conditioned problems

### Key Dependencies

```toml
numpy>=1.20.0      # Core numerical operations
pandas>=1.3.0      # Time series, DataFrame results
scipy>=1.7.0       # Sparse matrices, statistics
matplotlib>=3.4.0  # Visualization
```

---

## Codebase Architecture

### Module Responsibilities

#### `src/erra/__init__.py` (50 lines)
- **Role**: Public API surface
- **Exports**: `erra()`, `ERRAResult`, `plot_erra_results()`, `__version__`
- **Pattern**: Minimal, clean interface

#### `src/erra/erra_core.py` (843 lines)
- **Role**: Core ERRA implementation
- **Key Components**:
  - `ERRAResult` dataclass: Container for analysis outputs
  - `erra()` function: Main API (15+ parameters)
  - `_solve_rrd_robust()`: IRLS implementation
  - `_broken_stick()`: Lag compression
- **Pattern**: Single entry point with optional features via parameters

#### `src/erra/utils_core.py` (246 lines)
- **Role**: Shared utilities for core algorithm
- **Key Functions**:
  - `prepare_precipitation_matrix()`: Input normalization to 2D numpy
  - `build_design_matrix()`: Sliding window lag matrix construction
  - `solve_rrd()`: Ridge regression solver
  - `aggregate_time_series()`: Temporal aggregation
  - `apply_quantile_filter()`: Detrending
- **Pattern**: Pure functions, no side effects

#### `src/erra/nonlin.py` (345 lines)
- **Role**: Nonlinear Response Function (NRF) analysis
- **Key Functions**:
  - `create_xprime_matrix()`: Transform p(t) → x'(t) for intensity segments
  - `betaprime_to_nrf()`: Convert β' coefficients to NRF
  - `create_nrf_labels()`: Generate descriptive labels
- **Algorithm**: Implements Eq. 43 from Kirchner (2022)

#### `src/erra/splitting.py` (437 lines)
- **Role**: Non-stationary analysis via covariate splitting
- **Key Functions**:
  - `make_split_sets()`: Main splitting logic with hierarchical percentiles
  - `validate_split_params()`: Input validation
- **Features**: Global fallback, logging for small bins

#### `src/erra/utils.py` (500 lines)
- **Role**: Visualization and plotting
- **Key Functions**:
  - `plot_erra_results()`: Main plotting orchestrator
  - `_plot_rrd_with_error_bars()`: RRD visualization
  - `_plot_fitted_vs_observed()`: Model diagnostics
  - `_plot_residuals_analysis()`: Residual diagnostics
  - `_configure_chinese_fonts()`: Bilingual support
- **Pattern**: Isolated from core algorithm, optional dependency on matplotlib

### Data Flow

```
User Input (p, q, params)
    ↓
Input Normalization (utils_core: prepare_precipitation_matrix, convert_to_numpy_array)
    ↓
Optional: Time Aggregation (utils_core: aggregate_time_series if agg > 1)
    ↓
    ├─→ OPTION A: Splitting (splitting: make_split_sets)
    │   Output: p_split, labels, criteria
    │
    └─→ OPTION B: Nonlinear (nonlin: create_xprime_matrix)
        Output: p_xprime, xknot_values
    ↓
Quantile Filtering (utils_core: apply_quantile_filter if fq > 0)
    ↓
Design Matrix Construction (utils_core: build_design_matrix)
    Output: design (n × k(m+1)), response, weights
    ↓
    ├─→ Standard Solver (utils_core: solve_rrd)
    │
    └─→ Robust Solver (erra_core: _solve_rrd_robust with IRLS)
    ↓
Post-processing
    - If nonlinear: betaprime_to_nrf()
    - If broken-stick: _broken_stick()
    - Always: to_rrd_dataframe()
    ↓
Return ERRAResult
    ↓
Optional: Visualization (utils: plot_erra_results)
```

### Key Design Patterns

1. **Separation of Concerns**: Core algorithm, utilities, nonlinear, splitting, visualization all separate
2. **Flexible Input Handling**: Accepts pandas DataFrame/Series, numpy arrays, lists
3. **Typed Results**: All outputs in `ERRAResult` dataclass
4. **Optional Features**: Advanced features enabled via parameters (not separate functions)
5. **Defensive Programming**: Extensive validation with helpful error messages
6. **Memory Efficiency**: In-place operations where safe, copies where needed

---

## Code Conventions & Patterns

### Python Style

**Target Version**: Python 3.9+
**Line Length**: 88 characters (Black default)
**Formatter**: Black
**Linter**: Ruff + Flake8
**Type Checker**: mypy with `disallow_untyped_defs = true`

### Type Hints (REQUIRED)

All functions must have complete type hints:

```python
from __future__ import annotations  # ALWAYS first import

from typing import Dict, Iterable, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

def example_function(
    p: Iterable[Sequence[float]] | pd.DataFrame | pd.Series | np.ndarray,
    q: Sequence[float] | pd.Series | np.ndarray,
    m: int = 60,
    labels: Optional[list[str]] = None,
    mode: Literal["linear", "nonlinear"] = "linear",
) -> ERRAResult:
    """Function docstring here."""
    pass
```

**Conventions**:
- Use `from __future__ import annotations` in every module
- Use `|` for union types (Python 3.10+ style, enabled by future annotations)
- Use `Optional[X]` for nullable parameters
- Use `Literal` for string enums
- Prefer specific container types (`Sequence`, `Iterable`) over generic `list`, `tuple`

### Dataclass Pattern

Use `@dataclass` for structured results:

```python
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class ERRAResult:
    """Container for ERRA outputs / ERRA 结果容器。

    Attributes / 属性
    -----------------
    lags : np.ndarray
        Lag times corresponding to RRD coefficients (time units follow input data)
        RRD 系数对应的时滞（单位与输入时间步一致）
    rrd : pd.DataFrame
        Estimated runoff response distributions for each precipitation driver
        每个降水驱动变量的径流响应分布估计值
    """
    # Required fields first
    lags: np.ndarray
    rrd: pd.DataFrame
    fitted: np.ndarray

    # Optional fields with defaults
    lag_knots: Optional[pd.DataFrame] = None
    nrf: Optional[pd.DataFrame] = None
```

### Import Organization

```python
# 1. Future imports (always first)
from __future__ import annotations

# 2. Standard library
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

# 3. Third-party (alphabetical)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# 4. Internal (relative imports, alphabetical)
from .nonlin import create_xprime_matrix
from .splitting import make_split_sets
from .utils_core import solve_rrd
```

### Error Handling

**Pattern**: Fail fast with descriptive `ValueError`, include actionable suggestions:

```python
# Good: Specific error with actionable message
if fq < 0 or fq >= 1.0:
    raise ValueError(
        f"fq must be in [0, 1), got {fq}. "
        "Use fq=0 for no filtering or fq=0.1-0.3 to remove baseflow."
    )

# Good: Exception chaining for debugging
try:
    beta = np.linalg.solve(XtX, Xty)
except np.linalg.LinAlgError as exc:
    raise np.linalg.LinAlgError(
        "Regression matrix is singular; consider increasing nu (regularization) "
        f"or reducing m (max lag). Current: nu={nu}, m={m}"
    ) from exc

# Bad: Generic error without context
if fq >= 1:
    raise ValueError("Invalid fq")
```

### Magic Numbers

**Pattern**: Named constants with explanatory comments:

```python
# Constants at module level (UPPER_CASE)
_MAD_TO_STD_FACTOR = 1.4826  # Makes MAD consistent with std for normal distribution
_HUBER_TUNING_CONSTANT = 1.345  # Standard choice for 95% efficiency at normal
_MIN_PRECIPITATION_VALUE = 0  # Exclude zero precipitation in knot calculations
_EPSILON_WEIGHT = 1e-10  # Small value to prevent division by zero
```

### Logging Pattern

```python
import logging

logger = logging.getLogger(__name__)

# Usage: Informational warnings, not errors
logger.warning(
    "Criterion '%s' bin %s has only %d valid observations (< %d); "
    "consider merging bins or adjusting thresholds.",
    crit_label, bin_key, n_obs, min_bin_size,
)
```

**When to log**:
- Automatic fallbacks (e.g., switching from hierarchical to global percentiles)
- Small bin warnings in splitting
- Performance hints (future)

**When NOT to log**:
- Normal operation
- User errors (use exceptions instead)

### Defensive Null Handling

```python
# Pattern: Suppress warnings, use `where` parameter, provide fallbacks
with np.errstate(invalid="ignore", divide="ignore"):
    result = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > _EPSILON_WEIGHT,
    )

# Fallback for empty bins
fallback_value = 0.5 * (lower_bound + upper_bound)
empty_mask = denominator <= _EPSILON_WEIGHT
result[empty_mask] = fallback_value[empty_mask]
```

### Naming Conventions

- **Public API**: Short, domain-appropriate names (`erra`, not `run_erra_analysis`)
- **Internal functions**: Underscore prefix (`_solve_rrd`, `_broken_stick`)
- **Parameters**: Match R implementation where possible (e.g., `nu`, `m`, `nk`)
- **Variables**: Descriptive, avoid abbreviations except domain terms
  - Good: `design_matrix`, `response_vector`, `lag_knots`
  - Acceptable domain terms: `p` (precipitation), `q` (discharge), `rrd`, `nrf`
  - Bad: `tmp`, `arr`, `x`, `data`

---

## Development Workflows

### Setup

```powershell
# Clone repository
git clone https://github.com/licm13/ERRA.git
cd ERRA

# Install in editable mode with dev dependencies
python -m pip install -U pip
pip install -e .[dev]

# Verify installation
python -c "from erra import erra; print(erra.__module__)"
```

### Running Examples

```powershell
# Set data directory (for MOPEX examples)
$env:ERRA_DATA_DIR = "${PWD}\reference_materials\R_implementation\demonstration-scripts\Source data"

# Run example scripts (uses installed package)
python code/examples/master_demonstration.py
python code/examples/gao2025_dynamic_linkages.py

# Alternative: Set PYTHONPATH without installing
$env:PYTHONPATH = "${PWD}\src"
python code/examples/example.py
```

### Code Quality Tools

```powershell
# Format code with Black
black src/erra tests

# Lint with Ruff
ruff check src/erra tests

# Type check with mypy
mypy src/erra

# Run all quality checks
black src/erra tests && ruff check src/erra tests && mypy src/erra
```

### Testing

```powershell
# Run all tests
pytest tests/

# Run with verbose output
pytest -v tests/

# Run specific test
pytest tests/test_splitting.py::test_make_split_sets_global_percentiles

# Run with coverage (if coverage installed)
pytest --cov=erra tests/
```

---

## Testing Guidelines

### Current State

- **Coverage**: Minimal (only `tests/test_splitting.py` exists)
- **Framework**: pytest
- **Strategy**: Examples serve as integration tests; unit tests needed

### Test Structure

```python
import numpy as np
from pathlib import Path
import sys

# Add src to path (for test isolation)
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from erra.module import function_to_test


def _helper_function(param1, param2):
    """Fixture-like helpers use underscore prefix."""
    return {"key": "value"}


def test_function_name_scenario_expected_outcome():
    """Test with descriptive name following pytest convention.

    Pattern: test_<function>_<scenario>_<expected>
    Example: test_make_split_sets_global_percentiles_matches_expected
    """
    # Arrange
    input_data = np.array([1, 2, 3])
    expected = np.array([2, 4, 6])

    # Act
    result = function_to_test(input_data)

    # Assert
    np.testing.assert_allclose(result, expected)
```

### Assertion Patterns

```python
# For numpy arrays (preferred for float comparisons)
np.testing.assert_allclose(actual, expected, rtol=1e-5)
np.testing.assert_array_equal(actual, expected)  # Exact equality

# For exact matches
assert result == expected

# For membership
assert item in collection
assert "substring" in string

# For shapes
assert array.shape == (10, 5)

# For memory sharing (splitting tests)
assert np.shares_memory(q_out, q_in)
```

### Testing with Logging

```python
def test_function_logs_warning(caplog):
    """Verify warning messages using caplog fixture."""
    import logging

    with caplog.at_level(logging.WARNING):
        result = function_that_warns(data)

    # Check warning was logged
    warning_messages = " ".join(rec.message for rec in caplog.records)
    assert "expected warning text" in warning_messages
```

### What Needs Tests (Priority Order)

1. **High Priority**:
   - `utils_core.py` functions (pure functions, easy to test)
   - `nonlin.py` functions (complex logic, needs validation)
   - Edge cases in `erra_core.py` (error handling, validation)

2. **Medium Priority**:
   - `splitting.py` (partially covered, needs more edge cases)
   - Integration tests for `erra()` with various parameter combinations

3. **Low Priority**:
   - Plotting functions (visual output, harder to test)
   - Example scripts (manual verification OK)

---

## Bilingual Documentation

### Overview

All documentation is bilingual (English / Chinese 中文) to serve the international hydrology community. English is primary, Chinese translations follow.

### Module Docstrings

```python
"""Ensemble Rainfall-Runoff Analysis (ERRA) Package.

ERRA 集成降雨-径流分析包

This package provides a comprehensive Python implementation of the Ensemble
Rainfall-Runoff Analysis (ERRA) framework, including advanced features for
nonlinear and non-stationary hydrologic analysis.

本包提供了 ERRA 框架的完整 Python 实现，包括非线性和非平稳水文分析的高级功能。

Main Functions / 主要函数:
--------------------------
- erra(): Core ERRA analysis function / 核心 ERRA 分析函数
- plot_erra_results(): Comprehensive plotting utilities / 综合绘图工具

References / 参考文献:
---------------------
Kirchner, J.W. (2024). Characterizing nonlinear, nonstationary, and
heterogeneous hydrologic behavior using Ensemble Rainfall-Runoff Analysis
(ERRA): proof of concept. Hydrology and Earth System Sciences, 28, 4427-4454.
https://doi.org/10.5194/hess-28-4427-2024
"""
```

### Function Docstrings

Use NumPy docstring format with bilingual sections:

```python
def erra(
    p: ...,
    q: ...,
    m: int = 60,
    nu: float = 0.0,
) -> ERRAResult:
    """Estimate runoff response distributions using ERRA methodology.

    利用 ERRA 方法估算径流响应分布。

    This function implements the complete ERRA framework including linear
    RRD estimation via ridge regression, nonlinear analysis, non-stationary
    splitting, robust estimation (IRLS), and broken-stick compression.

    此函数实现完整的 ERRA 框架，包括通过岭回归进行线性 RRD 估计、
    非线性分析、非平稳分割、鲁棒估计（IRLS）和断棍压缩。

    Parameters / 参数
    ----------
    p : array-like
        Precipitation series (vector or matrix). Missing values (NaN) are
        allowed and will be removed together with the affected regression rows.

        降水时间序列，可以是向量或矩阵。允许缺失值（NaN），
        缺失值会导致相关的回归行被删除。

    m : int, default=60
        Maximum lag (inclusive) for the RRD. Typical values: 30-120 depending
        on time resolution and catchment response time.

        RRD 的最大时滞（包含）。典型值：30-120，
        取决于时间分辨率和流域响应时间。

        Recommended values / 推荐值:
        - Hourly data / 每小时数据: m=60-120 (captures 2.5-5 day response)
        - Daily data / 每日数据: m=30-60 (captures 1-2 month response)

    nu : float, default=0.0
        Tikhonov regularization strength (0-1). Higher values produce smoother
        RRDs but may underfit the data.

        Tikhonov 正则化强度（0-1）。较高的值产生更平滑的 RRD，
        但可能欠拟合数据。

        Recommended values / 推荐值:
        - Clean data / 清洁数据: nu=0 (no regularization)
        - Noisy data / 噪声数据: nu=0.01-0.1 (light smoothing)
        - Very noisy / 非常噪声: nu=0.1-0.5 (moderate smoothing)

    Returns / 返回
    -------
    ERRAResult
        Dataclass containing RRD estimates, standard errors, fitted values,
        residuals, and optional advanced outputs (NRF, split results, etc.).

        包含 RRD 估计、标准误差、拟合值、残差和可选高级输出
        （NRF、分割结果等）的数据类。

    Examples / 示例
    --------
    **Basic linear analysis / 基本线性分析:**

    >>> import numpy as np
    >>> from erra import erra
    >>> p = np.random.exponential(5, 1000)
    >>> q = np.random.gamma(2, 3, 1000)
    >>> result = erra(p=p, q=q, m=60, nu=0.1)
    >>> print(result.rrd)

    **Nonlinear analysis / 非线性分析:**

    >>> result = erra(
    ...     p=p, q=q, m=60,
    ...     xknots=[50, 80, 95],
    ...     xknot_type='percentiles'
    ... )
    >>> print(result.nrf)

    Notes / 注释
    -----
    The algorithm uses ridge regression with optional Tikhonov regularization
    to solve the ill-conditioned deconvolution problem. For details, see:

    该算法使用带可选 Tikhonov 正则化的岭回归来解决病态反卷积问题。详情请见：

    - Kirchner (2024): https://doi.org/10.5194/hess-28-4427-2024
    - Kirchner (2022): https://doi.org/10.3390/s22093291

    See Also / 另见
    --------
    plot_erra_results : Visualize ERRA outputs / 可视化 ERRA 输出
    ERRAResult : Result container structure / 结果容器结构
    """
```

### Dataclass Attribute Documentation

```python
@dataclass
class ERRAResult:
    """Container for ERRA outputs / ERRA 结果容器。

    Attributes / 属性
    -----------------
    lags : np.ndarray
        Lag times corresponding to RRD coefficients (time units follow input data)
        RRD 系数对应的时滞（单位与输入时间步一致）

    rrd : pd.DataFrame
        Estimated runoff response distributions for each precipitation driver
        每个降水驱动变量的径流响应分布估计值

    stderr : pd.DataFrame
        Standard errors of the RRD coefficients (same layout as rrd)
        RRD 系数的标准误差，与 rrd 具有相同结构
    """
    lags: np.ndarray
    rrd: pd.DataFrame
    stderr: pd.DataFrame
```

### Inline Comments

Use bilingual comments sparingly, only for key algorithmic steps:

```python
# Apply aggregation if requested
# 如果需要，应用聚合
if agg > 1:
    p_matrix, q_vec, wt = aggregate_time_series(p_matrix, q_vec, wt, agg)

# Build design matrix with lagged precipitation
# 构建带滞后降水的设计矩阵
design, response, weights = build_design_matrix(
    p_matrix, q_vec, wt_vec, m, dt
)
```

**Guideline**: Use inline bilingual comments for:
- Major algorithm steps
- Non-obvious transformations
- Key decision points

**Don't use** for:
- Obvious operations
- Every line
- Error messages (English only for consistency)

### Plot Labels

```python
def _bilingual_text(english: str, chinese: str = "", use_chinese: bool = True) -> str:
    """Create bilingual text for plots.

    创建双语图表文本。
    """
    if use_chinese and chinese:
        return f"{english} / {chinese}"
    return english

# Usage in plotting functions
plt.xlabel(_bilingual_text("Lag (time units)", "时滞 (时间单位)", use_chinese))
plt.ylabel(_bilingual_text("RRD Coefficient", "RRD系数", use_chinese))
plt.title(_bilingual_text(
    "Runoff Response Distribution",
    "径流响应分布",
    use_chinese
))
```

### Translation Guidelines

1. **Technical Terms**: Keep consistent with established hydrology literature
   - Runoff Response Distribution → 径流响应分布
   - Impulse response function → 脉冲响应函数
   - Ridge regression → 岭回归
   - Tikhonov regularization → Tikhonov 正则化

2. **Parameter Names**: English only (for code consistency)
   ```python
   # Good
   def erra(p, q, m=60, nu=0.0):

   # Bad - don't translate parameter names
   def erra(降水, 流量, m=60, nu=0.0):
   ```

3. **Error Messages**: English only
   ```python
   # Good
   raise ValueError("nu must be in [0, 1)")

   # Bad - don't translate error messages
   raise ValueError("nu 必须在 [0, 1) 范围内")
   ```

4. **Variable Names**: English only
   ```python
   # Good
   precipitation = df["p"]
   discharge = df["q"]

   # Bad
   降水 = df["p"]
   流量 = df["q"]
   ```

---

## Common Tasks

### Adding a New Parameter to `erra()`

1. **Add parameter to function signature** (`src/erra/erra_core.py`):
   ```python
   def erra(
       # ... existing params
       new_param: float = 1.0,
   ) -> ERRAResult:
   ```

2. **Document in docstring** (bilingual):
   ```python
   new_param : float, default=1.0
       Description of parameter in English.

       参数的中文描述。

       Recommended values / 推荐值:
       - Case 1: value1
       - Case 2: value2
   ```

3. **Validate input** (early in function):
   ```python
   if new_param < 0:
       raise ValueError(
           f"new_param must be non-negative, got {new_param}"
       )
   ```

4. **Use parameter** in algorithm

5. **Add to tests** (`tests/test_erra_core.py` - create if needed):
   ```python
   def test_erra_new_param_validation():
       with pytest.raises(ValueError):
           erra(p=dummy_p, q=dummy_q, new_param=-1)
   ```

6. **Update examples** if appropriate

### Adding a New Utility Function

1. **Choose correct module**:
   - Core utilities → `src/erra/utils_core.py`
   - Plotting → `src/erra/utils.py`
   - Nonlinear-specific → `src/erra/nonlin.py`
   - Splitting-specific → `src/erra/splitting.py`

2. **Write function** with full type hints:
   ```python
   def my_utility(
       data: np.ndarray,
       threshold: float = 0.5,
   ) -> tuple[np.ndarray, np.ndarray]:
       """Brief description in one line.

       一行简要描述。

       Longer description if needed.

       Parameters / 参数
       ----------
       data : np.ndarray
           Description
           描述

       Returns / 返回
       -------
       tuple[np.ndarray, np.ndarray]
           Description of return values
           返回值描述
       """
       # Implementation
       pass
   ```

3. **Add tests** (`tests/test_utils_core.py`):
   ```python
   def test_my_utility_basic_case():
       data = np.array([1, 2, 3])
       result1, result2 = my_utility(data)
       assert result1.shape == (3,)
   ```

4. **Export if public** (add to `__init__.py` if needed for public API)

### Adding a New Example

1. **Create example file** (`code/examples/my_example.py`):
   ```python
   """Brief description of what this example demonstrates.

   简要描述此示例演示的内容。
   """

   from pathlib import Path
   import numpy as np
   import pandas as pd

   from erra import erra, plot_erra_results


   def main():
       # Generate or load data
       # 生成或加载数据
       p = np.random.rand(1000)
       q = np.random.rand(1000)

       # Run ERRA analysis
       # 运行 ERRA 分析
       result = erra(p=p, q=q, m=60, nu=0.1)

       # Plot results
       # 绘制结果
       script_name = Path(__file__).stem
       figures_dir = Path(__file__).parent / "figures"

       plot_erra_results(
           result=result,
           observed_q=q,
           output_dir=figures_dir,
           filename_prefix=script_name,
           save_plots=True,
           show_plots=False,
           use_chinese=True,
       )

       print(f"Analysis complete. Figures saved to {figures_dir}/")


   if __name__ == "__main__":
       main()
   ```

2. **Test the example**:
   ```powershell
   python code/examples/my_example.py
   ```

3. **Document in README** if it's a significant example

### Fixing a Bug

1. **Reproduce the bug** with minimal example

2. **Write failing test** first (TDD approach):
   ```python
   def test_bug_description():
       """Test that reproduces bug #123."""
       # Minimal reproducing case
       result = function_with_bug(problematic_input)
       # What it should return
       assert result == expected_value
   ```

3. **Fix the bug** in source code

4. **Verify test passes**:
   ```powershell
   pytest tests/test_module.py::test_bug_description -v
   ```

5. **Add regression test** for edge case

6. **Update docstring** if behavior changed

### Refactoring Code

1. **Ensure tests exist** for code being refactored
   - If not, write tests first!

2. **Run tests before refactoring**:
   ```powershell
   pytest tests/ -v
   ```

3. **Make incremental changes**:
   - Extract function
   - Rename variable
   - Simplify logic

4. **Run tests after each change**

5. **Run type checker**:
   ```powershell
   mypy src/erra
   ```

6. **Run formatter and linter**:
   ```powershell
   black src/erra tests
   ruff check src/erra tests
   ```

7. **Verify examples still work**:
   ```powershell
   python code/examples/master_demonstration.py
   ```

---

## File Reference

### Configuration Files

#### `pyproject.toml`
- **Package metadata**: name, version, dependencies
- **Build system**: setuptools configuration
- **Tool config**: black, ruff, mypy, pytest settings
- **Edit when**: Adding dependencies, changing package metadata

Key sections:
```toml
[project]
name = "erra-hydrology"
version = "1.1.0"
requires-python = ">=3.9"
dependencies = [...]

[tool.black]
line-length = 88

[tool.mypy]
disallow_untyped_defs = true
```

#### `.gitignore`
- **Excludes**: `__pycache__/`, `*.pyc`, `build/`, `dist/`, `.vscode/`, virtual environments
- **Edit when**: Adding new build artifacts or IDE files to ignore

### Source Files (Priority Order for Contributions)

1. **`src/erra/erra_core.py`** (843 lines)
   - Most important file
   - Contains main `erra()` function and `ERRAResult`
   - Edit when: Adding features, fixing core bugs

2. **`src/erra/utils_core.py`** (246 lines)
   - Shared utilities
   - Pure functions, easy to test
   - Edit when: Adding utility functions, optimizing algorithms

3. **`src/erra/nonlin.py`** (345 lines)
   - Nonlinear analysis logic
   - Complex mathematics
   - Edit when: Fixing nonlinear bugs, adding NRF features

4. **`src/erra/splitting.py`** (437 lines)
   - Non-stationary splitting
   - Well-tested module
   - Edit when: Adding splitting features, improving bin logic

5. **`src/erra/utils.py`** (500 lines)
   - Plotting utilities
   - Isolated from core
   - Edit when: Adding plots, improving visualizations

6. **`src/erra/__init__.py`** (50 lines)
   - Public API definition
   - Edit when: Adding public exports

### Test Files

- **`tests/test_splitting.py`** (100 lines)
  - Only existing test file
  - Good patterns to follow

- **Tests needed** (create these!):
  - `tests/test_erra_core.py`
  - `tests/test_utils_core.py`
  - `tests/test_nonlin.py`
  - `tests/test_utils.py`

### Example Files

- **`code/examples/master_demonstration.py`** - Comprehensive demo of all features
- **`code/examples/gao2025_dynamic_linkages.py`** - Multiple drivers example
- **`code/examples/sharif_ameli2025_functional_simplicity.py`** - Functional simplicity concept
- **`code/examples/tu2025_permafrost_transition.py`** - Non-stationary example
- **`code/examples/complex_sensitivity_study.py`** - Stress test with multiple options

---

## Git Workflow

### Branch Naming

Pattern: `<ai-assistant>/<short-description>-<session-id>`

Examples:
- `claude/add-claude-documentation-01W1vtiB6Jq9wTCKgjJG52Mq`
- `copilot/improve-code-quality`
- `codex/optimize-build-design-matrix-performance`

### Commit Messages

**Format**: Imperative mood, descriptive, concise

```
# Good
Add CLAUDE.md documentation for AI assistants
Refactor ERRA core utilities and align naming
Optimize matrix builders and add performance benchmarks

# Bad
Added documentation
Updated files
Fixed stuff
```

### Typical Workflow

1. **Create feature branch**:
   ```bash
   git checkout -b claude/my-feature-description-session-id
   ```

2. **Make changes**, commit frequently:
   ```bash
   git add src/erra/module.py tests/test_module.py
   git commit -m "Add new feature with tests"
   ```

3. **Run quality checks** before pushing:
   ```powershell
   black src/erra tests
   ruff check src/erra tests
   mypy src/erra
   pytest tests/
   ```

4. **Push to remote**:
   ```bash
   git push -u origin claude/my-feature-description-session-id
   ```

5. **Create Pull Request** (via GitHub UI)

6. **After merge**, delete branch:
   ```bash
   git checkout main
   git pull
   git branch -d claude/my-feature-description-session-id
   ```

### Recent Development Patterns

Based on recent commits:
- **Refactoring**: Code quality improvements (naming, structure)
- **Performance**: Optimization of matrix operations
- **Examples**: Adding scientific case studies
- **Documentation**: Bilingual docs, improving clarity

---

## Quick Reference Cheat Sheet

### Most Common Commands

```powershell
# Setup
pip install -e .[dev]

# Run example
python code/examples/master_demonstration.py

# Quality checks
black src/erra tests && ruff check src/erra tests && mypy src/erra

# Test
pytest tests/ -v

# Type check single file
mypy src/erra/erra_core.py
```

### Import Pattern

```python
from __future__ import annotations
from typing import Optional
import numpy as np
from .utils_core import function_name
```

### Docstring Template

```python
def function_name(param: type) -> return_type:
    """One-line summary in English.

    一行中文摘要。

    Parameters / 参数
    ----------
    param : type
        Description
        描述

    Returns / 返回
    -------
    return_type
        Description
        描述
    """
```

### Test Pattern

```python
def test_function_scenario_expected():
    # Arrange
    input_data = ...

    # Act
    result = function(input_data)

    # Assert
    np.testing.assert_allclose(result, expected)
```

---

## Additional Resources

- **README.md**: User-facing documentation (bilingual)
- **.github/copilot-instructions.md**: Quick reference for common tasks
- **reference_materials/theory_pdfs/**: Theory documentation
- **reference_materials/papers/**: Application papers
- **reference_materials/R_implementation/**: Original R code

---

## Notes for AI Assistants

### When Starting a Task

1. **Read relevant source files** first
2. **Check existing tests** for patterns
3. **Look at examples** for usage patterns
4. **Verify assumptions** with user if unclear

### Before Committing Changes

1. **Type hints**: All functions must have complete type hints
2. **Docstrings**: Bilingual with recommended values
3. **Tests**: Add or update tests for changes
4. **Format**: Run `black src/erra tests`
5. **Lint**: Run `ruff check src/erra tests`
6. **Type check**: Run `mypy src/erra`
7. **Examples**: Verify existing examples still work

### Communication Style

- **Be precise**: Reference specific files and line numbers
- **Be helpful**: Suggest alternatives when saying "no"
- **Be bilingual**: Provide Chinese translations for user-facing text
- **Be explicit**: Don't assume knowledge of hydrology domain

### Domain Knowledge

- **Hydrology terms**: RRD, impulse response, convolution, deconvolution
- **Statistical terms**: Ridge regression, Tikhonov regularization, IRLS
- **Package conventions**: Match R implementation (nu, m, nk, fq)
- **Time series**: Understand lag, aggregation, filtering concepts

---

**Questions?** Consult `.github/copilot-instructions.md` for quick answers or README.md for user documentation.

**End of CLAUDE.md**

# 🤖 ERRA 开发专用 AI Prompt / AI Prompt for ERRA Development

本文档提供了针对 Codex/Claude/Cursor 等 AI 编码助手的优化 Prompt 模板。
This document provides optimized prompt templates for AI coding assistants like Codex/Claude/Cursor.

---

## 📋 通用开发 Prompt / General Development Prompt

```markdown
# ERRA Package Development Task

## 项目背景 / Project Background
ERRA (Ensemble Rainfall-Runoff Analysis) 是用于水文降雨-径流分析的 Python 包。
核心方法：集成反卷积、岭回归、Tikhonov 正则化。

ERRA is a Python package for hydrological rainfall-runoff analysis using ensemble deconvolution methods.
Core methods: Ridge regression with Tikhonov regularization.

## 代码库结构 / Repository Structure

### 核心模块 / Core Modules
- `src/erra/erra_core.py` - 主算法实现 (linear, nonlinear, robust estimation)
- `src/erra/nonlin.py` - 非线性响应函数 (Nonlinear Response Functions)
- `src/erra/splitting.py` - 数据分割工具 (Data splitting for non-stationarity)
- `src/erra/utils.py` - 绘图和工具函数 (Plotting utilities)

### 示例和测试 / Examples & Tests
- `examples/` - 应用示例脚本
- `tests/` - 单元测试
- `tutorials/` - Jupyter Notebook 教程

## 当前任务 / Current Task

### 任务描述 / Task Description
[在此描述具体任务，例如：实现并行计算模块、创建教学 Notebook、优化性能等]

### 功能要求 / Functional Requirements
1. [具体功能点 1]
2. [具体功能点 2]
3. [具体功能点 3]

### 性能要求 / Performance Requirements
- 执行速度 / Speed: [例如：处理 10000 个数据点 < 1 秒]
- 内存使用 / Memory: [例如：峰值内存 < 500 MB]
- 可扩展性 / Scalability: [例如：支持数据规模 up to 100K points]

### 接口规范 / Interface Specification

**输入格式 / Input Format:**
```python
# 示例输入
input_data = {
    'precipitation': np.ndarray,  # shape: (n_samples,) or (n_samples, n_drivers)
    'discharge': np.ndarray,      # shape: (n_samples,)
    'parameters': {
        'm': 60,      # Maximum lag
        'nu': 0.1,    # Regularization strength
        'dt': 1.0,    # Time step
    }
}
```

**输出格式 / Output Format:**
```python
# 期望输出
output = {
    'rrd': pd.DataFrame,       # Runoff Response Distribution
    'stderr': pd.DataFrame,    # Standard errors
    'fitted': np.ndarray,      # Fitted discharge
    'residuals': np.ndarray,   # Residuals
    'metrics': {
        'r_squared': float,
        'rmse': float,
    }
}
```

### 约束条件 / Constraints
- ✅ Python 版本: ≥3.9
- ✅ 依赖库: numpy, pandas, scipy, matplotlib
- ✅ 代码风格: 符合 PEP 8
- ✅ 类型提示: 所有函数必须有完整的类型注解
- ✅ 文档: NumPy 风格 docstring，中英文双语
- ✅ 测试: 单元测试覆盖率 ≥85%

### 代码风格示例 / Code Style Example

```python
from typing import Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd

def erra_analysis(
    precipitation: np.ndarray,
    discharge: np.ndarray,
    max_lag: int = 60,
    regularization: float = 0.1,
    time_step: float = 1.0,
    *,
    robust: bool = False,
    verbose: bool = True,
) -> Dict[str, Union[pd.DataFrame, np.ndarray, Dict]]:
    """Perform ERRA rainfall-runoff analysis.
    
    执行 ERRA 降雨-径流分析。
    
    This function estimates the Runoff Response Distribution (RRD) using
    ridge regression with Tikhonov regularization.
    
    该函数使用带 Tikhonov 正则化的岭回归估计径流响应分布 (RRD)。
    
    Parameters
    ----------
    precipitation : np.ndarray
        Precipitation time series, shape (n_samples,) or (n_samples, n_drivers)
        降水时间序列
    discharge : np.ndarray
        Discharge time series, shape (n_samples,)
        径流时间序列
    max_lag : int, optional
        Maximum lag in time steps (default: 60)
        最大滞后步数
        Recommended values / 推荐值:
        - Hourly data / 小时数据: 60-120
        - Daily data / 日数据: 30-60
    regularization : float, optional
        Tikhonov regularization strength (default: 0.1)
        正则化强度
        Range / 范围: [0, 1]
        - 0: No regularization / 无正则化
        - 0.01-0.1: Light smoothing / 轻度平滑
        - 0.1-0.5: Moderate smoothing / 中度平滑
    time_step : float, optional
        Time step in hours or days (default: 1.0)
        时间步长（小时或天）
    robust : bool, optional
        Use robust estimation (IRLS) for outlier handling (default: False)
        使用鲁棒估计处理异常值
    verbose : bool, optional
        Print progress information (default: True)
        打印进度信息
        
    Returns
    -------
    dict
        Dictionary containing:
        包含以下键的字典:
        - 'rrd': pd.DataFrame - Runoff Response Distribution / 径流响应分布
        - 'stderr': pd.DataFrame - Standard errors / 标准误差
        - 'fitted': np.ndarray - Fitted discharge / 拟合径流
        - 'residuals': np.ndarray - Residuals / 残差
        - 'metrics': dict - Performance metrics / 性能指标
        
    Raises
    ------
    ValueError
        If input arrays have incompatible shapes or invalid parameter values
        如果输入数组形状不兼容或参数值无效
        
    Examples
    --------
    Basic linear analysis / 基础线性分析:
    
    >>> import numpy as np
    >>> precip = np.random.exponential(5, 1000)
    >>> discharge = np.random.gamma(2, 3, 1000)
    >>> result = erra_analysis(precip, discharge, max_lag=60, regularization=0.1)
    >>> print(result['rrd'].head())
    
    With robust estimation / 使用鲁棒估计:
    
    >>> result_robust = erra_analysis(
    ...     precip, discharge, 
    ...     max_lag=60, 
    ...     regularization=0.1,
    ...     robust=True
    ... )
    
    Notes
    -----
    - Ensure data length is sufficient: n_samples > 4 * max_lag
      确保数据长度充足: n_samples > 4 * max_lag
    - Missing values should be handled before calling this function
      调用前应处理缺失值
    - For noisy data, increase regularization strength
      对于噪声数据，增加正则化强度
      
    References
    ----------
    .. [1] Kirchner, J.W. (2024). Characterizing nonlinear, nonstationary, 
           and heterogeneous hydrologic behavior using Ensemble Rainfall-Runoff 
           Analysis (ERRA): proof of concept. HESS, 28, 4427-4454.
    """
    # Validate inputs / 验证输入
    if precipitation.ndim not in [1, 2]:
        raise ValueError("Precipitation must be 1D or 2D array")
    
    if len(discharge) != len(precipitation):
        raise ValueError(
            f"Length mismatch: precipitation ({len(precipitation)}) "
            f"vs discharge ({len(discharge)})"
        )
    
    if len(discharge) < 4 * max_lag:
        raise ValueError(
            f"Insufficient data: need at least {4*max_lag} samples, "
            f"got {len(discharge)}"
        )
    
    # Implementation / 实现
    # [核心算法代码]
    
    # Return results / 返回结果
    return {
        'rrd': rrd_dataframe,
        'stderr': stderr_dataframe,
        'fitted': fitted_values,
        'residuals': residuals,
        'metrics': {
            'r_squared': calculate_r_squared(discharge, fitted_values),
            'rmse': calculate_rmse(discharge, fitted_values),
        }
    }


def calculate_r_squared(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate R-squared coefficient of determination.
    
    计算决定系数 R²。
    
    Parameters
    ----------
    observed : np.ndarray
        Observed values / 观测值
    predicted : np.ndarray
        Predicted values / 预测值
        
    Returns
    -------
    float
        R-squared value, range [0, 1] / R² 值，范围 [0, 1]
        
    Examples
    --------
    >>> obs = np.array([1, 2, 3, 4, 5])
    >>> pred = np.array([1.1, 2.0, 2.9, 4.2, 4.8])
    >>> r2 = calculate_r_squared(obs, pred)
    >>> print(f"R² = {r2:.3f}")
    """
    ss_res = np.sum((observed - predicted) ** 2)
    ss_tot = np.sum((observed - np.mean(observed)) ** 2)
    return 1 - (ss_res / ss_tot)
```

### 测试要求 / Testing Requirements

**单元测试示例 / Unit Test Example:**

```python
import pytest
import numpy as np
from erra import erra_analysis

class TestERRAAnalysis:
    """Test suite for ERRA analysis function."""
    
    def test_basic_functionality(self):
        """Test basic ERRA analysis with synthetic data."""
        # Setup / 设置
        np.random.seed(42)
        precip = np.random.exponential(5, 500)
        discharge = np.random.gamma(2, 3, 500)
        
        # Execute / 执行
        result = erra_analysis(precip, discharge, max_lag=30, regularization=0.1)
        
        # Assert / 断言
        assert 'rrd' in result
        assert 'stderr' in result
        assert 'fitted' in result
        assert len(result['fitted']) == len(discharge)
        assert result['metrics']['r_squared'] >= 0
        assert result['metrics']['r_squared'] <= 1
    
    def test_input_validation(self):
        """Test input validation and error handling."""
        precip = np.random.rand(100)
        discharge = np.random.rand(50)  # Mismatched length
        
        with pytest.raises(ValueError, match="Length mismatch:"): 
            erra_analysis(precip, discharge)
    
    def test_insufficient_data(self):
        """Test error handling for insufficient data."""
        precip = np.random.rand(100)
        discharge = np.random.rand(100)
        
        with pytest.raises(ValueError, match="Insufficient data:"):
            erra_analysis(precip, discharge, max_lag=50)
    
    def test_robust_estimation(self):
        """Test robust estimation with outliers."""
        np.random.seed(42)
        precip = np.random.exponential(5, 500)
        discharge = np.random.gamma(2, 3, 500)
        # Add outliers / 添加异常值
        discharge[100] = discharge[100] * 10
        discharge[200] = discharge[200] * 10
        
        result_normal = erra_analysis(precip, discharge, robust=False)
        result_robust = erra_analysis(precip, discharge, robust=True)
        
        # Robust should perform better / 鲁棒估计应该表现更好
        assert result_robust['metrics']['rmse'] <= result_normal['metrics']['rmse']
    
    @pytest.mark.parametrize("max_lag,regularization", [
        (30, 0.0),
        (60, 0.1),
        (90, 0.5),
    ])
    def test_parameter_combinations(self, max_lag, regularization):
        """Test various parameter combinations."""
        precip = np.random.exponential(5, 1000)
        discharge = np.random.gamma(2, 3, 1000)
        
        result = erra_analysis(
            precip, discharge,
            max_lag=max_lag,
            regularization=regularization
        )
        
        assert result is not None
        assert len(result['rrd']) == max_lag + 1
```

### 交付清单 / Deliverables Checklist

- [ ] ✅ 完整的函数/类实现 (Complete function/class implementation)
- [ ] ✅ 完整的类型注解 (Full type hints)
- [ ] ✅ NumPy 风格 docstring (NumPy-style docstrings)
- [ ] ✅ 中英文双语注释 (Bilingual comments)
- [ ] ✅ 单元测试文件 `tests/test_xxx.py` (Unit tests)
- [ ] ✅ 使用示例 (Usage examples in docstring)
- [ ] ✅ 性能基准测试（如适用）(Performance benchmarks if applicable)
- [ ] ✅ 错误处理和验证 (Error handling and validation)

### 额外说明 / Additional Notes

- 所有代码必须遵循 GPL-3.0 许可证
- 优先使用 NumPy 向量化操作，避免显式循环
- 大型数组操作考虑内存效率
- 关键算法添加性能计时（使用 `time.perf_counter()`）
- 复杂逻辑添加内联注释解释
```
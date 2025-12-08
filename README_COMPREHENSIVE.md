# 🌊 ERRA - Ensemble Rainfall-Runoff Analysis

## 📖 README for Developers / 开发者阅读指南

> **A comprehensive guide for new team members to quickly understand and contribute to the ERRA project**
> 
> **帮助新团队成员快速理解并参与 ERRA 项目的综合指南**

---

## 🎯 Project Overview / 项目全貌

### One-Sentence Summary / 一句话概括

**English:** ERRA is a Python package for advanced hydrological analysis that estimates rainfall-runoff response distributions using ensemble deconvolution with ridge regression, supporting nonlinear and non-stationary system behavior.

**中文：** ERRA 是一个高级水文分析 Python 包，使用集成反卷积和岭回归估计降雨-径流响应分布，支持非线性和非平稳系统行为分析。

### Core Technology Stack / 核心技术栈

- **Language / 编程语言:** Python 3.9+
- **Scientific Computing / 科学计算:** NumPy, SciPy, Pandas
- **Visualization / 可视化:** Matplotlib
- **Methodology / 方法论:** 
  - Ridge Regression (岭回归)
  - Tikhonov Regularization (Tikhonov 正则化)
  - Ensemble Deconvolution (集成反卷积)
  - Iteratively Reweighted Least Squares - IRLS (迭代加权最小二乘)

### What Problem Does ERRA Solve? / ERRA 解决什么问题？

**English:**
In hydrology, understanding how precipitation transforms into streamflow is critical for flood forecasting, water resource management, and climate change impact assessment. ERRA provides a data-driven approach to estimate this transformation as an **impulse response function (Runoff Response Distribution - RRD)**, capturing:

- **Nonlinear effects:** How response changes with precipitation intensity
- **Non-stationary behavior:** How response varies with antecedent conditions (wetness, temperature)
- **Multiple drivers:** Separating contributions from rain, snow, convective vs. stratiform precipitation

**中文：**
在水文学中，理解降水如何转化为河川流量对于洪水预报、水资源管理和气候变化影响评估至关重要。ERRA 提供了一种数据驱动的方法来估计这种转化，将其表示为**脉冲响应函数（径流响应分布 - RRD）**，捕捉：

- **非线性效应：** 响应如何随降水强度变化
- **非平稳行为：** 响应如何随前期条件（湿度、温度）变化
- **多个驱动因素：** 分离降雨、降雪、对流与层状降水的贡献


---

## 📂 File Structure / 目录结构详解

### Root Directory Overview / 根目录概览

```
ERRA/
├── 📦 src/erra/                    # Core Python package / 核心 Python 包
├── 📝 code/examples/               # Example scripts & demonstrations / 示例脚本与演示
├── 📚 reference_materials/         # Original R code & papers / 原始 R 代码与论文
├── 🧪 tests/                       # Unit tests / 单元测试
├── 📄 pyproject.toml               # Package configuration / 包配置
├── 📖 README.md                    # User documentation / 用户文档
└── 📖 README_COMPREHENSIVE.md      # This file / 本文件
```

---

### 🔍 Detailed Structure Analysis / 详细结构分析

#### 1️⃣ Core Package: `src/erra/` ⭐⭐⭐

**Purpose / 作用:** This is the heart of the project - the production-ready Python package that users install and import.

**这是项目的核心 - 用户安装和导入的生产就绪 Python 包。**

```
src/erra/
├── __init__.py          # Package entry point / 包入口
├── erra_core.py         # 🔴 Main algorithm implementation / 主算法实现
├── nonlin.py            # 🟠 Nonlinear response functions / 非线性响应函数
├── splitting.py         # 🟡 Non-stationary data splitting / 非平稳数据分割
├── utils.py             # 🟢 Plotting & utilities / 绘图与工具
└── utils_core.py        # 🔵 Low-level matrix operations / 底层矩阵运算
```

**Module Responsibilities / 模块职责:**

| File | Responsibility (EN) | 职责 (CN) | Dependency Level |
|------|---------------------|-----------|-----------------|
| `__init__.py` | Package exports, public API | 包导出，公共 API | Top Level |
| `erra_core.py` | Main `erra()` function, result container | 主 `erra()` 函数，结果容器 | High Level |
| `nonlin.py` | Intensity-based splitting (xknots) | 基于强度的分割 (xknots) | Mid Level |
| `splitting.py` | Covariate-based splitting (split_params) | 基于协变量的分割 (split_params) | Mid Level |
| `utils.py` | Visualization, plotting | 可视化，绘图 | High Level |
| `utils_core.py` | Design matrix construction, solver | 设计矩阵构建，求解器 | Low Level |

**Dependency Relationships / 依赖关系:**
```
        ┌──────────────┐
        │ __init__.py  │  ← User imports from here
        └──────┬───────┘
               │
        ┌──────▼───────┐
        │ erra_core.py │  ← Orchestrates everything
        └──────┬───────┘
         ┌─────┼─────┐
    ┌────▼───┐ │ ┌───▼──────┐
    │nonlin.py│ │ │splitting│  ← Feature modules
    └────┬───┘ │ └───┬──────┘
         └─────┼─────┘
         ┌─────▼─────┐
         │utils_core │  ← Low-level operations
         └───────────┘
```

---

#### 2️⃣ Examples & Demonstrations: `code/examples/` ⭐⭐

**Purpose / 作用:** Complete, runnable scripts demonstrating different ERRA use cases. Perfect for learning and testing.

**完整的可运行脚本，演示不同的 ERRA 用例。非常适合学习和测试。**

```
code/examples/
├── master_demonstration.py                    # 🌟 ALL features showcase / 全功能展示
├── gao2025_dynamic_linkages.py               # Multi-driver analysis / 多驱动分析
├── sharif_ameli2025_functional_simplicity.py # Spatial heterogeneity / 空间异质性
├── tu2025_permafrost_transition.py           # Time-varying response / 时变响应
├── complex_sensitivity_study.py              # Parameter sensitivity / 参数敏感性
├── example.py                                 # Basic MOPEX demo (bilingual) / 基础演示
├── example_en.py                              # Basic MOPEX demo (English only) / 纯英文演示
├── data_prep/                                 # Data fetching scripts / 数据获取脚本
│   ├── gao2025_fetch_data.py
│   ├── sharif_ameli2025_fetch_data.py
│   └── tu2025_fetch_data.py
└── figures/                                   # Generated output plots / 生成的输出图
```

**Example Script Purposes / 示例脚本用途:**

| Script | Features Demonstrated | Real-world Scenario | 演示功能 | 现实场景 |
|--------|----------------------|---------------------|----------|---------|
| `master_demonstration.py` | All: multi-driver, xknots, split_params, nk, robust | Comprehensive stress test | 全部：多驱动、非线性、分割、断棍、鲁棒 | 综合压力测试 |
| `gao2025_dynamic_linkages.py` | Multiple drivers, weights | Convective vs. stratiform rain | 多驱动、权重 | 对流雨 vs. 层状雨 |
| `sharif_ameli2025_functional_simplicity.py` | xknots | Intensity-dependent response | 非线性节点 | 强度依赖响应 |
| `tu2025_permafrost_transition.py` | Time-varying kernels | Climate change impact | 时变核 | 气候变化影响 |
| `complex_sensitivity_study.py` | nu, fq comparison | Regularization tuning | 正则化对比 | 正则化调优 |

---

#### 3️⃣ Reference Materials: `reference_materials/` ⭐

**Purpose / 作用:** Scientific foundation - original R implementation and academic papers.

**科学基础 - 原始 R 实现和学术论文。**

```
reference_materials/
├── R_implementation/
│   ├── erra_scripts_v1.06/              # Original R code by James Kirchner
│   └── demonstration-scripts/
│       ├── Source data/                  # MOPEX datasets (hourly streamflow)
│       └── Outputs from demo scripts/    # Reference R outputs for validation
├── papers/
│   └── application/                      # Real-world case studies
└── theory_pdfs/                          # Theory documentation PDFs
```

**Why This Matters / 为什么重要:**
- **Validation:** Compare Python outputs with R outputs to ensure correctness / **验证：** 对比 Python 与 R 输出确保正确性
- **Learning:** Understand the scientific basis / **学习：** 理解科学基础
- **Extension:** When adding new features, check R implementation first / **扩展：** 添加新功能时，先检查 R 实现

---

#### 4️⃣ Tests: `tests/` ⭐

**Purpose / 作用:** Unit tests to ensure code quality and catch regressions.

**单元测试以确保代码质量并捕获回归。**

```
tests/
└── test_splitting.py    # Tests for splitting.py module
```

**Current Coverage / 当前覆盖率:** Minimal - needs expansion (贡献机会！)


---

## 🔑 Core Code Navigation / 核心代码导航

### Entry Point / 入口点

**Start here when using ERRA / 使用 ERRA 时从这里开始:**

```python
from erra import erra, ERRAResult, plot_erra_results
```

**Defined in / 定义于:** `src/erra/__init__.py` (lines 38-39)

This imports:
- `erra()` - The main analysis function / 主分析函数
- `ERRAResult` - Container for results / 结果容器
- `plot_erra_results()` - Plotting utilities / 绘图工具

---

### Core Algorithm: `erra()` Function ⭐⭐⭐

**Location / 位置:** `src/erra/erra_core.py` (starts at ~line 105)

**Signature / 函数签名:**
```python
def erra(
    p: Union[np.ndarray, pd.DataFrame],  # Precipitation / 降水
    q: Union[np.ndarray, pd.Series],      # Discharge / 流量
    wt: Optional[np.ndarray] = None,      # Weights / 权重
    m: int = 60,                           # Maximum lag / 最大时滞
    nk: int = 0,                           # Broken-stick knots / 断棍节点数
    nu: float = 0.0,                       # Regularization / 正则化强度
    fq: float = 0.0,                       # Quantile filter / 分位数滤波
    dt: float = 1.0,                       # Time step / 时间步长
    agg: int = 1,                          # Aggregation / 聚合因子
    labels: Optional[List[str]] = None,    # Driver labels / 驱动标签
    xknots: Optional[List[float]] = None,  # Nonlinear knots / 非线性节点
    xknot_type: Literal['percentiles', 'values'] = 'percentiles',
    show_top_xknot: bool = False,
    split_params: Optional[Dict] = None,   # Non-stationary params / 非平稳参数
    robust: bool = False,                  # Use IRLS / 使用鲁棒估计
    robust_maxiter: int = 10,
    robust_tolerance: float = 1e-4,
) -> ERRAResult:
```

**What it does / 功能:**
1. **Preprocesses data:** Handle missing values, apply filters / **预处理数据：** 处理缺失值，应用滤波器
2. **Builds design matrix:** Creates lagged precipitation matrix / **构建设计矩阵：** 创建滞后降水矩阵
3. **Applies feature transformations:** Nonlinear (xknots) or splitting (split_params) / **应用特征转换：** 非线性或分割
4. **Solves regression:** Ridge regression with optional robustness / **求解回归：** 带可选鲁棒性的岭回归
5. **Returns structured results:** ERRAResult dataclass / **返回结构化结果：** ERRAResult 数据类

---

### Key Supporting Functions / 关键支持函数

#### Design Matrix Construction / 设计矩阵构建

**Function / 函数:** `build_design_matrix()` in `utils_core.py`

**Purpose / 目的:** Creates the lagged precipitation matrix X where each column is precipitation at lag τ.

**目的：** 创建滞后降水矩阵 X，其中每列是滞后 τ 的降水。

**Mathematical representation / 数学表示:**
```
X[t, τ] = P[t - τ]    for τ = 0, 1, ..., m
```

**Business meaning / 业务含义:** "How does today's precipitation, yesterday's precipitation, ... , m-days-ago precipitation affect today's streamflow?"

**"今天的降水、昨天的降水、...、m 天前的降水如何影响今天的河川流量？"**

---

#### Nonlinear Response Functions / 非线性响应函数

**Module / 模块:** `src/erra/nonlin.py`

**Key Function / 关键函数:** `create_xprime_matrix()`

**Purpose / 目的:** Transforms precipitation matrix into intensity-weighted segments.

**目的：** 将降水矩阵转换为强度加权的片段。

**Business scenario / 业务场景:** 
- **Question:** "Does a light rain (2mm) have the same runoff efficiency as a heavy rain (20mm)?"
- **Answer:** Usually no! Heavy rain generates more runoff per mm due to saturation effects.
- **ERRA solution:** Use xknots to split at [50th, 80th, 95th percentiles] and estimate separate responses.

**业务场景：**
- **问题：**"小雨（2mm）与大雨（20mm）的径流效率相同吗？"
- **答案：** 通常不是！由于饱和效应，大雨每毫米产生更多径流。
- **ERRA 解决方案：** 使用 xknots 在 [50th, 80th, 95th 百分位数] 处分割，估计单独的响应。

---

#### Non-stationary Splitting / 非平稳分割

**Module / 模块:** `src/erra/splitting.py`

**Key Function / 关键函数:** `make_split_sets()`

**Purpose / 目的:** Splits data based on external covariates (e.g., antecedent wetness, temperature).

**目的：** 根据外部协变量（如前期湿度、温度）分割数据。

**Business scenario / 业务场景:**
- **Question:** "Does the same rainfall produce more runoff when soil is already wet?"
- **Answer:** Yes! This is called non-stationarity in catchment response.
- **ERRA solution:** Use split_params with antecedent discharge as proxy for wetness, split at [50th, 90th percentiles], and estimate RRDs for dry/moderate/wet conditions separately.

**业务场景：**
- **问题：**"当土壤已经湿润时，相同的降雨是否会产生更多径流？"
- **答案：** 是的！这称为流域响应的非平稳性。
- **ERRA 解决方案：** 使用 split_params，以前期流量作为湿度代理，在 [50th, 90th 百分位数] 处分割，分别估计干燥/中等/湿润条件下的 RRD。

---

#### Visualization / 可视化

**Module / 模块:** `src/erra/utils.py`

**Key Function / 关键函数:** `plot_erra_results()`

**Generates / 生成:**
1. **RRD plot with error bars** / **带误差棒的 RRD 图**
   - Shows impulse response over time / 显示随时间的脉冲响应
   - Confidence intervals / 置信区间
   
2. **Fitted vs. Observed** / **拟合 vs. 观测**
   - Time series comparison / 时间序列对比
   - R² statistic / R² 统计量
   
3. **Residual Diagnostics** / **残差诊断**
   - Time series plot / 时间序列图
   - Histogram / 直方图
   - Q-Q plot / Q-Q 图
   - Residuals vs. fitted / 残差 vs. 拟合值
   
4. **Broken-stick representation** (if nk > 0) / **断棍表示**（如果 nk > 0）


---

## 📚 Source Code Reading Guide / 源码阅读指南

### For First-Time Contributors / 首次贡献者指南

**Recommended reading path / 推荐阅读路径:**

```
Step 1 → Read the API documentation
         阅读 API 文档
         Location: README.md (existing user guide)
         ↓
Step 2 → Run a simple example
         运行简单示例
         Location: code/examples/example_en.py
         ↓
Step 3 → Understand the main flow
         理解主流程
         Location: src/erra/__init__.py (see what's exported)
                   src/erra/erra_core.py (read erra() function docstring)
         ↓
Step 4 → Trace a simple case
         跟踪简单案例
         Debug code/examples/example_en.py with:
         - Breakpoint at: result = erra(...)
         - Step into erra() function
         - Watch variables: p_matrix, design_matrix, beta_rrd
         ↓
Step 5 → Understand low-level operations
         理解底层操作
         Location: src/erra/utils_core.py
         - build_design_matrix(): How X is constructed
         - solve_rrd(): Ridge regression solver
         ↓
Step 6 → Explore advanced features (optional)
         探索高级功能（可选）
         - Nonlinear: src/erra/nonlin.py
         - Non-stationary: src/erra/splitting.py
         - Robust: IRLS section in erra_core.py
```

---

### Data Flow Walkthrough / 数据流程详解

**Scenario / 场景:** User wants to analyze hourly rainfall-runoff relationship for 7 days (168 hours).

**用户想要分析 7 天（168 小时）的小时降雨-径流关系。**

#### Input Data / 输入数据

```python
import numpy as np
from erra import erra

# User provides / 用户提供:
precipitation = np.array([0.0, 2.1, 5.3, ...])  # 168 values / 168 个值
discharge = np.array([1.2, 1.5, 2.8, ...])      # 168 values / 168 个值
max_lag = 48  # hours / 小时
```

#### Step-by-Step Flow / 逐步流程

```
┌─────────────────────────────────────────────────────────────┐
│ 1. INPUT VALIDATION / 输入验证                               │
│    Function: erra() in erra_core.py                         │
│    - Check p and q have same length                         │
│    - Convert to numpy arrays                                │
│    - Handle missing values (NaN)                            │
│    ├─> p_matrix: (168, 1) [one precipitation driver]       │
│    └─> q_array: (168,)                                      │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. PREPROCESSING / 预处理                                    │
│    Functions: apply_quantile_filter(), aggregate_time_series│
│    - Apply quantile filter (fq) if requested                │
│    - Aggregate data (agg) if requested                      │
│    ├─> q_filtered: (168,) [detrended discharge]            │
│    └─> valid_mask: (168,) [boolean mask for valid data]    │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. FEATURE ENGINEERING / 特征工程                            │
│    A) Basic: build_design_matrix()                          │
│       Creates lagged precipitation matrix                   │
│       ├─> X: (168, 48) [each column is p shifted by τ]     │
│       └─> Each row represents one timestep                  │
│                                                              │
│    B) Advanced (if xknots): create_xprime_matrix()          │
│       Splits precipitation by intensity                     │
│       ├─> X': (168, 48 × n_segments)                        │
│       └─> Example: 3 segments → X': (168, 144)             │
│                                                              │
│    C) Advanced (if split_params): make_split_sets()         │
│       Creates separate datasets for different conditions    │
│       ├─> X_dry, X_moderate, X_wet                          │
│       └─> q_dry, q_moderate, q_wet                          │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. REGRESSION / 回归                                         │
│    Function: solve_rrd() in utils_core.py                   │
│    Solves: min ||Xβ - q||² + ν||Dβ||²                      │
│           β                                                  │
│    Where / 其中:                                             │
│    - X: Design matrix (lagged precipitation) / 设计矩阵      │
│    - β: RRD coefficients to estimate / 待估计的 RRD 系数     │
│    - q: Discharge (response) / 流量（响应）                  │
│    - ν: Regularization strength (nu) / 正则化强度           │
│    - D: Differencing matrix for smoothness / 差分矩阵        │
│    ├─> beta: (48,) [RRD coefficients]                      │
│    ├─> stderr: (48,) [standard errors]                     │
│    └─> fitted: (168,) [predicted discharge]                │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. ROBUST ITERATION (if robust=True) / 鲁棒迭代              │
│    Method: Iteratively Reweighted Least Squares (IRLS)      │
│    - Calculate residuals: res = q - fitted                  │
│    - Compute weights: w = 1 / (1 + (res/σ)²)               │
│    - Re-solve: min ||W^(1/2)(Xβ - q)||² + ν||Dβ||²        │
│    - Repeat until convergence                               │
│    └─> beta_robust: (48,) [robust RRD coefficients]        │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. POST-PROCESSING / 后处理                                  │
│    - Convert beta to DataFrame with lag index               │
│    - If xknots: convert beta' to NRF (betaprime_to_nrf)    │
│    - If nk: compute broken-stick representation             │
│    - Package everything into ERRAResult dataclass           │
│    └─> ERRAResult(                                          │
│         lags=[0, 1, 2, ..., 47],                           │
│         rrd=DataFrame(columns=['P'], index=lags),          │
│         stderr=DataFrame(...),                              │
│         fitted=array(...),                                  │
│         residuals=array(...),                               │
│         ...                                                  │
│       )                                                      │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ 7. VISUALIZATION / 可视化                                    │
│    Function: plot_erra_results() in utils.py                │
│    User calls:                                               │
│    plot_erra_results(result, observed_q=discharge, ...)     │
│    ├─> RRD plot: shows beta vs. lag                        │
│    ├─> Fitted vs. observed: shows fitted vs. discharge     │
│    ├─> Residuals: diagnostic plots                         │
│    └─> Saves figures to output directory                    │
└─────────────────────────────────────────────────────────────┘
```


---

## 🌍 Business Scenario Mapping / 业务场景映射

### Core Classes & Functions → Real-world Concepts / 核心类与函数 → 现实世界概念

| Code Element | Business Concept (EN) | 业务概念 (CN) | Explanation |
|--------------|----------------------|--------------|-------------|
| `erra()` | Catchment Response Estimator | 流域响应估计器 | Estimates how a watershed transforms precipitation into streamflow |
| `ERRAResult` | Analysis Report | 分析报告 | Complete record of analysis with response distributions and diagnostics |
| `rrd` (DataFrame) | Impulse Response Function | 脉冲响应函数 | "If 1mm rain falls now, how much streamflow increase at each future time?" |
| `lags` (array) | Time Delays | 时间延迟 | Hours/days between rainfall and runoff arrival |
| `nrf` (DataFrame) | Intensity-Dependent Response | 强度依赖响应 | "Light rain vs. heavy rain have different runoff efficiencies" |
| `xknots` (parameter) | Intensity Breakpoints | 强度分界点 | Thresholds to split rainfall into light/moderate/heavy categories |
| `split_params` (dict) | Condition-Based Segmentation | 基于条件的分割 | "Dry soil vs. wet soil produce different responses" |
| `nu` (regularization) | Smoothness Control | 平滑度控制 | Prevents overfitting by enforcing gradual changes in response |
| `fq` (quantile filter) | Trend Removal | 趋势去除 | Removes seasonal patterns and baseflow to focus on event responses |
| `robust=True` | Outlier Handling | 异常值处理 | Reduces influence of measurement errors and extreme events |
| `fitted` (array) | Model Prediction | 模型预测 | "What streamflow does the model predict based on observed rainfall?" |
| `residuals` (array) | Unexplained Variation | 未解释的变异 | Difference between observed and predicted streamflow |

---

### Real-world Use Cases / 实际应用案例

#### 🌧️ Case 1: Flood Forecasting / 洪水预报

**Scenario / 场景:** A water authority wants to predict flood risk 24 hours in advance.

**水务局希望提前 24 小时预测洪水风险。**

**ERRA Application / ERRA 应用:**
```python
# Estimate catchment response
result = erra(p=historical_rainfall, q=historical_streamflow, m=48, dt=1.0)

# Get response at 24-hour lag
response_24h = result.rrd.loc[24, 'P']

# Forecast: If 50mm rain falls now, expect:
forecast_increase = 50 * response_24h  # mm * (m³/s per mm) = m³/s increase
```

**Business value / 业务价值:** Early warning system for downstream communities / 下游社区的早期预警系统

---

#### 🌡️ Case 2: Climate Change Impact / 气候变化影响

**Scenario / 场景:** Researchers want to know if permafrost thawing is changing catchment response.

**研究人员想知道多年冻土融化是否正在改变流域响应。**

**ERRA Application / ERRA 应用:**
```python
# Split data into two periods
period1_data = data[data.year < 2000]
period2_data = data[data.year >= 2000]

result1 = erra(p=period1_data.precip, q=period1_data.discharge, m=60)
result2 = erra(p=period2_data.precip, q=period2_data.discharge, m=60)

# Compare response functions
response_change = result2.rrd - result1.rrd
```

**Business value / 业务价值:** Inform water resource adaptation strategies / 为水资源适应策略提供信息

---

#### 🌊 Case 3: Storm Type Separation / 风暴类型分离

**Scenario / 场景:** A hydrologist wants to separate the contribution of convective (intense, short) vs. stratiform (gentle, long) rainfall.

**水文学家希望分离对流（强烈、短暂）与层状（温和、长时间）降雨的贡献。**

**ERRA Application / ERRA 应用:**
```python
# Use multiple drivers
result = erra(
    p=pd.DataFrame({'Convective': conv_rain, 'Stratiform': strat_rain}),
    q=discharge,
    m=60,
    labels=['Convective', 'Stratiform']
)

# Compare responses
print(result.rrd['Convective'])  # Fast response, short duration
print(result.rrd['Stratiform'])   # Slow response, long duration
```

**Business value / 业务价值:** Better rainfall-runoff models for different storm types / 不同风暴类型的更好降雨-径流模型

---

#### 🏔️ Case 4: Soil Moisture Effects / 土壤湿度效应

**Scenario / 场景:** Engineers want to quantify how soil wetness affects flood generation.

**工程师希望量化土壤湿度如何影响洪水生成。**

**ERRA Application / ERRA 应用:**
```python
# Split by antecedent discharge (proxy for soil moisture)
split_params = {
    'crit': [discharge],
    'crit_label': ['Wetness'],
    'crit_lag': [24],  # Use discharge from 24 hours ago
    'pct_breakpts': [True],
    'breakpts': [[50, 90]],  # Dry, moderate, wet
}

result = erra(p=rainfall, q=discharge, m=48, split_params=split_params)

# Compare dry vs. wet conditions
print(result.split_labels)  # ['Dry', 'Moderate', 'Wet']
print(result.rrd)            # Separate RRDs for each condition
```

**Business value / 业务价值:** Design flood control systems that account for soil saturation / 设计考虑土壤饱和的洪水控制系统

---

## 🚀 Getting Started / 开始使用

### Installation / 安装

```bash
# Clone repository / 克隆仓库
git clone https://github.com/licm13/ERRA.git
cd ERRA

# Install in editable mode / 以可编辑模式安装
pip install -e .

# Or install with development tools / 或安装开发工具
pip install -e .[dev]
```

### Quick Test / 快速测试

```python
# Create a simple test / 创建简单测试
import numpy as np
from erra import erra, plot_erra_results

# Generate synthetic data / 生成合成数据
np.random.seed(42)
precip = np.random.exponential(2, 500)  # Rainfall events
discharge = np.convolve(precip, np.exp(-np.arange(50)/10))[:500]  # Smooth response

# Run ERRA / 运行 ERRA
result = erra(p=precip, q=discharge, m=30, nu=0.05)

# Plot results / 绘制结果
plot_erra_results(result, observed_q=discharge, save_plots=True, show_plots=True)

print("✅ Installation successful! Check ./figures/ for output plots.")
print("✅ 安装成功！检查 ./figures/ 目录查看输出图表。")
```

---

## 🛠️ Development Workflow / 开发工作流程

### For Adding New Features / 添加新功能

1. **Understand the requirement / 理解需求**
   - Read related papers in `reference_materials/papers/`
   - Check R implementation in `reference_materials/R_implementation/`

2. **Write tests first / 先写测试** (Test-Driven Development)
   - Add test case in `tests/test_<module>.py`
   - Run: `pytest tests/`

3. **Implement feature / 实现功能**
   - Modify core modules in `src/erra/`
   - Follow existing code style (Black formatting)

4. **Validate against R / 与 R 验证**
   - Run corresponding R script
   - Compare outputs (should match within numerical precision)

5. **Document / 文档化**
   - Update docstrings (bilingual)
   - Add example script in `code/examples/`
   - Update README.md

6. **Create pull request / 创建拉取请求**

---

### For Debugging Issues / 调试问题

**Common issues and where to look / 常见问题及排查位置:**

| Issue | Likely Location | What to Check |
|-------|----------------|---------------|
| Wrong RRD values | `utils_core.py`: `solve_rrd()` | Regularization strength (nu), design matrix construction |
| NaN in results | `erra_core.py`: preprocessing | Missing data handling, filter parameters (fq) |
| Slow performance | `utils_core.py`: `build_design_matrix()` | Use broken-stick (nk > 0) to reduce dimensions |
| Plotting errors | `utils.py`: `plot_erra_results()` | Check matplotlib backend, Chinese font configuration |
| Split not working | `splitting.py`: `make_split_sets()` | Validate split_params dictionary, check bin sizes |

---

## 📖 Recommended Reading Order / 推荐阅读顺序

### For Users (Just Want to Use ERRA) / 用户（只想使用 ERRA）

1. README.md (existing) - User guide / 用户指南
2. `code/examples/example_en.py` - Simple working example / 简单工作示例
3. `code/examples/master_demonstration.py` - Advanced features / 高级功能
4. API documentation in docstrings / docstring 中的 API 文档

### For Developers (Want to Modify ERRA) / 开发者（想要修改 ERRA）

1. This file (README_COMPREHENSIVE.md) - Architecture overview / 架构概述
2. `src/erra/__init__.py` - Public API / 公共 API
3. `src/erra/erra_core.py` - Main algorithm / 主算法
4. `src/erra/utils_core.py` - Core operations / 核心操作
5. `reference_materials/theory_pdfs/` - Mathematical foundation / 数学基础
6. `reference_materials/R_implementation/` - Reference implementation / 参考实现

### For Researchers (Want to Extend Theory) / 研究者（想要扩展理论）

1. `reference_materials/theory_pdfs/` - Kirchner 2022, 2024 papers
2. `reference_materials/papers/` - Application papers
3. `src/erra/nonlin.py` - Nonlinear implementation / 非线性实现
4. `src/erra/splitting.py` - Non-stationary implementation / 非平稳实现

---

## 🤝 Contributing Guidelines / 贡献指南

### Code Style / 代码风格

- **Python:** Follow PEP 8, use Black formatter
- **Docstrings:** Bilingual (English first, Chinese second) following NumPy style
- **Type hints:** Required for all function signatures
- **Comments:** Use for complex logic only, prefer self-documenting code

### Documentation Standards / 文档标准

```python
def example_function(param1: int, param2: str) -> float:
    """Brief description in English.
    
    简短的中文描述。
    
    Parameters / 参数
    ----------
    param1 : int
        Description in English / 英文描述
    param2 : str
        Description in English / 英文描述
        
    Returns / 返回
    -------
    float
        Description / 描述
    """
    pass
```

### Testing Requirements / 测试要求

- All new features must have unit tests
- Tests should cover both normal and edge cases
- Compare with R implementation when possible

---

## 📚 Additional Resources / 其他资源

### Key Papers / 关键论文

1. **Kirchner, J.W. (2024).** "Characterizing nonlinear, nonstationary, and heterogeneous hydrologic behavior using Ensemble Rainfall-Runoff Analysis (ERRA): proof of concept." *Hydrology and Earth System Sciences*, 28, 4427-4454.
   - 📄 https://doi.org/10.5194/hess-28-4427-2024

2. **Kirchner, J.W. (2022).** "Impulse response functions for heterogeneous, nonstationary, and nonlinear systems, estimated by deconvolution and demixing of noisy time series." *Sensors*, 22(9), 3291.
   - 📄 https://doi.org/10.3390/s22093291

### External Links / 外部链接

- **GitHub Repository:** https://github.com/licm13/ERRA
- **Issue Tracker:** https://github.com/licm13/ERRA/issues
- **Original R Code:** `reference_materials/R_implementation/erra_scripts_v1.06/`

---

## ❓ Frequently Asked Questions / 常见问题

### Q1: What's the difference between `code/erra.py` and `src/erra/`?

**A:** `code/erra.py` is a standalone demo script for quick testing in the `code/python-version-example/` folder. The actual production package is in `src/erra/`. Always use `from erra import erra` which imports from `src/erra/`.

**`code/erra.py` 是 `code/python-version-example/` 文件夹中用于快速测试的独立演示脚本。实际的生产包在 `src/erra/` 中。始终使用 `from erra import erra`，它从 `src/erra/` 导入。**

---

### Q2: When should I use `nu` (regularization)?

**A:** Use `nu > 0` when:
- Data is noisy (measurement errors)
- You want a smoother RRD (less oscillation)
- Sample size is small relative to number of lags

Recommended values / 推荐值:
- Clean data: `nu = 0`
- Slightly noisy: `nu = 0.01 - 0.1`
- Very noisy: `nu = 0.1 - 0.5`

---

### Q3: What's the difference between `xknots` and `split_params`?

**A:**
- **`xknots`:** Splits **precipitation** by intensity (endogenous) / 按强度分割**降水**（内生）
  - Example: Light rain vs. heavy rain
  
- **`split_params`:** Splits data by **external covariate** (exogenous) / 按**外部协变量**分割数据（外生）
  - Example: Dry conditions vs. wet conditions (using antecedent discharge)

You can use both simultaneously!

**两者可以同时使用！**

---

### Q4: How do I choose `m` (maximum lag)?

**A:** Rule of thumb / 经验法则:
- Set `m` large enough to capture the full response duration
- For hourly data: 48-120 hours (2-5 days)
- For daily data: 30-90 days (1-3 months)
- Check: RRD should decay to near zero by lag `m`

---

### Q5: What if I get NaN values in results?

**A:** Check / 检查:
1. Input data has too many missing values / 输入数据有太多缺失值
2. `m` is too large relative to data length / `m` 相对于数据长度太大
3. `nu` is too small (try increasing regularization) / `nu` 太小（尝试增加正则化）
4. Design matrix is rank-deficient (reduce `m` or increase `nu`) / 设计矩阵秩不足（减少 `m` 或增加 `nu`）

---

## 📝 Summary / 总结

This document provides a comprehensive guide for new developers to:

本文档为新开发者提供全面指南，以便：

✅ **Understand** the project architecture and core algorithms / **理解**项目架构和核心算法

✅ **Navigate** the codebase efficiently / **高效浏览**代码库

✅ **Modify** and extend ERRA for custom applications / **修改**和扩展 ERRA 以用于自定义应用

✅ **Debug** issues systematically / **系统地调试**问题

✅ **Contribute** high-quality code following project conventions / **贡献**遵循项目约定的高质量代码

---

## 🎯 Next Steps / 下一步

**For new team members / 对于新团队成员:**

1. ⬜ Read this document thoroughly / 仔细阅读本文档
2. ⬜ Install ERRA and run `code/examples/example_en.py` / 安装 ERRA 并运行示例
3. ⬜ Read `src/erra/erra_core.py` docstrings / 阅读核心文件的文档字符串
4. ⬜ Debug through a simple example to understand flow / 通过简单示例调试以理解流程
5. ⬜ Read relevant papers in `reference_materials/` / 阅读相关论文
6. ⬜ Start contributing! Pick an issue from GitHub / 开始贡献！从 GitHub 挑选一个问题

---

**Welcome to the ERRA team! / 欢迎加入 ERRA 团队！** 🌊📊

**Questions? Open an issue on GitHub or contact the maintainers.**

**有问题？在 GitHub 上提出问题或联系维护者。**

---

*Last updated: 2025-12-08*  
*Document version: 1.0*  
*Author: ERRA Development Team*

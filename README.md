# Ensemble Rainfall-Runoff Analysis (ERRA) Resource Hub / ERRA 资源中心

> English and Chinese instructions are provided side by side so that students can follow along no matter which language they prefer.
> 文档提供中英文双语说明，方便不同背景的读者快速上手。

## Overview / 项目概览

This repository consolidates the theoretical background and practical tooling for the Ensemble Rainfall-Runoff Analysis (ERRA) framework.  It extends the original material curated by James Kirchner with Python demonstrations, high-level documentation, and reproducible example workflows inspired by three recent hydrologic studies.

本仓库汇总了 Ensemble Rainfall-Runoff Analysis (ERRA) 方法的理论资料与实践工具。在原始资料的基础上，补充了 Python 示例、整体说明文档，以及基于三篇最新研究的可复现案例流程。

## Repository layout / 仓库目录结构

```
.
├── application/              # Three reference studies that applied ERRA / 三篇 ERRA 应用研究
├── code/                     # Theory PDFs and Python scripts from the original release / 理论文档与原始代码
├── examples/                 # New synthetic reproductions of the three ERRA studies / 三个研究的合成复刻示例
├── figures/                  # Placeholder for root-level figures (optional) / 根目录图片（可选）
└── README.md                 # This bilingual project overview / 本双语总览文档
```

- **`application/`** – Contains the three published manuscripts (PDF) that showcase how ERRA reveals hydrologic dynamics in different climates.
  **`application/`** 目录包含三篇使用 ERRA 的论文 PDF，展示其在不同气候区的应用方式。
- **`code/`** – Mirrors the theory pack provided by the ERRA authors, with both the original R assets and the maintained Python translation.
  **`code/`** 目录保存原作者的 R 代码与本仓库维护的 Python 实现。
- **`examples/`** – Newly added, holds runnable Python notebooks/scripts that reproduce the high-level analyses described in each paper using synthetic data (so you can explore the workflow without the proprietary observations).
  **`examples/`** 是新增目录，提供基于合成数据的 Python 脚本，逐一复刻三篇论文中的分析流程，让读者在没有原始数据的情况下也能上手。

## Getting started / 快速上手

1. **Create a Python environment** (Python ≥ 3.9 recommended).  Install dependencies:
   ```bash
   pip install numpy pandas matplotlib scipy
   ```
   1. **创建 Python 环境**（推荐 Python ≥ 3.9），并安装依赖：
      ```bash
      pip install numpy pandas matplotlib scipy
      ```
2. **Add the project to your `PYTHONPATH`** or run the scripts from within the repository so that `code/python` can be imported.
   2. **配置 `PYTHONPATH`** 或在仓库目录内执行脚本，以便正确导入 `code/python` 模块。
3. **Run an example script** to verify the setup:
   ```bash
   python code/python/example.py
   ```
   3. **运行示例脚本** 验证环境是否正确：
      ```bash
      python code/python/example.py
      ```

Each example script automatically creates a `figures/` folder alongside the script (if not present) and exports high-resolution plots with both English and Chinese labels.  This ensures that the workflow is self-documented and presentation-ready.

每个示例脚本会在脚本所在路径自动创建 `figures/` 文件夹（若不存在），并输出带有中英文注释的高分辨率图件，便于教学展示与报告。

## Python ERRA utilities / Python 版 ERRA 工具

The modernised ERRA engine lives in [`code/python/erra.py`](code/python/erra.py) and exposes:

- `erra(...)` – Compute runoff response distributions (RRDs) using ridge regression with optional quantile filtering and aggregation.
- `plot_erra_results(...)` – Produce a multi-panel diagnostic dashboard with Chinese font support and publication-quality defaults.

改写后的 ERRA 核心位于 [`code/python/erra.py`](code/python/erra.py)，提供以下接口：

- `erra(...)` – 使用岭回归估算径流响应分布，可选分位滤波与时间聚合。
- `plot_erra_results(...)` – 输出多图综合诊断结果，默认支持中文字体与出版级画质。

Import convenience is provided through `code/python/__init__.py`, so you can simply write:

```python
from erra import erra, plot_erra_results
```

通过 `code/python/__init__.py` 已经导出了常用函数，因此可以直接：

```python
from erra import erra, plot_erra_results
```

## Synthetic reproductions / 合成复刻示例

The newly added `examples/` directory contains four ready-to-run scripts:

1. **`gao2025_dynamic_linkages.py`** – Mimics Gao et al. (2025) by generating three precipitation drivers (convective bursts, stratiform rain, and base recharge) feeding a groundwater-dominated catchment.  The script documents how ERRA separates fast and slow runoff pathways.
2. **`sharif_ameli2025_functional_simplicity.py`** – Re-creates the Sharif & Ameli (2025) focus on stormflow generation with a synthetic landscape that toggles between wet/dry states, illustrating how ERRA highlights functional simplicity despite process heterogeneity.
3. **`tu2025_permafrost_transition.py`** – Emulates permafrost degradation impacts reported by Tu et al. (2025) by switching the runoff kernel mid-way, revealing a declining discharge sensitivity to precipitation inputs.
4. **`complex_sensitivity_study.py`** – A comprehensive stress test that mixes multiple precipitation drivers, variable observation weights, and non-stationary noise to probe ERRA’s robustness and the effect of regularisation.

`examples/` 目录新增了四个可运行脚本：

1. **`gao2025_dynamic_linkages.py`** – 以合成数据模拟 Gao 等（2025）研究的情景，生成对流暴雨、层状降雨与地下水补给三类驱动，展示 ERRA 如何区分快、慢径流路径。
2. **`sharif_ameli2025_functional_simplicity.py`** – 复刻 Sharif 和 Ameli（2025）的暴雨径流分析，通过湿润/干燥状态切换的合成流域说明“功能简洁性”的含义。
3. **`tu2025_permafrost_transition.py`** – 重现 Tu 等（2025）报道的多年冻土退化效应，在序列中途更改冲激响应核，展示流量对降水敏感度的下降。
4. **`complex_sensitivity_study.py`** – 综合性压力测试，混合多种降水驱动、可变观测权重与非平稳噪声，检验 ERRA 的稳健性与正则化参数的影响。

Each script is self-contained: run it via `python examples/<script_name>.py`.  The scripts print key diagnostics (design matrix size, regression strength, leading RRD coefficients) and create bilingual figures in `examples/figures/`.

每个脚本均可独立运行：执行 `python examples/<脚本名>.py` 即可。脚本会打印关键诊断信息（设计矩阵规模、正则化强度、主要 RRD 系数等），并在 `examples/figures/` 输出中英文对照的图件。

## Reproducing the theory / 查阅理论资料

- **Introductory notes**: `code/an-introduction-to-erra-v1.06.pdf`
- **Methodological references**: `code/Kirchner_2022_*.pdf`, `code/Kirchner_2024_*.pdf`
- **Original scripts**: `code/demonstration-scripts/`, `code/erra_scripts_v1.06/`

- **方法简介**：`code/an-introduction-to-erra-v1.06.pdf`
- **理论论文**：`code/Kirchner_2022_*.pdf`、`code/Kirchner_2024_*.pdf`
- **原始脚本**：`code/demonstration-scripts/`、`code/erra_scripts_v1.06/`

These materials pair with the synthetic examples so that you can understand both the conceptual reasoning and the practical implementation of ERRA in one place.

结合上述资料与合成示例，读者可以在同一仓库内完成 ERRA 理论学习与实操演练。

## Citing / 引用

If you build upon this repository, please cite the original ERRA publications alongside any derivative work you publish.  For classroom use, acknowledge both James Kirchner’s ERRA series and the synthetic workflows prepared here.

若在科研或教学中引用本仓库，请同时注明原始 ERRA 论文及本仓库整理的合成工作流程。

## Feedback / 反馈

Questions or suggestions can be submitted through the project issue tracker or by opening a pull request with proposed improvements.

如需反馈问题或提供建议，欢迎在项目 issue 区留言，或提交 pull request 贡献改进。

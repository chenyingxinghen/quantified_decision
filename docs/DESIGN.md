# 量化决策系统 · 设计文档

> 项目：`G:\quantified_decision`　|　生成日期：2026-07-24　|　基于代码静态分析（core / config / analysis / scripts / visualization / core/automation）

---

## 1. 系统概述

本系统是一套**数据驱动的量化选股 / 决策系统**，覆盖从原始行情与基本面数据到机器学习模型训练、历史回测、因子分析、实盘下单的完整闭环。核心设计目标是：**可复现、防前视泄露、训练/回测/选股三阶段因子一致**。

- **数据源**：统一为聚源 SQL Server（`JYDB`）。旧的 `stock_finance.db` 已废弃，财务改走 `jydb_features.db` 的 `pit_features`（Point-In-Time 对齐）。
- **核心产物**：多目标 ML 因子模型（`XGBoost` / `LightGBM`）与可选的神经网络模型，配合横截面归一化统计，用于回测与每日选股。
- **交付形态**：离线回测 + 因子分析（研究态），以及基于 `easytrader` 的 Windows 券商客户端实盘自动化（生产态）。

---

## 2. 架构分层

系统按依赖方向自下而上分为 11 层，配置层为被普遍依赖的叶子层，编排层（scripts + shell）驱动端到端流程。

| 层 | 目录 | 职责 |
|----|------|------|
| L0 配置 | `config/` | 数据库路径、市场限制、因子/模型/训练、策略退出、实盘、神经超参。单向依赖链：`data_config → jydb_config → factor_config → strategy_config → automation_config` |
| L1 数据 | `core/data/` | 原始→特征 ETL：bronze `jydb_raw.db` 抽取；silver `jydb_features.db`（PIT）、`stock_daily.db`（复权 OHLCV）、`stock_meta.db`（基础/行业/情绪） |
| L2 技术基元 | `core/analysis/` | K 线形态、趋势线、价格行为分析；被因子、回测、可视化复用 |
| L3 因子·模型 | `core/factors/` | **最核心层**：因子计算、多目标标签、特征工程、ML 模型封装、中央训练流水线 |
| L4 神经 | `core/neural/` | PyTorch 网络与训练器，**复用 L3 的因子缓存/标签/归一化**，仅替换模型 |
| L5 回测 | `core/backtest/` + `core/exit_rules` | 回测引擎、数据加载、组合/绩效、ML 主策略、统一退出规则 |
| L6 分析 | `analysis/` | 稳健性（RankIC/regime）、组合、SHAP、机制叙述、报告渲染 |
| L7 可视化 | `visualization/` | 回测、支撑线、因子诊断、K 线参数绘图 |
| L8 实盘 | `core/automation/` | easytrader 补丁、交易接口、实盘执行控制器（与回测共用退出规则） |
| L9 编排 | `scripts/` + `*.sh` | CLI 与实验流水线，连接 L0–L8 驱动端到端 |
| L10 测试 | `tests/` | 模块契约与端到端回归 |

---

## 3. 核心模块说明

### 3.1 配置层 `config/`
- `data_config.py`：最底层，定义 `DATABASE_PATH`、市场限制、并行数、支持市场，读取 `GEMINI_DATA_OUT` 等环境变量。
- `jydb_config.py`：聚源连接/路径配置，`pyodbc` 为可选 `try/except` 导入。
- `factor_config.py`：定义 `ModelConfig` / `TrainingConfig` / `FactorConfig` / `OptimizationConfig`。
- `strategy_config.py`：策略与退出参数（时间止损、止损止盈、趋势线周期、选股市场）。
- `automation_config.py` / `neural_config.py`：实盘与神经网络超参。

### 3.2 数据层 `core/data/`
- `jydb_raw_etl.py`：定义原始表抽取规格，向 `jydb_raw.db` 抽取。
- `jydb_feature_store.py`：**统一 PIT 特征库**，远端表→长表→SQLite；通过 `available_date <= 交易日` 约束实现 Point-In-Time 读取，从根本上防止前视泄露。
- `jydb_market_etl.py`：聚源行情→`stock_daily.db`（复权 OHLCV）。
- `market_sentiment_calculator.py`：市场情绪指标→`stock_meta.db`。

### 3.3 因子与模型层 `core/factors/`（核心）
- `comprehensive_factor_calculator.py`：**综合因子计算器**，统一编排全部因子子模块，保证训练/预测一致。
  - `quantitative_factors.py`（TA-Lib 技术指标）
  - `candlestick_pattern_factors.py`（K 线形态）
  - `fundamental_factors.py`（PIT 财务/`FinanceReportFetcher`）
  - `advanced_factors.py`（时间序列/风险因子）
  - `external_source_factors.py`（聚源特征库外部因子）
  - `feature_engineering.py`（比率/乘积变换）、`factor_filler.py`（缺失填充）
- `multi_objective_labels.py`：**多目标标签构造**，严格 forward-only（标签自 `t+1` 开盘起）并使用正交收益腿，是防泄露的关键设计。
- `ml_factor_model.py`：ML 因子模型封装（`MLFactorModel` / `EnsembleFactorModel` / `MultiObjectiveFactorModel`，支持 XGB/LGB/sklearn）。
- `train_ml_model.py`：中央训练流水线 `MLModelTrainer`——加载数据→算因子→准备标签→横截面归一化→多目标训练→评估→保存模型与 `norm_stats.pkl`、`factor_summary.json`、`selected_features.json`。
- `feature_selector.py` / `model_optimizer.py`：横截面特征选择与超参优化。

### 3.4 神经网络层 `core/neural/`
- `nn_models.py`：PyTorch 网络 `NeuralNet` 与多目标神经模型 `MultiObjectiveNeuralModel`。
- `trainer.py`：`NeuralTrainer` **复用 `MLModelTrainer` 的因子缓存/标签/归一化**，仅替换模型结构，输出 `neural_multi_objective_model.pkl`。
- `portfolio.py`：组合权重优化（最大夏普/风险平价/回撤惩罚）。

### 3.5 回测层 `core/backtest/` + `core/exit_rules.py`
- `engine.py`：`BacktestEngine`，协调策略/数据/组合/绩效/退出规则/趋势线。
- `data_handler.py`：行情加载与复权（`LazyMarketSnapshot` 懒加载）。
- `strategies/ml_factor_strategy.py`：**主回测策略**，完全依赖训练期因子缓存，加载模型 + `norm_stats`，可选组合优化。
- `exit_rules.py`：止损/止盈/时间止损/支撑破位统一判定（`evaluate_exit`），**回测与实盘共用**，保证研究态与生产态退出逻辑一致。

### 3.6 分析层 `analysis/`
- `common.py`：分析公共层（加载模型、构建/缓存 `full_dataset`、归一化、预测打分、regime 分类）。
- `robustness.py`：日频 RankIC、regime 异质性、分组收益。
- `shap_analysis.py`：SHAP/树模型重要性、类别贡献、跨模型一致性（SHAP 缺失时自动回退 gain + 排列重要性）。
- `report.py`：渲染图表并写出 `report.md` / `metrics.csv` / `figures/*.png`。

### 3.7 可视化层 `visualization/`
- `visualize_backtest.py`、`visualize_support_lines.py`、`diagnose_factors.py`、`visualize_candlestick_params.py`。

### 3.8 实盘自动化层 `core/automation/`
- `easytrader_patch.py`：easytrader 补丁（验证码识别、剪贴板、价格舍入、弹窗处理）。
- `trader_interface.py` / `execution_controller.py`：交易接口封装与实盘执行控制器（下单循环、涨跌停价计算、与回测同套退出规则，写入 `automation/tracking.json`）。
- 入口 `scripts/main_auto_trade.py`：选股 → `signals.json` → 执行控制器 → 券商下单。

### 3.9 编排层 `scripts/` + shell
- **数据**：`pull_jydb_parallel.py`、`build_intermediate_from_raw.py`（可断点续跑/可 RAM 盘）、`update_industry.py`、`update_daily_data.py`。
- **训练**：`train_model.py`、`train_neural_model.py`、`cloud_train.py`。
- **回测**：`run_backtest.py`、`run_backtests_parallel.py`、`make_backtest_plan.py`、`aggregate_by_objective.py`。
- **选股/实盘**：`select_stocks.py`、`select_neural_portfolio.py`、`main_auto_trade.py`。
- **实验流水线**（shell）：`run_2yr_backtests.sh`、`run_by_objective_parallel.sh`、`run_exp3yr_resilient.sh`（断点续跑）、`run_exp3yr_analysis.sh`。

---

## 4. 数据流（Pipeline）

```
聚源 JYDB (SQL Server)
  │ scripts/pull_jydb_parallel.py
  ▼
jydb_raw.db (bronze)
  │ scripts/build_intermediate_from_raw.py
  ├─► jydb_features.db (silver, PIT 特征/财务)
  ├─► stock_daily.db   (silver, 复权 OHLCV)
  └─► stock_meta.db    (基础/行业/情绪)
        │
        ▼ ComprehensiveFactorCalculator 编排（量化+K线+基本面+PIT+情绪+外部）
        → FeatureEngineer 变换 → FactorFiller 补缺
        ▼
{code}_factors.parquet  (因子缓存：训练/回测/选股/分析 共用)
        ├──────────────► MultiObjectiveLabelBuilder (forward-only 正交标签)
        │                        ▼
        │              MLModelTrainer (横截面归一化 + 多目标 XGB/LGB)
        │                        ▼
        │              models/*.pkl + norm_stats.pkl + *.json
        │                        │
        ├──────────────► BacktestEngine + MLFactorBacktestStrategy ─► backtest_result/*
        ├──────────────► select_stocks → signals.json → ExecutionController → 券商下单
        ├──────────────► analysis/run_analysis → analysis/output/*
        └──────────────► NeuralTrainer (复用缓存/标签/归一化) → neural_*.pkl
```

---

## 5. 关键设计决策

1. **单一事实来源 + 因子缓存共享**：`{code}_factors.parquet` 由训练流水线产出，回测、选股、分析直接消费，**从根本上消除训练/预测因子错位**。
2. **Point-In-Time 防前视**：`jydb_feature_store` 用 `available_date <= 交易日` 读取；`multi_objective_labels` 标签自 `t+1` 开盘起算并使用正交腿，确保回测无信息泄露。
3. **横截面归一化**：训练期计算 `norm_stats.pkl` 并在回测/选股时复用，避免未来信息混入标准化。
4. **双模型并行、接口一致**：树模型 `MultiObjectiveFactorModel` 与神经 `MultiObjectiveNeuralModel` 暴露相同 `predict` / `feature_names` / `save_model` / `load_model` 接口，回测与选股可无缝切换。
5. **退出规则共享**：`core/exit_rules.py` 同时被回测引擎与实盘执行控制器引用，保证研究态与生产态退出逻辑一致。
6. **bronze→silver 分层 ETL**：原始抽取与特征构建解耦，`build_intermediate_from_raw` 支持断点续跑与 RAM 盘加速，适合大批量重建。

---

## 6. 已知风险与技术债

| 项 | 位置 | 影响 | 建议 |
|----|------|------|------|
| 悬空依赖 `DataFetcher` | `visualization/visualize_backtest.py:24` | 运行该脚本即 `ImportError` | 删除该导入/使用，或补齐 `core/data` 中的 `DataFetcher` |
| 已废弃 `stock_finance.db` | `database/` | 无读取方（已迁 PIT） | 可清理，避免误用 |
| SHAP 云端不可用 | `analysis/shap_analysis.py` | 分析报告改用 gain+排列重要性 | 已在代码内优雅回退，功能不受影响 |
| 平台限定依赖 | `requirements.txt` | `pyodbc`/`easytrader`/`comtypes`/`paramiko` 仅本地 Windows | 云端已注释，按需单机安装 |

---

## 7. 运行入口速查

| 目标 | 命令 |
|------|------|
| 抽取原始数据 | `python scripts/pull_jydb_parallel.py` |
| 构建中间库 | `python scripts/build_intermediate_from_raw.py` |
| 训练树模型 | `python scripts/train_model.py` |
| 训练神经模型 | `python scripts/train_neural_model.py` |
| 单次回测 | `python scripts/run_backtest.py` |
| 每日选股 | `python scripts/select_stocks.py` |
| 因子分析 | `python analysis/run_analysis.py` |
| 端到端实验 | `bash run_exp3yr_resilient.sh` |
| 实盘交易 | `python scripts/main_auto_trade.py` |

> 依赖与可选包详见 `requirements.txt`（区分 Windows 本地与云端 Linux 镜像）。

# Quantified Decision 量化决策系统

全栈 A 股量化选股与交易系统，集成 130+ 技术/基本面/形态因子、XGBoost/LightGBM 机器学习模型、回测框架与自动化实盘交易。

## 功能特性

- **因子引擎** — 62 项技术指标 + 21 种 K 线形态 + 30+ 基本面因子 + 10+ 市场情绪指标
- **机器学习** — XGBoost/LightGBM 排序模型，正交优化标签（路径效率 × 收益显著性）
- **回测框架** — 模块化事件驱动回测，支持止盈/止损/时间/趋势突破四种退出机制
- **实盘自动化** — 基于 easytrader + 同花顺客户端的自动交易调度
- **Web 管理端** — Vue 3 + Element Plus 前端，FastAPI 后端，支持选股/模拟盘/数据管理

## 架构概览

```
quantified_decision/
├── config/             中心配置（数据源、策略参数、模型超参、自动化参数）
├── core/
│   ├── factors/        因子计算 & 机器学习模型（130+ 因子、XGBoost/LightGBM）
│   ├── backtest/       回测引擎（策略、投资组合、业绩分析、数据处理器）
│   ├── data/           Baostock 数据获取（日线、财务、股票列表）
│   ├── automation/     实盘交易（easytrader 封装、执行控制器）
│   └── analysis/       技术分析工具（趋势线、K 线形态）
├── scripts/            命令行入口（数据更新、训练、回测、选股、自动交易）
├── quantification-system/
│   ├── backend/        FastAPI Web API（7 个路由模块）
│   └── frontend/       Vue 3 + Vite SPA（Element Plus 界面）
├── models/             训练模型归档
├── database/           SQLite 数据库（日线、财务、元数据、用户数据）
├── backtest_result/    回测输出
└── visualization/      诊断可视化脚本
```

## 环境要求

- Python 3.10+
- Node.js 18+（仅 Web 前端需要）
- 同花顺客户端（仅实盘交易需要）

## 安装

```bash
# 克隆仓库
git clone <repo-url>
cd quantified_decision

# 创建虚拟环境
python -m venv .venv
.venv\Scripts\activate    # Windows
# source .venv/bin/activate  # Linux/Mac

# 安装依赖
pip install -r quantification-system/backend/requirements.txt
pip install xgboost lightgbm scikit-learn pandas numpy matplotlib seaborn baostock easytrader apscheduler joblib pyarrow tqdm

# 安装前端依赖（可选）
cd quantification-system/frontend
npm install
cd ../..
```

## 配置说明

所有配置集中在 `config/` 目录下：

| 配置文件 | 说明 |
|---------|------|
| `baostock_config.py` | 数据源配置、数据库路径、市场定义 |
| `factor_config.py` | 模型超参数、训练参数、因子参数 |
| `strategy_config.py` | 回测/交易策略参数（止损、佣金、持有期） |
| `automation_config.py` | 自动交易参数（交易窗口、仓位、过滤条件） |

## 使用指南

### 1. 数据准备

```bash
# 增量更新所有市场数据（推荐）
# 依次跑：个股行情 → 财务 → 指数日线/货币供应 → 因子缓存
python scripts/update_daily_data.py

# 全量更新
python scripts/update_daily_data.py --full

# 只刷指数与宏观（idx_* 因子的原料），不动个股与因子缓存
python scripts/update_daily_data.py --only-index-macro

# 业绩预告（fc_* 因子的原料）
python scripts/ingest_performance_forecast.py
```

### 2. 训练模型

两条入口按**模型族**分工，共用同一份数据基座
（`TrainingConfig.CURRENT_CACHE_DIR`）与同一套标签口径：

```bash
# 树模型（XGBoost / LightGBM），236 列面板
python scripts/train_tree_model.py
python scripts/train_tree_model.py --stocks 5480 --seed 42

# 仅更新因子缓存，不训练
python scripts/train_tree_model.py --update-cache-only

# NAM（当前生产载体），224 列面板。需要 torch 解释器
python -u scripts/train_nam_model.py --stocks 5480 --years 13 --end 2022-09-05 \
    --disable-gate --target returns --y-scale 2 --expert-hidden 16 --lr 2e-3 \
    --drop-groups forecast --select-holdout 0.4 --seed 42 \
    --save-model-dir models/nam_gate/<tag>_s42
```

面板差异只由显式开关决定（NAM 剔 `*_regime_*` + `--drop-groups forecast`），
不是版本差异。生产推理载体一律放 `models/mark/` 下。

### 3. 回测

```bash
python scripts/run_backtest.py
```

### 4. 选股

```bash
# 默认输出 Top 20
python scripts/select_stocks.py

# 自定义输出
python scripts/select_stocks.py --top 30 --min-confidence 65

# 指定模型（NAM 与树存档都支持，加载器自动识别）
python scripts/select_stocks.py --model models/mark/T115_idxrel_s42
```

### 5. 自动交易

```bash
python scripts/main_auto_trade.py
```

自动执行交易日调度：早盘信号生成 → 买入窗口执行 → 午后卖出窗口执行。

### 6. 分析与工具

```bash
# 因子重要性报告
python scripts/show_factor_importance.py

# 把已训练的 xgb+lgb 打包成等权集成存档
python scripts/build_ensemble.py

# 实测各板块涨跌停限额并落库（限额规则变更后需重跑）
python scripts/build_price_limit_history.py

# 财务数据连续性体检
python scripts/check_finance_data_gaps.py

# 行业分类更新
python scripts/update_industry.py
```

实验判定口径与同折对照臂在 `scripts/exp/`（见该目录 README）；
已关闭轴的一次性脚本在 `scripts/archive/`。

### 7. 启动 Web 管理端

```bash
# 启动后端（端口 8083）
cd quantification-system/backend
python main.py

# 启动前端开发服务器（端口 5173，代理到后端）
cd quantification-system/frontend
npm run dev

# 生产构建
npm run build  # 后端自动从 frontend/dist/ 提供静态文件
```

## 核心概念

### 正交优化标签

模型使用 **正交优化** 标签体系：
- **标签** = 路径效率（单位回撤收益率），衡量持有期内收益的平滑性
- **样本权重** = 收益显著性，放大高收益样本的影响

使模型偏好平滑、高收益的价格轨迹。

### 因子缓存

因子计算结果以 Parquet 格式按个股缓存，训练时自动增量更新，大幅提升重复运行速度。

### 策略对齐

回测策略（`MLFactorBacktestStrategy`）与实盘选股逻辑（`select_for_live`）使用相同的信号生成代码，保证回测与实盘信号一致性。

## 依赖概览

**Python 核心依赖：**

| 类别 | 包 |
|------|----|
| 机器学习 | xgboost, lightgbm, scikit-learn |
| 数据处理 | pandas, numpy, scipy, pyarrow |
| Web API | fastapi, uvicorn, pydantic |
| 数据源 | baostock |
| 自动化 | easytrader, pywinauto, apscheduler |
| 可视化 | matplotlib, seaborn |

**前端依赖：** Vue 3, Element Plus, ECharts, Axios, Vite

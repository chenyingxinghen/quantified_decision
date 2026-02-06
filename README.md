# 量化交易决策系统

基于威科夫理论和SMC流动性猎取策略的股票量化交易系统。

## 项目结构

```
quantified_decision/
├── core/                      # 核心功能模块
│   ├── data/                 # 数据获取和处理
│   │   ├── data_fetcher.py          # 主数据获取器
│   │   ├── browser_data_fetcher.py  # 浏览器数据获取器
│   │   └── akshare_patch.py         # AKShare补丁
│   ├── indicators/           # 技术指标
│   │   └── technical_indicators.py  # 技术指标计算
│   ├── analysis/             # 分析模块
│   │   ├── price_action_analyzer.py # 价格行为分析
│   │   ├── candlestick_patterns.py  # K线形态识别
│   │   └── trend_line_analyzer.py   # 趋势线分析
│   └── strategies/           # 交易策略
│       ├── wyckoff_strategy.py      # 威科夫Spring策略
│       └── smc_liquidity_strategy.py # SMC流动性策略
├── scripts/                   # 脚本工具
│   ├── init_data.py          # 数据初始化
│   ├── get_today.py          # 获取当日数据
│   ├── stock_screener.py     # 股票筛选器
│   ├── stock_analyzer.py     # 股票分析器
│   └── integrated_screener.py # 整合选股器
├── visualization/             # 可视化模块
│   ├── visualize_backtest.py        # 回测可视化
│   └── visualize_support_lines.py   # 支撑线可视化
├── quantitative_retest/       # 回测系统
│   ├── backtest_engine.py           # 回测引擎
│   ├── run_backtest.py              # 回测运行脚本
│   ├── performance_analyzer.py      # 性能分析器
│   ├── parameter_optimizer.py       # 参数优化器
│   ├── smart_parameter_tuner.py     # 智能参数调优
│   └── auto_backtest_iteration.py   # 自动迭代回测
├── config/                    # 配置文件
│   ├── config.py             # 主配置
│   └── strategy_config.py    # 策略配置
├── data/                      # 数据文件
│   ├── stock_data.db         # 股票数据库
│   └── cookie.json           # Cookie配置
├── docs/                      # 文档
│   ├── 使用说明.md
│   ├── 策略参数说明.md
│   ├── 回测可视化说明.md
│   └── AKSHARE_PATCH_DOC.md
├── utils/                     # 工具函数
└── requirements.txt           # 依赖包
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 初始化数据

```bash
python scripts/init_data.py
```

### 3. 运行选股器

```bash
python scripts/integrated_screener.py
```

### 4. 运行回测

```bash
python quantitative_retest/run_backtest.py
```

## 主要功能

### 数据获取 (core/data)
- 支持多市场股票数据获取（上证、深证、创业板、北交所）
- 自动处理滑动验证
- 增量更新和历史数据补充

### 技术分析 (core/indicators & core/analysis)
- 技术指标计算（MA、RSI、MACD、布林带等）
- 价格行为分析
- K线形态识别
- 趋势线分析

### 交易策略 (core/strategies)
- **威科夫Spring策略**: 识别底部区域的Spring形态，捕捉反转机会
- **SMC流动性策略**: 基于机构订单流的流动性猎取策略

### 回测系统 (quantitative_retest)
- 完整的回测引擎
- 性能分析和报告生成
- 参数优化
- 自动迭代优化

### 可视化 (visualization)
- K线图和交易点可视化
- 支撑/阻力线可视化
- 回测结果可视化

## 配置说明

主要配置文件位于 `config/` 目录：

- `config.py`: 数据库路径、市场配置、技术指标参数等
- `strategy_config.py`: 策略相关参数配置

## 文档

详细文档请查看 `docs/` 目录：

- [使用说明](docs/使用说明.md)
- [策略参数说明](docs/策略参数说明.md)
- [回测可视化说明](docs/回测可视化说明.md)
- [股票分析器使用说明](docs/股票分析器使用说明.md)

## 注意事项

1. 首次运行需要初始化数据库，可能需要较长时间
2. 数据获取可能遇到滑动验证，系统会自动切换到浏览器模式
3. 回测结果仅供参考，实际交易需谨慎

## 许可证

本项目仅供学习和研究使用。

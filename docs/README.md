# A股量化选股与回测系统

基于SMC流动性猎取和威科夫Spring策略的量化交易系统。

## 核心脚本

### 1. init_data.py - 数据初始化
初始化和更新股票数据库。

```bash
python init_data.py
```

功能：
- 初始化股票数据（首次使用）
- 增量/全量更新数据
- 查看数据库状态

### 2. integrated_screener.py - 选股系统
主动筛选当前买点的股票。

```bash
python integrated_screener.py
```

策略：
- SMC流动性猎取策略（快速反弹，高胜率）
- 威科夫Spring策略（底部反转，趋势跟随）

### 3. quantitative_retest/run_backtest.py - 回测系统
对策略进行历史数据回测。

```bash
python quantitative_retest/run_backtest.py
```

功能：
- 策略历史表现分析
- 生成收益曲线图
- 导出交易记录

## 项目结构

```
├── init_data.py              # 数据初始化脚本
├── integrated_screener.py    # 选股脚本
├── config.py                 # 配置文件
├── data_fetcher.py           # 数据获取模块
├── stock_screener.py         # 基础筛选模块
├── technical_indicators.py   # 技术指标计算
├── smc_liquidity_strategy.py # SMC策略实现
├── advanced_strategies.py    # 高级策略（威科夫等）
├── trend_line_analyzer.py    # 趋势线分析
├── price_action_analyzer.py  # 价格行为分析
├── stock_data.db             # 股票数据库
├── requirements.txt          # 依赖包
└── quantitative_retest/      # 回测模块
    ├── run_backtest.py       # 回测运行脚本
    ├── backtest_engine.py    # 回测引擎
    └── performance_analyzer.py # 性能分析
```

## 快速开始

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 初始化数据：
```bash
python init_data.py
# 选择 1 - 初始化股票数据
```

3. 运行选股：
```bash
python integrated_screener.py
# 选择策略进行筛选
```

4. 回测验证：
```bash
python quantitative_retest/run_backtest.py
# 选择策略进行回测
```

## 核心策略

### SMC流动性猎取策略
识别机构"扫止损"后的快速反弹机会。

特征：
- 流动性猎取：跌破支撑后快速收回
- 订单块回踩：机构建仓区域确认
- 公允价值缺口(FVG)：价格失衡区域
- 趋势线支撑：接近上升趋势线

### 威科夫Spring策略
识别底部反转的最后机会。

条件：
- Spring形态：跌破支撑后快速回升
- 均线向上：MA5 > MA20 > MA60
- 价格位置：在均线上方或接近均线
- 成交量确认：放量突破

## 依赖项

- Python 3.8+
- akshare - 股票数据获取
- pandas - 数据处理
- numpy - 数值计算
- talib - 技术指标
- matplotlib - 图表绘制
- scipy - 科学计算

## 注意事项

- 首次使用需要初始化数据，约需30-60分钟
- 建议定期更新数据以保持准确性
- 回测结果仅供参考，不构成投资建议
- 股市有风险，投资需谨慎

## 免责声明

本程序基于技术分析理论开发，仅供学习和研究使用。任何投资决策都存在风险，过往表现不代表未来收益。使用本程序进行投资决策的风险由用户自行承担。

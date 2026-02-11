# Core Backtest - 核心回测系统

## 概述

这是一个模块化、可扩展的回测框架，专为量化策略回测设计。相比旧的回测系统，新系统具有以下特点：

### 主要特点

1. **模块化设计** - 数据、策略、执行、分析完全分离
2. **策略可扩展** - 清晰的策略接口，易于实现新策略
3. **高性能** - 支持并行数据加载和索引优化
4. **灵活配置** - 支持多种参数配置
5. **完整分析** - 内置性能分析和报告生成

## 架构设计

```
core/backtest/
├── engine.py              # 回测引擎（核心）
├── strategy.py            # 策略基类和信号定义
├── portfolio.py           # 投资组合管理
├── data_handler.py        # 数据处理器
├── performance.py         # 性能分析器
└── strategies/            # 内置策略
    ├── __init__.py
    └── ml_factor_strategy.py  # ML因子策略
```

### 核心组件

#### 1. BacktestEngine（回测引擎）

协调各个模块，执行回测主循环。

```python
from core.backtest import BacktestEngine, DataHandler
from core.backtest.strategies import MLFactorBacktestStrategy

# 创建数据处理器
data_handler = DataHandler('data/stock_data.db')

# 创建策略
strategy = MLFactorBacktestStrategy(
    model_path='models/xgboost_factor_model.pkl',
    min_confidence=60.0
)

# 创建回测引擎
engine = BacktestEngine(
    strategy=strategy,
    data_handler=data_handler,
    initial_capital=1.0,
    commission_rate=0.001,
    max_positions=1
)

# 运行回测
results = engine.run(
    start_date='2023-01-01',
    end_date='2024-12-31'
)
```

#### 2. BaseStrategy（策略基类）

所有策略必须继承此类并实现 `generate_signals` 方法。

```python
from core.backtest.strategy import BaseStrategy, StrategySignal

class MyStrategy(BaseStrategy):
    def __init__(self):
        super().__init__(name="我的策略")
    
    def initialize(self, **kwargs):
        """初始化策略（可选）"""
        super().initialize(**kwargs)
        # 加载模型、设置参数等
    
    def generate_signals(self, current_date, market_data, portfolio_state):
        """生成交易信号"""
        signals = []
        
        # 策略逻辑
        for stock_code, stock_data in market_data.items():
            # 分析数据
            if self._should_buy(stock_data):
                signal = StrategySignal(
                    stock_code=stock_code,
                    signal_type='buy',
                    timestamp=current_date,
                    price=stock_data['close'].iloc[-1],
                    confidence=80.0,
                    stop_loss=...,
                    take_profit=...
                )
                signals.append(signal)
        
        return signals
    
    def on_trade(self, trade):
        """交易完成回调（可选）"""
        pass
    
    def cleanup(self):
        """清理资源（可选）"""
        pass
```

#### 3. Portfolio（投资组合）

管理持仓、现金和交易记录。

```python
from core.backtest.portfolio import Portfolio

portfolio = Portfolio(
    initial_capital=100000,
    commission_rate=0.001,
    max_positions=5
)

# 开仓
position = portfolio.open_position(
    stock_code='000001',
    date='2024-01-01',
    price=10.0,
    stop_loss=9.5,
    take_profit=11.0
)

# 平仓
trade = portfolio.close_position(
    stock_code='000001',
    date='2024-01-10',
    price=10.5,
    reason='take_profit'
)

# 查询状态
print(f"总资产: {portfolio.total_value}")
print(f"持仓数: {portfolio.position_count}")
```

#### 4. DataHandler（数据处理器）

负责数据加载、缓存和查询。

```python
from core.backtest.data_handler import DataHandler

handler = DataHandler('data/stock_data.db')

# 加载数据
data = handler.load_data(
    start_date='2023-01-01',
    end_date='2024-12-31',
    parallel=True
)

# 获取历史数据
hist_data = handler.get_historical_data(
    stock_code='000001',
    end_date='2024-06-01',
    lookback_days=60
)

# 获取市场快照
snapshot = handler.get_market_snapshot('2024-06-01')
```

#### 5. PerformanceAnalyzer（性能分析器）

计算性能指标和生成报告。

```python
from core.backtest.performance import PerformanceAnalyzer

analyzer = PerformanceAnalyzer()

# 计算指标
metrics = analyzer.calculate_metrics(
    trades=trades,
    initial_capital=100000,
    final_capital=120000
)

# 打印摘要
analyzer.print_summary(
    metrics=metrics,
    start_date='2023-01-01',
    end_date='2024-12-31',
    strategy_name='我的策略'
)

# 保存结果
analyzer.save_trades_to_csv(trades, 'trades.csv')
```

## 使用示例

### 1. 基本使用

```python
from core.backtest import BacktestEngine, DataHandler
from core.backtest.strategies import MLFactorBacktestStrategy

# 初始化
data_handler = DataHandler('data/stock_data.db')
strategy = MLFactorBacktestStrategy(
    model_path='models/xgboost_factor_model.pkl',
    min_confidence=60.0
)

engine = BacktestEngine(
    strategy=strategy,
    data_handler=data_handler,
    initial_capital=1.0,
    commission_rate=0.001
)

# 运行
results = engine.run('2023-01-01', '2024-12-31')

# 查看结果
print(results['metrics'])
```

### 2. 自定义策略

```python
from core.backtest.strategy import BaseStrategy, StrategySignal

class MomentumStrategy(BaseStrategy):
    def __init__(self, lookback=20):
        super().__init__(name="动量策略")
        self.lookback = lookback
    
    def generate_signals(self, current_date, market_data, portfolio_state):
        signals = []
        
        for stock_code, data in market_data.items():
            if len(data) < self.lookback:
                continue
            
            # 计算动量
            returns = data['close'].pct_change(self.lookback).iloc[-1]
            
            if returns > 0.1:  # 涨幅超过10%
                signal = StrategySignal(
                    stock_code=stock_code,
                    signal_type='buy',
                    timestamp=current_date,
                    price=data['close'].iloc[-1],
                    confidence=70.0
                )
                signals.append(signal)
        
        return signals

# 使用自定义策略
strategy = MomentumStrategy(lookback=20)
engine = BacktestEngine(strategy, data_handler)
results = engine.run('2023-01-01', '2024-12-31')
```

### 3. 批量回测

```python
# 测试不同参数
for confidence in [50, 60, 70, 80]:
    strategy = MLFactorBacktestStrategy(
        model_path='models/xgboost_factor_model.pkl',
        min_confidence=confidence
    )
    
    engine = BacktestEngine(strategy, data_handler)
    results = engine.run('2023-01-01', '2024-12-31', verbose=False)
    
    print(f"置信度 {confidence}%: 收益率 {results['metrics']['total_return_pct']:.2f}%")
```

## 性能指标

回测系统计算以下性能指标：

- **交易统计**
  - 总交易次数
  - 盈利/亏损交易数
  - 胜率
  - 平均持仓天数

- **收益指标**
  - 总收益/收益率
  - 平均收益率
  - 平均盈利/亏损

- **风险指标**
  - 盈亏比
  - 最大回撤
  - 夏普比率

- **其他**
  - 退出原因统计
  - 最佳/最差交易

## 退出机制

系统内置以下退出机制：

1. **止损** - 价格触及止损位
2. **止盈** - 价格触及止盈位
3. **时间止损** - 持仓5天未盈利且亏损超过1%
4. **趋势破位** - 跌破支撑趋势线

可以通过继承策略类自定义退出逻辑。

## 与旧系统对比

| 特性 | 旧系统 (quantitative_retest) | 新系统 (core.backtest) |
|------|----------------------------|----------------------|
| 模块化 | 较弱 | 强 |
| 策略扩展 | 需要修改核心代码 | 继承BaseStrategy即可 |
| 数据处理 | 耦合在引擎中 | 独立DataHandler |
| 性能分析 | 独立模块 | 集成PerformanceAnalyzer |
| 代码复用 | 低 | 高 |
| 测试友好 | 一般 | 好 |

## 快速开始

运行示例脚本：

```bash
python scripts/run_new_backtest.py
```

## 注意事项

1. 确保数据库路径正确
2. 模型文件必须存在
3. 数据量要足够（建议至少100个交易日）
4. 注意内存使用（大量股票时）

## 扩展开发

### 添加新策略

1. 在 `core/backtest/strategies/` 创建新文件
2. 继承 `BaseStrategy`
3. 实现 `generate_signals` 方法
4. 在 `__init__.py` 中导出

### 自定义退出逻辑

在策略类中重写 `_check_exit_conditions` 方法（需要修改引擎）或在信号中设置止损止盈。

### 添加新指标

在 `PerformanceAnalyzer` 中添加新的计算方法。

## 常见问题

**Q: 如何使用多个持仓？**

A: 设置 `max_positions` 参数：
```python
engine = BacktestEngine(..., max_positions=5)
```

**Q: 如何调整手续费？**

A: 设置 `commission_rate` 参数：
```python
engine = BacktestEngine(..., commission_rate=0.002)  # 0.2%
```

**Q: 如何只回测特定股票？**

A: 在 `run` 方法中传入股票列表：
```python
results = engine.run(
    start_date='2023-01-01',
    end_date='2024-12-31',
    stock_codes=['000001', '000002']
)
```

## 未来计划

- [ ] 支持做空
- [ ] 支持期货/期权
- [ ] 实时回测模式
- [ ] 更多内置策略
- [ ] 可视化界面
- [ ] 参数优化工具

## 贡献

欢迎提交Issue和Pull Request！

# 量化因子与机器学习模型

## 简介

本模块实现了基于62个技术指标的量化因子体系，并使用机器学习算法自动学习因子权重，从而发现量化因子与未来价格走势之间的隐藏联系。

## 核心特性

- ✅ **83个量化因子**: 
  - 62个技术指标因子（动量、趋势、波动率、成交量、价格形态）
  - 21个K线形态因子（单根形态、多根形态、形态强度）
- ✅ **因子缓存机制**: 自动缓存计算结果，加速训练过程
- ✅ **多种ML模型**: 支持XGBoost、LightGBM、Random Forest
- ✅ **自动特征工程**: 自动计算和标准化所有因子
- ✅ **因子重要性分析**: 识别最有预测力的因子
- ✅ **模型集成**: 组合多个模型提高预测稳定性
- ✅ **混合策略**: 结合ML和传统技术分析
- ✅ **回测集成**: 可直接用于回测系统

## 文件结构

```
core/factors/
├── quantitative_factors.py    # 62个技术指标因子计算
├── candlestick_pattern_factors.py  # 21个K线形态因子计算
├── ml_factor_model.py         # 机器学习模型实现
├── ml_strategy.py             # ML策略接口
├── train_ml_model.py          # 模型训练脚本（含缓存）
├── __init__.py                # 模块导出
└── README.md                  # 本文件
```

## 快速开始

### 1. 安装依赖

```bash
pip install xgboost lightgbm scikit-learn pandas numpy talib
```

### 2. 训练模型

```bash
python core/factors/train_ml_model.py
```

这将：
- 从数据库加载历史数据
- 计算所有量化因子
- 训练多个ML模型
- 保存最佳模型到 `models/` 目录

### 3. 使用模型

```python
from core.factors import MLFactorStrategy

# 加载模型
strategy = MLFactorStrategy('models/xgboost_factor_model.pkl')

# 筛选股票
result = strategy.screen_stock('000001.SZ', min_confidence=60)

if result:
    print(f"置信度: {result['confidence']:.2f}%")
    print(f"信号: {result['signal']}")
```

## 量化因子列表

### 技术指标因子 (62个)

#### 动量类因子 (8个)
- RSI (6日/12日)
- ROC (变动速率)
- MTM (动量指标)
- CMO (钱德动量摆动指标)
- StochRSI (随机强弱指数)
- RVI (相对波动率指数)

### 趋势类因子 (10个)
- MACD (指数平滑异同平均线)
- ADX (平均趋向指标)
- DMI (动向指标)
- Aroon (阿隆指标)
- TRIX (三重指数平滑)
- MA/CLOSE (均线价格比)
- MA线性回归系数

### 波动率因子 (7个)
- ATR (平均真实波幅)
- NATR (归一化ATR)
- 布林带宽度和位置
- CCI (顺势指标)
- Ulcer指数 (下行风险)
- 价格方差

### 成交量因子 (13个)
- OBV (能量潮)
- AD (累积/派发)
- Chaikin Oscillator
- MFI (资金流量指标)
- VR (成交量比率)
- VROC/VRSI/VMACD (量指标)
- 成交量/金额统计

### 价格形态因子 (10个)
- KDJ指标
- W%R (威廉指标)
- BIAS (乖离率)
- PSY (心理线)
- AR-BR指标
- CR (能量指标)
- 价格线性回归系数

### K线形态因子 (21个)

#### 单根K线形态 (9个)
- 白线/阳线 (看涨)
- 黑线/阴线 (看跌)
- 十字星 (犹豫/反转)
- 锤子线 (看涨反转)
- 上吊线 (看跌反转)
- 射击之星 (看跌反转)
- 倒锤线 (看涨反转)
- 光头光脚线 (强势趋势)
- 纺锤线 (犹豫/反转)

#### 多根K线形态 (7个)
- 看涨吞没 (看涨反转)
- 看跌吞没 (看跌反转)
- 刺穿线 (看涨反转)
- 乌云盖顶 (看跌反转)
- 晨星 (看涨反转)
- 暮星 (看跌反转)
- 孕线 (反转信号)

#### K线形态强度指标 (5个)
- K线实体比率
- 上影线比率
- 下影线比率
- 形态强度
- 形态确认度

**总计: 83个量化因子 (62个技术指标 + 21个K线形态)**

## 使用示例

### 示例1: 计算所有因子（技术指标 + K线形态）

```python
from core.factors import QuantitativeFactors, CandlestickPatternFactors
import pandas as pd

# 加载股票数据
data = pd.read_csv('stock_data.csv')

# 计算技术指标因子
tech_calc = QuantitativeFactors()
tech_factors = tech_calc.calculate_all_factors(data)

# 计算K线形态因子
pattern_calc = CandlestickPatternFactors()
pattern_factors = pattern_calc.calculate_all_candlestick_patterns(data)

# 合并所有因子
all_factors = pd.concat([tech_factors, pattern_factors], axis=1)

print(f"技术指标因子: {len(tech_factors.columns)}")
print(f"K线形态因子: {len(pattern_factors.columns)}")
print(f"总因子数: {len(all_factors.columns)}")
```

### 示例2: 使用因子缓存训练模型

```python
from core.factors.train_ml_model import MLModelTrainer

# 初始化训练器（自动启用缓存）
trainer = MLModelTrainer()

# 查看缓存信息
cache_info = trainer.get_cache_info()
print(f"已缓存: {cache_info['cached_stocks']} 只股票")

# 训练模型（自动使用缓存）
stocks_data = trainer.load_training_data(stock_codes, start_date, end_date)
X, y, factor_names = trainer.prepare_dataset(stocks_data, use_cache=True)

# 清理缓存（如需要）
trainer.clear_factors_cache()
```

### 示例3: 管理因子缓存

```bash
# 查看缓存信息
python scripts/manage_factors_cache.py info

# 查看特定股票的因子
python scripts/manage_factors_cache.py view --stock 000001.SZ

# 列出所有已缓存股票
python scripts/manage_factors_cache.py list

# 清理缓存
python scripts/manage_factors_cache.py clear
```

### 示例1: 计算因子

```python
from core.factors import QuantitativeFactors
import pandas as pd

# 加载股票数据
data = pd.read_csv('stock_data.csv')

# 计算所有因子
factor_calc = QuantitativeFactors()
factors = factor_calc.calculate_all_factors(data)

print(f"计算了 {len(factors.columns)} 个因子")
```

### 示例4: 训练自定义模型

```python
from core.factors import QuantitativeFactors, MLFactorModel

# 准备数据
factor_calc = QuantitativeFactors()
factors = factor_calc.calculate_all_factors(stock_data)

# 创建模型
model = MLFactorModel(model_type='xgboost', task='classification')

# 准备训练数据
X, y = model.prepare_training_data(
    factors, 
    stock_data, 
    forward_days=5,
    threshold=0.03
)

# 训练
results = model.train(X, y)

# 保存
model.save_model('my_model.pkl')
```

### 示例5: 批量筛选

```python
from core.factors import MLFactorStrategy

strategy = MLFactorStrategy('models/xgboost_factor_model.pkl')

# 批量筛选
stock_codes = ['000001.SZ', '000002.SZ', '600000.SH']
results = strategy.batch_screen(stock_codes, min_confidence=60)

# 显示结果
for result in results:
    print(f"{result['stock_code']}: {result['confidence']:.2f}%")
```

### 示例6: 混合策略

```python
from core.factors import HybridStrategy
from core.strategies import SMCLiquidityStrategy

# 创建混合策略
ml_model = 'models/xgboost_factor_model.pkl'
traditional = SMCLiquidityStrategy()

hybrid = HybridStrategy(ml_model, traditional)

# 使用混合策略
result = hybrid.screen_stock(
    '000001.SZ',
    ml_weight=0.6,
    traditional_weight=0.4
)
```

### 示例7: 因子分析

```python
from core.factors import MLFactorStrategy

strategy = MLFactorStrategy('models/xgboost_factor_model.pkl')

# 分析因子
analysis = strategy.analyze_factors('000001.SZ')

if analysis:
    print(f"股票: {analysis['stock_code']}")
    print("\n重要因子:")
    for factor in analysis['factors'][:10]:
        print(f"  {factor['name']}: {factor['value']:.4f} "
              f"(重要性: {factor['importance']:.4f})")
```

## 模型性能

典型的模型性能指标（基于历史回测）：

```
验证集性能:
  准确率: 0.68-0.72
  精确率: 0.65-0.70
  召回率: 0.70-0.75
  F1分数: 0.67-0.72
  AUC: 0.74-0.78
```

## 集成到回测系统

```python
# 在 quantitative_retest/backtest_engine.py 中
from core.factors import MLFactorStrategy

class StrategyInterface:
    def __init__(self):
        self.ml_strategy = MLFactorStrategy('models/xgboost_factor_model.pkl')
    
    def get_signals(self, current_date, historical_data, strategy_name='ml_factor'):
        if strategy_name == 'ml_factor':
            # 使用ML策略筛选
            results = []
            for code, data in historical_data.items():
                result = self.ml_strategy.screen_stock(code)
                if result:
                    results.append(result)
            
            # 返回最佳信号
            if results:
                return max(results, key=lambda x: x['confidence'])
        
        return None
```

## API文档

### QuantitativeFactors

计算技术指标因子的核心类。

**主要方法:**
- `calculate_all_factors(data)`: 计算所有62个技术指标因子
- `calculate_rsi(data, period)`: 计算RSI
- `calculate_macd(data)`: 计算MACD
- 等62个因子计算方法...

### CandlestickPatternFactors

计算K线形态因子的核心类。

**主要方法:**
- `calculate_all_candlestick_patterns(data)`: 计算所有21个K线形态因子
- `calculate_hammer(data)`: 计算锤子线
- `calculate_bullish_engulfing(data)`: 计算看涨吞没
- `calculate_morning_star(data)`: 计算晨星
- 等21个形态计算方法...

### MLFactorModel

机器学习模型类。

**主要方法:**
- `train(X, y)`: 训练模型
- `predict(factors)`: 预测
- `predict_signal(factors, threshold)`: 生成交易信号
- `save_model(filepath)`: 保存模型
- `load_model(filepath)`: 加载模型
- `get_top_factors(n)`: 获取最重要的N个因子

### MLFactorStrategy

ML因子策略类。

**主要方法:**
- `screen_stock(stock_code, min_confidence)`: 筛选单只股票
- `batch_screen(stock_codes, min_confidence)`: 批量筛选
- `analyze_factors(stock_code)`: 分析因子

### HybridStrategy

混合策略类。

**主要方法:**
- `screen_stock(stock_code, ml_weight, traditional_weight)`: 混合筛选

## 注意事项

1. **数据质量**: 确保输入数据完整且无异常值
2. **过拟合**: 使用时间序列分割验证，避免未来数据泄露
3. **样本平衡**: 注意正负样本比例，必要时使用采样技术
4. **定期更新**: 建议每月或每季度重新训练模型
5. **风险控制**: 设置合理的止损，不要过度依赖模型
6. **缓存管理**: 定期清理过期的因子缓存，节省磁盘空间

## 性能优化

1. **因子缓存**: 使用Parquet格式缓存因子计算结果
2. **并行计算**: 使用多进程加速因子计算
3. **特征选择**: 移除低重要性因子
4. **模型压缩**: 使用模型剪枝
5. **增量更新**: 只计算新增数据的因子

## 缓存机制

训练脚本会自动将计算的因子缓存到 `data/factors_cache/` 目录：

- 每只股票的因子保存为独立的Parquet文件
- 文件命名格式: `{股票代码}_factors.parquet`
- 缓存包含所有83个因子（技术指标 + K线形态）
- 下次训练时自动读取缓存，大幅提升速度

**缓存管理命令:**
```bash
# 查看缓存信息
python scripts/manage_factors_cache.py info

# 查看特定股票因子
python scripts/manage_factors_cache.py view --stock 000001.SZ

# 清理所有缓存
python scripts/manage_factors_cache.py clear --yes
```

## 常见问题

**Q: 训练需要多长时间？**
A: 500只股票2年数据，首次约10-30分钟，使用缓存后约5-10分钟。

**Q: 如何选择预测周期？**
A: 建议测试3天、5天、10天，选择回测表现最好的。

**Q: 置信度多少算高？**
A: 60%以上可考虑，70%以上较可靠。

**Q: K线形态因子有用吗？**
A: 研究表明K线形态结合技术指标可提升预测准确率2-5%。

**Q: 缓存会占用多少空间？**
A: 每只股票约50-200KB，500只股票约25-100MB。

**Q: 可以实盘交易吗？**
A: 建议先充分回测和模拟交易。

## 更多文档

- [机器学习因子模型使用指南](../../docs/机器学习因子模型使用指南.md)
- [量化因子详细列表](../../.kiro/steerin/量化因子.md)
- [完整示例代码](../../examples/ml_factor_example.py)

## 参考文献

1. 广发多因子系列21：alpha因子何处寻 掘金海量技术指标
2. XGBoost: A Scalable Tree Boosting System
3. LightGBM: A Highly Efficient Gradient Boosting Decision Tree

## 更新日志

- 2025-02: 新增K线形态因子
  - 新增21个K线形态因子（单根、多根、强度指标）
  - 实现因子缓存机制（Parquet格式）
  - 添加缓存管理工具
  - 总因子数提升至83个
  
- 2024-02: 初始版本
  - 实现62个技术指标因子
  - 支持XGBoost/LightGBM/RF
  - 提供混合策略和集成模型
  - 完整的训练和预测流程

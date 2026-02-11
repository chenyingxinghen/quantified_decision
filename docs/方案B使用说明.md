# 方案B：缓存完整特征使用说明

## 概述

方案B在训练时就将特征工程后的完整因子保存到缓存，回测时直接使用，无需重复特征工程。

## 优势

1. **性能最优** - 回测时无需重复特征工程计算
2. **一致性好** - 训练和回测使用完全相同的特征
3. **简化回测** - 回测代码更简单

## 劣势

1. **缓存更大** - 特征数从157个增加到235个，缓存文件增大约50%
2. **灵活性低** - 修改特征工程配置需要重新生成缓存
3. **初次训练慢** - 需要为所有股票应用特征工程

## 使用步骤

### 1. 清理旧缓存（可选）

如果之前使用方案A生成了缓存，建议先清理：

```bash
python scripts/manage_factors_cache.py clear --yes
```

### 2. 使用方案B训练模型

使用 `--cache-engineered` 参数：

```bash
python core/factors/train_ml_model.py --cache-engineered
```

输出示例：

```
================================================================================
机器学习因子模型训练（整合技术指标 + K线形态因子）
================================================================================

训练参数:
  股票数量: 500
  时间范围: 2022-01-01 至 2024-12-31
  预测周期: 5天
  分类阈值: 5.0%
  使用缓存: 是
  缓存策略: 方案B（缓存完整特征）  ← 确认这里

正在计算量化因子...
  缓存策略: 保存特征工程后的完整因子（方案B）  ← 确认这里
  ...

特征工程已在缓存时应用，跳过  ← 训练时跳过特征工程
  当前特征数: 235

训练完成！
================================================================================
✓ 缓存包含完整特征（方案B）
  回测时可以直接使用缓存，无需特征工程
  建议: 回测时设置 enable_feature_engineering=False
```

### 3. 验证缓存特征

使用检查脚本验证：

```bash
python scripts/check_feature_compatibility.py
```

应该看到：

```
4. 特征对比
   ✓ 完全匹配！模型和缓存特征一致

5. 建议
   ✓ 缓存可以正常使用
```

### 4. 回测时禁用特征工程

修改回测脚本，禁用特征工程：

```python
from core.backtest.strategies import MLFactorBacktestStrategy

strategy = MLFactorBacktestStrategy(
    model_path='models/xgboost_factor_model.pkl',
    min_confidence=60.0,
    use_cache=True,
    enable_feature_engineering=False,  # 禁用特征工程
)
```

或者修改 `scripts/run_backtest.py`：

```python
strategy = MLFactorBacktestStrategy(
    model_path=model_path,
    min_confidence=ML_FACTOR_MIN_CONFIDENCE,
    use_cache=use_cache,
    cache_dir=cache_dir,
    enable_feature_engineering=False,  # 改为False
    name="ML因子策略"
)
```

### 5. 运行回测

```bash
python scripts/run_backtest.py
```

输出应该显示：

```
策略初始化完成: ML因子策略
  模型: models/xgboost_factor_model.pkl
  模型特征数: 235
  最小置信度: 60.0%
  使用缓存: True
  缓存目录: data/factors_cache
  特征工程: 禁用  ← 确认禁用
```

## 性能对比

| 指标 | 方案A（基础因子缓存） | 方案B（完整特征缓存） |
|------|---------------------|---------------------|
| 缓存大小 | 100% | ~150% |
| 训练时间 | 快 | 中等 |
| 回测时间 | 中等 | 最快 |
| 灵活性 | 高 | 中等 |
| 推荐场景 | 开发调试 | 生产回测 |

## 切换方案

### 从方案A切换到方案B

1. 清理缓存
   ```bash
   python scripts/manage_factors_cache.py clear --yes
   ```

2. 使用方案B重新训练
   ```bash
   python core/factors/train_ml_model.py --cache-engineered
   ```

3. 修改回测脚本，禁用特征工程
   ```python
   enable_feature_engineering=False
   ```

### 从方案B切换到方案A

1. 清理缓存
   ```bash
   python scripts/manage_factors_cache.py clear --yes
   ```

2. 使用方案A重新训练（默认）
   ```bash
   python core/factors/train_ml_model.py
   ```

3. 修改回测脚本，启用特征工程
   ```python
   enable_feature_engineering=True
   ```

## 常见问题

### Q1: 如何知道当前使用的是哪个方案？

A: 运行检查脚本：

```bash
python scripts/check_feature_compatibility.py
```

- 如果显示"完全匹配"，说明是方案B
- 如果显示"缺少特征"，说明是方案A

### Q2: 方案B的缓存文件有多大？

A: 大约是方案A的1.5倍。例如：
- 方案A：3153个文件，约500MB
- 方案B：3153个文件，约750MB

### Q3: 可以混用两种方案吗？

A: 不建议。应该保持训练和回测使用相同的方案。

### Q4: 哪个方案更好？

A: 取决于使用场景：
- **开发调试**：方案A（灵活性高）
- **生产回测**：方案B（性能最优）
- **资源受限**：方案A（缓存更小）

### Q5: 如果回测时特征工程设置错误会怎样？

A: 策略初始化时会检测并警告：

```
检测到缓存包含完整特征，建议禁用特征工程
警告: 特征工程已启用，可能导致特征重复
```

## 最佳实践

1. **生产环境**
   - 使用方案B
   - 定期更新缓存（如每月）
   - 监控缓存大小

2. **开发环境**
   - 使用方案A
   - 快速迭代特征工程
   - 测试完成后切换到方案B

3. **资源管理**
   - 定期清理过期缓存
   - 使用 `manage_factors_cache.py` 管理
   - 监控磁盘空间

## 命令速查

```bash
# 清理缓存
python scripts/manage_factors_cache.py clear --yes

# 方案B训练
python core/factors/train_ml_model.py --cache-engineered

# 方案A训练（默认）
python core/factors/train_ml_model.py

# 检查特征兼容性
python scripts/check_feature_compatibility.py

# 查看缓存信息
python scripts/manage_factors_cache.py info

# 运行回测
python scripts/run_backtest.py
```

## 总结

方案B通过在训练时就保存完整特征，实现了最优的回测性能。虽然缓存文件更大，但对于生产环境的回测来说，这是值得的。

**推荐使用场景**：
- ✅ 生产环境回测
- ✅ 大规模回测
- ✅ 性能要求高
- ❌ 频繁修改特征工程
- ❌ 磁盘空间受限

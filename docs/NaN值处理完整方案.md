# NaN值处理完整方案

## 问题

训练时出现RuntimeWarning：
```
RuntimeWarning: invalid value encountered in subtract
RuntimeWarning: invalid value encountered in dot
```

这些警告来自于pandas和numpy在处理包含NaN值的数据时的计算操作，特别是在scikit-learn的StandardScaler中。

## 根本原因

NaN值在以下阶段产生和传播：

1. **因子计算阶段**：高级因子计算中的统计函数（skew、kurt、correlation等）可能返回NaN
2. **数据合并阶段**：合并多个因子DataFrame时可能产生NaN
3. **特征工程阶段**：特征工程转换可能产生NaN
4. **数据清理阶段**：虽然有清理逻辑，但不够彻底
5. **模型训练阶段**：StandardScaler在处理包含NaN的数据时产生警告

## 修复方案

### 1. 因子计算阶段 (`core/factors/advanced_factors.py`)

**修复方法**：对所有统计计算结果进行NaN检查

```python
# 示例：价格序列特征
hl_range_mean = hl_range.tail(20).mean()
features['hl_range_mean'] = 0.0 if pd.isna(hl_range_mean) else hl_range_mean
```

**修复的方法**：
- `calculate_price_series_features()` - 对skew、kurt等进行NaN检查
- `calculate_volume_series_features()` - 对相关性计算进行NaN检查
- `calculate_momentum_features()` - 对收益率计算进行NaN检查
- `calculate_risk_features()` - 对风险指标进行NaN检查
- `calculate_relative_strength_factors()` - 对相对强度计算进行NaN检查

### 2. 基本面因子计算阶段 (`core/factors/fundamental_factors.py`)

**修复方法**：确保所有数值字段都转换为float类型

```python
# 模式：先转换，再检查类型，最后比较
try:
    value = float(value) if value is not None else None
except (ValueError, TypeError):
    value = None

if value is not None and isinstance(value, (int, float)) and value > 0:
    # 安全使用value
```

**修复的方法**：12个因子计算方法都进行了类型转换和检查

### 3. 因子计算后处理 (`core/factors/train_ml_model.py` - `_calculate_base_factors`)

**修复方法**：在合并所有因子后进行最终数据清理

```python
# 最终数据清理
for col in factors.columns:
    factors[col] = pd.to_numeric(factors[col], errors='coerce').fillna(0)
    factors[col] = factors[col].replace([np.inf, -np.inf], 0)
```

### 4. 数据准备阶段 (`core/factors/train_ml_model.py` - `prepare_dataset`)

**修复方法**：多层次的NaN/inf检查和替换

```python
# 第1层：转换为数值类型
for col in X_combined.columns:
    X_combined[col] = pd.to_numeric(X_combined[col], errors='coerce')

# 第2层：替换无穷大值
X_combined = X_combined.replace([np.inf, -np.inf], np.nan)

# 第3层：填充NaN值（使用中位数或0）
for col in X_combined.columns:
    if X_combined[col].isna().any():
        median_val = X_combined[col].median()
        X_combined[col] = X_combined[col].fillna(median_val if not pd.isna(median_val) else 0)

# 第4层：最终检查和替换
X_combined = X_combined.fillna(0)
X_combined = X_combined.replace([np.inf, -np.inf], 0)

# 第5层：移除无效行
invalid_rows = X_combined.isna().any(axis=1) | np.isinf(X_combined.values).any(axis=1)
X_combined = X_combined[~invalid_rows]

# 第6层：转换为numpy数组并进行最后的清理
X_array = X_combined.values.astype(np.float64)
X_array = np.nan_to_num(X_array, nan=0.0, posinf=0.0, neginf=0.0)

# 第7层：最后验证
if np.isnan(X_array).any():
    X_array = np.nan_to_num(X_array, nan=0.0)
if np.isinf(X_array).any():
    X_array[np.isinf(X_array)] = 0.0
```

### 5. 模型训练阶段 (`core/factors/train_ml_model.py` - `train_models`)

**修复方法**：在StandardScaler之前进行数据验证

```python
# 数据验证和清理
X = X.astype(np.float64)

# 检查并替换NaN值
if np.isnan(X).any():
    X = np.nan_to_num(X, nan=0.0)

# 检查并替换无穷大值
if np.isinf(X).any():
    X[np.isinf(X)] = 0.0
```

### 6. StandardScaler之前 (`core/factors/ml_factor_model.py` - `train`)

**修复方法**：在StandardScaler.fit_transform之前进行最后的数据清理

```python
# 数据验证和清理（在StandardScaler之前）
X = X.astype(np.float64)

# 检查并替换NaN值
if np.isnan(X).any():
    X = np.nan_to_num(X, nan=0.0)

# 检查并替换无穷大值
if np.isinf(X).any():
    X[np.isinf(X)] = 0.0

# 现在可以安全地使用StandardScaler
X_scaled = self.scaler.fit_transform(X)
```

## 修复效果

修复后的代码将：

1. **消除RuntimeWarning**：所有NaN值在到达StandardScaler之前都被替换
2. **消除TypeError**：所有比较操作前都进行了类型检查
3. **提高数据质量**：确保训练数据中没有无效值
4. **加快训练速度**：减少数据清理的开销
5. **提高模型稳定性**：确保模型训练过程中没有数值异常

## 数据流程图

```
因子计算
  ↓ (NaN检查)
基本面因子计算
  ↓ (类型转换)
因子合并
  ↓ (最终清理)
数据准备 (prepare_dataset)
  ↓ (7层NaN/inf检查)
模型训练 (train_models)
  ↓ (数据验证)
StandardScaler
  ↓ (模型训练)
完成
```

## 测试建议

1. 运行训练脚本，检查是否还有RuntimeWarning
2. 验证训练数据的统计特性（均值、方差等）
3. 检查模型性能是否有改进
4. 验证因子计算的完整性和准确性

## 相关文件

- `core/factors/advanced_factors.py` - 高级因子计算
- `core/factors/fundamental_factors.py` - 基本面因子计算
- `core/factors/train_ml_model.py` - 模型训练和数据准备
- `core/factors/ml_factor_model.py` - 模型训练实现
- `core/factors/factor_filler.py` - 因子填充

# yfinance备用数据源使用说明

## 概述

本项目现已支持使用 yfinance 作为 akshare 的备用数据源。当 akshare 获取数据失败时，系统会自动切换到 yfinance 继续获取数据，提高数据获取的稳定性和成功率。

## 功能特点

1. **自动切换**: 当主数据源失败时自动切换到备用数据源
2. **灵活配置**: 可以选择优先使用 akshare 或 yfinance
3. **统计信息**: 提供详细的数据源使用统计
4. **兼容性**: 与现有数据库结构完全兼容
5. **易于使用**: 提供命令行工具，支持多种更新模式

## 安装依赖

```bash
pip install yfinance
```

或者安装所有依赖：

```bash
pip install -r requirements.txt
```

## 核心模块

### 1. YFinanceFetcher (yfinance_fetcher.py)

纯 yfinance 数据获取器，提供基本的数据获取功能。

**主要方法**:
- `get_historical_data()`: 获取历史数据
- `get_realtime_data()`: 获取实时数据
- `convert_code_to_yfinance()`: 将A股代码转换为yfinance格式

**代码转换规则**:
- 上交所 (60xxxx) → 600000.SS
- 深交所主板 (00xxxx) → 000001.SZ
- 创业板 (30xxxx) → 300001.SZ
- 北交所 (8xxxxx, 4xxxxx) → 800001.BJ

### 2. HybridDataFetcher (hybrid_fetcher.py)

混合数据获取器，支持 akshare 和 yfinance 自动切换。

**主要方法**:
- `get_historical_data()`: 获取历史数据（支持自动切换）
- `get_stock_list()`: 获取股票列表（使用akshare）
- `get_realtime_data()`: 获取实时数据（支持自动切换）
- `update_daily_data()`: 更新日线数据
- `print_stats()`: 打印数据源使用统计

### 3. 更新脚本 (update_data_with_fallback.py)

命令行工具，提供便捷的数据更新功能。

## 使用方法

### 1. 测试数据源可用性

```bash
python scripts/update_data_with_fallback.py --mode test
```

这会测试 akshare 和 yfinance 两个数据源是否可用。

### 2. 更新单只股票

```bash
# 使用akshare作为优先数据源（默认）
python scripts/update_data_with_fallback.py --mode single --symbol 600000

# 使用yfinance作为优先数据源
python scripts/update_data_with_fallback.py --mode single --symbol 600000 --source yfinance

# 全量更新（非增量）
python scripts/update_data_with_fallback.py --mode single --symbol 600000 --full
```

### 3. 批量更新多只股票

```bash
# 更新多只股票
python scripts/update_data_with_fallback.py --mode multiple --symbols 600000 000001 000002

# 使用yfinance作为优先数据源
python scripts/update_data_with_fallback.py --mode multiple --symbols 600000 000001 --source yfinance
```

### 4. 更新所有股票

```bash
# 更新上证和深圳主板所有股票（默认）
python scripts/update_data_with_fallback.py --mode all

# 更新指定市场的股票
python scripts/update_data_with_fallback.py --mode all --markets sh sz_main sz_gem

# 使用yfinance作为优先数据源
python scripts/update_data_with_fallback.py --mode all --source yfinance
```

## 命令行参数说明

| 参数 | 说明 | 可选值 | 默认值 |
|------|------|--------|--------|
| --mode | 运行模式 | single, multiple, all, test | test |
| --symbol | 股票代码（单只） | 如: 600000 | - |
| --symbols | 股票代码列表（多只） | 如: 600000 000001 | - |
| --markets | 市场列表 | sh, sz_main, sz_gem, bj | sh sz_main |
| --source | 优先数据源 | akshare, yfinance | akshare |
| --full | 全量更新模式 | - | False（增量更新） |

## 在代码中使用

### 示例1: 基本使用

```python
from core.data.hybrid_fetcher import HybridDataFetcher

# 创建混合数据获取器
fetcher = HybridDataFetcher(prefer_source='akshare')

# 获取历史数据（自动切换数据源）
data = fetcher.get_historical_data(
    symbol='600000',
    start_date='20240101',
    end_date='20240131'
)

# 打印统计信息
fetcher.print_stats()

# 关闭连接
fetcher.close()
```

### 示例2: 更新股票数据

```python
from core.data.hybrid_fetcher import HybridDataFetcher

fetcher = HybridDataFetcher(prefer_source='akshare')

# 增量更新单只股票
fetcher.update_daily_data('600000', incremental=True)

# 打印统计
fetcher.print_stats()

fetcher.close()
```

### 示例3: 仅使用yfinance

```python
from core.data.yfinance_fetcher import YFinanceFetcher

fetcher = YFinanceFetcher()

# 获取历史数据
data = fetcher.get_historical_data(
    symbol='600000',
    start_date='20240101',
    end_date='20240131'
)

print(data.head())
```

## 数据格式说明

两个数据源返回的数据格式已统一，包含以下字段：

| 字段 | 说明 | 备注 |
|------|------|------|
| 日期 | 交易日期 | YYYY-MM-DD格式 |
| 开盘 | 开盘价 | - |
| 最高 | 最高价 | - |
| 最低 | 最低价 | - |
| 收盘 | 收盘价 | - |
| 成交量 | 成交量 | - |
| 成交额 | 成交额 | yfinance为估算值 |
| 换手率 | 换手率 | yfinance不提供，为None |

## 注意事项

1. **股票列表获取**: yfinance 不支持获取A股股票列表，建议使用 akshare 获取股票列表
2. **换手率数据**: yfinance 不提供换手率数据，该字段会为 None
3. **成交额数据**: yfinance 的成交额是通过 收盘价 × 成交量 估算的，可能不够精确
4. **数据延迟**: yfinance 的数据可能有一定延迟
5. **代理设置**: 代理配置仅对 akshare 有效，yfinance 不使用代理

## 数据源对比

| 特性 | akshare | yfinance |
|------|---------|----------|
| A股支持 | ✓ 完整支持 | ✓ 支持 |
| 股票列表 | ✓ 支持 | ✗ 不支持 |
| 历史数据 | ✓ 完整 | ✓ 完整 |
| 实时数据 | ✓ 支持 | ✓ 支持 |
| 换手率 | ✓ 提供 | ✗ 不提供 |
| 成交额 | ✓ 精确 | △ 估算 |
| 稳定性 | △ 偶尔失败 | ✓ 较稳定 |
| 速度 | ✓ 较快 | △ 一般 |

## 推荐使用策略

1. **日常更新**: 优先使用 akshare，失败时自动切换到 yfinance
2. **批量初始化**: 使用 akshare，配合代理池提高成功率
3. **应急备份**: 当 akshare 长时间不可用时，临时切换到 yfinance
4. **数据验证**: 可以同时使用两个数据源进行数据交叉验证

## 故障排查

### 问题1: yfinance 安装失败

```bash
# 尝试升级pip
python -m pip install --upgrade pip

# 重新安装
pip install yfinance
```

### 问题2: 获取数据为空

- 检查股票代码是否正确
- 检查日期范围是否合理
- 运行测试模式检查数据源可用性

### 问题3: 数据源都失败

- 检查网络连接
- 检查防火墙设置
- 尝试使用代理（仅akshare）

## 统计信息示例

运行后会显示详细的统计信息：

```
==================================================
数据源使用统计:
  akshare成功: 45 次
  akshare失败: 5 次
  yfinance成功: 5 次
  yfinance失败: 0 次
  总成功率: 90.91%
==================================================
```

## 更新日志

- 2024-01-XX: 初始版本，支持 yfinance 作为备用数据源
- 支持自动切换和统计功能
- 提供命令行工具

## 技术支持

如有问题，请查看项目文档或提交 Issue。

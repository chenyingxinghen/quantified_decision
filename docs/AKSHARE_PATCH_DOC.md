# AkShare 请求补丁说明

## 概述

本项目实现了对 AkShare 库的请求方法进行补丁（patch），以在所有 HTTP 请求头中自动添加 `nid` 和 `create_time` 参数。

## 实现原理

1. **补丁机制**：通过替换 `akshare.request` 模块中的 `make_request_with_retry_json` 和 `make_request_with_retry_text` 函数
2. **参数注入**：在每次 HTTP 请求前，自动向请求头中添加以下参数：
   - `nid`: 32位随机字符串，用于标识请求
   - `create_time`: 当前时间戳（毫秒格式）

## 文件说明

- `akshare_patch.py`: 补丁实现模块
- 修改后的 `data_fetcher.py`: 自动应用补丁
- 修改后的 `t.py`: 自动应用补丁

## 使用方法

补丁会在导入以下模块时自动应用：
- `data_fetcher.py`
- `t.py`

无需额外操作，所有通过 AkShare 发起的请求都会自动包含所需的请求头参数。

## 技术细节

- `nid` 生成：使用 `random.choices()` 从数字和小写字母中随机选取32个字符
- `create_time` 格式：Unix 时间戳（毫秒）
- 原有功能完整性：补丁保留了原函数的所有原有功能，仅添加了请求头参数

## 测试

项目包含以下测试文件：
- `test_akshare_patch.py`: 测试补丁功能
- `test_integration.py`: 测试与现有模块的集成

## 注意事项

1. 补丁仅在导入时应用，重启Python进程后需重新应用
2. 如需取消补丁，可以调用 `akshare_patch.unpatch_akshare_requests()`
3. 补丁不影响原有的代理设置、重试机制等功能
"""
配置文件适配层（保持向后兼容）

本文件已废弃，所有配置已迁移到 baostock_config.py
为保持向后兼容性，本文件重新导出 baostock_config 中的配置项
"""

# 导入所有配置项
from config.baostock_config import *

# 向后兼容性警告
import warnings
warnings.warn(
    "config.config 已废弃，请直接使用 config.baostock_config",
    DeprecationWarning,
    stacklevel=2
)

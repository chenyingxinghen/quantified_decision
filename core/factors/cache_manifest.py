"""因子缓存版本契约：防止模型与同名但不同公式的缓存静默错配。"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Sequence

from config.factor_config import TrainingConfig


def manifest_path(directory: str) -> str:
    return os.path.join(os.path.abspath(directory), TrainingConfig.CACHE_MANIFEST_NAME)


def load_manifest(directory: str) -> Optional[Dict[str, Any]]:
    path = manifest_path(directory)
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f'因子缓存清单格式错误: {path}')
    return payload


def write_cache_manifest(cache_dir: str) -> Dict[str, Any]:
    """[2026-08-22] manifest 机制已移除：不再写缓存版本清单，直接返回空载体。

    版本演进改为靠重命名/删除缓存文件夹完成；所有模型默认指向 config 里的
    factors_cache，无需逐缓存写版本清单。
    """
    return {}


def validate_cache_manifest(cache_dir: str, expected_version: str = None) -> Dict[str, Any]:
    """[2026-08-22] manifest 机制已移除：不再做版本校验，直接返回空载体。"""
    return {}


def bind_model_to_cache(model_dir: str, cache_dir: str) -> Dict[str, Any]:
    """[2026-08-22] manifest 机制已移除：不再写模型→缓存绑定清单，直接返回空载体。

    版本演进改为靠重命名/删除缓存文件夹完成；所有模型默认指向 config 里的
    factors_cache，无需逐模型绑定。
    """
    return {}


def resolve_model_cache(model_dir: str, legacy_cache_dir: str) -> str:
    """返回单一通用因子缓存 factors_cache。

    2026-08-22 起移除 per-model manifest 版本契约：所有模型（NAM / 树 / mark / latest）
    共用同一个全集缓存，版本演进靠重命名/删除缓存文件夹完成。不再读取或校验
    模型目录里的绑定清单，默认即指向 config 里的 factors_cache。
    """
    return os.path.abspath(TrainingConfig.CACHE_DIR)


def resolve_ensemble_cache(model_dirs: Sequence[str], legacy_cache_dir: str) -> str:
    """集成模型缓存解析：单一通用缓存下所有子模型天然同缓存，直接返回 factors_cache。"""
    return os.path.abspath(TrainingConfig.CACHE_DIR)

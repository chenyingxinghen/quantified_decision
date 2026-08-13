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
    """在独立缓存目录写当前公式版本清单；不负责迁移或删除旧缓存。"""
    cache_dir = os.path.abspath(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    payload = {
        'manifest_type': 'factor_cache',
        'factor_definition_version': TrainingConfig.FACTOR_DEFINITION_VERSION,
        'cache_dir': cache_dir,
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
    }
    path = manifest_path(cache_dir)
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(tmp, path)
    return payload


def validate_cache_manifest(cache_dir: str, expected_version: str = None) -> Dict[str, Any]:
    payload = load_manifest(cache_dir)
    if payload is None:
        raise FileNotFoundError(f'独立因子缓存缺少版本清单: {manifest_path(cache_dir)}')
    actual = payload.get('factor_definition_version')
    expected = expected_version or TrainingConfig.FACTOR_DEFINITION_VERSION
    if actual != expected:
        raise RuntimeError(
            f'因子缓存版本错配: expected={expected}, actual={actual}, cache={cache_dir}'
        )
    return payload


def bind_model_to_cache(model_dir: str, cache_dir: str) -> Dict[str, Any]:
    """把模型绑定到已验证的独立缓存目录。"""
    cache_manifest = validate_cache_manifest(cache_dir)
    payload = {
        'manifest_type': 'model_factor_cache_binding',
        'factor_definition_version': cache_manifest['factor_definition_version'],
        'cache_dir': os.path.abspath(cache_dir),
    }
    os.makedirs(model_dir, exist_ok=True)
    path = manifest_path(model_dir)
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(tmp, path)
    return payload


def resolve_model_cache(model_dir: str, legacy_cache_dir: str) -> str:
    """新模型按绑定清单取缓存；旧模型无清单时兼容历史默认缓存。"""
    binding = load_manifest(model_dir)
    if binding is None:
        return os.path.abspath(legacy_cache_dir)
    if binding.get('manifest_type') != 'model_factor_cache_binding':
        raise ValueError(f'模型目录中的因子清单类型错误: {manifest_path(model_dir)}')
    cache_dir = binding.get('cache_dir')
    if not cache_dir:
        raise ValueError(f'模型因子清单缺少 cache_dir: {manifest_path(model_dir)}')
    validate_cache_manifest(cache_dir, binding.get('factor_definition_version'))
    return os.path.abspath(cache_dir)


def resolve_ensemble_cache(model_dirs: Sequence[str], legacy_cache_dir: str) -> str:
    """解析集成模型缓存，并拒绝不同公式版本/目录的模型混用。"""
    resolved = [resolve_model_cache(path, legacy_cache_dir) for path in model_dirs]
    normalized = {os.path.normcase(os.path.abspath(path)) for path in resolved}
    if len(normalized) != 1:
        details = ', '.join(f'{model} -> {cache}' for model, cache in zip(model_dirs, resolved))
        raise RuntimeError(f'集成模型绑定了不同因子缓存，拒绝混用: {details}')
    return resolved[0]

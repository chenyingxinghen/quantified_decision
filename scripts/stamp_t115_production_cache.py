"""把 T115 缓存与存档的公式版本重盖到当前 FACTOR_DEFINITION_VERSION（幂等）。

为什么要重盖而不是重建缓存
--------------------------
``build_idxrel_cache.py`` 的清单 note 里写着「通过判定后应把公式集成进
factor_calculator 并 bump FACTOR_DEFINITION_VERSION 全量重建」。公式已经集成
（``core/factors/index_relative_factors.py``），版本也 bump 了，但**全量重建
没有必要**：生产计算器的输出与缓存里那 5 列已验证**逐位一致**
（``tests/test_business_logic.py::IndexRelativeFactorTests``：4 个板块前缀抽样，
5 列最大绝对差 0.000e+00）。花几小时重算出一模一样的数字是浪费。

重盖的对象只有两类小文件：
  1. 缓存目录的 factor_cache_manifest.json（type=factor_cache）
  2. T115 四个存档目录的 factor_cache_manifest.json（type=model_factor_cache_binding）
两者的 factor_definition_version 必须相等，否则 resolve_model_cache 会拒绝。

其余缓存（factors_cache_2026-08-14-fwdadjust 等）与 T098/T113/T114 存档**不动**：
它们确实对应旧公式集，各自版本自洽，用旧模型跑回测仍然正常。

用法:
  python scripts/stamp_t115_production_cache.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config.factor_config import TrainingConfig
from core.factors.cache_manifest import manifest_path

CACHE_DIR = os.path.join(ROOT, 'database', 'system_data',
                         'factors_cache_2026-08-18-idxrel')
MODEL_DIRS = [os.path.join(ROOT, 'models', 'nam_gate', f'T115_idxrel_s{s}')
              for s in (42, 11, 23, 37)]

# 清单里记录的必须是**真实**路径：开发时 database/ models/ 常以 junction/软链
# 挂进 git worktree，abspath 会把临时 worktree 路径写死进绑定清单，等 worktree
# 被删就变成一条指向虚空的生产配置。realpath 穿透链接，拿到主仓的真路径。
CANONICAL_CACHE_DIR = os.path.realpath(CACHE_DIR)

STAMP_NOTE = (
    'T115 晋级后把 idx_* 公式集成进 core/factors/index_relative_factors.py，'
    '生产路径与本缓存里的 5 列已验证逐位一致（最大绝对差 0），'
    '故只重盖版本号，未全量重建基础 242 列。'
)


def _rewrite(path: str, updates: dict, dry_run: bool) -> bool:
    if not os.path.exists(path):
        print(f'  [跳过] 不存在: {path}')
        return False
    with open(path, 'r', encoding='utf-8') as f:
        payload = json.load(f)
    before = payload.get('factor_definition_version')
    before_dir = payload.get('cache_dir')
    want_dir = updates.get('cache_dir')
    if (before == TrainingConfig.FACTOR_DEFINITION_VERSION
            and (want_dir is None or before_dir == want_dir)):
        print(f'  [已是最新] {os.path.relpath(path, ROOT)}')
        return False
    payload.update(updates)
    print(f'  [重盖] {os.path.relpath(path, ROOT)}\n'
          f'         {before} -> {TrainingConfig.FACTOR_DEFINITION_VERSION}')
    if want_dir is not None and before_dir != want_dir:
        print(f'         cache_dir: {before_dir} -> {want_dir}')
    if dry_run:
        return True
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    version = TrainingConfig.FACTOR_DEFINITION_VERSION
    print(f'目标版本: {version}\n')

    print('缓存清单:')
    changed = _rewrite(manifest_path(CACHE_DIR), {
        'factor_definition_version': version,
        'cache_dir': CANONICAL_CACHE_DIR,
        'stamped_by': 'scripts/stamp_t115_production_cache.py',
        'stamp_note': STAMP_NOTE,
    }, a.dry_run)

    print('\n模型绑定清单:')
    for d in MODEL_DIRS:
        changed |= _rewrite(manifest_path(d), {
            'factor_definition_version': version,
            'cache_dir': CANONICAL_CACHE_DIR,
        }, a.dry_run)

    if not a.dry_run and changed:
        # 立刻自检：解析生产模型 -> 缓存，走的就是实盘那条路径
        from core.factors.cache_manifest import resolve_model_cache
        resolved = resolve_model_cache(MODEL_DIRS[0], TrainingConfig.CACHE_DIR)
        assert os.path.normcase(resolved) == os.path.normcase(CANONICAL_CACHE_DIR), resolved
        print(f'\n自检通过: {os.path.basename(MODEL_DIRS[0])} -> {resolved}')
    print('\n完成' + ('（dry-run，未落盘）' if a.dry_run else ''))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

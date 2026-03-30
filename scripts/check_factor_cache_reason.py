"""
检查因子缓存缺列原因。

用途：
1. 加载指定模型的 feature_names
2. 检查指定股票缓存缺失了哪些特征
3. 输出缺失特征的模式分布，辅助判断是否为特征工程版本升级导致
4. 给出与训练脚本一致的“是否应触发全量重算”判断

示例：
    python scripts/check_factor_cache_reason.py --codes 001356 001391
    python scripts/check_factor_cache_reason.py --model models/latest/lightgbm_factor_model.pkl --show-missing 30
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.factor_config import TrainingConfig
from core.factors.ml_factor_model import MLFactorModel


DEFAULT_CODES = ["001356", "001391"]
DEFAULT_MODEL = PROJECT_ROOT / "models" / "latest" / "lightgbm_factor_model.pkl"
DEFAULT_CACHE_DIR = PROJECT_ROOT / TrainingConfig.CACHE_DIR

ENGINEERED_PATTERNS = ("_div_", "_mul_", "_sub_", "_x_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查因子缓存为何缺少模型特征")
    parser.add_argument(
        "--codes",
        nargs="+",
        default=DEFAULT_CODES,
        help="要检查的股票代码，默认检查 001356 和 001391",
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL),
        help="模型路径，默认 models/latest/lightgbm_factor_model.pkl",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(DEFAULT_CACHE_DIR),
        help="因子缓存目录，默认使用 TrainingConfig.CACHE_DIR",
    )
    parser.add_argument(
        "--show-missing",
        type=int,
        default=20,
        help="每只股票最多展示多少个缺失特征名",
    )
    return parser.parse_args()


def load_model(model_path: Path) -> MLFactorModel:
    model_type = "lightgbm" if "lightgbm" in model_path.name.lower() else "xgboost"
    model = MLFactorModel(model_type=model_type)
    model.load_model(str(model_path))
    return model


def find_factor_summary(model_path: Path) -> Optional[Path]:
    candidate = model_path.parent / "factor_summary.json"
    if candidate.exists():
        return candidate
    return None


def load_factor_summary(summary_path: Optional[Path]) -> Optional[dict]:
    if not summary_path or not summary_path.exists():
        return None
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def categorize_feature(name: str) -> str:
    for pattern in ENGINEERED_PATTERNS:
        if pattern in name:
            return pattern
    return "base_or_other"


def format_time(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def summarize_categories(features: Iterable[str]) -> Counter:
    counter = Counter()
    for feature in features:
        counter[categorize_feature(feature)] += 1
    return counter


def print_header(title: str) -> None:
    print(f"\n{'=' * 80}")
    print(title)
    print(f"{'=' * 80}")


def main() -> int:
    args = parse_args()
    model_path = Path(args.model).resolve()
    cache_dir = Path(args.cache_dir).resolve()

    if not model_path.exists():
        print(f"错误: 模型不存在: {model_path}")
        return 1
    if not cache_dir.exists():
        print(f"错误: 缓存目录不存在: {cache_dir}")
        return 1

    model = load_model(model_path)
    model_features: List[str] = list(getattr(model, "feature_names", []) or [])
    summary_path = find_factor_summary(model_path)
    factor_summary = load_factor_summary(summary_path)

    print_header("模型信息")
    print(f"模型路径: {model_path}")
    print(f"模型类型: {getattr(model, 'model_type', 'unknown')}")
    print(f"模型特征数: {len(model_features)}")
    print(f"模型修改时间: {format_time(model_path.stat().st_mtime)}")
    if summary_path:
        print(f"factor_summary: {summary_path}")
    else:
        print("factor_summary: 未找到（不影响诊断）")
    if factor_summary:
        print(f"factor_summary.total_factors: {factor_summary.get('total_factors')}")
        print(f"factor_summary.engineered_factors: {factor_summary.get('engineered_factors')}")

    all_missing_sets = []

    for code in args.codes:
        cache_file = cache_dir / f"{code}_factors.parquet"
        print_header(f"股票 {code}")
        print(f"缓存文件: {cache_file}")

        if not cache_file.exists():
            print("状态: 缓存文件不存在")
            continue

        df = pd.read_parquet(cache_file)
        cache_columns = list(df.columns)
        missing = [f for f in model_features if f not in cache_columns]
        extra = [f for f in cache_columns if f not in model_features and f != "date"]

        all_missing_sets.append(set(missing))

        print(f"缓存行数: {len(df)}")
        print(f"缓存列数(含date): {len(cache_columns)}")
        print(f"缓存列数(不含date): {len([c for c in cache_columns if c != 'date'])}")
        print(f"缓存修改时间: {format_time(cache_file.stat().st_mtime)}")
        print(f"是否晚于模型: {'是' if cache_file.stat().st_mtime >= model_path.stat().st_mtime else '否'}")
        print(f"缺失特征数: {len(missing)}")
        print(f"额外特征数: {len(extra)}")

        missing_counter = summarize_categories(missing)
        present_engineered = sum(1 for c in cache_columns if any(p in c for p in ENGINEERED_PATTERNS))
        print(
            "缺失模式分布: "
            f"_div_={missing_counter['_div_']}, "
            f"_mul_={missing_counter['_mul_']}, "
            f"_sub_={missing_counter['_sub_']}, "
            f"_x_={missing_counter['_x_']}, "
            f"其它={missing_counter['base_or_other']}"
        )
        print(f"缓存中已存在的工程特征数: {present_engineered}")

        if missing:
            print(f"训练侧判断: 缓存缺少 {len(missing)} 个特征，应该触发全量重算")
            print(f"前 {min(args.show_missing, len(missing))} 个缺失特征:")
            for name in missing[: args.show_missing]:
                print(f"  - {name}")
        else:
            print("训练侧判断: 特征完整，不需要因缺列而重算")

        if extra:
            print(f"前 {min(10, len(extra))} 个缓存额外列:")
            for name in extra[:10]:
                print(f"  - {name}")

        hints: List[str] = []
        if missing:
            if cache_file.stat().st_mtime < model_path.stat().st_mtime:
                hints.append("缓存文件早于当前模型，疑似模型升级后缓存未重建")
            if missing_counter["_div_"] + missing_counter["_mul_"] + missing_counter["_sub_"] + missing_counter["_x_"] >= len(missing) * 0.6:
                hints.append("缺失项以特征工程组合列为主，疑似缓存生成时使用了旧版特征工程清单")
            if factor_summary and factor_summary.get("total_factors") == len(model_features):
                hints.append("模型侧特征清单完整，问题更像是缓存落盘版本偏旧，而不是模型元数据损坏")

        if hints:
            print("原因判断:")
            for hint in hints:
                print(f"  - {hint}")

    if len(all_missing_sets) >= 2:
        same = all_missing_sets[0]
        identical = all(m == same for m in all_missing_sets[1:])
        print_header("跨股票结论")
        print(f"各股票缺失集合是否完全一致: {'是' if identical else '否'}")
        if identical and same:
            print("结论: 这更像是缓存版本问题，而不是个股数据偶发缺失。")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

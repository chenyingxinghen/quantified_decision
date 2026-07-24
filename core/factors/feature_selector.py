"""面向横截面模型的可复现特征筛选。"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .feature_categories import classify_features


@dataclass
class FeatureSelectionReport:
    selected_features: List[str]
    dropped_low_coverage: List[str]
    dropped_constant: List[str]
    dropped_redundant: List[str]
    scores: Dict[str, float]
    category_min: Dict[str, int] = None  # 实际生效的类别下限（用于缓存键/审计）


class CrossSectionalFeatureSelector:
    """按目标相关性排序，再剔除高度冗余特征。

    该实现只在训练集上拟合。对单目标传入一维 y；多目标可传入二维
    ``targets``，最终得分为各目标绝对 Spearman 相关性的加权平均。

    category_min: 可选 ``{类别: 最少保留数}``。在相关性贪心剪枝之后，对未达下限的
    类别用其得分最高的候选补齐（即便与已选特征有一定冗余也保留，仅跳过近乎完全
    重复的 corr>0.99）。用于保证基本面/估值等弱信号类别不被纯相关性排名完全挤出
    （A 股短周期下财务类因子与未来收益相关性天然偏弱，但经济学上需要保留）。
    """

    def __init__(
        self,
        max_features: int = 120,
        min_coverage: float = 0.20,
        corr_threshold: float = 0.95,
        sample_size: int = 200_000,
        category_min: Optional[Dict[str, int]] = None,
    ):
        self.max_features = max_features
        self.min_coverage = min_coverage
        self.corr_threshold = corr_threshold
        self.sample_size = sample_size
        self.category_min = dict(category_min) if category_min else {}
        self.report_: Optional[FeatureSelectionReport] = None

    def fit(
        self,
        X: np.ndarray,
        feature_names: Sequence[str],
        targets: np.ndarray,
        target_weights: Optional[Sequence[float]] = None,
        feature_coverage: Optional[Sequence[float]] = None,
    ) -> "CrossSectionalFeatureSelector":
        X = np.asarray(X)
        y = np.asarray(targets)
        if y.ndim == 1:
            y = y[:, None]
        if X.shape[0] != y.shape[0] or X.shape[1] != len(feature_names):
            raise ValueError("X、targets 与 feature_names 维度不一致")
        if feature_coverage is not None and len(feature_coverage) != len(feature_names):
            raise ValueError("feature_coverage 必须与 feature_names 等长")
        weights = np.ones(y.shape[1], dtype=float) if target_weights is None else np.asarray(target_weights, dtype=float)
        if len(weights) != y.shape[1] or np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError("target_weights 必须与目标数一致且至少一个为正")
        weights = weights / weights.sum()

        if len(X) > self.sample_size:
            # 等距抽样保持时间覆盖，且结果完全可复现。
            idx = np.linspace(0, len(X) - 1, self.sample_size, dtype=int)
            X_fit, y_fit = X[idx], y[idx]
        else:
            X_fit, y_fit = X, y
        frame = pd.DataFrame(X_fit, columns=list(feature_names)).replace([np.inf, -np.inf], np.nan)

        # 覆盖率必须来自填充/归一化之前的原始训练矩阵。X 通常已经被处理成
        # 模型真正消费的横截面排名值，此时再用 notna() 会把中性填充值误判为
        # 有效观测，导致稀疏事件特征绕过覆盖率门槛。
        if feature_coverage is None:
            coverage = frame.notna().mean()
        else:
            coverage = pd.Series(
                np.asarray(feature_coverage, dtype=float), index=list(feature_names)
            )
        coverage = coverage.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        dropped_low_coverage = coverage[coverage < self.min_coverage].index.tolist()
        candidates = [name for name in feature_names if name not in dropped_low_coverage]
        nunique = frame[candidates].nunique(dropna=True)
        dropped_constant = nunique[nunique <= 1].index.tolist()
        candidates = [name for name in candidates if name not in dropped_constant]

        scores: Dict[str, float] = {}
        for name in candidates:
            x_rank = frame[name].rank(pct=True).to_numpy(dtype=float)
            per_target = []
            for target_idx in range(y_fit.shape[1]):
                target = y_fit[:, target_idx]
                mask = np.isfinite(x_rank) & np.isfinite(target)
                if mask.sum() < 30:
                    per_target.append(0.0)
                    continue
                target_rank = pd.Series(target[mask]).rank(pct=True).to_numpy(dtype=float)
                corr = np.corrcoef(x_rank[mask], target_rank)[0, 1]
                per_target.append(abs(float(corr)) if np.isfinite(corr) else 0.0)
            scores[name] = float(np.dot(per_target, weights))

        original_position = {name: idx for idx, name in enumerate(feature_names)}
        ordered = sorted(candidates, key=lambda name: (-scores[name], original_position[name]))
        # 先取较宽的候选集合再做相关性剪枝，避免对全部上千列构造相关矩阵。
        preselected = ordered[: max(self.max_features * 4, self.max_features)]
        corr = frame[preselected].corr(method="spearman").abs()
        selected: List[str] = []
        dropped_redundant: List[str] = []
        for name in preselected:
            if any(corr.loc[name, kept] > self.corr_threshold for kept in selected):
                dropped_redundant.append(name)
                continue
            selected.append(name)
            if len(selected) >= self.max_features:
                break

        # ── 类别下限保证（category_min）──
        # 在相关性贪心剪枝之后，对未达下限的类别用其得分最高的候选补齐。
        # 即便与已选特征有一定冗余也保留（仅跳过 corr>0.99 的近完全重复），
        # 以保证该经济学类别在模型中至少有一席之地。配额特征可能使最终数量
        # 略超 max_features，这是设计预期（用户的明确诉求优先于硬性数量上限）。
        if self.category_min:
            cat_of = classify_features(candidates)
            counts = Counter(cat_of[n] for n in selected)
            for cat_name, min_n in self.category_min.items():
                need = min_n - counts.get(cat_name, 0)
                if need <= 0:
                    continue
                # 该类别尚未入选的候选，按得分降序
                pool = [n for n in candidates if cat_of.get(n) == cat_name
                        and n not in selected]
                pool.sort(key=lambda n: -float(scores.get(n, 0.0)))
                added = 0
                for n in pool:
                    # 仅排除近完全重复，保留有信息量的弱信号特征
                    if any(corr.loc[n, kept] > 0.99 for kept in selected):
                        continue
                    selected.append(n)
                    added += 1
                    if added >= need:
                        break
                counts[cat_name] = counts.get(cat_name, 0) + added

        self.report_ = FeatureSelectionReport(
            selected_features=selected,
            dropped_low_coverage=dropped_low_coverage,
            dropped_constant=dropped_constant,
            dropped_redundant=dropped_redundant,
            scores=scores,
            category_min=self.category_min,
        )
        return self

    def transform(self, X: np.ndarray, feature_names: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
        if self.report_ is None:
            raise RuntimeError("请先调用 fit")
        position = {name: idx for idx, name in enumerate(feature_names)}
        missing = [name for name in self.report_.selected_features if name not in position]
        if missing:
            raise ValueError(f"输入缺少已选择特征: {missing[:10]}")
        indices = [position[name] for name in self.report_.selected_features]
        return np.asarray(X)[:, indices], list(self.report_.selected_features)

    def fit_transform(
        self, X, feature_names, targets, target_weights=None,
        feature_coverage=None,
    ):
        return self.fit(
            X, feature_names, targets, target_weights,
            feature_coverage=feature_coverage,
        ).transform(X, feature_names)

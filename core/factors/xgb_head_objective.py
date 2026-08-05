"""
头部加权 pairwise 目标函数（XGBoost 自定义 objective 工厂）。

背景：
    T027–T030 已证明当前瓶颈是「头部可预测性不足」而非标签错位——模型倾向于
    挑选「中上段」而非真实头部（预测头部对应的真实标签分位仅约 0.559）。
    本模块提供一个在 LambdaRank 之外的、显式聚焦真实头部的梯度约束：
    对每个 query（交易日），把「真实头部样本胜出」的 pair 赋予额外梯度权重，
    迫使模型更强烈地学习把头部样本排在前面。

设计：
    - 头部定义：每个 query 内按连续标签降序取前 head_frac（默认 10%）作为头部样本。
    - 头部 pair：每个头部样本对若干随机非头部样本形成 (i, j) pair，i 为头部胜者，
      gradient 权重 = 1 + alpha（alpha 即头部聚焦强度，单变量）。
    - 基础 pair：每个 query 额外采样少量随机 pair（权重 1.0），保留中段排序信号，
      避免模型只学头部而破坏整体 Rank IC。
    - 损失形状：加权 pairwise logistic；grad/hess 有界且 hess>0，训练稳定。

注意：
    - 工厂在 train_models 内、使用「训练期实际标签与 group」构造，保证与 DMatrix 行序一致。
    - 该模块不修改任何默认训练路径；仅当 TrainingConfig.XGBOOST_CUSTOM_OBJECTIVE
      被设置为本工厂返回的可调用对象时才生效。
"""

from typing import Callable, Optional, Tuple

import numpy as np


def make_head_weighted_objective(
    alpha: float = 2.0,
    head_frac: float = 0.10,
    head_losers_per_sample: int = 20,
    base_pairs_per_query: int = 20,
    seed: int = 42,
) -> Callable[[np.ndarray, np.ndarray], Callable]:
    """
    返回一个工厂：factory(y_train, group_train) -> obj(predt, dtrain)。

    y_train      : 训练期连续标签（长度 = 训练样本数），用于识别真实头部。
    group_train  : 每个 query 的样本数（per-query count）。
    obj          : 标准 XGBoost 自定义目标签名 (predt, dtrain) -> (grad, hess)。
    """

    def factory(y_train: np.ndarray, group_train: np.ndarray) -> Callable:
        y = np.asarray(y_train, dtype=np.float64)
        grp = np.asarray(group_train, dtype=np.int64)
        n = len(y)
        group_ptr = np.concatenate([[0], np.cumsum(grp)]).astype(np.int64)
        rng = np.random.default_rng(seed)

        i_list: list = []
        j_list: list = []
        w_list: list = []

        head_w = 1.0 + alpha
        G = len(grp)
        for g in range(G):
            s = int(group_ptr[g])
            e = int(group_ptr[g + 1])
            if e - s < 2:
                continue
            order = np.argsort(-y[s:e])  # 降序
            hsize = max(1, int((e - s) * head_frac))
            head = order[:hsize]
            losers = order[hsize:]
            if len(losers) == 0:
                continue
            # 头部 pair：每个头部样本挑选 head_losers_per_sample 个非头部对手
            for hi in head:
                sel = rng.choice(losers, size=min(head_losers_per_sample, len(losers)), replace=False)
                for lo in sel:
                    i_list.append(s + int(hi))
                    j_list.append(s + int(lo))
                    w_list.append(head_w)
            # 基础 pair：每个 query 随机采样 base_pairs_per_query 个 pair（i 标签 > j 标签）
            if base_pairs_per_query > 0 and len(losers) > 0:
                for _ in range(base_pairs_per_query):
                    a = rng.integers(s, e)
                    b = rng.integers(s, e)
                    if a == b:
                        continue
                    # 仅保留胜者标签更高的方向，避免与头部 pair 重复计数语义冲突
                    if y[a] >= y[b]:
                        i_list.append(a)
                        j_list.append(b)
                    else:
                        i_list.append(b)
                        j_list.append(a)
                    w_list.append(1.0)

        i_idx = np.asarray(i_list, dtype=np.int64)
        j_idx = np.asarray(j_list, dtype=np.int64)
        w_arr = np.asarray(w_list, dtype=np.float64)
        n_pairs = len(i_idx)
        if n_pairs == 0:
            # 退化为无操作目标（理论上不会发生）
            def _noop(predt, dtrain):
                grad = np.zeros(n, dtype=np.float64)
                hess = np.ones(n, dtype=np.float64) * 1e-3
                return grad, hess
            _noop.n_pairs = 0
            return _noop

        def obj(predt, dtrain):
            p = np.asarray(predt, dtype=np.float64).ravel()
            # 配对 logistic 梯度：sig = 1/(1+exp(-(p_i - p_j)))
            diff = p[i_idx] - p[j_idx]
            sig = 1.0 / (1.0 + np.exp(-diff))
            g_i = w_arr * sig
            g_j = -w_arr * sig
            h = w_arr * sig * (1.0 - sig)
            grad = (
                np.bincount(i_idx, weights=g_i, minlength=n)
                + np.bincount(j_idx, weights=g_j, minlength=n)
            )
            hess = (
                np.bincount(i_idx, weights=h, minlength=n)
                + np.bincount(j_idx, weights=h, minlength=n)
            )
            return grad, hess

        obj.n_pairs = n_pairs
        obj.head_w = head_w
        obj.head_frac = head_frac
        return obj

    factory.head_frac = head_frac
    factory.alpha = alpha
    return factory

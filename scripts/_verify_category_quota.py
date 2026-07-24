"""
验证类别下限（category_min）让 XGB 也纳入基本面因子。
用缓存数据集（mmap 避免 8.7GB 全量载入），对真实 829 特征池跑
CrossSectionalFeatureSelector，模拟 XGB(max_features=200) 与 LGB(max_features=400)，
对比「不加类别下限」与「加 category_min={'财务':3,'估值':5}」的选中类别分布。
"""
from __future__ import annotations
import sys, os, warnings, json
warnings.filterwarnings("ignore")
sys.path.insert(0, r"G:\quantified_decision")

import numpy as np
from collections import Counter
from analysis.common import classify_features
from core.factors.feature_selector import CrossSectionalFeatureSelector

BASE = r"G:\quantified_decision\analysis\full_dataset_exp3yr"
meta = __import__("pickle").load(open(os.path.join(BASE, "meta.pkl"), "rb"))
factor_names = list(meta["factor_names"])
print(f"全量候选池: {len(factor_names)} 特征, 类别 {dict(Counter(classify_features(factor_names).values()))}")

# mmap 载入 X（不占满内存）；arr_0 以 1D 落盘，需按 meta.shapes 还原 2D 再切片
# 仅取 4 万行做轻量验证（选择器内部会再等距抽样到 sample_size）
N = 40_000
x_shape = tuple(meta["shapes"][0])
X_flat = np.load(os.path.join(BASE, "arr_0.npy"), mmap_mode="r")
X = np.ascontiguousarray(X_flat.reshape(x_shape)[:N])
y_shape = tuple(meta["shapes"][1])
y = np.ascontiguousarray(np.load(os.path.join(BASE, "arr_1.npy"), mmap_mode="r").reshape(y_shape)[:N])
print(f"验证用 X={X.shape}, y={y.shape}", flush=True)

# 覆盖率：用训练集非空比例（这里用 full 的非空比例近似）
coverage = (~np.isnan(np.asarray(X))).mean(axis=0)


def run(max_features, category_min, label):
    sel = CrossSectionalFeatureSelector(
        max_features=max_features, min_coverage=0.20,
        corr_threshold=0.8, sample_size=200_000,
        category_min=category_min,
    )
    Xs, names = sel.fit_transform(np.asarray(X), factor_names, np.asarray(y),
                                  feature_coverage=coverage)
    dist = dict(Counter(classify_features(names).values()))
    fin = [f for f in names if classify_features([f])[f] == "财务"]
    print(f"\n[{label}] max_features={max_features} category_min={category_min}")
    print(f"  选中 {len(names)} 特征, 类别分布 {dist}")
    print(f"  选中的财务类特征: {fin}")
    return names


print("\n########## 旧逻辑（无类别下限）##########")
run(200, None, "XGB-旧")
run(400, None, "LGB-旧")

print("\n########## 新逻辑（类别下限 财务3/估值5）##########")
run(200, {"财务": 3, "估值": 5}, "XGB-新")
run(400, {"财务": 3, "估值": 5}, "LGB-新")

"""Synthetic integration check: NeuralTrainer reuses real data-infra functions
and produces a model loadable by the backtest loader path. No DB needed."""
import os, sys, tempfile, shutil
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config.jydb_config import DATABASE_PATH
from core.neural.trainer import NeuralTrainer
from core.neural.nn_models import MultiObjectiveNeuralModel

# --- synthetic dataset -------------------------------------------------------
rng = np.random.default_rng(7)
D = 120
N = 6000
# 构造若干带 skip/rank 前缀的列名，验证归一化路由
names = []
for i in range(D):
    if i < 10:
        names.append(f"is_pattern_{i}")          # 类 is_ 前缀 -> skip rank
    elif i < 20:
        names.append(f"rsi_{i}")                   # tech 前缀 -> rank
    elif i < 30:
        names.append(f"rank_feat_{i}")             # 已 rank
    else:
        names.append(f"feat_{i}")

dates = np.sort(rng.integers(0, 1000, size=N)).astype(str)
codes = rng.choice([f"60000{i}" for i in range(50)], size=N)
X = rng.standard_normal((N, D)).astype(np.float32)
# 注入一点横截面结构
X[:, 0] += np.arange(N) % 50 * 0.01

objs = ["rank_y_ret_5d", "rank_y_ret_20d", "rank_y_ret_60d",
        "rank_y_mdd_20d", "rank_y_downvol_20d", "rank_y_illiq_20d",
        "rank_y_tradable_20d"]
aligned = pd.DataFrame({"date": dates, "code": codes})
for o in objs:
    # 每个交易日内部做排序，制造可学习横截面信号
    s = pd.Series(X[:, 0] + rng.standard_normal(N) * 0.5)
    aligned[o] = s.groupby(dates).rank(pct=True).values.astype(np.float32)

dataset = {
    "X": X, "aligned": aligned, "factor_names": names,
    "dates": dates.astype(str), "returns": rng.standard_normal(N).astype(np.float32),
    "codes": codes, "unbuyable_mask": np.zeros(N, bool),
    "is_st_arr": np.zeros(N, bool), "w_sig_arr": np.ones(N, np.float32),
}

trainer = NeuralTrainer(db_path=DATABASE_PATH)
mk = dict(hidden_dims=(32, 16), dropout=0.1, epochs=4, patience=2,
          batch_size=1024, lr=1e-3)
multi, selected, norm_stats, results = trainer.train_multiobjective(
    dataset, objective_weights=None, model_kwargs=mk
)
print("selected features:", len(selected))
print("objectives trained:", list(multi.models.keys()))
for o, r in results.items():
    print(f"  {o}: val rank_ic={r['val_metrics']['rank_ic']:.4f}")

# save + reload via the EXACT path the backtest uses
with tempfile.TemporaryDirectory() as td:
    save_dir = os.path.join(td, "models")
    archive = trainer.save_artifacts(multi, selected, save_dir, norm_stats)
    pkl = os.path.join(archive, "neural_multi_objective_model.pkl")
    # 回测加载路径：MultiObjectiveNeuralModel.load_model
    reloaded = MultiObjectiveNeuralModel.load_model(pkl)
    df = pd.DataFrame(X[:200], columns=names)
    p1 = multi.predict(df)
    p2 = reloaded.predict(df)
    assert np.allclose(p1, p2, atol=1e-5), "reload mismatch"
    comp = reloaded.predict_components(df)
    assert set(comp) == set(multi.models)
    print("RELOAD OK; norm_stats keys:", list(norm_stats.keys()))
print("SYNTHETIC NN INTEGRATION CHECK PASSED")

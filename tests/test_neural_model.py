"""
神经网络模块的冒烟测试（无需数据库，纯数值验证接口与序列化）。

运行：
    .venv/Scripts/python.exe -m pytest tests/test_neural_model.py -q
或：
    .venv/Scripts/python.exe tests/test_neural_model.py
"""
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import torch

# 确保项目根目录在路径中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.neural.nn_models import (
    NeuralNet,
    NeuralNetFactorModel,
    MultiObjectiveNeuralModel,
)


def _make_data(n=2000, d=40, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    # 让 y 与部分特征弱相关，制造可学习信号
    logit = 0.3 * X[:, 0] + 0.2 * X[:, 1] - 0.25 * X[:, 2]
    p = 1.0 / (1.0 + np.exp(-logit))
    return X, p.astype(np.float32)


def test_neural_net_forward():
    net = NeuralNet(input_dim=10, hidden_dims=(16, 8), dropout=0.1)
    x = torch.randn(5, 10)
    out = net(x)
    assert out.shape == (5,)
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_single_model_train_predict_save_load():
    X, y = _make_data()
    n = len(X)
    Xtr, ytr = X[:n // 2], y[:n // 2]
    Xv, yv = X[n // 2:], y[n // 2:]
    feats = [f"f{i}" for i in range(X.shape[1])]

    model = NeuralNetFactorModel(
        input_dim=X.shape[1], feature_names=feats,
        hidden_dims=(32, 16), dropout=0.1, epochs=10, patience=3,
        batch_size=256, lr=1e-3,
    )
    model.fit(Xtr, ytr, Xv, yv, verbose=False)
    assert model.is_trained

    preds = model.predict(pd.DataFrame(Xv, columns=feats))
    assert preds.shape == (len(Xv),)
    assert np.all(np.isfinite(preds))
    assert float(preds.min()) >= 0.0 and float(preds.max()) <= 1.0

    # 与标签应有正相关（弱信号可学习）
    ic = np.corrcoef(preds, yv)[0, 1]
    assert ic > 0.05, f"模型未能学到信号，IC={ic:.4f}"

    # 序列化往返
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "net_factor_model.pkl")
        model.save_model(path)
        loaded = NeuralNetFactorModel.load_model(path)
        assert loaded.is_trained
        preds2 = loaded.predict(pd.DataFrame(Xv, columns=feats))
        assert np.allclose(preds, preds2, atol=1e-5)


def test_multi_objective_compatibility():
    X, _ = _make_data(d=20)
    feats = [f"f{i}" for i in range(X.shape[1])]
    mk = dict(hidden_dims=(16, 8), dropout=0.1, epochs=6, patience=2,
              batch_size=128, lr=1e-3)

    models = {}
    for obj in ("rank_y_ret_5d", "rank_y_ret_20d"):
        y = (np.arange(len(X)) % 7) / 6.0  # 任意可学习目标
        m = NeuralNetFactorModel(input_dim=X.shape[1], feature_names=feats, **mk)
        m.fit(X, y.astype(np.float32), verbose=False)
        models[obj] = m

    multi = MultiObjectiveNeuralModel(
        models, {"rank_y_ret_5d": 0.5, "rank_y_ret_20d": 0.5}
    )
    assert set(multi.models) == set(models)
    assert abs(sum(multi.weights.values()) - 1.0) < 1e-6

    df = pd.DataFrame(X, columns=feats)
    comp = multi.predict_components(df)
    assert set(comp) == set(models)
    total = multi.predict(df)
    assert total.shape == (len(X),)
    # 总分应为分量加权和
    manual = comp["rank_y_ret_5d"] * 0.5 + comp["rank_y_ret_20d"] * 0.5
    assert np.allclose(total, manual, atol=1e-6)

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "neural_multi_objective_model.pkl")
        multi.save_model(path)
        loaded = MultiObjectiveNeuralModel.load_model(path)
        total2 = loaded.predict(df)
        assert np.allclose(total, total2, atol=1e-5)
        assert set(loaded.models) == set(models)


if __name__ == "__main__":
    test_neural_net_forward()
    test_single_model_train_predict_save_load()
    test_multi_objective_compatibility()
    print("ALL NEURAL SMOKE TESTS PASSED")

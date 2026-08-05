"""
用已训练的 xgboost + lightgbm 单模型合成 EnsembleFactorModel，存为 ensemble_factor_model.pkl。

无需重训：直接读取现有归档目录里的两个 *_factor_model.pkl，等权 (0.5/0.5) 打包。
融合逻辑（截面 rank 后平均）在 EnsembleFactorModel.predict 内实现，见 [[ensemble-beats-single]]。

用法:
    python scripts/build_ensemble.py                       # 默认 models/latest, 等权 0.5/0.5
    python scripts/build_ensemble.py models/xl_7d_17y_...  # 指定目录
    python scripts/build_ensemble.py models/latest 0.4     # 指定 xgb 权重(lgb 自动=1-0.4)
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.factors.ml_factor_model import MLFactorModel, EnsembleFactorModel


def build(model_dir: str, weights=(0.5, 0.5)):
    xgb_path = os.path.join(model_dir, 'xgboost_factor_model.pkl')
    lgb_path = os.path.join(model_dir, 'lightgbm_factor_model.pkl')
    for p in (xgb_path, lgb_path):
        if not os.path.exists(p):
            print(f"缺少子模型: {p}")
            return None

    xgb_m = MLFactorModel(); xgb_m.load_model(xgb_path)
    lgb_m = MLFactorModel(); lgb_m.load_model(lgb_path)
    print(f"已加载子模型: xgb(is_trained={xgb_m.is_trained}), lgb(is_trained={lgb_m.is_trained})")

    ens = EnsembleFactorModel([xgb_m, lgb_m], list(weights))
    out_path = os.path.join(model_dir, 'ensemble_factor_model.pkl')
    ens.save_model(out_path)
    print(f"集成模型已保存: {out_path}")
    return out_path


if __name__ == '__main__':
    target = sys.argv[1] if len(sys.argv) > 1 else os.path.join('models', 'latest')
    if not os.path.isabs(target):
        target = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), target)
    # 可选第二参数: xgb 权重 (lgb = 1 - xgb)，缺省等权
    if len(sys.argv) > 2:
        w_xgb = float(sys.argv[2])
        weights = (w_xgb, 1.0 - w_xgb)
    else:
        weights = (0.5, 0.5)
    build(target, weights=weights)

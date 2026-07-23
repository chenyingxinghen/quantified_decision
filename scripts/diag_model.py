import pickle, numpy as np, pandas as pd, os, glob

CACHE = 'database/system_data/factors_cache'
m = pickle.load(open('models/latest/multi_objective_factor_model.pkl', 'rb'))
ms = m['model_states']

print("########## 各子模型 TOP15 特征重要性 ##########")
top_features = {}
for obj in ['rank_y_ret_5d','rank_y_ret_20d','rank_y_ret_60d','rank_y_mdd_20d','rank_y_downvol_20d','rank_y_illiq_20d']:
    st = ms[obj]
    fi = st['feature_importance']
    fn = st['feature_names']
    total = sum(fi.values())
    items = sorted(fi.items(), key=lambda x: -x[1])[:15]
    top15 = sum(v for _, v in items)
    print(f"\n=== {obj}  (TOP15 占 {top15/total*100:.1f}% of total) ===")
    for name, imp in items:
        print(f"   {imp:9.1f}  {name}")
    top_features[obj] = items[0][0]  # 头号特征

# 抽查 ret_20d 头号特征在个股上的时间波动（是否静态）
tf = top_features['rank_y_ret_20d']
print(f"\n########## 头号特征 '{tf}' 的时序波动抽查 (ret_20d 选择信号是否随时间变化) ##########")
files = glob.glob(os.path.join(CACHE, '*.parquet'))[:8]
stds = []
for f in files:
    try:
        df = pd.read_parquet(f)
    except Exception:
        continue
    if tf in df.columns and len(df) > 5:
        s = df[tf].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
        if len(s) > 5:
            stds.append((os.path.basename(f).split('_')[0], s.std(), s.mean(), len(s)))
if stds:
    for code, sd, mu, n in stds:
        print(f"   {code}: std={sd:.4f} mean={mu:.4f} n={n}  -> {'静态(≈0)' if sd < 1e-6 else '动态'}")
else:
    print("   头号特征不在缓存列中（可能是派生/组合特征），无法直接从缓存验证波动")

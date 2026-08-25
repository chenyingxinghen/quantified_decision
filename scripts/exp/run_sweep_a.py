# -*- coding: utf-8 -*-
"""
sweep A: 4-种子配对验证 scheme B 的熊市回撤增益是否系统性。

设计 (T069 配对协议):
  对每个种子 S, 仅改变 treatment:
    base_S    = NAM(disable-gate)             # 与 seed 无关的选股映射
    schemeB_S = base_S 专家冻结 + dual_scalar gate + mkt_pc1/pc2 + 收缩先验
  两者专家逐位相同, 唯一差异 = gate 是否开启 -> 干净的配对比较。

度量 (熊市主判窗 2022-09-05→2024-08-05, 真实成本 买0.0008/卖0.0013/Top20/排ST):
  回撤 / 收益 / 日频夏普 / 随机零假设分位
"""
import os
import sys
import json
import glob
import subprocess
import datetime

ROOT = r"G:/ai_proj/quantified_decision"
PY = r"C:/Users/29454/.workbuddy/binaries/python/versions/3.13.12/python.exe"
TRAIN = os.path.join(ROOT, "scripts", "train_nam_model.py")
BACKTEST = os.path.join(ROOT, "scripts", "run_backtest.py")
NULL = os.path.join(ROOT, "scripts", "exp", "diag_random_null.py")
CACHE = "database/system_data/factors_cache"

SEEDS = [42, 11, 23, 37]
COMMON = [
    "--stocks", "800", "--years", "13", "--end", "2022-09-05",
    "--target", "returns", "--y-scale", "2", "--lambda-lb", "0",
    "--epochs", "60", "--min-epochs", "20", "--skip-baseline",
    "--folds", "0.8:1.0", "--cache-dir", CACHE,
]
BT_ARGS = [
    "--start", "2022-09-05", "--end", "2024-08-05",
    "--max-positions", "20", "--min-confidence", "0",
    "--risk-min-price", "1", "--exclude-st",
    "--buy-cost", "0.0008", "--sell-cost", "0.0013",
]
DIAG = os.path.join(ROOT, "diagnose_output", "sweepA")
os.makedirs(DIAG, exist_ok=True)

TS = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RESULT_JSON = os.path.join(DIAG, f"sweepA_results_{TS}.json")


def run(label, args, logname):
    logp = os.path.join(DIAG, logname)
    with open(logp, "w") as f:
        p = subprocess.run([PY, args[0]] + args[1:], cwd=ROOT,
                           stdout=f, stderr=subprocess.STDOUT)
    return p.returncode == 0, logp


def find_trades_csv(model_basename):
    pat = os.path.join(ROOT, "backtest_result", model_basename, "nam",
                       "*_2022-09-05_to_2024-08-05", "backtest_trades.csv")
    hits = glob.glob(pat)
    return hits[0] if hits else None


def main():
    results = {}
    for s in SEEDS:
        base_dir = f"models/nam_gate/sweepA_base_s{s}"
        base_out = f"training_iterations/phase0/sweepA/base_s{s}"
        base_pkl = os.path.join(ROOT, base_dir, "nam_gate_factor_model.pkl")

        # ---- base (disable-gate) ----
        base_args = [TRAIN] + COMMON + [
            "--seed", str(s), "--disable-gate",
            "--output", base_out, "--save-model-dir", base_dir,
            "--plot-dir", base_out,
        ]
        ok, log = run(f"base_s{s}", base_args, f"base_s{s}.log")
        results[f"base_s{s}"] = {"train_ok": ok, "train_log": os.path.basename(log)}

        if not ok or not os.path.exists(base_pkl):
            print(f"[SKIP] base_s{s} train failed/missing, skip schemeB", flush=True)
            continue

        # ---- schemeB (frozen experts + gate) ----
        sb_dir = f"models/nam_gate/sweepA_schemeB_s{s}"
        sb_out = f"training_iterations/phase0/sweepA/schemeB_s{s}"
        sb_pkl = os.path.join(ROOT, sb_dir, "nam_gate_factor_model.pkl")
        sb_args = [TRAIN] + COMMON + [
            "--seed", str(s),
            "--gate-mode", "dual_scalar",
            "--gate-scalar-col", "mkt_pc1",
            "--gate-scalar-col2", "mkt_pc2",
            "--include-mkt",
            "--init-experts-from", base_pkl,
            "--freeze-experts", "--gate-wd", "0.1",
            "--output", sb_out, "--save-model-dir", sb_dir,
            "--plot-dir", sb_out,
        ]
        ok2, log2 = run(f"schemeB_s{s}", sb_args, f"schemeB_s{s}.log")
        results[f"schemeB_s{s}"] = {"train_ok": ok2,
                                    "train_log": os.path.basename(log2)}

        # ---- backtest both ----
        for tag, mdir, mpkl in [("base", base_dir, base_pkl),
                                 ("schemeB", sb_dir, sb_pkl)]:
            if not os.path.exists(mpkl):
                print(f"[SKIP] {tag}_s{s} pkl missing, skip backtest", flush=True)
                continue
            bt_tag = f"sweepA_{tag}_s{s}_bear"
            bt_args = [BACKTEST, "--model", mpkl] + BT_ARGS + ["--tag", bt_tag]
            okb, logb = run(f"bt_{tag}_s{s}", bt_args, f"bt_{tag}_s{s}.log")
            # metrics
            mdir_base = mdir.replace("models/nam_gate/", "")
            csvp = find_trades_csv(mdir_base)
            metrics = None
            if csvp:
                d = os.path.dirname(csvp)
                mp = os.path.join(d, "backtest_metrics.json")
                if os.path.exists(mp):
                    with open(mp) as f:
                        metrics = json.load(f)
                # random null
                outn = os.path.join(DIAG, f"null_{tag}_s{s}.json")
                null_args = [NULL, "--trades", csvp, "--n-sims", "2000",
                             "--buy-cost", "0.0008", "--sell-cost", "0.0013",
                             "--out", outn]
                okn, _ = run(f"null_{tag}_s{s}", null_args,
                             f"null_{tag}_s{s}.log")
                null_res = None
                if okn and os.path.exists(outn):
                    with open(outn) as f:
                        null_res = json.load(f)
                results[f"{tag}_s{s}"]["metrics"] = metrics
                results[f"{tag}_s{s}"]["random_null"] = null_res
            else:
                print(f"[WARN] {tag}_s{s} trades csv not found", flush=True)
            results[f"{tag}_s{s}"]["backtest_ok"] = okb

        # persist incremental
        with open(RESULT_JSON, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    with open(RESULT_JSON, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"=== SWEEP A DONE -> {RESULT_JSON} ===", flush=True)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

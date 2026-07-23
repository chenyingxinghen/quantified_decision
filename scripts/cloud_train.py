"""
scripts/cloud_train.py — 云端一键训练 + 回测 + 检验脚本（SUFE Gemini AI Platform）

完整流程（默认四个环节；--with-neural 时追加神经网络环节）:
  1. 数据处理:    (可选) 从 jydb_raw.db 重建中间库；训练前数据完整性自检（非阻断）
  2. 模型训练:    scripts/train_model.py（GBM 多目标，输出到 models/latest）
  3. 回测:        scripts/run_backtest.py（未给 --bt-start/--bt-end 时自动跑「样本内(训练区间)+样本外(训练后 oos_years 年)」双回测；给定则单次自定义）
  4. 分析报告:    analysis/run_analysis.py（稳健性/组合/SHAP → 报告 + 图表）
  4b/5. (可选) 神经网络: scripts.train_neural_model.py（输出 models/latest_neural）
        → 神经网络回测(backtest_neural) → 神经网络分析(analysis_neural, 自动跳过 SHAP)

全流程日志写入单一文件 <out>/pipeline.log（含每步命令、退出码、耗时、时间戳），
云端离线任务也能据此完整复盘。

环境变量（Gemini 平台挂载）:
  GEMINI_DATA_IN1  -> 输入数据根（含 jydb_raw.db / stock_daily.db 等）
  GEMINI_DATA_OUT  -> 输出目录（报告/日志落盘位置，缺省用项目内 analysis/output）

用法:
  # build + 三年训练 + 样本内(3年)回测 + 样本外(1年)回测 + 报告（本需求）
  python scripts/cloud_train.py --build-from-raw --train-years 3 --oos-years 1 --stocks 6000

  # 默认 6 年训练 + 样本内/样本外双回测 + 报告（数据已就绪，不重建）
  python scripts/cloud_train.py --stocks 6000

  # 单次自定义回测（给定 --bt-start/--bt-end 时只跑一次，不拆内外样本）
  python scripts/cloud_train.py --bt-start 2023-01-01 --bt-end 2026-01-01

  # 含神经网络模型（训练+回测+分析，自动跳过 SHAP）
  python scripts/cloud_train.py --build-from-raw --train-years 3 --oos-years 1 --with-neural
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import datetime
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from config.factor_config import TrainingConfig  # noqa: E402
from config.jydb_config import DATABASE_PATH  # noqa: E402


def _ts() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class PipelineLog:
    """同时写控制台与日志文件的简单 logger（带时间戳）。"""

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.f = open(path, "w", encoding="utf-8")
        self.path = path
        self(f"Pipeline log => {path}")

    def __call__(self, msg: str):
        line = f"[{_ts()}] {msg}"
        print(line)
        self.f.write(line + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


def _run(cmd, log: PipelineLog, fatal: bool = True):
    """运行子命令，实时把输出同时写到控制台和日志文件。"""
    log(">>> " + " ".join(cmd))
    t0 = time.time()
    p = subprocess.Popen(
        cmd, cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    for line in p.stdout:
        sys.stdout.write(line)
        log.f.write(f"[{_ts()}] {line}")
        log.f.flush()
    rc = p.wait()
    dt = (time.time() - t0) / 60.0
    log(f"<<< 退出码={rc}, 耗时={dt:.1f} 分钟")
    if rc != 0 and fatal:
        raise RuntimeError(f"命令失败 (exit={rc}): {' '.join(cmd)}")
    return rc


def _raw_db_path():
    return os.path.join(os.path.dirname(DATABASE_PATH), "jydb_raw.db")


def main():
    ap = argparse.ArgumentParser(description="云端一键训练 + 回测 + 检验")
    ap.add_argument("--build-from-raw", action="store_true",
                    help="从 jydb_raw.db 重建中间库（数据未预处理时）")
    ap.add_argument("--fast", action="store_true",
                    help="高速构建模式（GEMINI_BUILD_FAST）：SQLite 关闭 fsync，配合断点续跑")
    ap.add_argument("--stocks", type=int, default=getattr(TrainingConfig, "STOCK_NUM", 6000))
    ap.add_argument("--out", default=None, help="分析报告/日志输出目录（默认 analysis/output 或 GEMINI_DATA_OUT）")
    ap.add_argument("--workers", type=int, default=getattr(TrainingConfig, "N_JOBS_FACTOR_CALC", 15))
    ap.add_argument("--q", type=int, default=10)
    ap.add_argument("--cost", type=float, default=0.001)
    ap.add_argument("--shap-sample", type=int, default=5000)
    ap.add_argument("--start", default=None, help="训练开始日期（缺省按 --train-years/--oos-years 自动推算）")
    ap.add_argument("--end", default=None, help="训练结束日期（缺省 = 现在 - oos_years 年，为样本外留出空间）")
    ap.add_argument("--train-years", type=float, default=6.0,
                    help="训练窗口长度（年）。默认 6；本需求用 3 表示三年训练")
    ap.add_argument("--oos-years", type=float, default=1.0,
                    help="训练结束日到现在的留白年数 = 样本外回测长度（年）。默认 1")
    ap.add_argument("--bt-start", default=None, help="回测开始日期（与 --bt-end 同时给则单次自定义回测，否则自动跑样本内+样本外）")
    ap.add_argument("--bt-end", default=None, help="回测结束日期")
    ap.add_argument("--skip-verify", action="store_true", help="跳过训练前数据完整性自检")
    ap.add_argument("--with-neural", action="store_true",
                    help="额外训练神经网络模型并做神经网络回测/分析（需 torch，耗时更长）")
    ap.add_argument("--log", default=None, help="流水线日志路径（缺省 <out>/pipeline.log）")
    args = ap.parse_args()

    py = sys.executable
    out_dir = args.out or os.environ.get("GEMINI_DATA_OUT",
                                         os.path.join(PROJECT_ROOT, "analysis", "output"))
    os.makedirs(out_dir, exist_ok=True)
    log_path = args.log or os.path.join(out_dir, "pipeline.log")
    log = PipelineLog(log_path)

    model_path = os.path.join("models", "latest", "multi_objective_factor_model.pkl")
    cache = os.path.join(PROJECT_ROOT, "analysis", "full_dataset.parquet")

    # ── 推算训练窗口与回测窗口 ──
    # 训练窗 [train_start, train_end]；train_end 默认 = 现在 - oos_years，
    # 从而把最近 oos_years 年留给样本外回测；train_start = train_end - train_years。
    now = datetime.datetime.now()
    oos_years = float(args.oos_years)
    train_years = float(args.train_years)
    if args.end:
        train_end = args.end
    else:
        train_end = (now - datetime.timedelta(days=365 * oos_years)).strftime("%Y-%m-%d")
    if args.start:
        train_start = args.start
    else:
        te = datetime.datetime.strptime(train_end, "%Y-%m-%d")
        train_start = (te - datetime.timedelta(days=365 * train_years)).strftime("%Y-%m-%d")

    if args.bt_start and args.bt_end:
        bt_mode = "custom"
        bt_is_start = bt_oos_start = args.bt_start
        bt_is_end = bt_oos_end = args.bt_end
    else:
        bt_mode = "dual"
        bt_is_start, bt_is_end = train_start, train_end
        te = datetime.datetime.strptime(train_end, "%Y-%m-%d")
        bt_oos_start = train_end
        bt_oos_end = (te + datetime.timedelta(days=365 * oos_years)).strftime("%Y-%m-%d")

    log("=" * 70)
    log("云端一键流水线启动")
    log(f"输出目录: {out_dir}")
    log(f"训练股票数: {args.stocks}, workers: {args.workers}")
    log(f"训练窗口: {train_start} ~ {train_end}（{train_years:g} 年）")
    if bt_mode == "dual":
        log(f"回测(样本内): {bt_is_start} ~ {bt_is_end}")
        log(f"回测(样本外): {bt_oos_start} ~ {bt_oos_end}（{oos_years:g} 年）")
    else:
        log(f"回测(自定义): {bt_is_start} ~ {bt_is_end}")
    log("=" * 70)

    # ── 步骤 0: (可选) 从原始库重建 + 数据完整性自检 ──
    auto_build = args.build_from_raw or os.path.exists(_raw_db_path())
    if args.fast:
        os.environ["GEMINI_BUILD_FAST"] = "1"
    if auto_build:
        log("[1/4] 数据处理: 从 jydb_raw.db 重建中间库 ...")
        build_cmd = [py, "-m", "scripts.build_intermediate_from_raw", "--mode", "both"]
        if args.fast:
            build_cmd.append("--fast")
        _run(build_cmd, log)
    else:
        log("[1/4] 数据处理: 跳过重建（未检测到 jydb_raw.db 且未指定 --build-from-raw）")

    if not args.skip_verify:
        log("[1/4] 数据处理: 训练前数据完整性自检（非阻断）...")
        try:
            _run([py, "-m", "scripts.verify_data_completeness",
                  "--recent-years", "4", "--train-years", str(int(train_years))], log, fatal=False)
        except Exception as e:  # noqa: BLE001
            log(f"  自检异常（已忽略）: {e}")

    # ── 步骤 1: 训练模型 ──
    log("[2/4] 模型训练: scripts.train_model ...")
    train_cmd = [py, "-m", "scripts.train_model", "--workers", str(args.workers),
                 "--start", train_start, "--end", train_end]
    _run(train_cmd, log)

    # ── 步骤 2: 回测（样本内 + 样本外，或单次自定义）──
    if bt_mode == "custom":
        log("[3/4] 回测(自定义窗口): scripts.run_backtest ...")
        bt_cmd = [py, "-m", "scripts.run_backtest",
                  "--model", model_path,
                  "--output", os.path.join(out_dir, "backtest"),
                  "--start", bt_is_start, "--end", bt_is_end]
        _run(bt_cmd, log)
    else:
        log("[3/4] 回测(样本内): scripts.run_backtest（训练区间）...")
        bt_is = [py, "-m", "scripts.run_backtest",
                 "--model", model_path,
                 "--output", os.path.join(out_dir, "backtest_insample"),
                 "--start", bt_is_start, "--end", bt_is_end]
        _run(bt_is, log)
        log("[3/4] 回测(样本外): scripts.run_backtest（训练后一年）...")
        bt_oos = [py, "-m", "scripts.run_backtest",
                  "--model", model_path,
                  "--output", os.path.join(out_dir, "backtest_oos"),
                  "--start", bt_oos_start, "--end", bt_oos_end]
        _run(bt_oos, log)

    # ── 步骤 3: 分析 + 报告 ──
    log("[4/4] 分析报告: analysis.run_analysis ...")
    analysis_cmd = [
        py, "-m", "analysis.run_analysis",
        "--model", model_path,
        "--out", out_dir,
        "--cache", cache,
        "--stocks", str(args.stocks),
        "--q", str(args.q),
        "--cost", str(args.cost),
        "--shap-sample", str(args.shap_sample),
        "--start", train_start, "--end", train_end,
    ]
    _run(analysis_cmd, log)

    # ── 步骤 4b (可选): 神经网络模型（与 GBM 平行，共用同一数据基础设施）──
    if args.with_neural:
        neural_model_dir = os.path.join("models", "latest_neural")
        neural_model_path = os.path.join(neural_model_dir, "neural_multi_objective_model.pkl")

        log("[4b/5] 神经网络训练: scripts.train_neural_model ...")
        nn_train_cmd = [py, "-m", "scripts.train_neural_model",
                        "--workers", str(args.workers), "--stocks", str(args.stocks),
                        "--start", train_start, "--end", train_end]
        _run(nn_train_cmd, log)

        log("[5b/5] 神经网络回测: scripts.run_backtest（神经网络模型）...")
        if bt_mode == "custom":
            nn_bt_cmd = [py, "-m", "scripts.run_backtest",
                         "--model", neural_model_path,
                         "--output", os.path.join(out_dir, "backtest_neural"),
                         "--start", bt_is_start, "--end", bt_is_end]
            _run(nn_bt_cmd, log, fatal=False)
        else:
            nn_bt_is = [py, "-m", "scripts.run_backtest",
                        "--model", neural_model_path,
                        "--output", os.path.join(out_dir, "backtest_neural_insample"),
                        "--start", bt_is_start, "--end", bt_is_end]
            _run(nn_bt_is, log, fatal=False)
            nn_bt_oos = [py, "-m", "scripts.run_backtest",
                         "--model", neural_model_path,
                         "--output", os.path.join(out_dir, "backtest_neural_oos"),
                         "--start", bt_oos_start, "--end", bt_oos_end]
            _run(nn_bt_oos, log, fatal=False)

        log("[5b/5] 神经网络分析报告: analysis.run_analysis（SHAP 对神经网络不适用，自动跳过）...")
        nn_analysis_cmd = [
            py, "-m", "analysis.run_analysis",
            "--model", neural_model_path,
            "--out", os.path.join(out_dir, "analysis_neural"),
            "--cache", cache,
            "--stocks", str(args.stocks),
            "--q", str(args.q),
            "--cost", str(args.cost),
            "--no-shap",
            "--start", train_start, "--end", train_end,
        ]
        _run(nn_analysis_cmd, log, fatal=False)

    log("=" * 70)
    log("全部流程完成。产物:")
    log(f"  - 模型:        {model_path}")
    if bt_mode == "custom":
        log(f"  - 回测结果:    {os.path.join(out_dir, 'backtest')}/")
    else:
        log(f"  - 回测(样本内): {os.path.join(out_dir, 'backtest_insample')}/")
        log(f"  - 回测(样本外): {os.path.join(out_dir, 'backtest_oos')}/")
    log(f"  - 分析报告:    {os.path.join(out_dir, 'report.md')}")
    log(f"  - 指标表:      {os.path.join(out_dir, 'metrics.csv')}")
    log(f"  - 图表:        {os.path.join(out_dir, 'figures')}/")
    log(f"  - 流水线日志:  {log_path}")
    if args.with_neural:
        log(f"  - 神经网络模型: {os.path.join('models', 'latest_neural')}/")
        if bt_mode == "custom":
            log(f"  - 神经网络回测: {os.path.join(out_dir, 'backtest_neural')}/")
        else:
            log(f"  - 神经网络回测(内): {os.path.join(out_dir, 'backtest_neural_insample')}/")
            log(f"  - 神经网络回测(外): {os.path.join(out_dir, 'backtest_neural_oos')}/")
        log(f"  - 神经网络分析: {os.path.join(out_dir, 'analysis_neural', 'report.md')}")
    log("=" * 70)
    log.close()


if __name__ == "__main__":
    main()

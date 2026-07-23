"""
scripts/cloud_train.py — 云端一键训练 + 回测 + 检验脚本（SUFE Gemini AI Platform）

完整流程（默认四个环节；--with-neural 时追加神经网络环节）:
  1. 数据处理:    (可选) 从 jydb_raw.db 重建中间库；训练前数据完整性自检（非阻断）
  2. 模型训练:    scripts/train_model.py（GBM 多目标，输出到 models/latest）
  3. 回测:        scripts/run_backtest.py（多目标模型 → 资金曲线/交易/绩效）
  4. 分析报告:    analysis/run_analysis.py（稳健性/组合/SHAP → 报告 + 图表）
  4b/5. (可选) 神经网络: scripts.train_neural_model.py（输出 models/latest_neural）
        → 神经网络回测(backtest_neural) → 神经网络分析(analysis_neural, 自动跳过 SHAP)

全流程日志写入单一文件 <out>/pipeline.log（含每步命令、退出码、耗时、时间戳），
云端离线任务也能据此完整复盘。

环境变量（Gemini 平台挂载）:
  GEMINI_DATA_IN1  -> 输入数据根（含 jydb_raw.db / stock_daily.db 等）
  GEMINI_DATA_OUT  -> 输出目录（报告/日志落盘位置，缺省用项目内 analysis/output）

用法:
  python scripts/cloud_train.py --build-from-raw --stocks 6000
  python scripts/cloud_train.py                       # 仅训练+回测+分析（数据已就绪）
  python scripts/cloud_train.py --bt-start 2025-01-01 --bt-end 2026-01-01
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
    ap.add_argument("--stocks", type=int, default=getattr(TrainingConfig, "STOCK_NUM", 6000))
    ap.add_argument("--out", default=None, help="分析报告/日志输出目录（默认 analysis/output 或 GEMINI_DATA_OUT）")
    ap.add_argument("--workers", type=int, default=getattr(TrainingConfig, "N_JOBS_FACTOR_CALC", 15))
    ap.add_argument("--q", type=int, default=10)
    ap.add_argument("--cost", type=float, default=0.001)
    ap.add_argument("--shap-sample", type=int, default=5000)
    ap.add_argument("--start", default=None, help="训练开始日期")
    ap.add_argument("--end", default=None, help="训练结束日期")
    ap.add_argument("--bt-start", default=None, help="回测开始日期（缺省用策略默认近期窗口）")
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

    log("=" * 70)
    log("云端一键流水线启动")
    log(f"输出目录: {out_dir}")
    log(f"训练股票数: {args.stocks}, workers: {args.workers}")
    log("=" * 70)

    # ── 步骤 0: (可选) 从原始库重建 + 数据完整性自检 ──
    auto_build = args.build_from_raw or os.path.exists(_raw_db_path())
    if auto_build:
        log("[1/4] 数据处理: 从 jydb_raw.db 重建中间库 ...")
        _run([py, "-m", "scripts.build_intermediate_from_raw", "--mode", "both"], log)
    else:
        log("[1/4] 数据处理: 跳过重建（未检测到 jydb_raw.db 且未指定 --build-from-raw）")

    if not args.skip_verify:
        log("[1/4] 数据处理: 训练前数据完整性自检（非阻断）...")
        try:
            _run([py, "-m", "scripts.verify_data_completeness",
                  "--recent-years", "4", "--train-years", "6"], log, fatal=False)
        except Exception as e:  # noqa: BLE001
            log(f"  自检异常（已忽略）: {e}")

    # ── 步骤 1: 训练模型 ──
    log("[2/4] 模型训练: scripts.train_model ...")
    train_cmd = [py, "-m", "scripts.train_model", "--workers", str(args.workers)]
    if args.start:
        train_cmd += ["--start", args.start]
    if args.end:
        train_cmd += ["--end", args.end]
    _run(train_cmd, log)

    # ── 步骤 2: 回测 ──
    log("[3/4] 回测: scripts.run_backtest（多目标模型）...")
    bt_cmd = [py, "-m", "scripts.run_backtest",
              "--model", model_path,
              "--output", os.path.join(out_dir, "backtest")]
    if args.bt_start:
        bt_cmd += ["--start", args.bt_start]
    if args.bt_end:
        bt_cmd += ["--end", args.bt_end]
    _run(bt_cmd, log)

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
    ]
    if args.start:
        analysis_cmd += ["--start", args.start]
    if args.end:
        analysis_cmd += ["--end", args.end]
    _run(analysis_cmd, log)

    # ── 步骤 4b (可选): 神经网络模型（与 GBM 平行，共用同一数据基础设施）──
    if args.with_neural:
        neural_model_dir = os.path.join("models", "latest_neural")
        neural_model_path = os.path.join(neural_model_dir, "neural_multi_objective_model.pkl")

        log("[4b/5] 神经网络训练: scripts.train_neural_model ...")
        nn_train_cmd = [py, "-m", "scripts.train_neural_model",
                        "--workers", str(args.workers), "--stocks", str(args.stocks)]
        if args.start:
            nn_train_cmd += ["--start", args.start]
        if args.end:
            nn_train_cmd += ["--end", args.end]
        _run(nn_train_cmd, log)

        log("[5b/5] 神经网络回测: scripts.run_backtest（神经网络模型）...")
        nn_bt_cmd = [py, "-m", "scripts.run_backtest",
                     "--model", neural_model_path,
                     "--output", os.path.join(out_dir, "backtest_neural")]
        if args.bt_start:
            nn_bt_cmd += ["--start", args.bt_start]
        if args.bt_end:
            nn_bt_cmd += ["--end", args.bt_end]
        _run(nn_bt_cmd, log, fatal=False)

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
        ]
        if args.start:
            nn_analysis_cmd += ["--start", args.start]
        if args.end:
            nn_analysis_cmd += ["--end", args.end]
        _run(nn_analysis_cmd, log, fatal=False)

    log("=" * 70)
    log("全部流程完成。产物:")
    log(f"  - 模型:        {model_path}")
    log(f"  - 回测结果:    {os.path.join(out_dir, 'backtest')}/")
    log(f"  - 分析报告:    {os.path.join(out_dir, 'report.md')}")
    log(f"  - 指标表:      {os.path.join(out_dir, 'metrics.csv')}")
    log(f"  - 图表:        {os.path.join(out_dir, 'figures')}/")
    log(f"  - 流水线日志:  {log_path}")
    if args.with_neural:
        log(f"  - 神经网络模型: {os.path.join('models', 'latest_neural')}/")
        log(f"  - 神经网络回测: {os.path.join(out_dir, 'backtest_neural')}/")
        log(f"  - 神经网络分析: {os.path.join(out_dir, 'analysis_neural', 'report.md')}")
    log("=" * 70)
    log.close()


if __name__ == "__main__":
    main()

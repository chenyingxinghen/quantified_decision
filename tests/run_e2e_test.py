"""
端到端连通性测试
流程: 训练 → 因子重要性(SHAP) → 回测
所有输出报告统一放在临时文件夹
"""
import os
import sys
import subprocess
import shutil
import json
from pathlib import Path
from datetime import datetime

# ──────────────────────────── helpers ────────────────────────────
ENCODE = sys.stdout.encoding or 'utf-8'

def safe_print(text):
    try:
        print(text, flush=True)
    except UnicodeEncodeError:
        print(text.encode('gbk', 'replace').decode('gbk'), flush=True)


def run_cmd(cmd, label=""):
    safe_print(f"\n{'='*60}")
    safe_print(f"[STEP] {label}")
    safe_print(f"CMD: {' '.join(cmd)}")
    safe_print('='*60)
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    result = subprocess.run(cmd, env=env, capture_output=True, text=True,
                            encoding='utf-8', errors='replace')
    stdout = result.stdout
    stderr = result.stderr
    try:
        safe_print(stdout)
    except Exception:
        pass
    if result.returncode != 0:
        safe_print(f"[ERROR] Command failed (exit {result.returncode})")
        safe_print(stderr)
        sys.exit(1)
    return stdout, stderr


def copy_glob(src_dir, dst_dir, exts=('.png', '.csv', '.json', '.txt')):
    """递归拷贝 src_dir 下符合扩展名的文件到 dst_dir（扁平化）"""
    src = Path(src_dir)
    dst = Path(dst_dir)
    copied = []
    for f in src.rglob('*'):
        if f.is_file() and f.suffix.lower() in exts:
            shutil.copy2(f, dst / f.name)
            copied.append(f.name)
    return copied


# ──────────────────────────── main ────────────────────────────
def main():
    root_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    os.chdir(root_dir)

    # ── 0. 创建临时输出目录 ──────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_dir = root_dir / f"temp_reports_{timestamp}"
    temp_dir.mkdir(exist_ok=True)
    safe_print(f"\n[INFO] 报告输出目录: {temp_dir}\n")

    # ─────────────────────────────────────────────────────────────
    # STEP 1 — 训练模型
    # 使用少量数据做连通性测试（2 只股票，5 个月数据）
    # ─────────────────────────────────────────────────────────────
    train_stdout, _ = run_cmd([
        sys.executable, "scripts/train_model.py",
        "--start", "2022-12-01",
        "--end",   "2023-04-30",
        "--stocks", "10",
        "--force",
    ], label="训练模型")

    # 把训练日志写到报告目录
    train_log_path = temp_dir / "train_log.txt"
    train_log_path.write_text(train_stdout, encoding='utf-8')
    safe_print(f"[OK] 训练日志已写入: {train_log_path.name}")

    # 找到最新训练的归档目录（models/xxx 非 latest/mark）
    models_root = root_dir / "models"
    archive_dirs = sorted(
        [d for d in models_root.iterdir()
         if d.is_dir() and d.name not in ('latest', 'mark')],
        key=lambda d: d.stat().st_mtime
    )
    if not archive_dirs:
        safe_print("[ERROR] 未找到模型归档目录")
        sys.exit(1)
    model_archive = archive_dirs[-1]
    safe_print(f"[OK] 最新模型归档: {model_archive.name}")

    # 复制模型配置文件（factor_summary, training_config）到报告目录
    for fname in ('factor_summary.json', 'training_config.json'):
        src = model_archive / fname
        if src.exists():
            shutil.copy2(src, temp_dir / fname)
            safe_print(f"[OK] 复制: {fname}")

    # 生成统计性检验报告（从训练 log 中提取关键指标）
    stats_report = _extract_stats_report(train_stdout, model_archive)
    stats_path = temp_dir / "stats_report.txt"
    stats_path.write_text(stats_report, encoding='utf-8')
    safe_print(f"[OK] 统计性检验报告: {stats_path.name}")

    # ─────────────────────────────────────────────────────────────
    # STEP 2 — SHAP / 因子重要性
    # ─────────────────────────────────────────────────────────────
    model_path = root_dir / "models" / "latest" / "multi_objective_factor_model.pkl"
    if not model_path.exists():
        safe_print(f"[ERROR] 模型文件不存在: {model_path}")
        sys.exit(1)

    importance_prefix = temp_dir / "importance"
    run_cmd([
        sys.executable, "scripts/show_factor_importance.py",
        "--model", str(model_path),
        "--save",
        "--output", str(importance_prefix),
        "--top", "30",
    ], label="SHAP 因子重要性分析")
    safe_print("[OK] SHAP 因子重要性图表 & CSV 已保存至报告目录")

    # ─────────────────────────────────────────────────────────────
    # STEP 3 — 回测（与训练数据时间段紧接）
    # 训练: 2022-12 ~ 2023-04，回测: 2023-05 ~ 2023-07
    # ─────────────────────────────────────────────────────────────
    backtest_out_dir = temp_dir / "backtest"
    backtest_out_dir.mkdir(exist_ok=True)

    run_cmd([
        sys.executable, "scripts/run_backtest.py",
        "--model", str(model_path),
        "--start", "2023-05-01",
        "--end",   "2023-07-31",
        "--output", str(backtest_out_dir),
    ], label="回测")
    safe_print("[OK] 回测完成，结果保存至报告目录")

    # ─────────────────────────────────────────────────────────────
    # STEP 4 — 汇总报告
    # ─────────────────────────────────────────────────────────────
    safe_print("\n" + "="*60)
    safe_print("端到端连通性测试 — 完成!")
    safe_print(f"报告目录: {temp_dir}")
    safe_print("="*60)
    safe_print("生成文件:")
    all_files = sorted(temp_dir.rglob('*'))
    for f in all_files:
        if f.is_file():
            rel = f.relative_to(temp_dir)
            size_kb = f.stat().st_size / 1024
            safe_print(f"  {rel}  ({size_kb:.1f} KB)")
    safe_print("="*60)


def _extract_stats_report(train_stdout: str, model_archive: Path) -> str:
    """从训练日志和配置文件中提取关键统计指标，生成文字报告"""
    lines = []
    lines.append("=" * 60)
    lines.append("统计性检验报告 — 模型评估摘要")
    lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 60)

    # 从 training_config.json 读取训练信息
    cfg_path = model_archive / "training_config.json"
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
            lines.append("\n【训练配置】")
            for k, v in cfg.items():
                lines.append(f"  {k}: {v}")
        except Exception:
            pass

    # 从 factor_summary.json 读取因子统计
    fs_path = model_archive / "factor_summary.json"
    if fs_path.exists():
        try:
            fs = json.loads(fs_path.read_text(encoding='utf-8'))
            lines.append("\n【因子统计】")
            lines.append(f"  总因子数: {fs.get('total_factors', '?')}")
            lines.append(f"  技术因子: {fs.get('technical_factors', '?')}")
            lines.append(f"  基本面因子: {fs.get('fundamental_factors', '?')}")
            lines.append(f"  高级时序因子: {fs.get('advanced_factors', '?')}")
            lines.append(f"  工程因子: {fs.get('engineered_factors', '?')}")
        except Exception:
            pass

    # 从训练日志中提取关键评估指标（Rank IC, AUC, ndcg 等）
    lines.append("\n【模型评估指标（训练日志提取）】")
    keywords = [
        'Rank IC', 'ndcg', 'AUC', 'Top-1', 'Top-5',
        '最佳选股模型', 'lightgbm', 'xgboost',
        'LIGHTGBM', 'XGBOOST', 'Early stopping',
        'best iteration',
    ]
    for line in train_stdout.splitlines():
        line_stripped = line.strip()
        if any(kw.lower() in line_stripped.lower() for kw in keywords):
            lines.append(f"  {line_stripped}")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


if __name__ == "__main__":
    main()

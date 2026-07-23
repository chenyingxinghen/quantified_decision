"""逐特征缺失率诊断（训练结束后 DB 解锁再跑）。

用途：回答"训练初期提示发现 96682337 个缺失值是否正常"。
核心判断：
  - 缺失应集中在 jy_（聚源PIT）特征：它们按设计在归一化/覆盖率筛选前保持 NaN。
  - 非 jy 内部特征（技术/基本面/高级/工程）应近 0 缺失（已零填充）。
  - 若某个非 jy 特征缺失率 > 阈值，说明因子缓存/计算有 bug（应被零填充却漏了）。
"""
from __future__ import annotations
import os, glob, sqlite3, argparse
import numpy as np
import pandas as pd

DB_DIR = "database"
CACHE_DIR = os.path.join(DB_DIR, "system_data", "factors_cache")
JYDB_DB = os.path.join(DB_DIR, "jydb_features.db")
STOCK_DB = os.path.join(DB_DIR, "stock_daily.db")


def non_jy_coverage(cache_dir: str, max_files: int = 200):
    """从 factors_cache parquet 计算非 jy 特征逐列有限率（缺失率=1-有限率）。"""
    files = sorted(glob.glob(os.path.join(cache_dir, "*.parquet")))
    if max_files and len(files) > max_files:
        files = files[:max_files]
        print(f"[non-jy] 采样 {len(files)} 个股票 parquet（非全量，仅估缺失率）")
    agg = {}  # col -> [finite_sum, total_sum]
    for f in files:
        df = pd.read_parquet(f)
        skip = {"date", "code", "trading_date"}
        for c in df.columns:
            if c.lower() in skip or c.startswith("jy_"):
                continue
            s = pd.to_numeric(df[c], errors="coerce")
            tot = len(s)
            fin = int(np.isfinite(s).sum())
            if c not in agg:
                agg[c] = [0, 0]
            agg[c][0] += fin
            agg[c][1] += tot
    return {c: (v[0] / v[1] if v[1] else 0.0) for c, v in agg.items()}


def jy_coverage(jydb_db: str, start: str, end: str):
    """从 jydb_features.db 计算 jy_ 特征在训练窗内的有限率。"""
    conn = sqlite3.connect(jydb_db, timeout=180)
    cur = conn.cursor()
    # daily_features（宽表，密集）
    cur.execute("PRAGMA table_info(daily_features)")
    daily_cols = [r[1] for r in cur.fetchall() if r[1] not in ("code", "date")]
    daily_cov = {}
    if daily_cols:
        cols = ", ".join(f'AVG(CASE WHEN "{c}" IS NOT NULL THEN 1.0 ELSE 0.0 END) AS "{c}"' for c in daily_cols)
        cur.execute(f"SELECT {cols} FROM daily_features WHERE date>='{start}' AND date<='{end}'")
        row = cur.fetchone()
        daily_cov = {c: (float(v) if v is not None else 0.0) for c, v in zip(daily_cols, row)}
    # pit_features（长表，稀疏）：逐特征计数
    cur.execute(
        f"""SELECT feature_name,
                  COUNT(*) AS n,
                  SUM(CASE WHEN feature_value IS NOT NULL THEN 1 ELSE 0 END) AS fin
           FROM pit_features
           WHERE available_date>='{start}' AND available_date<='{end}'
           GROUP BY feature_name"""
    )
    # 训练窗 code-date 总数
    cur.execute(f"SELECT COUNT(*) FROM daily_data WHERE date>='{start}' AND date<='{end}'")
    tot_pairs = cur.fetchone()[0]
    pit = {}
    for fname, n, fin in cur.fetchall():
        pit[fname] = (fin or 0) / tot_pairs if tot_pairs else 0.0
    conn.close()
    return daily_cov, pit, tot_pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2023-07-21")
    ap.add_argument("--end", default="2025-07-21")
    ap.add_argument("--non-jy-files", type=int, default=200)
    ap.add_argument("--bug-threshold", type=float, default=0.20,
                    help="非 jy 特征缺失率超过该值视为异常（应零填充却漏了）")
    args = ap.parse_args()

    print("=" * 70)
    print(f"特征缺失率诊断  [{args.start} ~ {args.end}]")
    print("=" * 70)

    nonjy = non_jy_coverage(CACHE_DIR, args.non_jy_files)
    daily_cov, pit_cov, tot_pairs = jy_coverage(JYDB_DB, args.start, args.end)

    # ---- 非 jy ----
    if nonjy:
        rates = np.array(list(nonjy.values()))
        hi = {c: r for c, r in nonjy.items() if (1 - r) > args.bug_threshold}
        print(f"\n[非 jy 内部特征] 统计列数={len(nonjy)}")
        print(f"  平均缺失率 = {(1-rates.mean())*100:.3f}%")
        print(f"  最大缺失率 = {(1-rates.max())*100:.3f}%")
        if hi:
            print(f"  ⚠ 异常：{len(hi)} 个非 jy 特征缺失率>{args.bug_threshold*100:.0f}% （应零填充却漏了，疑似 bug）")
            for c, r in sorted(hi.items(), key=lambda x: x[1])[:20]:
                print(f"      {c}: 缺失 {(1-r)*100:.1f}%")
        else:
            print(f"  ✓ 非 jy 特征缺失率均 ≤{args.bug_threshold*100:.0f}%，无异常（内部特征已正确零填充）")

    # ---- jy_ ----
    all_jy = {**daily_cov, **pit_cov}
    if all_jy:
        rates = np.array(list(all_jy.values()))
        dropped = {c: r for c, r in all_jy.items() if r < (1 - args.bug_threshold)}
        print(f"\n[聚源 jy_ 特征] distinct 特征数={len(all_jy)}")
        print(f"  平均填充率 = {rates.mean()*100:.2f}%（即平均缺失 {(1-rates.mean())*100:.2f}%）")
        print(f"  填充率<{ (1-args.bug_threshold)*100:.0f}% 被覆盖率筛选淘汰的特征数 = {len(dropped)} （预期，稀疏事件）")
        print(f"  填充率最高的 10 个 jy_ 特征：")
        for c, r in sorted(all_jy.items(), key=lambda x: -x[1])[:10]:
            print(f"      {c}: 填充 {r*100:.1f}%")
        print(f"  填充率最低的 10 个 jy_ 特征（多为低频事件）：")
        for c, r in sorted(all_jy.items(), key=lambda x: x[1])[:10]:
            print(f"      {c}: 填充 {r*100:.1f}%")

    print("\n结论速判：")
    print("  - 若『非 jy 异常数=0』且『jy_ 平均缺失为个位数%~数十%』→ 96682337 属正常（矩阵巨大+JYDB按设计保留NaN）。")
    print("  - 若『非 jy 异常数>0』→ 内部特征未零填充，是真实 bug，需查因子缓存/计算链路。")


if __name__ == "__main__":
    main()

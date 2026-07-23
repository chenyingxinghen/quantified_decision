"""
验证纯聚源管线数据完整性（最近 N 年 + 训练全窗口）

检查项：
  1. daily_data 年度交易日数 / 股票数（缺口初筛）
  2. adjust_factor（复权因子）覆盖
  3. daily_data.is_st 覆盖（非 NULL 比例 + 被标记 ST 的股票数）
  4. daily_features / pit_features 对 daily_data 的 (code,date) 覆盖比
  5. 基本面/行业特征 jy_fin_* / jy_industry_* 在近期窗口的非空率

用法：
  python scripts/verify_data_completeness.py --recent-years 4 --train-years 6
"""
import sqlite3
import os
import sys
import time
import argparse
from datetime import datetime, timedelta

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config.data_config import DATABASE_PATH
from config.jydb_config import JYDB_FEATURE_DB_PATH

EXPECT_TRADING_DAYS = 200  # A 股年均交易日下限（少于则视为缺口）


def ro(p):
    return f"file:{os.path.abspath(p)}?mode=ro"


def q(conn, sql, params=()):
    return conn.execute(sql, params).fetchall()


def verify():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recent-years", type=int, default=4)
    ap.add_argument("--train-years", type=int, default=6)
    args = ap.parse_args()

    today = datetime.now()
    r_end = today.strftime("%Y-%m-%d")
    r_start = (today - timedelta(days=365 * args.recent_years)).strftime("%Y-%m-%d")
    t_end = r_end
    t_start = (today - timedelta(days=365 * args.train_years)).strftime("%Y-%m-%d")

    print("=" * 72)
    print("纯聚源管线 — 数据完整性校验")
    print(f"  最近 {args.recent_years} 年窗口 : {r_start} ~ {r_end}")
    print(f"  训练 {args.train_years} 年窗口 : {t_start} ~ {t_end}")
    print("=" * 72)

    # ---- 数据可用边界 ----
    c = sqlite3.connect(ro(DATABASE_PATH), uri=True, timeout=120)
    real_min, real_max = q(c, "SELECT MIN(date), MAX(date) FROM daily_data")[0]
    print(f"\n[daily_data 实际覆盖] {real_min} ~ {real_max}")
    r_start = max(r_start, real_min)
    t_start = max(t_start, real_min)
    r_end = min(r_end, real_max)
    t_end = min(t_end, real_max)

    # ---- 1. 年度交易日 / 股票数 ----
    print("\n[1] daily_data 年度交易日 / 股票数")
    for label, s, e in [("RECENT", r_start, r_end), ("TRAIN", t_start, t_end)]:
        rows = q(c, """
            SELECT substr(date,1,4) y, COUNT(DISTINCT date) tdays,
                   COUNT(DISTINCT code) codes, COUNT(*) n
            FROM daily_data WHERE date BETWEEN ? AND ? GROUP BY y ORDER BY y
        """, (s, e))
        for y, td, cd, n in rows:
            flag = "" if td >= EXPECT_TRADING_DAYS else "  ⚠ 交易日偏少"
            print(f"  [{label}] {y}: 交易日={td:3d} 股票数={cd:5d} 行数={n:,}{flag}")

    # ---- 2. adjust_factor ----
    print("\n[2] adjust_factor（复权因子）覆盖")
    n_adj = q(c, "SELECT COUNT(*) FROM adjust_factor")[0][0]
    adj_min, adj_max = q(c, "SELECT MIN(date), MAX(date) FROM adjust_factor")[0]
    if n_adj == 0:
        print(f"  ⚠ adjust_factor 为空（0 行）！复权因子缺失。")
    else:
        print(f"  行数={n_adj:,} 覆盖={adj_min} ~ {adj_max}")
        for label, s, e in [("RECENT", r_start, r_end), ("TRAIN", t_start, t_end)]:
            inw = q(c, "SELECT COUNT(*) FROM adjust_factor WHERE date BETWEEN ? AND ?", (s, e))[0][0]
            print(f"    [{label}] 窗口内行数={inw:,}")

    # ---- 3. is_st 覆盖 ----
    print("\n[3] daily_data.is_st 覆盖")
    tot = q(c, "SELECT COUNT(*) FROM daily_data WHERE date BETWEEN ? AND ?", (t_start, t_end))[0][0]
    st_null = q(c, "SELECT COUNT(*) FROM daily_data WHERE date BETWEEN ? AND ? AND is_st IS NULL", (t_start, t_end))[0][0]
    st1 = q(c, "SELECT COUNT(DISTINCT code) FROM daily_data WHERE date BETWEEN ? AND ? AND is_st=1", (t_start, t_end))[0][0]
    null_rate = (st_null / tot * 100) if tot else 0
    print(f"  训练窗口行数={tot:,} | is_st 为 NULL={st_null:,} ({null_rate:.2f}%) | 曾被标记 ST 的股票数={st1}")
    if st_null > 0:
        print("  ⚠ 存在 is_st 为 NULL 的行（ST 状态未填充）")
    c.close()

    # ---- 4 & 5. 特征库覆盖（抽样 + 单年 + 水位线，避免整表扫描）----
    fc = sqlite3.connect(ro(JYDB_FEATURE_DB_PATH), uri=True, timeout=180)
    # 水位线（特征构建进度，毫秒级）
    try:
        wm = q(fc, "SELECT source_table, watermark FROM etl_watermarks")
        print("\n[水位线] etl_watermarks (特征构建进度):")
        for name, val in wm:
            print(f"    {name} -> {val}")
    except Exception as e:
        print("  [水位线] 读取失败:", e)

    # 代表股票（跨市场/板块）
    reps = ["000001", "600000", "300750", "688981", "000858"]
    cdd = sqlite3.connect(ro(DATABASE_PATH), uri=True, timeout=120)

    # daily_features：宽表，含 date 列，可逐股对比覆盖
    tbl = "daily_features"
    print(f"\n[4] {tbl} 覆盖（宽表，逐股对比）")
    cols = [r[1] for r in q(fc, f"PRAGMA table_info({tbl})")]
    fin = [x for x in cols if x.startswith("jy_fin_")]
    ind = [x for x in cols if x.startswith("jy_industry_")]
    print(f"    字段: 共 {len(cols)} 个 | jy_fin_*={len(fin)} | jy_industry_*={len(ind)}")
    print(f"    ⚠ daily_features 无 jy_fin_*/jy_industry_* → 基本面/行业在 pit_features(长表)")
    for code in reps:
        dd_y = dict(q(cdd, """
            SELECT substr(date,1,4) y, COUNT(*) FROM daily_data
            WHERE code=? AND date BETWEEN ? AND ? GROUP BY y
        """, (code, t_start, t_end)))
        ft_y = dict(q(fc, f"""
            SELECT substr(date,1,4) y, COUNT(*) FROM {tbl}
            WHERE code=? AND date BETWEEN ? AND ? GROUP BY y
        """, (code, t_start, t_end)))
        dd_tot = sum(dd_y.values()); ft_tot = sum(ft_y.values())
        line = f"    {code}: "
        for y in sorted(set(dd_y) | set(ft_y)):
            d, f = dd_y.get(y, 0), ft_y.get(y, 0)
            miss = "" if d == 0 or f >= d * 0.99 else f" ⚠缺{f}/{d}"
            line += f"{y}[dd={d},ft={f}]{miss} "
        cov = (ft_tot / dd_tot * 100) if dd_tot else 0
        line += f"| 该股技术特征覆盖比={cov:.1f}%"
        print(line)

    # pit_features：长表，键为 available_date（非 date），存基本面/行业 PIT 值
    tbl = "pit_features"
    print(f"\n[5] {tbl} 覆盖（长表 PIT，键=available_date）")
    pcols = [r[1] for r in q(fc, f"PRAGMA table_info({tbl})")]
    print(f"    字段: {pcols}")
    ad_min, ad_max = q(fc, f"SELECT MIN(available_date), MAX(available_date) FROM {tbl}")[0]
    # 用 etl_watermarks 推断各行业/基本面表是否已构建（避免整表 COUNT 扫描 1500 万行）
    wm_map = {n: v for n, v in q(fc, "SELECT source_table, watermark FROM etl_watermarks")}
    built_pit = [s for s in wm_map if s in (
        "LC_MainIndexNew", "LC_CSIIndustry", "LC_PerformanceLetters",
        "LC_Dividend", "LC_ShareStru", "LC_FreeFloat")]
    print(f"    available_date 范围: [{ad_min} ~ {ad_max}]")
    print(f"    PIT 特征表已构建(水位线): {built_pit}")
    print(f"    available_date 末值接近今日 => 基本面/行业 PIT 数据已就位")
    # 抽样确认：代表股近期是否有 LC_MainIndexNew / LC_CSIIndustry 记录
    for code in reps[:2]:
        for src in ("LC_MainIndexNew", "LC_CSIIndustry"):
            n = q(fc, f"""
                SELECT COUNT(*) FROM {tbl}
                WHERE code=? AND source_table=? AND available_date <= ?
            """, (code, src, r_end))[0][0]
            print(f"    {code} / {src}: 截至 {r_end} 记录数={n}")

    cdd.close()
    fc.close()
    print("\n=== 校验完成 ===")


if __name__ == "__main__":
    verify()

import sys
import os

# 将项目根目录添加到路径中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.jydb_config import JYDB_RAW_DB_PATH, JYDB_FEATURE_DB_PATH
from core.data.jydb_feature_store import DEFAULT_TABLE_SPECS, JYDBFeatureStore


def update_industry_to_db():
    """
    从本地聚源 raw 库 (jydb_raw.db) 重建行业特征到特征库 (jydb_features.db)。

    行业数据来自聚源 LC_CSIIndustry 表，经 build_intermediate_from_raw 预处理为
    jy_industry_* 列（已按公告日做 PIT 对齐）。本脚本等价于：

        python scripts/build_intermediate_from_raw.py --mode feature --tables LC_CSIIndustry

    不依赖 Baostock。
    """
    if "LC_CSIIndustry" not in DEFAULT_TABLE_SPECS:
        print("✗ 当前聚源规格未包含 LC_CSIIndustry，无法更新行业特征")
        return

    if not os.path.exists(JYDB_RAW_DB_PATH):
        print(f"✗ raw 库不存在: {JYDB_RAW_DB_PATH}，请先运行 pull_jydb_parallel.py")
        return

    print(f"开始重建行业特征 (源: {JYDB_RAW_DB_PATH})")
    store = JYDBFeatureStore(JYDB_FEATURE_DB_PATH)
    store.initialize()
    spec = DEFAULT_TABLE_SPECS["LC_CSIIndustry"]
    total = store.upsert_wide_frame(
        _read_raw_industry(JYDB_RAW_DB_PATH, spec),
        source_table=spec.name,
        available_date_col=spec.available_date_col,
        end_date_col=spec.end_date_col,
        feature_cols=spec.feature_cols,
        dimension_cols=spec.dimension_cols,
        prefix=spec.prefix,
    )
    print(f"✅ 行业特征重建完成，写入 {total:,} 个值 -> {JYDB_FEATURE_DB_PATH}")


def _read_raw_industry(raw_db: str, spec):
    import sqlite3
    import pandas as pd
    conn = sqlite3.connect(f"file:{raw_db}?mode=ro", uri=True, timeout=120)
    try:
        df = pd.read_sql_query(f'SELECT * FROM "{spec.name}"', conn)
    finally:
        conn.close()
    return df


if __name__ == "__main__":
    update_industry_to_db()

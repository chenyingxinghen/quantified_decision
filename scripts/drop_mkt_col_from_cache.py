"""从 factors_cache_phase0 删除 mkt_lpr_5y 列（lpr_5y 仅 85 非空，已判定为装饰列）。

用法:
    .venv/Scripts/python.exe scripts/drop_mkt_col_from_cache.py --cache-dir database/system_data/factors_cache_phase0
"""
import argparse
import glob
import os
import sys

import pandas as pd

COL_TO_DROP = 'mkt_lpr_5y'


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache-dir', required=True)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.cache_dir, '*.parquet')))
    if not files:
        print('[ABORT] 缓存目录无 parquet')
        return 1
    print(f'共 {len(files)} 个 parquet，删除列: {COL_TO_DROP}')

    dropped = 0
    errors = 0
    for i, f in enumerate(files):
        try:
            df = pd.read_parquet(f)
            if COL_TO_DROP not in df.columns:
                continue
            df = df.drop(columns=[COL_TO_DROP])
            df.to_parquet(f, index=False)
            dropped += 1
            if (i + 1) % 1000 == 0:
                print(f'  进度 {i + 1}/{len(files)}，已删 {dropped} 个')
        except Exception as e:
            errors += 1
            print(f'[ERR] {os.path.basename(f)}: {e}')

    print(f'完成: 含该列已删 {dropped} 个，错误 {errors}，总 {len(files)}')
    return 0 if errors == 0 else 2


if __name__ == '__main__':
    sys.exit(main())

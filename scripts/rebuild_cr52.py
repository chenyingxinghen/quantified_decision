"""
删除 cr_52 数值爆炸的坏缓存文件，使 calculate_cr 的 clip 修复在下次训练时生效。

被删的 parquet 会在训练 Step 0（batch_update_factor_cache / _scan_one_cache）
中因"文件不存在"被判定为需更新，从而用修复后的代码重算。

只打印计数，不打印股票代码，避免 Windows 控制台编码错误。
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pyarrow.parquet as pq
from config.factor_config import TrainingConfig

THRESH = 500.0  # 与 calculate_cr 的 clip 上界对齐；>500 即为修复前写入的存量不一致缓存


def main():
    cache_dir = TrainingConfig.CACHE_DIR
    files = [f for f in os.listdir(cache_dir) if f.endswith('.parquet')]
    print(f"扫描 {len(files)} 个缓存文件...", flush=True)

    bad_files = []
    scanned = 0
    for f in files:
        path = os.path.join(cache_dir, f)
        try:
            names = pq.read_schema(path).names
            cr_cols = [c for c in names if c.startswith('cr_')]
            if not cr_cols:
                continue
            t = pq.read_table(path, columns=cr_cols).to_pandas()
            mx = np.nanmax(np.abs(t.values)) if t.size else 0.0
            if mx > THRESH:
                bad_files.append(path)
        except Exception:
            # 读不了的缓存也删，触发重算
            bad_files.append(path)
        scanned += 1
        if scanned % 500 == 0:
            print(f"  已扫描 {scanned}/{len(files)}, 累计坏值 {len(bad_files)}", flush=True)

    print(f"坏缓存文件: {len(bad_files)} 只", flush=True)

    deleted = 0
    for path in bad_files:
        try:
            os.remove(path)
            deleted += 1
        except Exception:
            pass
    print(f"已删除 {deleted} 个坏缓存，训练时将自动重算。", flush=True)


if __name__ == '__main__':
    main()

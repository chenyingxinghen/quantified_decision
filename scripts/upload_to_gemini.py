"""上传 database/ 到 Gemini 平台数据集（SFTP，可断点续传）。

前置条件
--------
1. 必须在能访问 gemini2.sufe.edu.cn 的网络下运行（上海财经大学校园网或 VPN）。
   该域名解析到内网地址 10.2.170.10，校外/非 VPN 环境连不通。
2. 先在平台「数据 -> 创建数据 -> 数据集」建好数据集，并在数据集详情页
   点「上传 -> SFTP 传输」拿到连接信息（地址/用户名/密码/端口）。
3. 安装依赖：pip install paramiko

凭据通过环境变量传入（不要写死在脚本里，避免误提交）：
    export GEMINI_SFTP_HOST=gemini2.sufe.edu.cn
    export GEMINI_SFTP_PORT=19003
    export GEMINI_SFTP_USER=<你的用户名>
    export GEMINI_SFTP_PASS=<你的密码>
    python scripts/upload_to_gemini.py

行为说明
--------
- 上传 database/ 的「内容」到远端 /upload（数据集根目录直接包含各 .db 文件）。
  这样在训练任务里 GEMINI_DATA_IN1=/gemini/data-1，代码无需改路径。
- 自动跳过临时文件：*.db-wal *.db-shm *.db-journal Thumbs.db *.tmp
- 断点续传：远端已存在且大小一致的同名文件会被跳过，可反复重跑补全。
- 大文件显示进度。
"""
import os
import sys
import time
import argparse

try:
    import paramiko
except ImportError:
    sys.exit("缺少 paramiko，请先执行: pip install paramiko")

# 项目根目录（scripts/ 的上一级）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EXCLUDE_SUFFIXES = (".db-wal", ".db-shm", ".db-journal", ".tmp")
EXCLUDE_NAMES = {"Thumbs.db"}
# 因子缓存是训练时自动重算的，且云端只读挂载用不到（会写到 $GEMINI_DATA_OUT），
# 上传既浪费带宽又占存储，直接跳过整个 factors_cache 目录。
EXCLUDE_DIRS = {"factors_cache"}
# raw-only 上传白名单：只传源库 jydb_raw.db（6.7GB）；其余（jydb_features.db /
# stock_daily.db / stock_meta.db / 因子缓存）均在云端由 build_intermediate_from_raw.py
# 与训练 Step0 重新产出。设为 None 则恢复上传 database/ 下全部（除排除项）。
WHITELIST = {"jydb_raw.db"}


def should_skip(name: str) -> bool:
    low = name.lower()
    if low in EXCLUDE_NAMES:
        return True
    return low.endswith(EXCLUDE_SUFFIXES)


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", default=os.path.join(PROJECT_ROOT, "database"),
                    help="本地待上传目录（默认 project_root/database）")
    ap.add_argument("--remote-base", default="/upload",
                    help="远端基目录（默认 /upload）")
    ap.add_argument("--dry-run", action="store_true", help="只打印将要上传的文件，不实际上传")
    args = ap.parse_args()

    host = os.getenv("GEMINI_SFTP_HOST", "gemini2.sufe.edu.cn")
    port = int(os.getenv("GEMINI_SFTP_PORT", "19003"))
    user = os.getenv("GEMINI_SFTP_USER")
    pwd = os.getenv("GEMINI_SFTP_PASS")
    if not user or not pwd:
        sys.exit("请在环境变量中设置 GEMINI_SFTP_USER 和 GEMINI_SFTP_PASS")

    local_base = os.path.abspath(args.local)
    if not os.path.isdir(local_base):
        sys.exit(f"本地目录不存在: {local_base}")

    # 收集待上传文件
    tasks = []  # (local_path, remote_path)
    total_bytes = 0
    for root, dirs, files in os.walk(local_base):
        # 剪掉不需要上传的目录（如 factors_cache）
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for f in files:
            if should_skip(f):
                continue
            if WHITELIST and f not in WHITELIST:
                continue
            lp = os.path.join(root, f)
            rel = os.path.relpath(lp, local_base)
            # 把 database/ 的内容直接放到 /upload 下（不带 database 前缀）
            rp = os.path.join(args.remote_base, rel).replace("\\", "/")
            tasks.append((lp, rp))
            total_bytes += os.path.getsize(lp)

    print(f"待上传文件数: {len(tasks)}，总大小: {human(total_bytes)}")
    print(f"本地: {local_base}")
    print(f"远端: {host}:{port}{args.remote_base}")
    if args.dry_run:
        for lp, rp in tasks:
            print(f"  {human(os.path.getsize(lp)):>10}  {rp}")
        return

    # 建立连接
    t0 = time.time()
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    print("正在连接 SFTP ...")
    client.connect(hostname=host, port=port, username=user, password=pwd, timeout=30)
    sftp = client.open_sftp()

    # 确保远端基目录存在
    def mkdir_p(path: str):
        parts = [p for p in path.split("/") if p]
        cur = ""
        for p in parts:
            cur = (cur + "/" + p) if cur else "/" + p
            try:
                sftp.stat(cur)
            except IOError:
                try:
                    sftp.mkdir(cur)
                except IOError:
                    pass

    mkdir_p(args.remote_base)

    uploaded = skipped = 0
    uploaded_bytes = 0
    for lp, rp in tasks:
        size = os.path.getsize(lp)
        # 断点续传：大小一致则跳过
        try:
            rstat = sftp.stat(rp)
            if rstat.st_size == size:
                print(f"[跳过] {rp} ({human(size)})")
                skipped += 1
                continue
        except IOError:
            pass
        # 创建远端目录
        rdir = os.path.dirname(rp)
        if rdir:
            mkdir_p(rdir)
        print(f"[上传] {rp} ({human(size)})")
        sftp.put(lp, rp)
        uploaded += 1
        uploaded_bytes += size

    sftp.close()
    client.close()
    dt = time.time() - t0
    print("\n=== 上传完成 ===")
    print(f"新增上传: {uploaded} 个文件 / {human(uploaded_bytes)}")
    print(f"已存在跳过: {skipped} 个文件")
    print(f"耗时: {dt/60:.1f} 分钟")
    print("回到平台数据集详情页点「确认完成」关闭传输通道。")


if __name__ == "__main__":
    main()

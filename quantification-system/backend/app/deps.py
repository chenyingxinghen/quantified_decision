"""
共享依赖 — 提供数据库路径、模型实例等可复用的 Depends
"""

import os
import sys
import sqlite3
from functools import lru_cache
from typing import Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import DATABASE_PATH
from config.strategy_config import ML_FACTOR_MODEL_PATH


def get_db_path() -> str:
    """返回 SQLite 数据库绝对路径"""
    return DATABASE_PATH


def get_db_connection():
    """创建一个新的 SQLite 连接（非线程共享）"""
    conn = sqlite3.connect(get_db_path(), timeout=30.0)
    conn.row_factory = sqlite3.Row
    return conn


def get_project_root() -> str:
    return PROJECT_ROOT


# ── User 专用数据库 (包含配置、会话、模拟盘持仓) ──────────────────────────
USER_DB_PATH = os.path.join(PROJECT_ROOT, "data", "user_data.db")


def get_user_db():
    """User SQLite 连接"""
    conn = sqlite3.connect(USER_DB_PATH, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def get_paper_db():
    """兼容性别名，后续可逐步移除"""
    return get_user_db()


def init_user_db():
    """初始化 user_data 数据库表"""
    # 如果旧的 paper_trading.db 存在且 user_data.db 不存在，则进行更名
    旧路径 = os.path.join(PROJECT_ROOT, "data", "paper_trading.db")
    if os.path.exists(旧路径) and not os.path.exists(USER_DB_PATH):
        try:
            os.rename(旧路径, USER_DB_PATH)
            print(f"✅ 已将 {旧路径} 重命名为 {USER_DB_PATH}")
        except Exception as e:
            print(f"⚠️ 重命名数据库失败: {e}")

    conn = get_user_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS positions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            code        TEXT NOT NULL,
            name        TEXT DEFAULT '',
            buy_date    TEXT NOT NULL,
            buy_price   REAL NOT NULL,
            quantity    INTEGER NOT NULL DEFAULT 1,
            status      TEXT NOT NULL DEFAULT 'active',   -- active / closed
            sell_date   TEXT,
            sell_price  REAL,
            sell_reason TEXT,
            profit_pct  REAL,
            notes       TEXT DEFAULT '',
            created_at  TEXT DEFAULT (datetime('now','localtime'))
        );
        
        CREATE TABLE IF NOT EXISTS users (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            username    TEXT UNIQUE NOT NULL,
            password    TEXT NOT NULL,
            source      TEXT DEFAULT '',
            created_at  TEXT DEFAULT (datetime('now','localtime'))
        );
        
        CREATE TABLE IF NOT EXISTS user_configs (
            username    TEXT PRIMARY KEY,
            config_json TEXT NOT NULL
        );
        
        CREATE TABLE IF NOT EXISTS sessions (
            token       TEXT PRIMARY KEY,
            username    TEXT NOT NULL,
            created_at  TEXT DEFAULT (datetime('now','localtime'))
        );
    """)
    # Add columns if not exists
    try:
        conn.execute("ALTER TABLE positions ADD COLUMN username TEXT DEFAULT 'guest'")
    except sqlite3.OperationalError:
        pass
    try:
        conn.execute("ALTER TABLE positions ADD COLUMN monitoring INTEGER DEFAULT 1")
    except sqlite3.OperationalError:
        pass
    
    conn.commit()
    conn.close()

# 启动时初始化
init_user_db()

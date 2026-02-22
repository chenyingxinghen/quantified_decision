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


# ── Paper‑trading 专用数据库 ──────────────────────────────────
PAPER_DB_PATH = os.path.join(PROJECT_ROOT, "data", "paper_trading.db")


def get_paper_db():
    """Paper‑trading SQLite 连接"""
    conn = sqlite3.connect(PAPER_DB_PATH, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_paper_db():
    """初始化 paper_trading 数据库表"""
    conn = get_paper_db()
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
    # Add username column if not exists
    try:
        conn.execute("ALTER TABLE positions ADD COLUMN username TEXT DEFAULT 'guest'")
    except sqlite3.OperationalError:
        pass  # Column already exists
    
    conn.commit()
    conn.close()

# 启动时初始化
init_paper_db()

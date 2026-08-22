"""
用户鉴权与配置中心
"""
import hashlib
import secrets
import uuid
import time
from typing import Optional
from fastapi import APIRouter, HTTPException, Request, Header
from pydantic import BaseModel

from app.deps import get_user_db

router = APIRouter(prefix="/api/auth", tags=["用户鉴权"])

# 口令哈希与会话参数
_PBKDF2_ROUNDS = 240_000
SESSION_TTL_DAYS = 30
MIN_PASSWORD_LEN = 8
MAX_USERNAME_LEN = 32

# ── 简单的请求频率限制 (Rate Limiting) ──
RATE_LIMIT_CACHE = {}

def check_rate_limit(client_ip: str, limit: int = 5, window: int = 60):
    now = time.time()
    if client_ip not in RATE_LIMIT_CACHE:
        RATE_LIMIT_CACHE[client_ip] = []
    
    timestamps = RATE_LIMIT_CACHE[client_ip]
    timestamps = [ts for ts in timestamps if now - ts < window]
    RATE_LIMIT_CACHE[client_ip] = timestamps
    
    if len(timestamps) >= limit:
        raise HTTPException(status_code=429, detail="请求过于频繁，请稍后再试")
    
    timestamps.append(now)

class RegisterRequest(BaseModel):
    username: str
    password: str
    source: str = "organic"

class LoginRequest(BaseModel):
    username: str
    password: str

class ConfigData(BaseModel):
    config_json: str

def hash_password(password: str) -> str:
    """PBKDF2-HMAC-SHA256，随机盐，格式 ``pbkdf2$<迭代数>$<盐hex>$<摘要hex>``。

    原实现是裸 `sha256(password)`：无盐、单轮。无盐意味着相同口令产生相同摘要，
    彩虹表直接命中；单轮意味着即使加盐，离线爆破也几乎不花代价。
    旧格式（64 位纯 hex）仍可校验，并在下次登录成功时自动升级（见 verify_password）。
    """
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _PBKDF2_ROUNDS)
    return f"pbkdf2${_PBKDF2_ROUNDS}${salt.hex()}${dk.hex()}"


def _is_legacy_hash(stored: str) -> bool:
    return len(stored) == 64 and not stored.startswith("pbkdf2$")


def verify_password(password: str, stored: str) -> bool:
    """校验口令，兼容旧的裸 sha256 记录。全程用 compare_digest 防时序侧信道。"""
    if not stored:
        return False
    if _is_legacy_hash(stored):
        return secrets.compare_digest(
            hashlib.sha256(password.encode()).hexdigest(), stored
        )
    try:
        scheme, rounds, salt_hex, digest_hex = stored.split("$")
        if scheme != "pbkdf2":
            return False
        dk = hashlib.pbkdf2_hmac(
            "sha256", password.encode(), bytes.fromhex(salt_hex), int(rounds)
        )
        return secrets.compare_digest(dk.hex(), digest_hex)
    except (ValueError, TypeError):
        return False


def get_current_user_from_token(token: Optional[str]) -> str:
    """token → username。无效 / 过期 / 未提供一律降级为 "guest"。

    注意 sessions 现在有有效期：原实现的 token 一旦签发**永不失效**，
    登出只删自己那一条，泄露的 token 可以一直用下去。
    """
    if not token or token in ("guest", "null", "undefined"):
        return "guest"
    conn = get_user_db()
    try:
        row = conn.execute(
            "SELECT username FROM sessions WHERE token = ? "
            "AND datetime(created_at) > datetime('now','localtime',?)",
            (token, f"-{SESSION_TTL_DAYS} days"),
        ).fetchone()
        if row:
            return row["username"]
        # 顺手清理过期会话，避免 sessions 表无限增长
        conn.execute(
            "DELETE FROM sessions WHERE datetime(created_at) <= "
            "datetime('now','localtime',?)",
            (f"-{SESSION_TTL_DAYS} days",),
        )
        conn.commit()
    finally:
        conn.close()
    return "guest"

@router.post("/register")
async def register(req: RegisterRequest, request: Request):
    client_ip = request.client.host if request.client else "127.0.0.1"
    check_rate_limit(client_ip, limit=5, window=60) # 限制注册频率

    username = (req.username or "").strip()
    if not username or len(username) > MAX_USERNAME_LEN:
        raise HTTPException(status_code=400, detail=f"用户名需为 1~{MAX_USERNAME_LEN} 个字符")
    if username in ("guest", "null", "undefined"):
        raise HTTPException(status_code=400, detail="该用户名为保留字")
    if len(req.password or "") < MIN_PASSWORD_LEN:
        raise HTTPException(status_code=400, detail=f"密码至少 {MIN_PASSWORD_LEN} 位")

    conn = get_user_db()
    try:
        row = conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
        if row:
            raise HTTPException(status_code=400, detail="用户名已存在")

        conn.execute("INSERT INTO users (username, password, source) VALUES (?, ?, ?)",
                     (username, hash_password(req.password), req.source))
        conn.commit()
    finally:
        conn.close()

    return {"message": "注册成功"}

@router.post("/login")
async def login(req: LoginRequest, request: Request):
    client_ip = request.client.host if request.client else "127.0.0.1"
    check_rate_limit(client_ip, limit=5, window=60) # 限制登录频率

    conn = get_user_db()
    try:
        row = conn.execute("SELECT password FROM users WHERE username = ?", (req.username,)).fetchone()

        if not row or not verify_password(req.password, row["password"]):
            raise HTTPException(status_code=401, detail="用户名或密码错误")

        # 登录成功且仍是旧的裸 sha256 记录 → 就地升级为 PBKDF2
        if _is_legacy_hash(row["password"]):
            conn.execute("UPDATE users SET password = ? WHERE username = ?",
                         (hash_password(req.password), req.username))

        token = secrets.token_urlsafe(32)
        conn.execute("INSERT INTO sessions (token, username) VALUES (?, ?)", (token, req.username))
        conn.commit()
    finally:
        conn.close()

    return {"token": token, "username": req.username, "message": "登录成功"}

@router.post("/logout")
async def logout(token: str = Header(None, alias="Token")):
    if token and token not in ("guest", "null", "undefined"):
        conn = get_user_db()
        try:
            conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
            conn.commit()
        finally:
            conn.close()
    return {"message": "已登出"}

@router.get("/user/info")
async def get_user_info(token: str = Header(None, alias="Token")):
    username = get_current_user_from_token(token)
    return {"username": username, "is_logged_in": username != "guest"}

@router.get("/config")
async def get_user_config(token: str = Header(None, alias="Token")):
    username = get_current_user_from_token(token)
    if username == "guest":
        return {"config_json": None}
        
    conn = get_user_db()
    try:
        row = conn.execute("SELECT config_json FROM user_configs WHERE username = ?", (username,)).fetchone()
        return {"config_json": row["config_json"] if row else None}
    finally:
        conn.close()

@router.post("/config")
async def save_user_config(req: ConfigData, token: str = Header(None, alias="Token")):
    username = get_current_user_from_token(token)
    if username == "guest":
        return {"message": "未登录无云端同步"}
        
    conn = get_user_db()
    try:
        conn.execute(
            "INSERT INTO user_configs (username, config_json) VALUES (?, ?) ON CONFLICT(username) DO UPDATE SET config_json=excluded.config_json",
            (username, req.config_json)
        )
        conn.commit()
    finally:
        conn.close()
    return {"message": "配置已同步至云端"}

"""
用户鉴权与配置中心
"""
import hashlib
import uuid
import time
from typing import Optional
from fastapi import APIRouter, HTTPException, Request, Header
from pydantic import BaseModel

from app.deps import get_paper_db

router = APIRouter(prefix="/api/auth", tags=["用户鉴权"])

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
    return hashlib.sha256(password.encode()).hexdigest()

def get_current_user_from_token(token: Optional[str]) -> str:
    if not token or token == "guest" or token == "null" or token == "undefined":
        return "guest"
    conn = get_paper_db()
    row = conn.execute("SELECT username FROM sessions WHERE token = ?", (token,)).fetchone()
    conn.close()
    if row:
        return row["username"]
    return "guest"

@router.post("/register")
async def register(req: RegisterRequest, request: Request):
    client_ip = request.client.host if request.client else "127.0.0.1"
    check_rate_limit(client_ip, limit=5, window=60) # 限制注册频率

    conn = get_paper_db()
    try:
        row = conn.execute("SELECT id FROM users WHERE username = ?", (req.username,)).fetchone()
        if row:
            raise HTTPException(status_code=400, detail="用户名已存在")
        
        hashed_pw = hash_password(req.password)
        conn.execute("INSERT INTO users (username, password, source) VALUES (?, ?, ?)",
                     (req.username, hashed_pw, req.source))
        conn.commit()
    finally:
        conn.close()
        
    return {"message": "注册成功"}

@router.post("/login")
async def login(req: LoginRequest, request: Request):
    client_ip = request.client.host if request.client else "127.0.0.1"
    check_rate_limit(client_ip, limit=5, window=60) # 限制登录频率
    
    conn = get_paper_db()
    try:
        row = conn.execute("SELECT password FROM users WHERE username = ?", (req.username,)).fetchone()
        
        if not row or row["password"] != hash_password(req.password):
            raise HTTPException(status_code=401, detail="用户名或密码错误")
            
        token = str(uuid.uuid4())
        conn.execute("INSERT INTO sessions (token, username) VALUES (?, ?)", (token, req.username))
        conn.commit()
    finally:
        conn.close()
    
    return {"token": token, "username": req.username, "message": "登录成功"}

@router.post("/logout")
async def logout(token: str = Header(None)):
    if token and token not in ("guest", "null", "undefined"):
        conn = get_paper_db()
        conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
        conn.commit()
        conn.close()
    return {"message": "已登出"}

@router.get("/user/info")
async def get_user_info(token: str = Header(None)):
    username = get_current_user_from_token(token)
    return {"username": username, "is_logged_in": username != "guest"}

@router.get("/config")
async def get_user_config(token: str = Header(None)):
    username = get_current_user_from_token(token)
    if username == "guest":
        return {"config_json": None}
        
    conn = get_paper_db()
    row = conn.execute("SELECT config_json FROM user_configs WHERE username = ?", (username,)).fetchone()
    conn.close()
    
    return {"config_json": row["config_json"] if row else None}

@router.post("/config")
async def save_user_config(req: ConfigData, token: str = Header(None)):
    username = get_current_user_from_token(token)
    if username == "guest":
        return {"message": "未登录无云端同步"}
        
    conn = get_paper_db()
    conn.execute(
        "INSERT INTO user_configs (username, config_json) VALUES (?, ?) ON CONFLICT(username) DO UPDATE SET config_json=excluded.config_json",
        (username, req.config_json)
    )
    conn.commit()
    conn.close()
    return {"message": "配置已同步至云端"}

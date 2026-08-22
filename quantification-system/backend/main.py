"""
量化决策系统 — FastAPI 后端入口
"""

import sys
import os

# 将原始项目根目录加入 sys.path，以便 import config / core / scripts 等模块
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
import os

# 配置日志
import logging
from logging.handlers import RotatingFileHandler
import time

# 禁用tqdm的进度条输出（如果导入的话）
try:
    from tqdm import auto as tqdm_lib
    # 通过环境变量禁用tqdm
    os.environ['TQDM_DISABLE'] = '1'
except ImportError:
    pass

# 创建日志目录
log_dir = os.path.join(PROJECT_ROOT, 'quantification-system','backend')
os.makedirs(log_dir, exist_ok=True)

# 配置日志格式和处理器
log_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 创建文件处理器，使用RotatingFileHandler避免日志文件过大
file_handler = RotatingFileHandler(
    os.path.join(log_dir, 'server.log'),
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5,
    encoding='utf-8'
)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(log_formatter)

# 配置根日志记录器
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.addHandler(file_handler)

# 添加控制台处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(log_formatter)
root_logger.addHandler(console_handler)

logger = logging.getLogger(__name__)

# 防止第三方库刷屏
logging.getLogger("numexpr.utils").setLevel(logging.WARNING)
logging.getLogger("apscheduler.scheduler").setLevel(logging.WARNING)

from app.scheduler import start_scheduler, stop_scheduler

from app.routers import stock_selector, paper_trading, analysis, config_center, auth, fundamentals, data_center

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("量化系统启动中...")
    start_scheduler()
    yield
    # Shutdown
    logger.info("量化系统关闭中...")
    stop_scheduler()

app = FastAPI(
    title="量化决策系统 API",
    description="Stock selection, paper trading, data center, technical analysis & configuration",
    version="1.0.0",
    lifespan=lifespan,
)

# 简单的全局请求频率限制 Middleware
import time
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self.cache = {}
        self.last_cleanup = time.time()
    
    async def dispatch(self, request: Request, call_next):
        # 排除静态资源和前端页面的限流，只限制 API 
        if request.url.path.startswith("/api/"):
            client_ip = request.client.host if request.client else "127.0.0.1"
            now = time.time()
            
            # 修复问题13：定期清理过期的 key，防止内存无限增长
            if now - self.last_cleanup > 300:  # 每5分钟清理一次
                expired_ips = [ip for ip, timestamps in self.cache.items() 
                              if not timestamps or now - timestamps[-1] > 300]
                for ip in expired_ips:
                    del self.cache[ip]
                self.last_cleanup = now
            
            if client_ip not in self.cache:
                self.cache[client_ip] = []
            
            timestamps = self.cache[client_ip]
            timestamps = [ts for ts in timestamps if now - ts < 60]
            self.cache[client_ip] = timestamps
            
            if len(timestamps) >= 120: # 稍微宽容点的 API 限制: 60秒内 120 次
                return JSONResponse(status_code=429, content={"detail": "API 请求过于频繁，请稍后再试"})
            
            timestamps.append(now)
            
        return await call_next(request)

app.add_middleware(RateLimitMiddleware)

# CORS —— 允许前端访问。
#
# 原先是 allow_origins=["*"] + allow_credentials=True：这个组合按 CORS 规范本身
# 就是非法的，更重要的是本服务的鉴权走自定义 `Token` 请求头而不是 Cookie，
# 通配放行等于允许**任意网站**的 JS 携带用户 token 调用本 API。
# 改为显式白名单；需要额外来源时用环境变量 QUANT_CORS_ORIGINS（逗号分隔）追加。
_default_origins = [
    "http://localhost:8083", "http://127.0.0.1:8083",
    "http://localhost:5173", "http://127.0.0.1:5173",   # vite dev server
]
_extra_origins = [
    o.strip() for o in os.environ.get("QUANT_CORS_ORIGINS", "").split(",") if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_default_origins + _extra_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由 (移除 /quant 前缀，由网关负责 namespace)
app.include_router(auth.router)
app.include_router(stock_selector.router) 
app.include_router(paper_trading.router)
app.include_router(analysis.router)
app.include_router(config_center.router)
app.include_router(fundamentals.router)
app.include_router(data_center.router)

# 托管前端静态文件
FRONTEND_DIST = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend", "dist"))

# 具体的静态资源路径映射 (优先处理)
if os.path.exists(FRONTEND_DIST):
    app.mount("/assets", StaticFiles(directory=os.path.join(FRONTEND_DIST, "assets")), name="quant_assets")

    # 处理根路径下的所有前端页面 (SPA Fallback)
    @app.get("/{path:path}")
    async def serve_quant_frontend(path: str):
        # 路径必须**收敛回** FRONTEND_DIST 内部。
        #
        # 原实现是 `os.path.join(FRONTEND_DIST, path)` 后直接 FileResponse。
        # os.path.join 遇到绝对路径会丢弃前缀（`join(dist, "G:/...")` → `G:/...`），
        # 加上 `../` 也能逐级上跳，于是这个 SPA 兜底路由等于一个任意文件下载接口：
        # 可以读到 config/trader.json（券商账号密码）、database/*.db、models/*.pkl。
        # 这里改为 realpath 归一化后校验公共前缀，越界一律回落到 index.html。
        index_html = os.path.join(FRONTEND_DIST, "index.html")
        if path:
            candidate = os.path.realpath(os.path.join(FRONTEND_DIST, path))
            root = os.path.realpath(FRONTEND_DIST)
            try:
                inside = os.path.commonpath([candidate, root]) == root
            except ValueError:      # 跨盘符（Windows）时 commonpath 抛异常 = 必然越界
                inside = False
            if inside and os.path.isfile(candidate):
                return FileResponse(candidate)
        # 默认为 index.html (SPA)
        return FileResponse(index_html)


@app.get("/api/health")
async def health_check():
    return {"status": "ok", "mode": "integrated"}

if __name__ == "__main__":
    import uvicorn
    print(f"量化系统(前后端整合版) 启动中，监听端口: 8083")
    uvicorn.run(app, host="127.0.0.1", port=8083)

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
import os

from app.routers import stock_selector, paper_trading, data_center, analysis, config_center, auth

app = FastAPI(
    title="量化决策系统 API",
    description="Stock selection, paper trading, data center, technical analysis & configuration",
    version="1.0.0",
)

# 简单的全局请求频率限制 Middleware
import time
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self.cache = {}
    
    async def dispatch(self, request: Request, call_next):
        # 排除静态资源和前端页面的限流，只限制 API 
        if request.url.path.startswith("/quant/api/"):
            client_ip = request.client.host if request.client else "127.0.0.1"
            now = time.time()
            if client_ip not in self.cache:
                self.cache[client_ip] = []
            
            timestamps = self.cache[client_ip]
            timestamps = [ts for ts in timestamps if now - ts < 60]
            self.cache[client_ip] = timestamps
            
            if len(timestamps) >= 120: # 稍微宽容点的 API 限制: 60秒内 300 次
                return JSONResponse(status_code=429, content={"detail": "API 请求过于频繁，请稍后再试"})
            
            timestamps.append(now)
            
        return await call_next(request)

app.add_middleware(RateLimitMiddleware)

# CORS —— 允许前端开发服务器访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由 (增加 /quant 前缀以匹配子系统架构)
app.include_router(auth.router, prefix="/quant")
app.include_router(stock_selector.router, prefix="/quant") 
app.include_router(paper_trading.router, prefix="/quant")
app.include_router(data_center.router, prefix="/quant")
app.include_router(analysis.router, prefix="/quant")
app.include_router(config_center.router, prefix="/quant")

# 托管前端静态文件
FRONTEND_DIST = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend", "dist"))

# 具体的静态资源路径映射 (优先处理)
if os.path.exists(FRONTEND_DIST):
    app.mount("/quant/assets", StaticFiles(directory=os.path.join(FRONTEND_DIST, "assets")), name="quant_assets")

    # 处理 /quant 路径下的所有前端页面 (SPA Fallback)
    @app.get("/quant/{path:path}")
    async def serve_quant_frontend(path: str):
        # 尝试直接查找文件
        file_path = os.path.join(FRONTEND_DIST, path)
        if path and os.path.isfile(file_path):
            return FileResponse(file_path)
        # 默认为 index.html (SPA)
        return FileResponse(os.path.join(FRONTEND_DIST, "index.html"))


@app.get("/quant/api/health")
async def health_check():
    return {"status": "ok", "mode": "integrated"}

if __name__ == "__main__":
    import uvicorn
    print(f"量化系统(前后端整合版) 启动中，监听端口: 8083")
    uvicorn.run(app, host="0.0.0.0", port=8083)

"""
选股与信号 API
"""

import os, sys, json, glob, traceback
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Query, Header
from pydantic import BaseModel

from app.deps import get_db_path, get_db_connection, get_project_root

# 后台任务状态
_selection_task = {"running": False, "progress": "", "items": [], "error": None, "file": None}

router = APIRouter(prefix="/api/stock-selector", tags=["选股与信号"])

# ── 智能加载模型 ──────────────────────────────────────────

def find_latest_model(base_dir: str) -> Optional[str]:
    import glob
    if not os.path.exists(base_dir): return None
    if os.path.isfile(base_dir) and base_dir.endswith('.pkl'): return base_dir
    
    # 寻找带 pkl 的子目录
    pkls = glob.glob(os.path.join(base_dir, "**", "*.pkl"), recursive=True)
    if not pkls: return None
    # 按修改时间排序
    pkls.sort(key=os.path.getmtime, reverse=True)
    return pkls[0]

def load_smart_model(model_path: str):
    from core.factors.ml_factor_model import MLFactorModel, EnsembleFactorModel
    
    if os.path.isdir(model_path):
        xgb_path = os.path.join(model_path, 'xgboost_factor_model.pkl')
        lgb_path = os.path.join(model_path, 'lightgbm_factor_model.pkl')
        if os.path.exists(xgb_path) and os.path.exists(lgb_path):
            m1 = MLFactorModel(model_type='xgboost')
            m1.load_model(xgb_path)
            m2 = MLFactorModel(model_type='lightgbm')
            m2.load_model(lgb_path)
            return EnsembleFactorModel(models=[m1, m2], weights=[0.5, 0.5])
        
        latest = find_latest_model(model_path)
        if latest: return load_smart_model(latest)
        return None

    if not os.path.exists(model_path): return None
    try:
        return EnsembleFactorModel.load_model(model_path)
    except Exception:
        m = MLFactorModel()
        m.load_model(model_path)
        return m

# ── 惰性加载重型模块 ────────────────────────────────────────
_model = None
_current_model_path = None
_factor_calculator = None
_candlestick = None


def _load_model(force_path: str = None):
    global _model, _current_model_path
    from config.strategy_config import ML_FACTOR_MODEL_PATH
    
    target_path = force_path or os.path.join(get_project_root(), "models", "mark")
    if not os.path.exists(target_path):
        target_path = os.path.join(get_project_root(), ML_FACTOR_MODEL_PATH)

    if _model is None or _current_model_path != target_path:
        _model = load_smart_model(target_path)
        _current_model_path = target_path
    return _model


# ── 模型列表 ──────────────────────────────────────────────

@router.get("/models")
async def list_available_models():
    """列出 models/mark 下的所有可用模型目录，并检测包含的模型类型"""
    mark_dir = os.path.join(get_project_root(), "models", "mark")
    if not os.path.exists(mark_dir):
        return {"models": []}
    
    models = []
    for d in os.listdir(mark_dir):
        path = os.path.join(mark_dir, d)
        if os.path.isdir(path):
            # 检测包含的模型类型
            types = []
            if os.path.exists(os.path.join(path, "xgboost_factor_model.pkl")):
                types.append("xgboost")
            if os.path.exists(os.path.join(path, "lightgbm_factor_model.pkl")):
                types.append("lgbm")
            
            # 如果没有标准命名的，检查是否有任何 pkl
            if not types:
                has_pkl = any(f.endswith(".pkl") for f in os.listdir(path))
                if has_pkl:
                    types.append("custom")

            if types:
                models.append({
                    "name": d,
                    "path": os.path.join("models", "mark", d),
                    "mtime": os.path.getmtime(path),
                    "types": types
                })
    
    # # 也包含 models/latest
    # latest_dir = os.path.join(get_project_root(), "models", "latest")
    # if os.path.exists(latest_dir):
    #     types = []
    #     if os.path.exists(os.path.join(latest_dir, "xgboost_factor_model.pkl")):
    #         types.append("xgboost")
    #     if os.path.exists(os.path.join(latest_dir, "lightgbm_factor_model.pkl")):
    #         types.append("lgbm")
        
    #     models.append({
    #         "name": "latest",
    #         "path": "models/latest",
    #         "mtime": os.path.getmtime(latest_dir),
    #         "types": types or ["custom"]
    #     })

    models.sort(key=lambda x: x["mtime"], reverse=True)
    return {"models": models}


# ── 选股结果 ───────────────────────────────────────────────
def _load_factor_calculator():
    global _factor_calculator
    if _factor_calculator is None:
        from core.factors.comprehensive_factor_calculator import ComprehensiveFactorCalculator
        _factor_calculator = ComprehensiveFactorCalculator(db_path=get_db_path())
    return _factor_calculator


def _load_candlestick():
    global _candlestick
    if _candlestick is None:
        from core.analysis.candlestick_patterns import CandlestickPatterns
        _candlestick = CandlestickPatterns()
    return _candlestick


# ── 选股结果 ───────────────────────────────────────────────

@router.get("/latest")
async def get_latest_selection():
    """获取最近一次选股结果 (CSV)"""
    result_dir = os.path.join(get_project_root(), "backtest_result")
    csvs = sorted(glob.glob(os.path.join(result_dir, "selected_stocks_*.csv")))
    if not csvs:
        return {"items": [], "file": None}

    import pandas as pd
    latest = csvs[-1]
    df = pd.read_csv(latest, dtype={'stock_code': str})
    for col in ['stock_code']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.zfill(6)
    return {
        "items": json.loads(df.to_json(orient="records", force_ascii=False)),
        "file": os.path.basename(latest),
    }


class RunSelectionRequest(BaseModel):
    top_n: int = 20
    apply_filter: bool = False
    min_confidence: float = 0
    model_path: Optional[str] = None
    model_types: List[str] = ["lgbm", "xgboost"]
    guest_config: Optional[str] = None  # 游客本地配置 JSON 字符串，用于基础筛选条件
    markets: Optional[List[str]] = None
    max_zcfzl: Optional[float] = None


@router.post("/run")
async def run_selection(req: RunSelectionRequest, token: Optional[str] = Header(None, alias="Token")):
    """异步执行选股流程"""
    global _selection_task
    if _selection_task["running"]:
        raise HTTPException(status_code=409, detail="已有选股任务在运行中")

    _selection_task = {
        "running": True, 
        "progress": "正在启动...", 
        "items": [], 
        "error": None,
        "file": None
    }

    import threading
    from scripts.select_stocks import select_stocks
    from config.strategy_config import ML_FACTOR_MODEL_PATH

    def _run(tk: Optional[str]):
        global _selection_task
        from datetime import datetime
        try:
            # 1. 确定基准目录
            if req.model_path:
                base_path = os.path.join(get_project_root(), req.model_path)
            else:
                base_path = os.path.join(get_project_root(), "models", "mark")
                if not os.path.exists(base_path):
                    base_path = os.path.join(get_project_root(), ML_FACTOR_MODEL_PATH)
            
            # 2. 获取用户配置作为覆盖
            user_filters = {}
            try:
                from app.routers.auth import get_current_user_from_token
                username = get_current_user_from_token(tk)
                if username != "guest":
                    # 登录用户：从数据库读取配置
                    from app.deps import get_user_db
                    conn_u = get_user_db()
                    conf_row = conn_u.execute(
                        "SELECT config_json FROM user_configs WHERE username = ?", (username,)
                    ).fetchone()
                    conn_u.close()
                    if conf_row and conf_row["config_json"]:
                        user_conf = json.loads(conf_row["config_json"])
                        user_filters = {
                            "min_market_cap": user_conf.get("MIN_MARKET_CAP"),
                            "max_pe":         user_conf.get("MAX_PE"),
                            "min_price":      user_conf.get("MIN_PRICE"),
                            "max_price":      user_conf.get("MAX_PRICE"),
                            "max_zcfzl":      user_conf.get("MAX_ZCFZL"),
                            "include_st":     user_conf.get("INCLUDE_ST"),
                            "apply_filter":   user_conf.get("ENABLE_FUNDAMENTAL_FILTER"),
                            "markets":        user_conf.get("SELECTOR_MARKETS"),
                        }
                        user_filters = {k: v for k, v in user_filters.items() if v is not None}
                else:
                    # 游客用户：从请求体携带的 guest_config 字段读取
                    if req.guest_config:
                        try:
                            guest_conf = json.loads(req.guest_config)
                            user_filters = {
                                "min_market_cap": guest_conf.get("MIN_MARKET_CAP"),
                                "max_pe":         guest_conf.get("MAX_PE"),
                                "min_price":      guest_conf.get("MIN_PRICE"),
                                "max_price":      guest_conf.get("MAX_PRICE"),
                                "max_zcfzl":      guest_conf.get("MAX_ZCFZL"),
                                "include_st":     guest_conf.get("INCLUDE_ST"),
                                "apply_filter":   guest_conf.get("ENABLE_FUNDAMENTAL_FILTER"),
                                "markets":        guest_conf.get("SELECTOR_MARKETS"),
                            }
                            user_filters = {k: v for k, v in user_filters.items() if v is not None}
                        except Exception as ge:
                            print(f"解析游客配置失败: {ge}")
            except Exception as e:
                print(f"Loading user config for selection failed: {e}")

            # 3. 收集需要运行的模型路径
            run_configs = []
            if os.path.isdir(base_path):
                if "xgboost" in req.model_types:
                    p = os.path.join(base_path, "xgboost_factor_model.pkl")
                    if os.path.exists(p): run_configs.append(("xgboost", p))
                if "lgbm" in req.model_types:
                    p = os.path.join(base_path, "lightgbm_factor_model.pkl")
                    if os.path.exists(p): run_configs.append(("lgbm", p))
                if not run_configs:
                    run_configs.append(("default", base_path))
            else:
                run_configs.append(("default", base_path))

            all_results = []
            executed_types = []
            for m_type, p in run_configs:
                _selection_task["progress"] = f"正在使用 {m_type} 模型进行选股..."
                executed_types.append(m_type)
                results = select_stocks(
                    model_path=p,
                    min_confidence=req.min_confidence,
                    top_n=req.top_n,
                    apply_filter=user_filters.get("apply_filter", req.apply_filter),
                    workers=4,
                    save_csv=False,
                    min_market_cap=user_filters.get("min_market_cap"),
                    max_pe=user_filters.get("max_pe"),
                    max_zcfzl=req.max_zcfzl or user_filters.get("max_zcfzl"),
                    min_price=user_filters.get("min_price"),
                    max_price=user_filters.get("max_price"),
                    include_st=user_filters.get("include_st"),
                    markets=req.markets or user_filters.get("markets"),
                )
                
                for r in results:
                    r["model_type"] = m_type
                all_results.extend(results)

            # 4. 汇总分析
            code_counts = {}
            for r in all_results:
                code = str(r.get("stock_code", "")).zfill(6)
                r["stock_code"] = code
                code_counts[code] = code_counts.get(code, 0) + 1
            
            for r in all_results:
                r["is_resonance"] = code_counts[r["stock_code"]] > 1

            final_results = sorted(all_results, key=lambda x: (x.get("model_type", ""), -x.get("confidence", 0.0)))

            import pandas as pd
            safe_results = []
            for r in final_results:
                item = {k: (v if isinstance(v, (int, float, str, bool)) or v is None else str(v)) for k, v in r.items()}
                safe_results.append(item)
            
            if safe_results:
                result_dir = os.path.join(get_project_root(), "backtest_result")
                os.makedirs(result_dir, exist_ok=True)
                types_str = "_".join(executed_types)
                csv_path = os.path.join(result_dir, f"selected_stocks_{types_str}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
                pd.DataFrame(safe_results).to_csv(csv_path, index=False, encoding='utf-8-sig')
                _selection_task["file"] = os.path.basename(csv_path)

            _selection_task["items"] = safe_results
            _selection_task["progress"] = "完成"

        except Exception as e:
            _selection_task["error"] = str(e)
            _selection_task["progress"] = "失败"
            traceback.print_exc()
        finally:
            _selection_task["running"] = False

    t = threading.Thread(target=_run, args=(token,), daemon=True)
    t.start()
    return {"message": "选股任务已启动"}


@router.get("/run-status")
async def get_selection_status():
    """获取异步选股任务的状态"""
    return _selection_task


# ── 因子看板 ───────────────────────────────────────────────

@router.get("/factors/{code}")
async def get_stock_factors(code: str, days: int = Query(default=250, ge=30, le=2000)):
    """获取单只股票的详细因子得分"""
    try:
        import pandas as pd
        conn = get_db_connection()
        query = f"""
            SELECT * FROM daily_data
            WHERE code = ? ORDER BY date DESC LIMIT ?
        """
        df = pd.read_sql_query(query, conn, params=(code, days))
        conn.close()
        if df.empty:
            raise HTTPException(status_code=404, detail=f"股票 {code} 无数据")

        df = df.sort_values("date").reset_index(drop=True)
        calc = _load_factor_calculator()
        factors = calc.calculate_all_factors(code, df, apply_feature_engineering=False)
        if factors is None or factors.empty:
            raise HTTPException(status_code=500, detail="因子计算失败")

        # 获取模型重要性
        importance = {}
        try:
            model = _load_model()
            if hasattr(model, 'feature_importance') and model.feature_importance:
                importance = model.feature_importance
            elif hasattr(model, 'models'):
                # 对集成模型的重要性进行汇总
                for m in model.models:
                    if hasattr(m, 'feature_importance'):
                        for f, v in m.feature_importance.items():
                            importance[f] = importance.get(f, 0) + v
        except Exception:
            pass

        # 取最后一行作为"最新因子快照"
        latest = factors.iloc[-1]
        factor_list = []
        for col in factors.columns:
            v = latest[col]
            try:
                import numpy as np
                val = float(v) if not (pd.isna(v) or np.isinf(v)) else None
            except:
                val = None
            
            factor_list.append({
                "name": col,
                "value": val,
                "importance": float(importance.get(col, 0))
            })

        # 按重要性排序
        factor_list.sort(key=lambda x: x['importance'], reverse=True)

        return {
            "code": code, 
            "factors": {item['name']: item['value'] for item in factor_list},
            "factor_details": factor_list, # 额外提供带排序和重要性的列表
            "factor_count": len(factor_list),
            "latest_date": str(df['date'].iloc[-1])
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ── K 线形态信号 ──────────────────────────────────────────

@router.get("/signals/{code}")
async def get_candlestick_signals(code: str, days: int = Query(default=100, ge=10, le=1000)):
    """获取 K 线形态买卖信号"""
    try:
        import pandas as pd
        conn = get_db_connection()
        df = pd.read_sql_query(
            "SELECT * FROM daily_data WHERE code = ? ORDER BY date DESC LIMIT ?",
            conn, params=(code, days),
        )
        conn.close()
        if df.empty:
            raise HTTPException(status_code=404, detail=f"股票 {code} 无数据")

        df = df.sort_values("date").reset_index(drop=True)
        cd = _load_candlestick()
        bullish = cd.detect_all_bullish_patterns(df)
        bearish = cd.detect_all_bearish_patterns(df)
        bullish_score = cd.get_total_bullish_score(df)
        bearish_score = cd.get_total_bearish_score(df)

        return {
            "code": code,
            "bullish_patterns": bullish,
            "bearish_patterns": bearish,
            "bullish_score": bullish_score,
            "bearish_score": bearish_score,
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ── 股票检索 (已迁移至 /api/fundamental/search) ──────────────────────────────────────────
# search 与 fundamental 端点已统一迁移至 app/routers/fundamentals.py

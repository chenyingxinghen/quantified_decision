"""
配置中心 API — 读写 factor_config / strategy_config
"""

import os, sys, re, traceback, importlib
from typing import Any, Dict
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.deps import get_project_root

router = APIRouter(prefix="/api/config", tags=["配置中心"])

# ── 配置文件路径 ──────────────────────────────────────────
FACTOR_CONFIG_PATH = os.path.join(get_project_root(), "config", "factor_config.py")
STRATEGY_CONFIG_PATH = os.path.join(get_project_root(), "config", "strategy_config.py")


def _parse_config_values(filepath: str) -> Dict[str, Any]:
    """从 Python 配置文件中解析顶层常量和类属性"""
    result = {}
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # 解析简单赋值行:  NAME = value  or  NAME = value  # comment
    # 也解析类体内的属性
    current_class = None
    for line in content.split("\n"):
        stripped = line.strip()

        # 检测 class 定义
        cls_match = re.match(r"^class\s+(\w+)", stripped)
        if cls_match:
            current_class = cls_match.group(1)
            continue

        # 跳过注释、空行、def
        if not stripped or stripped.startswith("#") or stripped.startswith("def ") \
                or stripped.startswith("@") or stripped.startswith("\"\"\"") \
                or stripped.startswith("'"):
            continue

        # 匹配赋值
        m = re.match(r"^([A-Z_][A-Z0-9_]*)\s*=\s*(.+?)(?:\s*#.*)?$", stripped)
        if m:
            name = m.group(1)
            raw_val = m.group(2).strip()
            key = f"{current_class}.{name}" if current_class else name
            # 尝试安全 eval
            try:
                val = eval(raw_val, {"__builtins__": {"True": True, "False": False, "None": None,
                                                       "float": float, "int": int, "range": range}})
            except Exception:
                val = raw_val
            result[key] = {"value": val, "raw": raw_val, "class": current_class}
    return result


def _update_config_file(filepath: str, updates: Dict[str, Any]) -> int:
    """
    更新配置文件中的常量值。
    updates: { "FactorConfig.RSI_PERIOD": 14, "ATR_PERIOD": 10, ... }
    返回成功更新的字段数。
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    count = 0
    for full_key, new_val in updates.items():
        # 提取纯变量名
        var_name = full_key.split(".")[-1]
        # 用正则替换赋值行
        pattern = rf"^(\s*{re.escape(var_name)}\s*=\s*)(.+?)(\s*#.*)?$"
        
        replacement_val = str(new_val)
        if isinstance(new_val, str):
            # 如果字符串本身看起来像是一个列表或字典（以 [ 或 { 开头结尾），则不加引号
            # 否则视作普通字符串，需要 repr() 加引号
            if not ((new_val.strip().startswith('[') and new_val.strip().endswith(']')) or 
                    (new_val.strip().startswith('{') and new_val.strip().endswith('}'))):
                replacement_val = repr(new_val)
        else:
            replacement_val = str(new_val)

        new_content, n = re.subn(
            pattern,
            lambda m: f"{m.group(1)}{replacement_val}{m.group(3) if m.group(3) else ''}",
            content,
            count=1,
            flags=re.MULTILINE,
        )
        if n > 0:
            content = new_content
            count += 1

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

    # 重新加载模块
    try:
        if "factor_config" in filepath:
            import config.factor_config
            importlib.reload(config.factor_config)
        elif "strategy_config" in filepath:
            import config.strategy_config
            importlib.reload(config.strategy_config)
    except Exception:
        pass

    return count


# ── factor_config ─────────────────────────────────────────

@router.get("/factor")
async def get_factor_config():
    """读取因子配置"""
    try:
        return {"config": _parse_config_values(FACTOR_CONFIG_PATH)}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


class ConfigUpdate(BaseModel):
    updates: Dict[str, Any]


@router.put("/factor")
async def update_factor_config(req: ConfigUpdate):
    """更新因子配置"""
    try:
        count = _update_config_file(FACTOR_CONFIG_PATH, req.updates)
        return {"updated": count, "message": f"已更新 {count} 个参数"}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ── strategy_config ───────────────────────────────────────

@router.get("/strategy")
async def get_strategy_config():
    """读取策略配置"""
    try:
        return {"config": _parse_config_values(STRATEGY_CONFIG_PATH)}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/strategy")
async def update_strategy_config(req: ConfigUpdate):
    """更新策略配置"""
    try:
        count = _update_config_file(STRATEGY_CONFIG_PATH, req.updates)
        return {"updated": count, "message": f"已更新 {count} 个参数"}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

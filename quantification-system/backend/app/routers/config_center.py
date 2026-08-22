"""
配置中心 API — 读写 factor_config / strategy_config
"""

import os, sys, re, traceback, importlib
from typing import Any, Dict
from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel

from app.deps import get_project_root
from app.routers.auth import get_current_user_from_token

router = APIRouter(prefix="/api/config", tags=["配置中心"])

# ── 配置文件路径 ──────────────────────────────────────────
FACTOR_CONFIG_PATH = os.path.join(get_project_root(), "config", "factor_config.py")
STRATEGY_CONFIG_PATH = os.path.join(get_project_root(), "config", "strategy_config.py")

# ── 写入侧的安全约束 ──────────────────────────────────────
#
# 这两个 PUT 端点会把值写进**真实的 .py 源文件**，而这些文件正是回测引擎和实盘
# 交易器启动时 import 的。原实现有两个问题，叠加起来是一个匿名远程代码执行：
#   1. 没有任何鉴权参数 —— 任何能访问到端口的人都能改；
#   2. `replacement_val = str(new_val)` 把用户输入原样拼进赋值行右侧，
#      传一个带换行的字符串就能往配置文件里追加任意 Python 语句，
#      下次 import config.strategy_config 时执行。
# 现在：必须登录；键必须是文件里**已存在**的大写常量；值必须是白名单标量/列表，
# 并且一律经 repr() 重新序列化（repr 会转义换行与引号，写不出可执行语句）。

_KEY_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
_MAX_STR_LEN = 128


def _safe_literal(value: Any, _depth: int = 0) -> Any:
    """只放行 bool/int/float/None/短字符串，以及它们组成的一层列表。"""
    if _depth > 1:
        raise ValueError("嵌套层级过深")
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("不接受 NaN / Inf")
        return value
    if isinstance(value, str):
        if len(value) > _MAX_STR_LEN:
            raise ValueError(f"字符串过长（>{_MAX_STR_LEN}）")
        if "\n" in value or "\r" in value:
            raise ValueError("字符串不得包含换行")
        return value
    if isinstance(value, (list, tuple)):
        return [_safe_literal(v, _depth + 1) for v in value]
    raise ValueError(f"不支持的值类型: {type(value).__name__}")


def _require_user(token: str) -> str:
    username = get_current_user_from_token(token)
    if username == "guest":
        raise HTTPException(status_code=401, detail="修改全局配置需要登录")
    return username



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

    安全约束见文件头：键必须是文件中已存在的大写常量，值必须通过 _safe_literal
    校验并由 repr() 重新序列化 —— 用户输入永远不会被原样拼进源码。
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    count = 0
    rejected = []
    for full_key, new_val in updates.items():
        var_name = full_key.split(".")[-1]

        if not _KEY_RE.match(var_name):
            rejected.append(f"{full_key}: 非法键名")
            continue
        try:
            safe_val = _safe_literal(new_val)
        except ValueError as e:
            rejected.append(f"{full_key}: {e}")
            continue

        # repr() 保证写出去的一定是一个合法且封闭的 Python 字面量
        replacement_val = repr(safe_val)

        pattern = rf"^(\s*{re.escape(var_name)}\s*=\s*)(.+?)(\s*#.*)?$"
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
        else:
            # 键在文件里不存在 —— 拒绝新增，避免通过"新常量"绕过任何审查
            rejected.append(f"{full_key}: 配置项不存在")

    if rejected:
        raise HTTPException(status_code=400, detail="；".join(rejected))

    # 写回前先确认整个文件仍是合法 Python，避免把配置写坏导致回测/实盘 import 失败
    import ast
    try:
        ast.parse(content)
    except SyntaxError as e:
        raise HTTPException(status_code=400, detail=f"更新后配置文件语法非法，已放弃写入: {e}")

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
async def update_factor_config(req: ConfigUpdate, token: str = Header(None, alias="Token")):
    """更新因子配置（需登录）"""
    _require_user(token)
    try:
        count = _update_config_file(FACTOR_CONFIG_PATH, req.updates)
        return {"updated": count, "message": f"已更新 {count} 个参数"}
    except HTTPException:
        raise
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
async def update_strategy_config(req: ConfigUpdate, token: str = Header(None, alias="Token")):
    """更新策略配置（需登录）

    ⚠ 这些参数会被回测引擎与实盘交易器直接 import（止损倍数、时间止损、持仓数、
    费率）。改动即刻对下一次调度生效，不存在灰度。
    """
    _require_user(token)
    try:
        count = _update_config_file(STRATEGY_CONFIG_PATH, req.updates)
        return {"updated": count, "message": f"已更新 {count} 个参数"}
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

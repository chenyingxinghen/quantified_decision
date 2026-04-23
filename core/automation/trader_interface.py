"""
easytrader 交易接口封装

负责与同花顺/各券商客户端进行 GUI 交互。
"""

import sys
import os
import time
import json
import logging
import pandas as pd
from typing import List, Dict, Optional, Any

# 添加项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from core.automation.easytrader_patch import RobustClientTrader, get_patched_trader
from config.automation_config import TRADER_TYPE, CONFIG_JSON_PATH, DRY_RUN

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(PROJECT_ROOT,'database','system_data','automation','logs', "trader.log"), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AutoTraderInterface")

class AutoTrader:
    """自动化交易接口包装类 (已集成 Easytrader 补丁)"""
    
    def __init__(self):
        self.user = None
        self.is_connected = False
        self.dry_run = DRY_RUN
        
        # 确保数据目录存在
        os.makedirs(os.path.join(PROJECT_ROOT, "data", "automation"), exist_ok=True)
        
    def wait_for_desktop(self, timeout_seconds: int = 120) -> bool:
        """
        等待交互式桌面恢复可用。
        改用 PostMessage 探测：PostMessage 不需要前台权限，
        能成功说明窗口消息队列可用；再尝试重连验证完整可用性。
        """
        import pywinauto
        import ctypes
        logger.info(f"等待桌面恢复，超时 {timeout_seconds}s ...")
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            try:
                windows = pywinauto.Desktop(backend="win32").windows()
                if not windows:
                    time.sleep(10)
                    continue
                # 用 PostMessage(WM_NULL=0) 探测任意可见窗口的消息队列是否响应
                # PostMessage 不依赖前台权限，比 GetCursorPos 更准确反映窗口可操作性
                WM_NULL = 0x0000
                probe_hwnd = windows[0].handle
                result = ctypes.windll.user32.PostMessageW(probe_hwnd, WM_NULL, 0, 0)
                if not result:
                    logger.debug("PostMessage 探测失败，桌面消息队列不可用，继续等待...")
                    time.sleep(10)
                    continue
                # 申请前台权限后尝试重连
                pid = ctypes.windll.kernel32.GetCurrentProcessId()
                ctypes.windll.user32.AllowSetForegroundWindow(pid)
                if self.connect():
                    time.sleep(2)
                    logger.info("桌面已恢复，客户端重连成功。")
                    return True
            except Exception as e:
                logger.debug(f"桌面探测: {e}")
            time.sleep(10)
        logger.error(f"桌面恢复超时 ({timeout_seconds}s)。")
        return False

    def connect(self):
        """连接交易客户端"""
        if self.dry_run:
            self.is_connected = True
            return True
            
        try:
            logger.info(f"正在连接 {TRADER_TYPE} 客户端...")
            self.user = get_patched_trader('ths') 
            exe_path = r'F:\同花顺\同花顺\xiadan.exe'
            self.user.connect(exe_path)
            self.is_connected = True
            logger.info("客户端连接成功。")
            return True
        except Exception as e:
            logger.error(f"客户端连接失败: {e}")
            self.is_connected = False
            return False

    def get_balance(self) -> Dict:
        """获取资金状况"""
        if self.dry_run:
            return {"可用金额": 1000000.0, "总资产": 1000000.0, "可用": 1000000.0}
            
        if not self.is_connected:
            if not self.connect(): return None
            
        try:
            res = self.user.balance
            if isinstance(res, list) and len(res) > 0:
                return res[0]
            return res
        except Exception as e:
            err_msg = str(e)
            if "no active desktop" in err_msg.lower() or "moving mouse cursor" in err_msg.lower():
                logger.error(f"获取资金失败：桌面不可交互，标记连接失效。")
                self.is_connected = False
                return None
            logger.error(f"获取资金失败: {e}")
            return None

    def get_positions(self, force_refresh: bool = False) -> List[Dict]:
        """
        获取当前持仓。
        返回语义：None=获取失败，[]=确认空仓，[...]=持仓列表
        force_refresh=True 时发送 F5 强制刷新 GUI 数据，仅在买卖后需要读取最新持仓时使用。
        """
        if self.dry_run:
            return []

        if not self.is_connected:
            if not self.connect(): return None

        try:
            if force_refresh:
                try:
                    self.user._switch_left_menus_by_shortcut("{F5}", sleep=1.5)
                except Exception:
                    pass

            positions = self.user.position

            # 空仓时做资金交叉验证，防止验证码阻挡导致误判
            if isinstance(positions, list) and len(positions) == 0:
                try:
                    balance = self.user.balance
                    if isinstance(balance, list) and len(balance) > 0:
                        balance = balance[0]
                    market_value = float(
                        balance.get('参考市值', balance.get('股票市值', balance.get('市值', 0))) or 0
                    )
                    if market_value > 0:
                        logger.warning(
                            f"持仓列表为空，但资金表显示股票市值={market_value}，"
                            "疑似 GUI 被验证码干扰，返回 None。"
                        )
                        return None
                except Exception as e:
                    logger.warning(f"持仓空仓交叉验证失败: {e}，保守返回 None。")
                    return None

            return positions
        except Exception as e:
            err_msg = str(e)
            if "no active desktop" in err_msg.lower() or "moving mouse cursor" in err_msg.lower():
                logger.error(f"获取持仓失败：桌面不可交互，标记连接失效。原始错误: {e}")
                self.is_connected = False
                return None
            logger.error(f"获取持仓失败: {e}")
            return None

    def buy(self, stock_code: str, amount: int, price: Optional[float] = None) -> Dict:
        """执行买入指令"""
        logger.info(f"买入: {stock_code} x{amount} @ {price or '市价'}")
        
        if self.dry_run:
            return {"status": "success", "msg": "dry_run", "entrust_no": "999999"}
            
        if not self.is_connected:
            if not self.connect(): return {"status": "error", "msg": "not_connected"}
            
        try:
            if price is None:
                return {"status": "error", "msg": "未提供价格，GUI 自动化需要明确价格。"}

            res = self.user.buy(stock_code, price=price, amount=amount)
            logger.debug(f"买入原始响应: {res}")
            if not res:
                return {"status": "error", "message": "trader returned empty result"}
            if isinstance(res, dict):
                if 'entrust_no' in res or res.get('status') == 'success':
                    return res
                return {"status": "error", "message": res.get('message', res.get('msg', 'unknown_fail'))}
            return {"status": "unknown", "message": str(res)}
        except Exception as e:
            logger.error(f"买入执行异常: {e}")
            return {"status": "error", "message": str(e)}

    def sell(self, stock_code: str, amount: int, price: Optional[float] = None) -> Dict:
        """执行卖出指令"""
        logger.info(f"卖出: {stock_code} x{amount} @ {price or '市价'}")
        
        if self.dry_run:
            return {"status": "success", "msg": "dry_run", "entrust_no": "888888"}
            
        if not self.is_connected:
            if not self.connect(): return {"status": "error", "msg": "not_connected"}
            
        try:
            if price is None:
                return {"status": "error", "msg": "未提供价格，GUI 自动化需要明确价格。"}

            res = self.user.sell(stock_code, price=price, amount=amount)
            logger.debug(f"卖出原始响应: {res}")
            if not res:
                return {"status": "error", "message": "trader returned empty result"}
            if isinstance(res, dict):
                if 'entrust_no' in res or res.get('status') == 'success':
                    return res
                return {"status": "error", "message": res.get('message', res.get('msg', 'unknown_fail'))}
            return {"status": "unknown", "message": str(res)}
        except Exception as e:
            logger.error(f"卖出执行异常: {e}")
            return {"status": "error", "message": str(e)}

    def sell_all(self, stock_code: str, price: Optional[float] = None) -> Dict:
        """全仓卖出某只股票"""
        positions = self.get_positions()
        if positions is None:
            logger.error(f"获取持仓失败，无法执行 {stock_code} 全仓卖出。")
            return {"status": "error", "msg": "fetch_positions_failed"}
            
        target = None
        for p in positions:
            p_code = p.get('证券代码', p.get('stock_code', ''))
            if stock_code in p_code or p_code in stock_code:
                target = p
                break
        
        if target:
            sell_price = price or target.get('当前价', target.get('last_price', target.get('现价')))
            try:
                if sell_price: sell_price = float(sell_price)
            except:
                sell_price = None

            try:
                amount = int(float(target.get('可用余额', target.get('可卖数量', 0)) or 0))
            except (ValueError, TypeError):
                amount = 0
                
            if amount > 0:
                return self.sell(stock_code, amount, price=sell_price)
            else:
                logger.warning(f"{stock_code} 可用持仓为 0，跳过卖出。")
                return {"status": "skipped", "msg": "zero_balance"}
        else:
            logger.warning(f"未找到 {stock_code} 的持仓记录。")
            return {"status": "skipped", "msg": "no_position"}

    def cancel_all(self):
        """撤销所有未成交委托"""
        if self.dry_run:
            return {"status": "success"}
        if not self.is_connected:
            if not self.connect(): return {"status": "error", "msg": "not_connected"}
        try:
            res = self.user.cancel_all()
            logger.info(f"撤单完成: {res}")
            return res
        except Exception as e:
            logger.error(f"撤单失败: {e}")
            return {"status": "error", "message": str(e)}

    def test_captcha(self) -> bool:
        """测试验证码识别填写"""
        if self.dry_run:
            return True
        if not self.is_connected:
            if not self.connect(): return False
        try:
            pos = self.get_positions()
            if pos is not None:
                logger.info("验证码测试通过，持仓获取成功。")
                return True
            else:
                logger.error("验证码测试失败：持仓获取返回 None。")
                return False
        except Exception as e:
            logger.error(f"验证码测试异常: {e}")
            return False

if __name__ == "__main__":
    # 简单的冒烟测试
    trader = AutoTrader()
    # 如果不是模拟模式，请谨慎运行以下代码
    if trader.dry_run:
        print("Dry Run 模式测试:")
        print("Balance:", trader.get_balance())
        print("Positions:", trader.get_positions())
        trader.buy("002397", 200, price=4.1)
    else:
        logger.info("准备进行真实交易测试...")
        if trader.connect():
            # 尝试先获取资金，验证连接质量
            balance = trader.get_balance()
            logger.info(f"当前资金状况: {balance}")
            # trader.buy("002397", 200, price=4.1)
            # 测试验证码识别
            # trader.test_captcha()

            print(trader.get_positions())
        else:
            logger.error("连接测试失败")

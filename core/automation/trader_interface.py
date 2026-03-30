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
        
    def connect(self):
        """连接交易客户端"""
        if self.dry_run:
            logger.info("模拟模式 (Dry Run) 已启用，跳过真实连接步骤。")
            self.is_connected = True
            return True
            
        try:
            logger.info(f"正在启动 {TRADER_TYPE} 交易客户端并连接...")
            
            # 使用补丁后的 RobustClientTrader
            self.user = get_patched_trader('ths') 
            
            # 这里的路径通常在配置文件或硬编码。
            exe_path = r'F:\同花顺\同花顺\xiadan.exe'
            logger.info(f"连接路径: {exe_path}")
            
            self.user.connect(exe_path)
            self.is_connected = True
            logger.info("交易客户端连接成功！")
            return True
        except Exception as e:
            logger.error(f"连接交易客户端失败: {e}")
            self.is_connected = False
            return False

    def get_balance(self) -> Dict:
        """获取资金状况"""
        if self.dry_run:
            return {"可用金额": 1000000.0, "总资产": 1000000.0, "可用": 1000000.0}
            
        if not self.is_connected:
            if not self.connect(): return {}
            
        try:
            # easytrader 的 balance 返回通常是列表
            res = self.user.balance
            if isinstance(res, list) and len(res) > 0:
                return res[0]
            return res
        except Exception as e:
            logger.error(f"获取资金失败: {e}", exc_info=True)
            return {}

    def get_positions(self) -> List[Dict]:
        """
        获取当前持仓。
        返回语义：
          - None  : 获取失败（GUI 异常、连接断开等）
          - []    : 确认空仓（经资金交叉验证）
          - [...]  : 正常持仓列表
        """
        if self.dry_run:
            return []

        if not self.is_connected:
            if not self.connect(): return None

        try:
            # 先切换到撤单页再切回，确保触发完整的 WMCopy 流程（含验证码检测）
            try:
                self.user._switch_left_menus(['撤单[F3]'])
                time.sleep(0.5)
            except Exception:
                pass

            positions = self.user.position

            # 空列表时做资金交叉验证，防止验证码阻挡导致误判为空仓
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
                            "疑似 GUI 读取被干扰（验证码未处理），返回 None 以触发上层保护。"
                        )
                        return None
                except Exception as e:
                    logger.warning(f"持仓空列表交叉验证资金时出错: {e}，保守返回 None。")
                    return None

            return positions
        except Exception as e:
            logger.error(f"获取持仓失败: {e}")
            return None

    def buy(self, stock_code: str, amount: int, price: Optional[float] = None) -> Dict:
        """执行买入指令"""
        logger.info(f"尝试买入: {stock_code}, 数量: {amount}, 价格: {price or '市价'}")
        
        if self.dry_run:
            logger.info(f"模拟买入成功: {stock_code}, {amount}股")
            return {"status": "success", "msg": "dry_run", "entrust_no": "999999"}
            
        if not self.is_connected:
            if not self.connect(): return {"status": "error", "msg": "not_connected"}
            
        try:
            # 对于 GUI 自动化（如同花顺），通常需要填写具体价格。
            # 如果 price 为 None，easytrader 内部格式化可能会报 NoneType 错误。
            if price is None:
                err_msg = "买入失败: 未提供价格。GUI 自动化建议提供具体价格（如涨停价或最新价）。"
                logger.error(err_msg)
                return {"status": "error", "msg": err_msg}

            # # 截图保存，用于调试
            # try:
            #     shot_path = os.path.join(PROJECT_ROOT, "logs", f"buy_{stock_code}_after.png")
            #     self.user._main.capture_as_image().save(shot_path)
            #     logger.info(f"已保存买入后截图: {shot_path}")
            # except:
            #     pass
                
            res = self.user.buy(stock_code, price=price, amount=amount)
            logger.info(f"买入响应: {res}")
            if not res:
                return {"status": "error", "message": "trader returned empty result"}
            if isinstance(res, dict):
                if 'entrust_no' in res or res.get('status') == 'success':
                    return res
                return {"status": "error", "message": res.get('message', res.get('msg', 'unknown_fail'))}
            return {"status": "unknown", "message": str(res)}
        except Exception as e:
            logger.error(f"买入执行失败: {e}")
            return {"status": "error", "message": str(e)}

    def sell(self, stock_code: str, amount: int, price: Optional[float] = None) -> Dict:
        """执行卖出指令"""
        logger.info(f"尝试卖出: {stock_code}, 数量: {amount}, 价格: {price or '市价'}")
        
        if self.dry_run:
            logger.info(f"模拟卖出成功: {stock_code}, {amount}股")
            return {"status": "success", "msg": "dry_run", "entrust_no": "888888"}
            
        if not self.is_connected:
            if not self.connect(): return {"status": "error", "msg": "not_connected"}
            
        try:
            if price is None:
                err_msg = "卖出失败: 未提供价格。GUI 自动化建议提供具体价格（如跌停价或最新价）。"
                logger.error(err_msg)
                return {"status": "error", "msg": err_msg}

            res = self.user.sell(stock_code, price=price, amount=amount)
            logger.info(f"卖出响应: {res}")
            if not res:
                return {"status": "error", "message": "trader returned empty result"}
            if isinstance(res, dict):
                if 'entrust_no' in res or res.get('status') == 'success':
                    return res
                return {"status": "error", "message": res.get('message', res.get('msg', 'unknown_fail'))}
            return {"status": "unknown", "message": str(res)}
        except Exception as e:
            logger.error(f"卖出执行失败: {e}")
            return {"status": "error", "message": str(e)}

    def sell_all(self, stock_code: str, price: Optional[float] = None) -> Dict:
        """全仓卖出某只股票"""
        positions = self.get_positions()
        if positions is None:
            logger.error(f"  获取持仓失败，无法执行 {stock_code} 的全仓卖出。")
            return {"status": "error", "msg": "fetch_positions_failed"}
            
        # 匹配代码 (部分券商代码带后缀，部分不带)
        target = None
        for p in positions:
            p_code = p.get('证券代码', p.get('stock_code', ''))
            if stock_code in p_code or p_code in stock_code:
                target = p
                break
        
        if target:
            # 优先使用传入价格，若无则尝试从持仓中获取当前价
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
                logger.warning(f"{stock_code} 可用持仓为 0，无法卖出。")
                return {"status": "skipped", "msg": "zero_balance"}
        else:
            logger.warning(f"未找到 {stock_code} 的持仓记录。")
            return {"status": "skipped", "msg": "no_position"}

    def cancel_all(self):
        """撤销所有未成交委托"""
        if self.dry_run:
            logger.info("模拟模式：已模拟撤单。")
            return {"status": "success"}
        if not self.is_connected:
            if not self.connect(): return {"status": "error", "msg": "not_connected"}
        try:
            res = self.user.cancel_all()
            logger.info(f"撤单响应: {res}")
            return res
        except Exception as e:
            logger.error(f"撤单失败: {e}")
            return {"status": "error", "message": str(e)}

    def test_captcha(self) -> bool:
        """测试验证码识别填写"""
        logger.info("开始测试验证码识别填写流程...")
        if self.dry_run:
            logger.info("模拟模式下不测试真实验证码。")
            return True
            
        if not self.is_connected:
            if not self.connect(): return False
            
        try:
            logger.info("尝试触发持仓查询以测试验证码识别...")
            pos = self.get_positions()
            if pos is not None:
                logger.info("获取持仓成功！(如遇到验证码，应已自动填好)")
                return True
            else:
                logger.error("验证码测试失败：未能获取到持仓。")
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

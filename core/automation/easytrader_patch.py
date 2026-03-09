import sys
import os
import re
import logging
import time
import pytesseract
import easytrader
import easyutils
from easytrader.clienttrader import ClientTrader

logger = logging.getLogger("EasytraderPatch")

# --- OCR System Config: Set Tesseract Path ---
TESSERACT_PATH = r"F:\Tesseract-OCR"
TESSERACT_EXE = os.path.join(TESSERACT_PATH, "tesseract.exe")

if os.path.exists(TESSERACT_EXE):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE
    # 同时加入系统环境变量供子进程调用 (easytrader 某些策略可能直接调用命令行)
    if TESSERACT_PATH not in os.environ["PATH"]:
        os.environ["PATH"] += os.pathsep + TESSERACT_PATH
    logger.info(f"Tesseract OCR path set to: {TESSERACT_EXE}")
else:
    logger.error(f"Target Tesseract path NOT found: {TESSERACT_EXE}")

# --- Fix 0: Monkey-patch easytrader's captcha recognition ---
import easytrader.utils.captcha as easy_captcha
from PIL import Image, ImageFilter

def improved_captcha_recognize(img_path):
    """
    改进的验证码识别：增加图像预处理以应对同花顺安全校验验证码
    """
    try:
        # 1. 载入并转换为灰度图
        # im = Image.open(img_path).convert("L")
        im = Image.open(img_path)
        
        # 5. 调用 Tesseract，限定只识别数字并使用单行模式
        config = '--psm 7 -c tessedit_char_whitelist=0123456789'
        num = pytesseract.image_to_string(im, config=config).strip()
        
        # 6. 过滤非数字
        import re
        num = "".join(re.findall(r'\d', num))
        
        logger.info(f"Improved OCR result for {img_path}: [{num}]")
        return num
    except Exception as e:
        logger.error(f"OCR patch evaluation failed: {e}")
        # 降级回原始逻辑（如果可能）或者返回空
        try:
            return pytesseract.image_to_string(Image.open(img_path)).strip()
        except:
            return ""

# 执行补丁
easy_captcha.captcha_recognize = improved_captcha_recognize

# --- Fix 0.1: Monkey-patch Copy strategy for better CAPTCHA interaction ---
import easytrader.grid_strategies as easy_strategies
import pywinauto.keyboard

def patched_get_clipboard_data(self):
    """
    改进的验证码交互逻辑：增加输入延迟，提升填入成功率
    """
    if easy_strategies.Copy._need_captcha_reg:
        top = self._trader.app.top_window()
        # 增加识别“验证码”关键字的鲁棒性，使用 child_window + found_index 避免歧义
        if top.child_window(class_name="Static", title_re=".*验证码.*", found_index=0).exists(timeout=1):
            file_path = "logs/tmp.png"
            count = 5
            found = False
            while count > 0:
                # 截图前微调：确保窗口在前台
                top.set_focus()
                
                # 截取验证码图片 (ID 0x965 是同花顺验证码图片的标准 ID)
                # 使用 child_window 配合 control_id 更准确
                top.child_window(control_id=0x965, class_name="Static").capture_as_image().save(file_path)

                captcha_num = easy_captcha.captcha_recognize(file_path).strip()
                captcha_num = "".join(captcha_num.split())
                
                if len(captcha_num) == 4:
                    logger.info(f"Trying CAPTCHA: [{captcha_num}] (Attempt {6-count}/5)")
                    
                    edit_ctrl = top.child_window(control_id=0x964, class_name="Edit")
                    edit_ctrl.set_focus()
                    
                    # 关键改进：改用 type_keys 模拟真实录入，并增加录入前后的等待
                    edit_ctrl.type_keys("{BACKSPACE 10}{DELETE 10}", pause=0.01)
                    edit_ctrl.type_keys(captcha_num, with_spaces=True, pause=0.1)
                    
                    time.sleep(0.3) # 给 GUI 一点反应时间
                    top.type_keys("{ENTER}") # 使用窗口对象的 type_keys 比全局 SendKeys 更稳
                    
                    # 等待窗体消失或报错
                    time.sleep(1.0) # 稍微增加等待时间
                    try:
                        if not top.exists(timeout=0.1):
                            logger.info("CAPTCHA window disappeared, assuming success.")
                            found = True
                            break
                    except Exception:
                        found = True
                        break
                        
                count -= 1
                if count > 0:
                    # 点击图片刷新验证码
                    logger.info("CAPTCHA failed, refreshing...")
                    try:
                        top.child_window(control_id=0x965, class_name="Static").click()
                        time.sleep(0.5) # 等待刷新
                    except:
                        pass
                    
            if not found:
                try: top.child_window(title="取消", class_name="Button").click()
                except: pass
        else:
            easy_strategies.Copy._need_captcha_reg = False
            
    # 原有的剪切板获取逻辑
    count = 5
    while count > 0:
        try:
            import pywinauto.clipboard
            return pywinauto.clipboard.GetData()
        except Exception as e:
            count -= 1
            time.sleep(0.2)
    return ""

# 执行补丁
easy_strategies.Copy._get_clipboard_data = patched_get_clipboard_data

# --- Fix 1: Monkey-patch easyutils to prevent NoneType format error ---
# This fixes the "unsupported format string passed to NoneType.__format__" error
_original_round_price = easyutils.round_price_by_code

def patched_round_price_by_code(price, code):
    if price is None:
        logger.warning(f"Detected None price for stock {code}, defaulting to '0.00'")
        return "0.00"
    try:
        return _original_round_price(price, code)
    except Exception as e:
        logger.error(f"Error rounding price {price} for {code}: {e}")
        return str(price)

easyutils.round_price_by_code = patched_round_price_by_code

from easytrader import pop_dialog_handler

# --- Fix 3: More Robust Pop-up Handler to capture entrust_no ---
class PatchedTradePopDialogHandler(pop_dialog_handler.TradePopDialogHandler):
    """
    增强版弹窗处理器：使用更通用的方式抓取标题和内容。
    """
    def _extract_content(self):
        """深度扫描弹窗内的所有文本控件"""
        try:
            # 尝试直接通过 top_window 获取，避免 pywinauto 的 app.Dialog 魔术方法崩溃
            top = self._app.top_window()
            # 确保我们真的在看一个弹窗，而不是主窗体
            # 注意：PatchedTradePopDialogHandler 实例创建时 app 是一样的，但我们需要确保我们没在扫主窗体
            
            elements = top.descendants()
            # logger.info(f"扫到 {len(elements)} 个候选控件")
            
            extracted_texts = []
            for e in elements:
                try:
                    txt = e.window_text()
                    cls = e.class_name()
                    if txt and len(txt) > 0 and len(txt) < 100: # 过滤掉超长文本（通常是主窗体上的数据列表）
                        if txt not in ["确定", "取消", "是", "否", "OK", "Cancel", "确认", "点击激活"]:
                            extracted_texts.append(txt)
                except:
                    continue
            
            content = " ".join(extracted_texts) if extracted_texts else top.window_text()
            return content
        except Exception as e:
            logger.error(f"提取内容失败: {e}")
            return ""

    def handle(self, title):
        # 如果标题为空，尝试从窗口本身获取
        if not title:
            try: title = self._app.top_window().window_text()
            except: title = "未知弹窗"
            
        logger.info(f"正在处理弹窗: [{title}]")
        content = self._extract_content()
        logger.info(f"内容详情: {content}")

        # 核心修复：处理“成功”但在不同标题下的情况
        if ("提示" in title or "信息" in title) and "成功" in content:
            try:
                entrust_no = self._extract_entrust_id(content)
                logger.info(f"捕获到委托单号: {entrust_no}")
                self._submit_by_click()
                return {"entrust_no": entrust_no, "message": "success"}
            except:
                logger.warning("匹配单号失败，仅返回成功状态")

        return super().handle(title)

class RobustClientTrader(ClientTrader):
    """
    增强型交易客户端：修正窗口锁定逻辑，增强弹窗检测。避开拷贝检测。
    """
    from easytrader import grid_strategies
    # 强制尝试 WMCopy，因为它包含了验证码自动填入逻辑
    # 如果 WMCopy 经常报验证码错误，可能是因为点击提交太快
    grid_strategy = grid_strategies.WMCopy

    def __init__(self, window_title_re=None):
        super().__init__()
        self._window_title_re = window_title_re or r".*股票交易.*|.*模拟炒股.*|同花顺.*"

    def connect(self, exe_path=None, **kwargs):
        import pywinauto
        connect_path = exe_path or self._config.DEFAULT_EXE_PATH
        os.startfile(connect_path)
        time.sleep(5)
        logger.info(f"尝试连接: {connect_path}")
        
        try:
            self._app = pywinauto.Application().connect(path=connect_path, timeout=10)
            
            # 1. 寻找可见的且符合特征的主窗体
            self._main = None
            all_windows = self._app.windows()
            
            visible_windows = [w for w in all_windows if w.is_visible()]
            logger.info(f"可见窗口列表: {[w.window_text() for w in visible_windows]}")
            
            # 优先匹配有标题的可见 Afx 窗口
            for window in visible_windows:
                title = window.window_text()
                cls = window.class_name()
                if "Afx" in cls and title and re.search(self._window_title_re, title):
                    self._main = self._app.window(handle=window.handle)
                    logger.info(f"命中目标主窗体: [{title}] (Class: {cls})")
                    break
            
            # 兜底：如果没找到精确匹配，找任意一个有标题的可见窗口
            if self._main is None:
                for window in visible_windows:
                    title = window.window_text()
                    if title and re.search(self._window_title_re, title) and "Visual Studio" not in title:
                        self._main = self._app.window(handle=window.handle)
                        logger.info(f"模糊匹配到可见窗体: [{title}]")
                        break
            
            if self._main is None: 
                self._main = self._app.top_window()
                logger.warning(f"最终兜底使用 top_window: {self._main.window_text()}")

            logger.info(f"最终绑定窗体: [{self._main.window_text()}] (Handle: {self._main.handle})")
            
            # 手动激活主窗体
            try:
                self._main.set_focus()
            except:
                pass
                
            try: self._init_toolbar()
            except: logger.warning("工具栏初始化失败")
        except Exception as e:
            logger.error(f"连接失败: {e}", exc_info=True)
            raise

    def is_exist_pop_dialog(self):
        """增强版弹窗识别：通过 handle, class, size 多重校验"""
        self.wait(0.5)
        try:
            top = self._app.top_window()
            top_handle = top.handle
            main_handle = self._main.handle
            cls = top.class_name()
            title = top.window_text()
            
            # 如果 handle 相同，绝对不是弹窗
            if top_handle == main_handle:
                return False
                
            # 获取窗口尺寸
            rect = top.rectangle()
            width = rect.width()
            height = rect.height()
            
            # 主窗体通常很大 (如 1024x768+)，交易弹窗通常很小 (如 200x150, 400x300)
            # 如果宽度大于主窗体的 80% 且高度大于 80%，极有可能是 misidentify 了主窗体，或者是半透明通知
            # (这里设置一个保守阈值)
            if width > 800 and height > 500:
                logger.debug(f"跳过大尺寸疑似主窗体: [{title}] {width}x{height}")
                return False

            # 特征识别
            # 1. 类名如果是 #32770 (标准对话框)
            if cls == "#32770":
                logger.info(f"侦测到标准对话框: [{title}]")
                return True
                
            # 2. 标题包含关键字
            if any(k in title for k in ["提示", "确认", "下单", "警告", "信息"]):
                logger.info(f"侦测到关键字弹窗: [{title}] (Class: {cls})")
                return True
                
            # 3. 如果是 Afx 窗口但尺寸很小且 handle 不一样
            if "Afx" in cls and width < 600 and height < 400:
                logger.info(f"侦测到疑似小型 Afx 弹窗: [{title}] ({width}x{height})")
                return True

            return False
        except Exception as e:
            logger.debug(f"弹窗检测异常: {e}")
            return False

    def _fill_stock_code(self, stock_code):
        logger.info(f"填写代码: {stock_code}")
        # 兼容性处理：如果 _config 没这些属性，使用同花顺标准 ID
        control_id = getattr(self._config, 'TRADE_STOCK_CONTROL_ID', 1032)
        edit = self._main.child_window(control_id=control_id)
        edit.set_focus()
        edit.click_input()
        try: edit.set_edit_text("")
        except: pass
        edit.type_keys("{BACKSPACE 20}{DELETE 20}", pause=0.01)
        edit.type_keys(stock_code, pause=0.1)
        edit.type_keys("{TAB}")

    def _fill_price(self, price):
        logger.info(f"填写价格: {price}")
        control_id = getattr(self._config, 'TRADE_PRICE_CONTROL_ID', 1033)
        edit = self._main.child_window(control_id=control_id)
        edit.set_focus()
        edit.click_input()
        try: edit.set_edit_text("")
        except: pass
        edit.type_keys("{BACKSPACE 20}{DELETE 20}", pause=0.01)
        edit.type_keys(str(price), pause=0.1)
        edit.type_keys("{TAB}")

    def _fill_amount(self, amount):
        logger.info(f"填写数量: {amount}")
        control_id = getattr(self._config, 'TRADE_AMOUNT_CONTROL_ID', 1034)
        edit = self._main.child_window(control_id=control_id)
        edit.set_focus()
        edit.click_input()
        try: edit.set_edit_text("")
        except: pass
        edit.type_keys("{BACKSPACE 20}{DELETE 20}", pause=0.01)
        edit.type_keys(str(amount), pause=0.1)
        edit.type_keys("{TAB}")

    def buy(self, stock_code, price, amount):
        logger.info(f"==> 执行定制买入流程: {stock_code} {price} {amount}")
        self._switch_left_menus(['买入[F1]'])
        return self._do_trade(stock_code, price, amount)

    def sell(self, stock_code, price, amount):
        logger.info(f"==> 执行定制卖出流程: {stock_code} {price} {amount}")
        self._switch_left_menus(['卖出[F2]'])
        return self._do_trade(stock_code, price, amount)

    def _do_trade(self, stock_code, price, amount):
        self.wait(1.0) 
        self._main.set_focus()
        
        self._fill_stock_code(stock_code)
        self.wait(0.5)
        self._fill_price(price)
        self.wait(0.5)
        self._fill_amount(amount)
        self.wait(0.5)
        
        self._main.set_focus()
        self._submit_trade()
        
        return self._handle_pop_dialogs(PatchedTradePopDialogHandler)

    def _submit_trade(self):
        """增强提交：点击 + 回车补偿"""
        logger.info("正在尝试点击 [提交] 按钮...")
        try:
            self._main.set_focus()
            control_id = getattr(self._config, 'TRADE_SUBMIT_CONTROL_ID', 1006)
            btn = self._main.child_window(control_id=control_id)
            try: btn.set_focus()
            except: pass
            
            btn.click_input()
            self.wait(1.0) 
            
            if not self.is_exist_pop_dialog():
                logger.info("点击后未见弹窗，尝试通过发送 {ENTER} 键补发提交...")
                self._main.set_focus()
                self._main.type_keys("{ENTER}")
                self.wait(1.0)
        except Exception as e:
            logger.warning(f"提交操作异常: {e}")
            self._main.type_keys("{ENTER}")

    def _get_pop_dialog_title(self):
        """覆盖标题获取逻辑：优先从窗口标题栏读取"""
        try:
            top = self._app.top_window()
            title = top.window_text()
            if title: return title
            return super()._get_pop_dialog_title()
        except:
            return "提示"

    def _handle_pop_dialogs(self, handler_class=pop_dialog_handler.PopDialogHandler):
        logger.info(f"进入弹窗处理循环，使用处理器: {handler_class.__name__}")
        if handler_class == pop_dialog_handler.TradePopDialogHandler:
            handler_class = PatchedTradePopDialogHandler
        res = super()._handle_pop_dialogs(handler_class)
        logger.info(f"退出弹窗处理循环，结果: {res}")
        return res

    def _switch_left_menus(self, path, sleep=0.5):
        """修复菜单切换，增加快捷键补偿"""
        try:
            self.close_pop_dialog()
            super()._switch_left_menus(path, sleep)
        except Exception as e:
            logger.warning(f"菜单切换失败 {path}, 尝试直接发送快捷键补偿...")
            first_menu = path[0] if path else ""
            if "买入" in first_menu: self._main.type_keys("{F1}")
            elif "卖出" in first_menu: self._main.type_keys("{F2}")
            elif "撤单" in first_menu: self._main.type_keys("{F3}")
            elif "查询" in first_menu: self._main.type_keys("{F4}")
            time.sleep(sleep)

def get_patched_trader(broker='ths'):
    """工厂方法：返回打过补丁的交易对象"""
    if broker == 'ths':
        return RobustClientTrader()
    # 可以根据需要扩展其他券商
    return RobustClientTrader()

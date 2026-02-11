"""
量化因子计算模块
基于广发多因子系列文档，实现62个技术指标因子

因子分类：
1. 动量类因子 (Momentum)
2. 趋势类因子 (Trend)
3. 波动率因子 (Volatility)
4. 成交量因子 (Volume)
5. 价格形态因子 (Price Pattern)
"""

import pandas as pd
import numpy as np
import talib
from typing import Dict, Optional
from config.factor_config import FactorConfig


class QuantitativeFactors:
    """量化因子计算器"""
    
    def __init__(self, config=None):
        """初始化因子计算器"""
        self.factor_names = []
        self.config = config if config is not None else FactorConfig
    
    def _ensure_length(self, arr: np.ndarray, target_length: int) -> np.ndarray:
        """
        确保数组长度与目标长度一致
        
        参数:
            arr: 输入数组
            target_length: 目标长度
        
        返回:
            长度正确的数组
        """
        if arr is None:
            return np.zeros(target_length, dtype=float)
        
        arr = np.asarray(arr, dtype=float)
        
        if len(arr) == target_length:
            return arr
        elif len(arr) < target_length:
            # 如果数组太短，在前面填充NaN
            padded = np.full(target_length, np.nan, dtype=float)
            padded[-len(arr):] = arr
            return padded
        else:
            # 如果数组太长，截取后面部分
            return arr[-target_length:]
    
    # ==================== 动量类因子 ====================
    
    def calculate_rsi(self, data: pd.DataFrame, period: int = 6) -> np.ndarray:
        """RSI - 相对强弱指标"""
        return talib.RSI(data['close'].values, timeperiod=period)
    
    def calculate_roc(self, data: pd.DataFrame, period: int = 12) -> np.ndarray:
        """ROC - 变动速率"""
        return talib.ROC(data['close'].values, timeperiod=period)
    
    def calculate_mtm(self, data: pd.DataFrame, period: int = 12) -> np.ndarray:
        """MTM - 动量指标"""
        close = data['close'].values
        mtm = np.zeros_like(close)
        mtm[period:] = close[period:] - close[:-period]
        return mtm
    
    def calculate_cmo(self, data: pd.DataFrame, period: int = 14) -> np.ndarray:
        """CMO - 钱德动量摆动指标"""
        return talib.CMO(data['close'].values, timeperiod=period)
    
    def calculate_stochrsi(self, data: pd.DataFrame, period: int = 14) -> tuple:
        """StochRSI - 随机强弱指数"""
        fastk, fastd = talib.STOCHRSI(data['close'].values, timeperiod=period)
        return fastk, fastd
    
    def calculate_rvi(self, data: pd.DataFrame, period: int = 10) -> np.ndarray:
        """RVI - 相对波动率指数"""
        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        
        n = len(close)
        rvi = np.full(n, 50.0, dtype=float)  # 默认值50
        
        if n < period:
            return rvi
        
        # 计算价格变化
        price_change = np.diff(close, prepend=close[0])
        
        # 计算每日波动
        daily_range = high - low
        
        # 分别计算上涨日和下跌日的波动
        for i in range(period, n):
            window_change = price_change[i-period+1:i+1]
            window_range = daily_range[i-period+1:i+1]
            
            # 确保数据是有效的
            window_change = window_change[np.isfinite(window_change)]
            window_range = window_range[np.isfinite(window_range)]
            
            # 计算标准差
            up_std = np.std(window_range[window_change > 0]) if len(window_range[window_change > 0]) > 0 else 0
            down_std = np.std(window_range[window_change < 0]) if len(window_range[window_change < 0]) > 0 else 0
            
            # 计算RVI
            denominator = up_std + down_std
            if denominator > 0:
                rvi[i] = 100 * up_std / denominator
        
        return rvi
    
    # ==================== 趋势类因子 ====================
    
    def calculate_macd(self, data: pd.DataFrame) -> tuple:
        """MACD - 指数平滑异同平均线"""
        macd, signal, hist = talib.MACD(
            data['close'].values,
            fastperiod=self.config.MACD_FAST,
            slowperiod=self.config.MACD_SLOW,
            signalperiod=self.config.MACD_SIGNAL
        )
        return macd, signal, hist
    
    def calculate_adx(self, data: pd.DataFrame, period: int = 14) -> np.ndarray:
        """ADX - 平均趋向指标"""
        return talib.ADX(
            data['high'].values,
            data['low'].values,
            data['close'].values,
            timeperiod=period
        )
    
    def calculate_dmi(self, data: pd.DataFrame, period: int = 14) -> tuple:
        """DMI - 动向指标"""
        plus_di = talib.PLUS_DI(
            data['high'].values,
            data['low'].values,
            data['close'].values,
            timeperiod=period
        )
        minus_di = talib.MINUS_DI(
            data['high'].values,
            data['low'].values,
            data['close'].values,
            timeperiod=period
        )
        return plus_di, minus_di
    
    def calculate_aroon(self, data: pd.DataFrame, period: int = 25) -> tuple:
        """Aroon - 阿隆指标"""
        aroon_up, aroon_down = talib.AROON(
            data['high'].values,
            data['low'].values,
            timeperiod=period
        )
        return aroon_up, aroon_down
    
    def calculate_trix(self, data: pd.DataFrame, period: int = 30) -> np.ndarray:
        """TRIX - 三重指数平滑平均线"""
        return talib.TRIX(data['close'].values, timeperiod=period)
    
    def calculate_ma_ratio(self, data: pd.DataFrame, period: int = 20) -> np.ndarray:
        """MA/CLOSE - 均线价格比"""
        close = data['close'].values
        ma = talib.SMA(close, timeperiod=period)
        # 避免除以零，并填充NaN
        ratio = np.where(close != 0, ma / close, 1.0)
        return ratio
    
    def calculate_ma_slope(self, data: pd.DataFrame, period: int = 20) -> np.ndarray:
        """MA线性回归系数"""
        ma = talib.SMA(data['close'].values, timeperiod=period)
        slopes = np.zeros_like(ma, dtype=float)
        
        for i in range(period, len(ma)):
            if not np.isnan(ma[i-period:i]).any():
                y = ma[i-period:i]
                x = np.arange(period)
                slope = np.polyfit(x, y, 1)[0]
                slopes[i] = slope
        
        return slopes
    
    # ==================== 波动率因子 ====================
    
    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> np.ndarray:
        """ATR - 平均真实波幅"""
        return talib.ATR(
            data['high'].values,
            data['low'].values,
            data['close'].values,
            timeperiod=period
        )
    
    def calculate_natr(self, data: pd.DataFrame, period: int = 14) -> np.ndarray:
        """NATR - 归一化平均真实波幅"""
        return talib.NATR(
            data['high'].values,
            data['low'].values,
            data['close'].values,
            timeperiod=period
        )
    
    def calculate_bollinger_bands(self, data: pd.DataFrame, period: int = 20) -> tuple:
        """布林带"""
        upper, middle, lower = talib.BBANDS(
            data['close'].values,
            timeperiod=period,
            nbdevup=self.config.BB_STD,
            nbdevdn=self.config.BB_STD
        )
        # 计算布林带宽度和位置
        # 使用 np.errstate 抑制除法警告，然后用 np.nan_to_num 处理结果
        with np.errstate(divide='ignore', invalid='ignore'):
            bb_width = (upper - lower) / middle
            bb_position = (data['close'].values - lower) / (upper - lower)
        
        # 将无效值替换为合理的默认值
        bb_width = np.nan_to_num(bb_width, nan=0.0, posinf=0.0, neginf=0.0)
        bb_position = np.nan_to_num(bb_position, nan=0.5, posinf=1.0, neginf=0.0)
        
        return bb_width, bb_position
    
    def calculate_cci(self, data: pd.DataFrame, period: int = 14) -> np.ndarray:
        """CCI - 顺势指标"""
        return talib.CCI(
            data['high'].values,
            data['low'].values,
            data['close'].values,
            timeperiod=period
        )
    
    def calculate_ulcer_index(self, data: pd.DataFrame, period: int = 14) -> np.ndarray:
        """Ulcer指标 - 衡量下行风险"""
        close = data['close'].values
        ulcer = np.zeros_like(close, dtype=float)
        
        for i in range(period, len(close)):
            window = close[i-period:i]
            window = window[np.isfinite(window)]
            if len(window) > 0:
                max_close = np.max(window)
                if max_close > 0:
                    drawdowns = ((window - max_close) / max_close) ** 2
                    ulcer[i] = np.sqrt(np.mean(drawdowns))
        
        return ulcer
    
    def calculate_price_variance(self, data: pd.DataFrame, period: int = 20) -> np.ndarray:
        """价格绝对方差均值"""
        close = pd.to_numeric(data['close'], errors='coerce').replace([np.inf, -np.inf], np.nan)
        return close.rolling(period).var().fillna(0).values
    
    # ==================== 成交量因子 ====================
    
    def calculate_obv(self, data: pd.DataFrame) -> np.ndarray:
        """OBV - 能量潮"""
        return talib.OBV(data['close'].values, data['volume'].values)
    
    def calculate_ad(self, data: pd.DataFrame) -> np.ndarray:
        """AD - 累积/派发指标"""
        return talib.AD(
            data['high'].values,
            data['low'].values,
            data['close'].values,
            data['volume'].values
        )
    
    def calculate_adosc(self, data: pd.DataFrame) -> np.ndarray:
        """Chaikin Oscillator - 佳庆指标"""
        return talib.ADOSC(
            data['high'].values,
            data['low'].values,
            data['close'].values,
            data['volume'].values,
            fastperiod=self.config.ADOSC_FAST,
            slowperiod=self.config.ADOSC_SLOW
        )
    
    def calculate_mfi(self, data: pd.DataFrame, period: int = 14) -> np.ndarray:
        """MFI - 资金流量指标"""
        return talib.MFI(
            data['high'].values,
            data['low'].values,
            data['close'].values,
            data['volume'].values,
            timeperiod=period
        )
    
    def calculate_vr(self, data: pd.DataFrame, period: int = 26) -> np.ndarray:
        """VR - 成交量比率"""
        close = data['close'].values
        volume = data['volume'].values
        
        vr = np.zeros_like(close, dtype=float)
        for i in range(period, len(close)):
            # 获取当前周期内的价格变化
            price_change = close[i-period+1:i+1] - close[i-period:i]
            vol_window = volume[i-period+1:i+1]  # 修正：使vol_window与price_change长度一致
            
            # 计算上升和下降成交量
            up_vol = np.sum(vol_window[price_change > 0])
            down_vol = np.sum(vol_window[price_change < 0])
            
            if down_vol > 0:
                vr[i] = up_vol / down_vol
            elif up_vol > 0:
                vr[i] = up_vol
        
        return vr
    
    def calculate_vroc(self, data: pd.DataFrame, period: int = 12) -> np.ndarray:
        """VROC - 量变动速率"""
        return talib.ROC(data['volume'].values, timeperiod=period)
    
    def calculate_vrsi(self, data: pd.DataFrame, period: int = 6) -> np.ndarray:
        """VRSI - 量相对强弱"""
        return talib.RSI(data['volume'].values, timeperiod=period)
    
    def calculate_vmacd(self, data: pd.DataFrame) -> tuple:
        """VMACD - 量指数平滑异同平均线"""
        macd, signal, hist = talib.MACD(
            data['volume'].values,
            fastperiod=self.config.VMACD_FAST,
            slowperiod=self.config.VMACD_SLOW,
            signalperiod=self.config.VMACD_SIGNAL
        )
        return macd, signal, hist
    
    def calculate_volume_ma(self, data: pd.DataFrame, period: int = 20) -> np.ndarray:
        """成交量移动平均"""
        return talib.SMA(data['volume'].values, timeperiod=period)
    
    def calculate_volume_std(self, data: pd.DataFrame, period: int = 20) -> np.ndarray:
        volume = pd.to_numeric(data['volume'], errors='coerce').replace([np.inf, -np.inf], np.nan)
        return volume.rolling(period).std().fillna(0).values
    
    def calculate_amount_ma(self, data: pd.DataFrame, period: int = 20) -> np.ndarray:
        """成交金额移动平均"""
        if 'amount' in data.columns:
            return talib.SMA(data['amount'].values, timeperiod=period)
        return np.zeros(len(data))
    
    def calculate_amount_std(self, data: pd.DataFrame, period: int = 20) -> np.ndarray:
        if 'amount' in data.columns:
            amount = pd.to_numeric(data['amount'], errors='coerce').replace([np.inf, -np.inf], np.nan)
            return amount.rolling(period).std().fillna(0).values
        return np.zeros(len(data))
    
    # ==================== 价格形态因子 ====================
    
    def calculate_kdj(self, data: pd.DataFrame, n: int = 9) -> tuple:
        """KDJ指标"""
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        
        # 计算RSV
        rsv = np.zeros_like(close, dtype=float)
        for i in range(n, len(close)):
            highest = np.max(high[i-n:i])
            lowest = np.min(low[i-n:i])
            if highest != lowest:
                rsv[i] = (close[i] - lowest) / (highest - lowest) * 100
            else:
                rsv[i] = 50  # 当最高价等于最低价时，设为50
        
        # 计算K、D、J
        k = pd.Series(rsv).ewm(alpha=1/3, adjust=False).mean().values
        d = pd.Series(k).ewm(alpha=1/3, adjust=False).mean().values
        j = 3 * k - 2 * d
        
        return k, d, j
    
    def calculate_willr(self, data: pd.DataFrame, period: int = 14) -> np.ndarray:
        """W%R - 威廉指标"""
        return talib.WILLR(
            data['high'].values,
            data['low'].values,
            data['close'].values,
            timeperiod=period
        )
    
    def calculate_bias(self, data: pd.DataFrame, period: int = 6) -> np.ndarray:
        """BIAS - 乖离率"""
        ma = talib.SMA(data['close'].values, timeperiod=period)
        # 使用 np.divide 安全处理除以零的情况
        with np.errstate(divide='ignore', invalid='ignore'):
            bias = np.divide(data['close'].values - ma, ma, where=ma!=0, out=np.zeros_like(ma)) * 100
        bias = np.nan_to_num(bias, nan=0.0, posinf=0.0, neginf=0.0)
        return bias
    
    def calculate_psy(self, data: pd.DataFrame, period: int = 12) -> np.ndarray:
        """PSY - 心理线"""
        close = data['close'].values
        psy = np.zeros_like(close, dtype=float)
        
        for i in range(1, len(close)):
            # 确定窗口起始位置
            start = max(0, i - period + 1)
            window_size = i - start + 1
            
            if window_size > 1 and start > 0:
                # 计算上涨天数（当日收盘价 > 前一日收盘价）
                price_change = close[start:i+1] - close[start-1:i]
                up_days = np.sum(price_change > 0)
                psy[i] = up_days / window_size * 100
        
        return psy
    
    def calculate_ar_br(self, data: pd.DataFrame, period: int = 26) -> tuple:
        """AR-BR指标"""
        high = data['high'].values
        low = data['low'].values
        open_price = data['open'].values
        close = data['close'].values
        
        ar = np.zeros_like(close, dtype=float)
        br = np.zeros_like(close, dtype=float)
        
        for i in range(period, len(close)):
            # AR指标：(H-O)/(O-L)
            ho_sum = np.sum(high[i-period:i] - open_price[i-period:i])
            ol_sum = np.sum(open_price[i-period:i] - low[i-period:i])
            
            if ol_sum > 0:
                ar[i] = ho_sum / ol_sum * 100
            
            # BR指标：(H-Cy)/(Cy-L)，其中Cy是前一日收盘价
            # 确保索引不会越界
            if i >= period + 1:
                hc_sum = np.sum(high[i-period:i] - close[i-period-1:i-1])
                cl_sum = np.sum(close[i-period-1:i-1] - low[i-period:i])
                
                if cl_sum > 0:
                    br[i] = hc_sum / cl_sum * 100
        
        return ar, br
    
    def calculate_cr(self, data: pd.DataFrame, period: int = 26) -> np.ndarray:
        """CR - 能量指标"""
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        
        # 计算中间价
        mid = (high + low + close) / 3
        
        cr = np.zeros_like(close, dtype=float)
        for i in range(period, len(close)):
            # 确保索引不会越界
            if i >= period + 1:
                # 当前周期内的高点与前一日中间价的差
                p1_sum = np.sum(np.maximum(0, high[i-period:i] - mid[i-period-1:i-1]))
                # 前一日中间价与当前周期内低点的差
                p2_sum = np.sum(np.maximum(0, mid[i-period-1:i-1] - low[i-period:i]))
                
                if p2_sum > 0:
                    cr[i] = p1_sum / p2_sum * 100
        
        return cr
    
    def calculate_price_slope(self, data: pd.DataFrame, period: int = 20) -> np.ndarray:
        """价格线性回归系数"""
        close = data['close'].values
        slopes = np.zeros_like(close, dtype=float)
        
        for i in range(period, len(close)):
            y = close[i-period:i]
            x = np.arange(period)
            if len(y) == period and not np.isnan(y).any():
                slope = np.polyfit(x, y, 1)[0]
                slopes[i] = slope
        
        return slopes
    
    # ==================== 综合计算方法 ====================
    
    def calculate_all_factors(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有量化因子
        
        参数:
            data: 包含OHLCV的DataFrame
        
        返回:
            包含所有因子的DataFrame
        """
        if len(data) < 60:
            return None
        
        target_length = len(data)
        factors = pd.DataFrame(index=data.index)
        
        try:
            # 动量类因子
            factors[f'rsi_{self.config.RSI_PERIOD}'] = self._ensure_length(self.calculate_rsi(data, self.config.RSI_PERIOD), target_length)
            factors['rsi_12'] = self._ensure_length(self.calculate_rsi(data, 12), target_length)
            factors[f'roc_{self.config.ROC_PERIOD}'] = self._ensure_length(self.calculate_roc(data, self.config.ROC_PERIOD), target_length)
            factors[f'mtm_{self.config.MTM_PERIOD}'] = self._ensure_length(self.calculate_mtm(data, self.config.MTM_PERIOD), target_length)
            factors[f'cmo_{self.config.CMO_PERIOD}'] = self._ensure_length(self.calculate_cmo(data, self.config.CMO_PERIOD), target_length)
            stochrsi_k, stochrsi_d = self.calculate_stochrsi(data, self.config.STOCHRSI_PERIOD)
            factors['stochrsi_k'] = self._ensure_length(stochrsi_k, target_length)
            factors['stochrsi_d'] = self._ensure_length(stochrsi_d, target_length)
            factors[f'rvi_{self.config.RVI_PERIOD}'] = self._ensure_length(self.calculate_rvi(data, self.config.RVI_PERIOD), target_length)
            
            # 趋势类因子
            macd, macd_signal, macd_hist = self.calculate_macd(data)
            factors['macd'] = self._ensure_length(macd, target_length)
            factors['macd_signal'] = self._ensure_length(macd_signal, target_length)
            factors['macd_hist'] = self._ensure_length(macd_hist, target_length)
            factors[f'adx_{self.config.ADX_PERIOD}'] = self._ensure_length(self.calculate_adx(data, self.config.ADX_PERIOD), target_length)
            plus_di, minus_di = self.calculate_dmi(data, self.config.DMI_PERIOD)
            factors['plus_di'] = self._ensure_length(plus_di, target_length)
            factors['minus_di'] = self._ensure_length(minus_di, target_length)
            aroon_up, aroon_down = self.calculate_aroon(data, self.config.AROON_PERIOD)
            factors['aroon_up'] = self._ensure_length(aroon_up, target_length)
            factors['aroon_down'] = self._ensure_length(aroon_down, target_length)
            factors[f'trix_{self.config.TRIX_PERIOD}'] = self._ensure_length(self.calculate_trix(data, self.config.TRIX_PERIOD), target_length)
            factors[f'ma_ratio_{self.config.MA_RATIO_PERIOD}'] = self._ensure_length(self.calculate_ma_ratio(data, self.config.MA_RATIO_PERIOD), target_length)
            factors[f'ma_slope_{self.config.MA_SLOPE_PERIOD}'] = self._ensure_length(self.calculate_ma_slope(data, self.config.MA_SLOPE_PERIOD), target_length)
            
            # 波动率因子
            factors[f'atr_{self.config.ATR_PERIOD}'] = self._ensure_length(self.calculate_atr(data, self.config.ATR_PERIOD), target_length)
            factors[f'natr_{self.config.NATR_PERIOD}'] = self._ensure_length(self.calculate_natr(data, self.config.NATR_PERIOD), target_length)
            bb_width, bb_position = self.calculate_bollinger_bands(data, self.config.BB_PERIOD)
            factors['bb_width'] = self._ensure_length(bb_width, target_length)
            factors['bb_position'] = self._ensure_length(bb_position, target_length)
            factors[f'cci_{self.config.CCI_PERIOD}'] = self._ensure_length(self.calculate_cci(data, self.config.CCI_PERIOD), target_length)
            factors[f'ulcer_{self.config.ULCER_PERIOD}'] = self._ensure_length(self.calculate_ulcer_index(data, self.config.ULCER_PERIOD), target_length)
            factors[f'price_var_{self.config.PRICE_VAR_PERIOD}'] = self._ensure_length(self.calculate_price_variance(data, self.config.PRICE_VAR_PERIOD), target_length)
            
            # 成交量因子
            factors['obv'] = self._ensure_length(self.calculate_obv(data), target_length)
            factors['ad'] = self._ensure_length(self.calculate_ad(data), target_length)
            factors['adosc'] = self._ensure_length(self.calculate_adosc(data), target_length)
            factors[f'mfi_{self.config.MFI_PERIOD}'] = self._ensure_length(self.calculate_mfi(data, self.config.MFI_PERIOD), target_length)
            factors[f'vr_{self.config.VR_PERIOD}'] = self._ensure_length(self.calculate_vr(data, self.config.VR_PERIOD), target_length)
            factors[f'vroc_{self.config.VROC_PERIOD}'] = self._ensure_length(self.calculate_vroc(data, self.config.VROC_PERIOD), target_length)
            factors[f'vrsi_{self.config.VRSI_PERIOD}'] = self._ensure_length(self.calculate_vrsi(data, self.config.VRSI_PERIOD), target_length)
            vmacd, vmacd_signal, vmacd_hist = self.calculate_vmacd(data)
            factors['vmacd'] = self._ensure_length(vmacd, target_length)
            factors['vmacd_signal'] = self._ensure_length(vmacd_signal, target_length)
            factors[f'vol_ma_{self.config.VOLUME_MA_PERIOD}'] = self._ensure_length(self.calculate_volume_ma(data, self.config.VOLUME_MA_PERIOD), target_length)
            factors[f'vol_std_{self.config.VOLUME_STD_PERIOD}'] = self._ensure_length(self.calculate_volume_std(data, self.config.VOLUME_STD_PERIOD), target_length)
            factors[f'amount_ma_{self.config.AMOUNT_MA_PERIOD}'] = self._ensure_length(self.calculate_amount_ma(data, self.config.AMOUNT_MA_PERIOD), target_length)
            factors[f'amount_std_{self.config.AMOUNT_STD_PERIOD}'] = self._ensure_length(self.calculate_amount_std(data, self.config.AMOUNT_STD_PERIOD), target_length)
            
            # 价格形态因子
            k, d, j = self.calculate_kdj(data, self.config.KDJ_N)
            factors['kdj_k'] = self._ensure_length(k, target_length)
            factors['kdj_d'] = self._ensure_length(d, target_length)
            factors['kdj_j'] = self._ensure_length(j, target_length)
            factors[f'willr_{self.config.WILLR_PERIOD}'] = self._ensure_length(self.calculate_willr(data, self.config.WILLR_PERIOD), target_length)
            factors[f'bias_{self.config.BIAS_PERIOD}'] = self._ensure_length(self.calculate_bias(data, self.config.BIAS_PERIOD), target_length)
            factors[f'psy_{self.config.PSY_PERIOD}'] = self._ensure_length(self.calculate_psy(data, self.config.PSY_PERIOD), target_length)
            ar, br = self.calculate_ar_br(data, self.config.AR_BR_PERIOD)
            factors[f'ar_{self.config.AR_BR_PERIOD}'] = self._ensure_length(ar, target_length)
            factors[f'br_{self.config.AR_BR_PERIOD}'] = self._ensure_length(br, target_length)
            factors[f'cr_{self.config.CR_PERIOD}'] = self._ensure_length(self.calculate_cr(data, self.config.CR_PERIOD), target_length)
            
            # 填充NaN值
            factors = factors.ffill().fillna(0)
            
            # 保存因子名称
            self.factor_names = factors.columns.tolist()
            
            return factors
            
        except Exception as e:
            print(f"计算因子时出错: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_factor_names(self) -> list:
        """获取所有因子名称"""
        return self.factor_names

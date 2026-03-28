"""
量化因子计算模块

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
        """RVI - 相对波动率指数 (向量化版)"""
        close = data['close']
        daily_range = data['high'] - data['low']
        price_change = close.diff()
        
        # 分别计算上涨日和下跌日的波动的滚动标准差
        up_std = daily_range.where(price_change > 0).rolling(window=period).std()
        down_std = daily_range.where(price_change < 0).rolling(window=period).std()
        
        with np.errstate(divide='ignore', invalid='ignore'):
            rvi = 100 * up_std / (up_std + down_std)
            
        return rvi.fillna(50.0).values

    def calculate_ulcer_index(self, data: pd.DataFrame, period: int = 14) -> np.ndarray:
        """Ulcer指标 (向量化版)"""
        close = data['close']
        max_close = close.rolling(window=period).max()
        
        with np.errstate(divide='ignore', invalid='ignore'):
            drawdowns_sq = ((close - max_close) / max_close) ** 2
            
        ulcer = np.sqrt(drawdowns_sq.rolling(window=period).mean())
        return np.nan_to_num(ulcer.values, nan=0.0)

    def calculate_kdj(self, data: pd.DataFrame, n: int = 9) -> tuple:
        """KDJ指标 (向量化版)"""
        high = data['high']
        low = data['low']
        close = data['close']
        
        low_min = low.rolling(window=n).min()
        high_max = high.rolling(window=n).max()
        
        with np.errstate(divide='ignore', invalid='ignore'):
            rsv = (close - low_min) / (high_max - low_min) * 100
        rsv = rsv.fillna(50.0)
        
        # 计算K、D、J
        k = rsv.ewm(alpha=1/3, adjust=False).mean()
        d = k.ewm(alpha=1/3, adjust=False).mean()
        j = 3 * k - 2 * d
        
        return k.values, d.values, j.values
    
    # ==================== 趋势类因子 ====================
    
    def calculate_macd(self, data: pd.DataFrame) -> tuple:
        """MACD - 指数平滑异同平均线 (相对变化版)"""
        macd, signal, hist = talib.MACD(
            data['close'].values,
            fastperiod=self.config.MACD_FAST,
            slowperiod=self.config.MACD_SLOW,
            signalperiod=self.config.MACD_SIGNAL
        )
        close = data['close'].values
        with np.errstate(divide='ignore', invalid='ignore'):
            macd = np.where(close!=0, macd / close, 0)
            signal = np.where(close!=0, signal / close, 0)
            hist = np.where(close!=0, hist / close, 0)
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
        """MA线性回归系数 (向量化版)"""
        ma = talib.SMA(data['close'].values, timeperiod=period)
        slopes = talib.LINEARREG_SLOPE(ma, timeperiod=period)
        return np.nan_to_num(slopes, nan=0.0)
    
    def calculate_price_slope(self, data: pd.DataFrame, period: int = 20) -> np.ndarray:
        """价格线性回归系数 (向量化版)"""
        close = data['close'].values
        slopes = talib.LINEARREG_SLOPE(close, timeperiod=period)
        return np.nan_to_num(slopes, nan=0.0)

    # ==================== 波动率因子 ====================
    
    def calculate_atr(self, data: pd.DataFrame, period: int = 14) -> np.ndarray:
        """ATR - 平均真实波幅 (相对变化版)"""
        atr = talib.ATR(
            data['high'].values,
            data['low'].values,
            data['close'].values,
            timeperiod=period
        )
        close = data['close'].values
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(close!=0, atr / close, 0)
    
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
        with np.errstate(divide='ignore', invalid='ignore'):
            bb_width = (upper - lower) / middle
            bb_position = (data['close'].values - lower) / (upper - lower)
        
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

    def calculate_price_variance(self, data: pd.DataFrame, period: int = 20) -> np.ndarray:
        """价格收益率方差 (相对变化版)"""
        returns = data['close'].pct_change().fillna(0)
        return returns.rolling(period).var().fillna(0).values
    
    # ==================== 成交量因子 ====================
    
    def calculate_obv(self, data: pd.DataFrame) -> np.ndarray:
        """OBV - 能量潮 (相对变化版：去除趋势)"""
        obv = talib.OBV(data['close'].values, data['volume'].values)
        vol_ma = talib.SMA(data['volume'].values, timeperiod=20)
        with np.errstate(divide='ignore', invalid='ignore'):
            obv_rel = np.where(vol_ma!=0, (obv - pd.Series(obv).rolling(20).mean().values) / vol_ma, 0)
        return np.nan_to_num(obv_rel)
    
    def calculate_ad(self, data: pd.DataFrame) -> np.ndarray:
        """AD - 累积/派发指标 (相对变化版)"""
        ad = talib.AD(
            data['high'].values,
            data['low'].values,
            data['close'].values,
            data['volume'].values
        )
        vol_ma = talib.SMA(data['volume'].values, timeperiod=20)
        with np.errstate(divide='ignore', invalid='ignore'):
            ad_rel = np.where(vol_ma!=0, (ad - pd.Series(ad).rolling(20).mean().values) / vol_ma, 0)
        return np.nan_to_num(ad_rel)
    
    def calculate_adosc(self, data: pd.DataFrame) -> np.ndarray:
        """Chaikin Oscillator - 佳庆指标 (相对变化版)"""
        adosc = talib.ADOSC(
            data['high'].values,
            data['low'].values,
            data['close'].values,
            data['volume'].values,
            fastperiod=self.config.ADOSC_FAST,
            slowperiod=self.config.ADOSC_SLOW
        )
        vol_ma = talib.SMA(data['volume'].values, timeperiod=20)
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.nan_to_num(np.where(vol_ma!=0, adosc / vol_ma, 0))
    
    def calculate_mfi(self, data: pd.DataFrame, period: int = 14) -> np.ndarray:
        """MFI - 资金流量指标"""
        return talib.MFI(
            data['high'].values,
            data['low'].values,
            data['close'].values,
            data['volume'].values,
            timeperiod=period
        )

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
        """换手率(或相对成交量)移动平均 (相对变化版)"""
        vol_ma = talib.SMA(data['volume'].values, timeperiod=period)
        vol_base = talib.SMA(data['volume'].values, timeperiod=period*5)
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.nan_to_num(np.where(vol_base!=0, vol_ma / vol_base, 1.0))
    
    def calculate_volume_std(self, data: pd.DataFrame, period: int = 20) -> np.ndarray:
        """成交量波动率 (相对变化版)"""
        volume = pd.to_numeric(data['volume'], errors='coerce').replace([np.inf, -np.inf], np.nan)
        vol_std = volume.rolling(period).std().fillna(0).values
        vol_ma = talib.SMA(data['volume'].values, timeperiod=period)
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.nan_to_num(np.where(vol_ma!=0, vol_std / vol_ma, 0.0))
    
    def calculate_amount_ma(self, data: pd.DataFrame, period: int = 20) -> np.ndarray:
        """相对成交金额移动平均 (相对变化版)"""
        if 'amount' in data.columns:
            amt_ma = talib.SMA(data['amount'].values, timeperiod=period)
            amt_base = talib.SMA(data['amount'].values, timeperiod=period*5)
            with np.errstate(divide='ignore', invalid='ignore'):
                return np.nan_to_num(np.where(amt_base!=0, amt_ma / amt_base, 1.0))
        return np.zeros(len(data))
    
    def calculate_amount_std(self, data: pd.DataFrame, period: int = 20) -> np.ndarray:
        """成交金额波动率 (相对变化版)"""
        if 'amount' in data.columns:
            amount = pd.to_numeric(data['amount'], errors='coerce').replace([np.inf, -np.inf], np.nan)
            amt_std = amount.rolling(period).std().fillna(0).values
            amt_ma = talib.SMA(data['amount'].values, timeperiod=period)
            with np.errstate(divide='ignore', invalid='ignore'):
                return np.nan_to_num(np.where(amt_ma!=0, amt_std / amt_ma, 0.0))
        return np.zeros(len(data))
    
    # ==================== 价格形态因子 ====================

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
        with np.errstate(divide='ignore', invalid='ignore'):
            bias = np.divide(data['close'].values - ma, ma, where=ma!=0, out=np.zeros_like(ma)) * 100
        bias = np.nan_to_num(bias, nan=0.0, posinf=0.0, neginf=0.0)
        return bias

    def calculate_vr(self, data: pd.DataFrame, period: int = 26) -> np.ndarray:
        """VR - 成交量比率 (向量化版)"""
        close = data['close']
        volume = data['volume']
        
        price_diff = close.diff()
        
        up_vol = volume.where(price_diff > 0, 0).rolling(window=period).sum()
        down_vol = volume.where(price_diff < 0, 0).rolling(window=period).sum()
        
        with np.errstate(divide='ignore', invalid='ignore'):
            vr = up_vol / down_vol
            
        return np.nan_to_num(vr.values, nan=0.0, posinf=up_vol.max() if not up_vol.empty else 0.0)

    def calculate_psy(self, data: pd.DataFrame, period: int = 12) -> np.ndarray:
        """PSY - 心理线 (向量化版)"""
        close = data['close']
        price_diff = close.diff()
        
        up_days = (price_diff > 0).astype(int).rolling(window=period).sum()
        psy = (up_days / period) * 100
        
        return psy.fillna(0).values

    def calculate_ar_br(self, data: pd.DataFrame, period: int = 26) -> tuple:
        """AR-BR指标 (向量化版)"""
        high = data['high']
        low = data['low']
        open_p = data['open']
        close = data['close']
        prev_close = close.shift(1)
        
        # AR: (H-O) / (O-L)
        ho_sum = (high - open_p).rolling(window=period).sum()
        ol_sum = (open_p - low).rolling(window=period).sum()
        
        with np.errstate(divide='ignore', invalid='ignore'):
            ar = (ho_sum / ol_sum) * 100
            
        # BR: (H-Cy) / (Cy-L)
        hc_sum = (high - prev_close).rolling(window=period).sum()
        cl_sum = (prev_close - low).rolling(window=period).sum()
        
        with np.errstate(divide='ignore', invalid='ignore'):
            br = (hc_sum / cl_sum) * 100
            
        return ar.fillna(0).values, br.fillna(0).values

    def calculate_cr(self, data: pd.DataFrame, period: int = 26) -> np.ndarray:
        """CR - 能量指标 (向量化版)"""
        high = data['high']
        low = data['low']
        close = data['close']
        
        # 中间价 = (H+L+C)/3
        mid = (high + low + close) / 3
        prev_mid = mid.shift(1)
        
        p1 = (high - prev_mid).clip(lower=0).rolling(window=period).sum()
        p2 = (prev_mid - low).clip(lower=0).rolling(window=period).sum()
        
        with np.errstate(divide='ignore', invalid='ignore'):
            cr = (p1 / p2) * 100
            
        return cr.fillna(0).values
    
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

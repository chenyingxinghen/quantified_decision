"""
基本面因子模块
基于stock_info_extended表构建基本面量化因子
"""

import pandas as pd
import numpy as np
import sqlite3
from typing import Optional, List
from config.config import DATABASE_PATH
from core.factors.advanced_factors import RelativeStrengthFactors


class FundamentalFactors:
    """基本面因子计算器"""
    
    def __init__(self, db_path=DATABASE_PATH):
        """初始化"""
        self.db_path = db_path
    
    def get_stock_info(self, code: str) -> Optional[pd.Series]:
        """
        获取单只股票的基本面信息
        
        参数:
            code: 股票代码
        
        返回:
            包含基本面信息的Series
        """
        with sqlite3.connect(self.db_path) as conn:
            query = "SELECT * FROM stock_info_extended WHERE code = ?"
            df = pd.read_sql_query(query, conn, params=(code,))
        
        if len(df) > 0:
            return df.iloc[0]
        return None
    
    def get_all_stocks_info(self) -> pd.DataFrame:
        """
        获取所有股票的基本面信息
        
        返回:
            包含所有股票基本面信息的DataFrame
        """
        with sqlite3.connect(self.db_path) as conn:
            query = "SELECT * FROM stock_info_extended"
            df = pd.read_sql_query(query, conn)
        return df
    
    # ==================== 价值因子 ====================
    
    def calculate_value_factors(self, code: str) -> dict:
        """
        计算价值因子
        
        参数:
            code: 股票代码
        
        返回:
            价值因子字典
        """
        info = self.get_stock_info(code)
        if info is None:
            return {}
        
        # 安全转换数值字段
        pe_ratio = info.get('pe_ratio')
        pb_ratio = info.get('pb_ratio')
        forward_pe = info.get('forward_pe')
        peg_ratio = info.get('peg_ratio')
        price_to_sales = info.get('price_to_sales')
        enterprise_to_revenue = info.get('enterprise_to_revenue')
        enterprise_to_ebitda = info.get('enterprise_to_ebitda')
        
        try:
            pe_ratio = float(pe_ratio) if pe_ratio is not None else None
            pb_ratio = float(pb_ratio) if pb_ratio is not None else None
            forward_pe = float(forward_pe) if forward_pe is not None else None
            peg_ratio = float(peg_ratio) if peg_ratio is not None else None
            price_to_sales = float(price_to_sales) if price_to_sales is not None else None
            enterprise_to_revenue = float(enterprise_to_revenue) if enterprise_to_revenue is not None else None
            enterprise_to_ebitda = float(enterprise_to_ebitda) if enterprise_to_ebitda is not None else None
        except (ValueError, TypeError):
            pe_ratio = None
            pb_ratio = None
            forward_pe = None
            peg_ratio = None
            price_to_sales = None
            enterprise_to_revenue = None
            enterprise_to_ebitda = None
        
        # 计算反转因子时确保值是数值类型
        inv_pe = None
        inv_pb = None
        
        if pe_ratio is not None and isinstance(pe_ratio, (int, float)) and pe_ratio > 0:
            inv_pe = 1.0 / pe_ratio
        
        if pb_ratio is not None and isinstance(pb_ratio, (int, float)) and pb_ratio > 0:
            inv_pb = 1.0 / pb_ratio
        
        factors = {
            'pe_ratio': pe_ratio,
            'pb_ratio': pb_ratio,
            'forward_pe': forward_pe,
            'peg_ratio': peg_ratio,
            'price_to_sales': price_to_sales,
            'enterprise_to_revenue': enterprise_to_revenue,
            'enterprise_to_ebitda': enterprise_to_ebitda,
            'inv_pe': inv_pe,
            'inv_pb': inv_pb,
        }
        
        return factors
    
    # ==================== 质量因子 ====================
    
    def calculate_quality_factors(self, code: str) -> dict:
        """
        计算质量因子
        
        参数:
            code: 股票代码
        
        返回:
            质量因子字典
        """
        info = self.get_stock_info(code)
        if info is None:
            return {}
        
        # 安全转换所有数值字段
        roe = info.get('return_on_equity')
        roa = info.get('return_on_assets')
        profit_margin = info.get('profit_margins')
        gross_margin = info.get('gross_margins')
        operating_margin = info.get('operating_margins')
        ebitda_margin = info.get('ebitda_margins')
        current_ratio = info.get('current_ratio')
        quick_ratio = info.get('quick_ratio')
        debt_to_equity = info.get('debt_to_equity')
        free_cashflow = info.get('free_cashflow')
        market_cap = info.get('market_cap')
        operating_cashflow = info.get('operating_cashflow')
        
        try:
            roe = float(roe) if roe is not None else None
            roa = float(roa) if roa is not None else None
            profit_margin = float(profit_margin) if profit_margin is not None else None
            gross_margin = float(gross_margin) if gross_margin is not None else None
            operating_margin = float(operating_margin) if operating_margin is not None else None
            ebitda_margin = float(ebitda_margin) if ebitda_margin is not None else None
            current_ratio = float(current_ratio) if current_ratio is not None else None
            quick_ratio = float(quick_ratio) if quick_ratio is not None else None
            debt_to_equity = float(debt_to_equity) if debt_to_equity is not None else None
            free_cashflow = float(free_cashflow) if free_cashflow is not None else None
            market_cap = float(market_cap) if market_cap is not None else None
            operating_cashflow = float(operating_cashflow) if operating_cashflow is not None else None
        except (ValueError, TypeError):
            pass  # 保持None值
        
        factors = {
            'roe': roe,
            'roa': roa,
            'profit_margin': profit_margin,
            'gross_margin': gross_margin,
            'operating_margin': operating_margin,
            'ebitda_margin': ebitda_margin,
            'current_ratio': current_ratio,
            'quick_ratio': quick_ratio,
            'debt_to_equity': debt_to_equity,
        }
        
        # 现金流质量 - 确保类型检查
        fcf_to_market_cap = None
        if (free_cashflow is not None and isinstance(free_cashflow, (int, float)) and
            market_cap is not None and isinstance(market_cap, (int, float)) and market_cap > 0):
            fcf_to_market_cap = free_cashflow / market_cap
        
        factors['fcf_to_market_cap'] = fcf_to_market_cap
        
        # 经营现金流质量
        ocf_to_market_cap = None
        if (operating_cashflow is not None and isinstance(operating_cashflow, (int, float)) and
            market_cap is not None and isinstance(market_cap, (int, float)) and market_cap > 0):
            ocf_to_market_cap = operating_cashflow / market_cap
        
        factors['ocf_to_market_cap'] = ocf_to_market_cap
        
        return factors
    
    # ==================== 成长因子 ====================
    
    def calculate_growth_factors(self, code: str) -> dict:
        """
        计算成长因子
        
        参数:
            code: 股票代码
        
        返回:
            成长因子字典
        """
        info = self.get_stock_info(code)
        if info is None:
            return {}
        
        # 安全转换数值字段
        revenue_growth = info.get('revenue_growth')
        earnings_growth = info.get('earnings_growth')
        earnings_quarterly_growth = info.get('earnings_quarterly_growth')
        
        try:
            revenue_growth = float(revenue_growth) if revenue_growth is not None else None
            earnings_growth = float(earnings_growth) if earnings_growth is not None else None
            earnings_quarterly_growth = float(earnings_quarterly_growth) if earnings_quarterly_growth is not None else None
        except (ValueError, TypeError):
            revenue_growth = None
            earnings_growth = None
            earnings_quarterly_growth = None
        
        factors = {
            'revenue_growth': revenue_growth,
            'earnings_growth': earnings_growth,
            'earnings_quarterly_growth': earnings_quarterly_growth,
        }
        
        return factors
    
    # ==================== 股息因子 ====================
    
    def calculate_dividend_factors(self, code: str) -> dict:
        """
        计算股息因子
        
        参数:
            code: 股票代码
        
        返回:
            股息因子字典
        """
        info = self.get_stock_info(code)
        if info is None:
            return {}
        
        # 安全转换数值字段
        dividend_yield = info.get('dividend_yield')
        payout_ratio = info.get('payout_ratio')
        five_year_avg_dividend_yield = info.get('five_year_avg_dividend_yield')
        
        try:
            dividend_yield = float(dividend_yield) if dividend_yield is not None else None
            payout_ratio = float(payout_ratio) if payout_ratio is not None else None
            five_year_avg_dividend_yield = float(five_year_avg_dividend_yield) if five_year_avg_dividend_yield is not None else None
        except (ValueError, TypeError):
            dividend_yield = None
            payout_ratio = None
            five_year_avg_dividend_yield = None
        
        factors = {
            'dividend_yield': dividend_yield,
            'payout_ratio': payout_ratio,
            'five_year_avg_dividend_yield': five_year_avg_dividend_yield,
        }
        
        return factors
    
    # ==================== 风险因子 ====================
    
    def calculate_risk_factors(self, code: str) -> dict:
        """
        计算风险因子
        
        参数:
            code: 股票代码
        
        返回:
            风险因子字典
        """
        info = self.get_stock_info(code)
        if info is None:
            return {}
        
        # 安全转换数值字段
        beta = info.get('beta')
        
        try:
            beta = float(beta) if beta is not None else None
        except (ValueError, TypeError):
            beta = None
        
        factors = {
            'beta': beta,
        }
        
        return factors
    
    # ==================== 现金流因子 ====================
    
    def calculate_cashflow_factors(self, code: str) -> dict:
        """
        计算现金流因子
        
        参数:
            code: 股票代码
        
        返回:
            现金流因子字典
        """
        info = self.get_stock_info(code)
        if info is None:
            return {}
        
        factors = {}
        
        # 安全转换所有可能的数值字段
        operating_cashflow = info.get('operating_cashflow')
        market_cap = info.get('market_cap')
        total_cash = info.get('total_cash')
        total_debt = info.get('total_debt')
        free_cashflow = info.get('free_cashflow')
        
        try:
            operating_cashflow = float(operating_cashflow) if operating_cashflow is not None else None
            market_cap = float(market_cap) if market_cap is not None else None
            total_cash = float(total_cash) if total_cash is not None else None
            total_debt = float(total_debt) if total_debt is not None else None
            free_cashflow = float(free_cashflow) if free_cashflow is not None else None
        except (ValueError, TypeError):
            operating_cashflow = None
            market_cap = None
            total_cash = None
            total_debt = None
            free_cashflow = None
        
        # 经营现金流质量
        if (operating_cashflow is not None and isinstance(operating_cashflow, (int, float)) and
            market_cap is not None and isinstance(market_cap, (int, float)) and market_cap > 0):
            factors['ocf_to_market_cap'] = operating_cashflow / market_cap
        
        # 现金充裕度
        if (total_cash is not None and isinstance(total_cash, (int, float)) and
            market_cap is not None and isinstance(market_cap, (int, float)) and market_cap > 0):
            factors['cash_to_market_cap'] = total_cash / market_cap
        
        # 现金流与债务比
        if (operating_cashflow is not None and isinstance(operating_cashflow, (int, float)) and
            total_debt is not None and isinstance(total_debt, (int, float)) and total_debt > 0):
            factors['ocf_to_debt'] = operating_cashflow / total_debt
        
        # 自由现金流收益率
        if (free_cashflow is not None and isinstance(free_cashflow, (int, float)) and
            market_cap is not None and isinstance(market_cap, (int, float)) and market_cap > 0):
            factors['fcf_yield'] = free_cashflow / market_cap
        
        return factors
    
    # ==================== 股本结构因子 ====================
    
    def calculate_ownership_factors(self, code: str) -> dict:
        """
        计算股本结构因子
        
        参数:
            code: 股票代码
        
        返回:
            股本结构因子字典
        """
        info = self.get_stock_info(code)
        if info is None:
            return {}
        
        factors = {}
        
        # 安全转换数值字段
        held_percent_insiders = info.get('held_percent_insiders')
        held_percent_institutions = info.get('held_percent_institutions')
        float_shares = info.get('float_shares')
        shares_outstanding = info.get('shares_outstanding')
        
        try:
            held_percent_insiders = float(held_percent_insiders) if held_percent_insiders is not None else None
            held_percent_institutions = float(held_percent_institutions) if held_percent_institutions is not None else None
            float_shares = float(float_shares) if float_shares is not None else None
            shares_outstanding = float(shares_outstanding) if shares_outstanding is not None else None
        except (ValueError, TypeError):
            held_percent_insiders = None
            held_percent_institutions = None
            float_shares = None
            shares_outstanding = None
        
        factors['held_percent_insiders'] = held_percent_insiders
        factors['held_percent_institutions'] = held_percent_institutions
        
        # 流通比例
        if (float_shares is not None and isinstance(float_shares, (int, float)) and
            shares_outstanding is not None and isinstance(shares_outstanding, (int, float)) and 
            shares_outstanding > 0):
            factors['float_ratio'] = float_shares / shares_outstanding
        
        # 机构+内部人持股总比例
        total_held = 0.0
        if held_percent_insiders is not None and isinstance(held_percent_insiders, (int, float)):
            total_held += held_percent_insiders
        if held_percent_institutions is not None and isinstance(held_percent_institutions, (int, float)):
            total_held += held_percent_institutions
        
        factors['total_held_percent'] = total_held
        
        return factors
    
    # ==================== 流动性因子 ====================
    
    def calculate_liquidity_factors(self, code: str) -> dict:
        """
        计算流动性因子
        
        参数:
            code: 股票代码
        
        返回:
            流动性因子字典
        """
        info = self.get_stock_info(code)
        if info is None:
            return {}
        
        factors = {}
        
        # 安全转换数值字段
        average_volume = info.get('average_volume')
        average_volume_10days = info.get('average_volume_10days')
        avg_vol = average_volume
        float_shares = info.get('float_shares')
        current_price = info.get('current_price')
        market_cap = info.get('market_cap')
        
        try:
            avg_vol = float(avg_vol) if avg_vol is not None else None
            float_shares = float(float_shares) if float_shares is not None else None
            current_price = float(current_price) if current_price is not None else None
            market_cap = float(market_cap) if market_cap is not None else None
            average_volume_10days = float(average_volume_10days) if average_volume_10days is not None else None
        except (ValueError, TypeError):
            avg_vol = None
            float_shares = None
            current_price = None
            market_cap = None
            average_volume_10days = None
        
        factors['average_volume'] = avg_vol
        factors['average_volume_10days'] = average_volume_10days
        
        # 换手率（成交量/流通股本）
        if (avg_vol is not None and isinstance(avg_vol, (int, float)) and
            float_shares is not None and isinstance(float_shares, (int, float)) and float_shares > 0):
            factors['turnover_ratio'] = avg_vol / float_shares
        
        # 成交金额（成交量 * 价格）
        if (avg_vol is not None and isinstance(avg_vol, (int, float)) and
            current_price is not None and isinstance(current_price, (int, float))):
            factors['avg_amount'] = avg_vol * current_price
        
        # 流动性比率（成交金额/市值）
        if (avg_vol is not None and isinstance(avg_vol, (int, float)) and
            current_price is not None and isinstance(current_price, (int, float)) and
            market_cap is not None and isinstance(market_cap, (int, float)) and market_cap > 0):
            factors['liquidity_ratio'] = (avg_vol * current_price) / market_cap
        
        return factors
    
    # ==================== 规模因子 ====================
    
    def calculate_size_factors(self, code: str) -> dict:
        """
        计算规模因子
        
        参数:
            code: 股票代码
        
        返回:
            规模因子字典
        """
        info = self.get_stock_info(code)
        if info is None:
            return {}
        
        factors = {}
        
        # 安全转换数值字段
        market_cap = info.get('market_cap')
        book_value = info.get('book_value')
        total_debt = info.get('total_debt')
        total_cash = info.get('total_cash')
        
        try:
            market_cap = float(market_cap) if market_cap is not None else None
            book_value = float(book_value) if book_value is not None else None
            total_debt = float(total_debt) if total_debt is not None else None
            total_cash = float(total_cash) if total_cash is not None else None
        except (ValueError, TypeError):
            market_cap = None
            book_value = None
            total_debt = None
            total_cash = None
        
        # 市值（对数）
        if market_cap is not None and isinstance(market_cap, (int, float)) and market_cap > 0:
            factors['log_market_cap'] = np.log(market_cap)
        
        # 账面价值（对数）
        if book_value is not None and isinstance(book_value, (int, float)) and book_value > 0:
            factors['log_book_value'] = np.log(book_value)
        
        # 企业价值
        if (market_cap is not None and isinstance(market_cap, (int, float)) and
            total_debt is not None and isinstance(total_debt, (int, float)) and
            total_cash is not None and isinstance(total_cash, (int, float))):
            enterprise_value = market_cap + total_debt - total_cash
            if isinstance(enterprise_value, (int, float)) and enterprise_value > 0:
                factors['log_enterprise_value'] = np.log(enterprise_value)
        
        return factors
    
    # ==================== 动量因子（基于基本面） ====================
    
    def calculate_price_momentum_factors(self, code: str) -> dict:
        """
        计算价格动量因子（基于基本面数据）
        
        参数:
            code: 股票代码
        
        返回:
            动量因子字典
        """
        info = self.get_stock_info(code)
        if info is None:
            return {}
        
        factors = {}
        
        # 安全转换数值字段
        current_price = info.get('current_price')
        fifty_two_week_high = info.get('fifty_two_week_high')
        fifty_two_week_low = info.get('fifty_two_week_low')
        fifty_day_average = info.get('fifty_day_average')
        two_hundred_day_average = info.get('two_hundred_day_average')
        
        try:
            current_price = float(current_price) if current_price is not None else None
            fifty_two_week_high = float(fifty_two_week_high) if fifty_two_week_high is not None else None
            fifty_two_week_low = float(fifty_two_week_low) if fifty_two_week_low is not None else None
            fifty_day_average = float(fifty_day_average) if fifty_day_average is not None else None
            two_hundred_day_average = float(two_hundred_day_average) if two_hundred_day_average is not None else None
        except (ValueError, TypeError):
            current_price = None
            fifty_two_week_high = None
            fifty_two_week_low = None
            fifty_day_average = None
            two_hundred_day_average = None
        
        # 52周高低点相对位置
        if (current_price is not None and isinstance(current_price, (int, float)) and
            fifty_two_week_high is not None and isinstance(fifty_two_week_high, (int, float)) and
            fifty_two_week_low is not None and isinstance(fifty_two_week_low, (int, float))):
            
            high = fifty_two_week_high
            low = fifty_two_week_low
            current = current_price
            
            if isinstance(high, (int, float)) and isinstance(low, (int, float)) and high > low:
                factors['price_52w_position'] = (current - low) / (high - low)
        
        # 相对均线位置
        if (current_price is not None and isinstance(current_price, (int, float)) and
            fifty_day_average is not None and isinstance(fifty_day_average, (int, float)) and 
            fifty_day_average > 0):
            factors['price_to_ma50'] = current_price / fifty_day_average
        
        if (current_price is not None and isinstance(current_price, (int, float)) and
            two_hundred_day_average is not None and isinstance(two_hundred_day_average, (int, float)) and 
            two_hundred_day_average > 0):
            factors['price_to_ma200'] = current_price / two_hundred_day_average
        
        return factors
    
    # ==================== 行业和板块因子 ====================
    
    def calculate_industry_sector_factors(self, code: str) -> dict:
        """
        计算行业和板块因子（分类特征）
        
        参数:
            code: 股票代码
        
        返回:
            行业和板块因子字典
        """
        info = self.get_stock_info(code)
        if info is None:
            return {}
        
        factors = {
            'sector': info.get('sector'),
            'industry': info.get('industry'),
        }
        
        return factors
    
    # ==================== 交叉因子 ====================
    
    def calculate_cross_factors(self, code: str) -> dict:
        """
        计算交叉因子（多个基本面指标的组合）
        
        参数:
            code: 股票代码
        
        返回:
            交叉因子字典
        """
        info = self.get_stock_info(code)
        if info is None:
            return {}
        
        factors = {}
        
        # 安全转换所有可能的数值字段
        roe = info.get('return_on_equity')
        revenue_growth = info.get('revenue_growth')
        pe_ratio = info.get('pe_ratio')
        earnings_growth = info.get('earnings_growth')
        pb_ratio = info.get('pb_ratio')
        debt_to_equity = info.get('debt_to_equity')
        free_cashflow = info.get('free_cashflow')
        profit_margins = info.get('profit_margins')
        market_cap = info.get('market_cap')
        dividend_yield = info.get('dividend_yield')
        payout_ratio = info.get('payout_ratio')
        held_percent_institutions = info.get('held_percent_institutions')
        
        try:
            roe = float(roe) if roe is not None else None
            revenue_growth = float(revenue_growth) if revenue_growth is not None else None
            pe_ratio = float(pe_ratio) if pe_ratio is not None else None
            earnings_growth = float(earnings_growth) if earnings_growth is not None else None
            pb_ratio = float(pb_ratio) if pb_ratio is not None else None
            debt_to_equity = float(debt_to_equity) if debt_to_equity is not None else None
            free_cashflow = float(free_cashflow) if free_cashflow is not None else None
            profit_margins = float(profit_margins) if profit_margins is not None else None
            market_cap = float(market_cap) if market_cap is not None else None
            dividend_yield = float(dividend_yield) if dividend_yield is not None else None
            payout_ratio = float(payout_ratio) if payout_ratio is not None else None
            held_percent_institutions = float(held_percent_institutions) if held_percent_institutions is not None else None
        except (ValueError, TypeError):
            pass  # 保持None值
        
        # 成长质量因子：ROE * 营收增长
        if (roe is not None and isinstance(roe, (int, float)) and
            revenue_growth is not None and isinstance(revenue_growth, (int, float))):
            factors['roe_growth'] = roe * revenue_growth
        
        # 价值成长因子：PE * 盈利增长（PEG的变体）
        if (pe_ratio is not None and isinstance(pe_ratio, (int, float)) and
            earnings_growth is not None and isinstance(earnings_growth, (int, float)) and
            pe_ratio > 0 and earnings_growth > 0):
            factors['pe_earnings_growth'] = pe_ratio * earnings_growth
        
        # 质量价值因子：ROE / PB
        if (roe is not None and isinstance(roe, (int, float)) and
            pb_ratio is not None and isinstance(pb_ratio, (int, float)) and pb_ratio > 0):
            factors['roe_to_pb'] = roe / pb_ratio
        
        # 盈利能力与杠杆：ROE / 资产负债率
        if (roe is not None and isinstance(roe, (int, float)) and
            debt_to_equity is not None and isinstance(debt_to_equity, (int, float)) and debt_to_equity > 0):
            factors['roe_to_leverage'] = roe / debt_to_equity
        
        # 现金流质量：自由现金流 / 净利润
        if (free_cashflow is not None and isinstance(free_cashflow, (int, float)) and
            profit_margins is not None and isinstance(profit_margins, (int, float)) and
            market_cap is not None and isinstance(market_cap, (int, float))):
            net_income = profit_margins * market_cap
            if isinstance(net_income, (int, float)) and net_income > 0:
                factors['fcf_to_net_income'] = free_cashflow / net_income
        
        # 估值效率：市值 / 营收增长
        if (market_cap is not None and isinstance(market_cap, (int, float)) and
            revenue_growth is not None and isinstance(revenue_growth, (int, float)) and revenue_growth > 0):
            factors['mcap_to_revenue_growth'] = market_cap / revenue_growth
        
        # 股息质量：股息率 * 派息比率
        if (dividend_yield is not None and isinstance(dividend_yield, (int, float)) and
            payout_ratio is not None and isinstance(payout_ratio, (int, float))):
            factors['dividend_quality'] = dividend_yield * (1 - payout_ratio)
        
        # 机构认可度：机构持股 * ROE
        if (held_percent_institutions is not None and isinstance(held_percent_institutions, (int, float)) and
            roe is not None and isinstance(roe, (int, float))):
            factors['institution_quality'] = held_percent_institutions * roe
        
        return factors
    
    # ==================== 综合计算 ====================
    
    def calculate_all_fundamental_factors(self, code: str) -> dict:
        """
        计算所有基本面因子
        
        参数:
            code: 股票代码
        
        返回:
            包含所有基本面因子的字典
        """
        all_factors = {}
        
        # 获取并清理基本面信息
        info = self.get_stock_info(code)
        if info is None:
            return {}
        
        # 安全转换数值字段（防御性处理）
        clean_info = info.copy()
        numeric_cols = [
            'pe_ratio', 'pb_ratio', 'forward_pe', 'peg_ratio', 'price_to_sales',
            'enterprise_to_revenue', 'enterprise_to_ebitda', 'return_on_equity',
            'return_on_assets', 'profit_margins', 'gross_margins', 'operating_margins',
            'ebitda_margins', 'current_ratio', 'quick_ratio', 'debt_to_equity',
            'revenue_growth', 'earnings_growth', 'earnings_quarterly_growth',
            'dividend_yield', 'payout_ratio', 'five_year_avg_dividend_yield',
            'beta', 'operating_cashflow', 'market_cap', 'total_cash', 'total_debt',
            'free_cashflow', 'held_percent_insiders', 'held_percent_institutions',
            'float_shares', 'shares_outstanding', 'average_volume', 'average_volume_10days',
            'current_price', 'book_value', 'fifty_two_week_high', 'fifty_two_week_low',
            'fifty_day_average', 'two_hundred_day_average'
        ]
        
        for col in numeric_cols:
            if col in clean_info:
                val = pd.to_numeric(clean_info[col], errors='coerce')
                clean_info[col] = val if np.isfinite(val) else None

        # 合并各类因子
        all_factors.update(self.calculate_value_factors(code))
        all_factors.update(self.calculate_quality_factors(code))
        all_factors.update(self.calculate_growth_factors(code))
        all_factors.update(self.calculate_dividend_factors(code))
        all_factors.update(self.calculate_risk_factors(code))
        all_factors.update(self.calculate_price_momentum_factors(code))
        all_factors.update(self.calculate_cashflow_factors(code))
        all_factors.update(self.calculate_ownership_factors(code))
        all_factors.update(self.calculate_liquidity_factors(code))
        all_factors.update(self.calculate_size_factors(code))
        all_factors.update(self.calculate_industry_sector_factors(code))
        all_factors.update(self.calculate_cross_factors(code))
        
        # 添加相对强度因子
        rs_calculator = RelativeStrengthFactors(self.db_path)
        all_factors.update(rs_calculator.calculate_relative_strength_factors(code, clean_info))
        
        return all_factors
    
    def calculate_all_stocks_factors(self, codes: Optional[List[str]] = None) -> pd.DataFrame:
        """
        批量计算多只股票的基本面因子
        
        参数:
            codes: 股票代码列表，如果为None则计算所有股票
        
        返回:
            包含所有股票基本面因子的DataFrame
        """
        if codes is None:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT code FROM stock_info_extended")
                codes = [row[0] for row in cursor.fetchall()]
        
        results = []
        for code in codes:
            factors = self.calculate_all_fundamental_factors(code)
            factors['code'] = code
            results.append(factors)
        
        df = pd.DataFrame(results)
        
        # 将code列移到第一列
        if 'code' in df.columns:
            cols = ['code'] + [col for col in df.columns if col != 'code']
            df = df[cols]
        
        return df
    
    # ==================== 因子标准化 ====================
    
    def normalize_factors(self, df: pd.DataFrame, method='zscore') -> pd.DataFrame:
        """
        标准化因子
        
        参数:
            df: 因子DataFrame
            method: 标准化方法 ('zscore', 'minmax', 'rank')
        
        返回:
            标准化后的DataFrame
        """
        result = df.copy()
        
        # 排除非数值列
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        
        if method == 'zscore':
            # Z-score标准化
            for col in numeric_cols:
                mean = result[col].mean()
                std = result[col].std()
                if isinstance(std, (int, float)) and std > 0:
                    result[col] = (result[col] - mean) / std
        
        elif method == 'minmax':
            # Min-Max标准化到[0, 1]
            for col in numeric_cols:
                min_val = result[col].min()
                max_val = result[col].max()
                if isinstance(max_val, (int, float)) and isinstance(min_val, (int, float)) and max_val > min_val:
                    result[col] = (result[col] - min_val) / (max_val - min_val)
        
        elif method == 'rank':
            # 排名标准化到[0, 1]
            for col in numeric_cols:
                result[col] = result[col].rank(pct=True)
        
        return result


# 使用示例
if __name__ == '__main__':
    # 创建因子计算器
    calculator = FundamentalFactors()
    
    # 计算单只股票的因子
    print("=" * 60)
    print("单只股票因子计算示例")
    print("=" * 60)
    factors = calculator.calculate_all_fundamental_factors('000001')
    for key, value in factors.items():
        print(f"{key}: {value}")
    
    # 批量计算因子
    print("\n" + "=" * 60)
    print("批量因子计算示例")
    print("=" * 60)
    df = calculator.calculate_all_stocks_factors(['000001', '000002', '600000'])
    print(df)
    
    # 标准化因子
    print("\n" + "=" * 60)
    print("因子标准化示例")
    print("=" * 60)
    normalized = calculator.normalize_factors(df, method='zscore')
    print(normalized)

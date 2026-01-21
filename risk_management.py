# 风险管理模块 - ATR动态止损与仓位管理
import pandas as pd
import numpy as np
from datetime import datetime
from advanced_config import (
    ATR_STOP_LOSS_PARAMS, 
    POSITION_MANAGEMENT, 
    RISK_CONTROL_PARAMS
)

class RiskManager:
    def __init__(self):
        self.atr_params = ATR_STOP_LOSS_PARAMS
        self.position_params = POSITION_MANAGEMENT
        self.risk_params = RISK_CONTROL_PARAMS
        self.active_positions = {}
        self.daily_pnl = []
        
    def calculate_atr_stop_loss(self, data, entry_price, strategy_type='conservative'):
        """
        基于ATR计算动态止损位
        """
        if len(data) < self.atr_params['atr_period'] + 1:
            return entry_price * (1 - self.atr_params['min_stop_loss_percent'])
        
        # 计算ATR
        atr = self._calculate_atr(data, self.atr_params['atr_period'])
        
        if len(atr) == 0:
            return entry_price * (1 - self.atr_params['min_stop_loss_percent'])
        
        current_atr = atr[-1]
        
        # 根据策略类型选择ATR倍数
        if strategy_type == 'conservative':
            multiplier = self.atr_params['atr_multiplier_conservative']
        elif strategy_type == 'aggressive':
            multiplier = self.atr_params['atr_multiplier_aggressive']
        elif strategy_type == 'breakout':
            multiplier = self.atr_params['atr_multiplier_breakout']
        else:
            multiplier = self.atr_params['atr_multiplier_conservative']
        
        # 计算止损距离
        stop_distance = current_atr * multiplier
        stop_loss_price = entry_price - stop_distance
        
        # 应用最小和最大止损限制
        min_stop = entry_price * (1 - self.atr_params['max_stop_loss_percent'])
        max_stop = entry_price * (1 - self.atr_params['min_stop_loss_percent'])
        
        stop_loss_price = max(min_stop, min(stop_loss_price, max_stop))
        
        return stop_loss_price
    
    def calculate_position_size(self, account_balance, entry_price, stop_loss_price, risk_per_trade=None):
        """
        基于固定风险百分比计算仓位大小
        """
        if risk_per_trade is None:
            risk_per_trade = self.position_params['max_single_position_risk']
        
        # 计算每股风险
        risk_per_share = entry_price - stop_loss_price
        
        if risk_per_share <= 0:
            return 0
        
        # 计算总风险金额
        total_risk_amount = account_balance * risk_per_trade
        
        # 计算股数
        shares = int(total_risk_amount / risk_per_share)
        
        # 确保不超过账户余额
        max_shares = int(account_balance / entry_price)
        shares = min(shares, max_shares)
        
        return shares
    
    def validate_new_position(self, symbol, entry_price, stop_loss_price, position_size, account_balance):
        """
        验证新仓位是否符合风险管理规则
        """
        validation_result = {
            'approved': True,
            'warnings': [],
            'adjusted_size': position_size
        }
        
        # 1. 检查单笔风险
        position_value = entry_price * position_size
        risk_amount = (entry_price - stop_loss_price) * position_size
        risk_percentage = risk_amount / account_balance
        
        if risk_percentage > self.position_params['max_single_position_risk']:
            validation_result['warnings'].append(
                f"单笔风险 {risk_percentage:.2%} 超过限制 {self.position_params['max_single_position_risk']:.2%}"
            )
            # 调整仓位大小
            max_risk_amount = account_balance * self.position_params['max_single_position_risk']
            validation_result['adjusted_size'] = int(max_risk_amount / (entry_price - stop_loss_price))
        
        # 2. 检查总风险敞口
        total_current_risk = self._calculate_total_risk()
        new_total_risk = (total_current_risk + risk_amount) / account_balance
        
        if new_total_risk > self.position_params['max_total_risk']:
            validation_result['warnings'].append(
                f"总风险敞口 {new_total_risk:.2%} 超过限制 {self.position_params['max_total_risk']:.2%}"
            )
            validation_result['approved'] = False
        
        # 3. 检查相关性持仓数量
        if len(self.active_positions) >= self.position_params['max_correlation_positions']:
            validation_result['warnings'].append(
                f"持仓数量 {len(self.active_positions)} 达到上限"
            )
            validation_result['approved'] = False
        
        return validation_result
    
    def update_trailing_stop(self, symbol, current_price, data):
        """
        更新追踪止损
        """
        if symbol not in self.active_positions:
            return None
        
        position = self.active_positions[symbol]
        
        # 计算当前ATR
        atr = self._calculate_atr(data, self.atr_params['atr_period'])
        if len(atr) == 0:
            return position['stop_loss']
        
        current_atr = atr[-1]
        
        # 计算新的追踪止损位
        if position['direction'] == 'long':
            new_stop = current_price - current_atr * self.atr_params['atr_multiplier_conservative']
            # 止损只能向上调整
            if new_stop > position['stop_loss']:
                position['stop_loss'] = new_stop
                position['last_update'] = datetime.now()
        
        return position['stop_loss']
    
    def check_risk_limits(self, account_balance):
        """
        检查风险限制
        """
        alerts = []
        
        # 检查日亏损限制
        if len(self.daily_pnl) > 0:
            today_pnl = self.daily_pnl[-1] if self.daily_pnl else 0
            daily_loss_pct = abs(today_pnl) / account_balance if today_pnl < 0 else 0
            
            if daily_loss_pct > self.risk_params['daily_loss_limit']:
                alerts.append({
                    'type': 'daily_loss_limit',
                    'message': f"日亏损 {daily_loss_pct:.2%} 超过限制 {self.risk_params['daily_loss_limit']:.2%}",
                    'severity': 'high'
                })
        
        # 检查连续亏损
        consecutive_losses = self._count_consecutive_losses()
        if consecutive_losses >= self.risk_params['max_consecutive_losses']:
            alerts.append({
                'type': 'consecutive_losses',
                'message': f"连续亏损 {consecutive_losses} 次，达到限制",
                'severity': 'high'
            })
        
        # 检查总风险敞口
        total_risk_pct = self._calculate_total_risk() / account_balance
        if total_risk_pct > self.position_params['max_total_risk']:
            alerts.append({
                'type': 'total_risk_exposure',
                'message': f"总风险敞口 {total_risk_pct:.2%} 超过限制",
                'severity': 'medium'
            })
        
        return alerts
    
    def add_position(self, symbol, entry_price, stop_loss, position_size, strategy):
        """
        添加新仓位到风险管理系统
        """
        self.active_positions[symbol] = {
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'position_size': position_size,
            'strategy': strategy,
            'direction': 'long',  # 简化为只考虑做多
            'entry_time': datetime.now(),
            'last_update': datetime.now()
        }
    
    def remove_position(self, symbol, exit_price=None):
        """
        移除仓位并记录盈亏
        """
        if symbol in self.active_positions:
            position = self.active_positions[symbol]
            
            if exit_price:
                pnl = (exit_price - position['entry_price']) * position['position_size']
                self._record_trade_pnl(pnl)
            
            del self.active_positions[symbol]
    
    def get_position_status(self, symbol):
        """
        获取仓位状态
        """
        if symbol not in self.active_positions:
            return None
        
        position = self.active_positions[symbol]
        return {
            'symbol': symbol,
            'entry_price': position['entry_price'],
            'current_stop_loss': position['stop_loss'],
            'position_size': position['position_size'],
            'strategy': position['strategy'],
            'days_held': (datetime.now() - position['entry_time']).days,
            'last_update': position['last_update']
        }
    
    def generate_risk_report(self, account_balance):
        """
        生成风险管理报告
        """
        total_positions = len(self.active_positions)
        total_risk = self._calculate_total_risk()
        risk_percentage = total_risk / account_balance if account_balance > 0 else 0
        
        # 计算各策略风险分布
        strategy_risk = {}
        for symbol, position in self.active_positions.items():
            strategy = position['strategy']
            risk = (position['entry_price'] - position['stop_loss']) * position['position_size']
            strategy_risk[strategy] = strategy_risk.get(strategy, 0) + risk
        
        # 计算近期表现
        recent_pnl = sum(self.daily_pnl[-30:]) if len(self.daily_pnl) >= 30 else sum(self.daily_pnl)
        
        report = {
            'timestamp': datetime.now(),
            'account_balance': account_balance,
            'total_positions': total_positions,
            'total_risk_amount': total_risk,
            'total_risk_percentage': risk_percentage,
            'available_risk': account_balance * self.position_params['max_total_risk'] - total_risk,
            'strategy_risk_distribution': strategy_risk,
            'recent_30day_pnl': recent_pnl,
            'consecutive_losses': self._count_consecutive_losses(),
            'risk_alerts': self.check_risk_limits(account_balance)
        }
        
        return report
    
    def _calculate_atr(self, data, period=14):
        """计算平均真实波幅"""
        if len(data) < period + 1:
            return []
        
        high = data['high']
        low = data['low']
        close = data['close']
        
        # 计算真实波幅
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # 计算ATR
        atr = tr.rolling(window=period).mean()
        
        return atr.dropna().values
    
    def _calculate_total_risk(self):
        """计算总风险敞口"""
        total_risk = 0
        for position in self.active_positions.values():
            risk = (position['entry_price'] - position['stop_loss']) * position['position_size']
            total_risk += risk
        return total_risk
    
    def _count_consecutive_losses(self):
        """计算连续亏损次数"""
        if not self.daily_pnl:
            return 0
        
        consecutive = 0
        for pnl in reversed(self.daily_pnl):
            if pnl < 0:
                consecutive += 1
            else:
                break
        
        return consecutive
    
    def _record_trade_pnl(self, pnl):
        """记录交易盈亏"""
        today = datetime.now().date()
        
        # 如果是新的一天，添加新记录
        if not self.daily_pnl or len(self.daily_pnl) == 0:
            self.daily_pnl.append(pnl)
        else:
            # 累加到当日盈亏
            self.daily_pnl[-1] += pnl
    
    def export_risk_metrics(self, filename=None):
        """导出风险指标到文件"""
        if filename is None:
            filename = f"risk_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        # 准备数据
        positions_data = []
        for symbol, position in self.active_positions.items():
            positions_data.append({
                'symbol': symbol,
                'entry_price': position['entry_price'],
                'stop_loss': position['stop_loss'],
                'position_size': position['position_size'],
                'strategy': position['strategy'],
                'entry_time': position['entry_time'],
                'risk_amount': (position['entry_price'] - position['stop_loss']) * position['position_size']
            })
        
        if positions_data:
            df = pd.DataFrame(positions_data)
            df.to_csv(filename, index=False, encoding='utf-8-sig')
            print(f"风险指标已导出到: {filename}")
        else:
            print("当前无持仓，无法导出风险指标")
    
    def reset_daily_tracking(self):
        """重置日度跟踪数据（每日开盘前调用）"""
        # 添加新的一天的PnL记录
        self.daily_pnl.append(0)
        
        # 保留最近90天的数据
        if len(self.daily_pnl) > 90:
            self.daily_pnl = self.daily_pnl[-90:]
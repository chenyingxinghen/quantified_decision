"""
特征工程增强模块
提供交叉因子、衍生因子和特征变换功能
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any
from sklearn.preprocessing import PolynomialFeatures
from config import TrainingConfig


class FeatureEngineer:
    """特征工程器 - 生成交叉因子和衍生因子"""
    
    def __init__(self):
        """初始化特征工程器"""
        self.generated_features = []
    
    def create_ratio_features(self, df: pd.DataFrame, 
                             numerator_cols: List[str], 
                             denominator_cols: List[str],
                             return_dict: bool = False) -> Any:
        """
        创建比率特征
        """
        new_features = {}
        
        # 预先检查并缓存数值
        cached_series = {}
        target_cols = list(set(numerator_cols + denominator_cols))
        for col in target_cols:
            if col in df.columns:
                cached_series[col] = pd.to_numeric(df[col], errors='coerce')

        for num_col in numerator_cols:
            if num_col not in cached_series:
                continue
            
            for den_col in denominator_cols:
                if den_col not in cached_series or num_col == den_col:
                    continue
                
                feature_name = f'{num_col}_div_{den_col}'
                
                # 避免除以零和NaN
                denominator = cached_series[den_col].replace(0, np.nan)
                ratio = cached_series[num_col] / denominator
                
                new_features[feature_name] = ratio.replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
                self.generated_features.append(feature_name)
        
        if return_dict:
            return new_features
            
        if new_features:
            return pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)
        return df
    
    def create_product_features(self, df: pd.DataFrame, 
                               col_pairs: List[tuple],
                               return_dict: bool = False) -> Any:
        """
        创建乘积特征
        """
        new_features = {}
        
        # 预加载
        unique_cols = list(set([c for pair in col_pairs for c in pair]))
        cached_series = {c: pd.to_numeric(df[c], errors='coerce') for c in unique_cols if c in df.columns}

        for col1, col2 in col_pairs:
            if col1 not in cached_series or col2 not in cached_series:
                continue
            
            feature_name = f'{col1}_mul_{col2}'
            product = cached_series[col1] * cached_series[col2]
            new_features[feature_name] = product.replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
            self.generated_features.append(feature_name)
        
        if return_dict:
            return new_features
            
        if new_features:
            return pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)
        return df
    
    def create_difference_features(self, df: pd.DataFrame, 
                                  col_pairs: List[tuple],
                                  return_dict: bool = False) -> Any:
        """
        创建差值特征
        """
        new_features = {}
        
        # 预计算
        unique_cols = list(set([c for pair in col_pairs for c in pair]))
        cached_series = {c: pd.to_numeric(df[c], errors='coerce') for c in unique_cols if c in df.columns}
        
        for col1, col2 in col_pairs:
            if col1 not in cached_series or col2 not in cached_series:
                continue
            
            feature_name = f'{col1}_sub_{col2}'
            diff = cached_series[col1] - cached_series[col2]
            new_features[feature_name] = diff.replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
            self.generated_features.append(feature_name)
        
        if return_dict:
            return new_features
            
        if new_features:
            return pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)
        return df
    
    def create_polynomial_features(self, df: pd.DataFrame, 
                                  columns: List[str], 
                                  degree: int = 2) -> pd.DataFrame:
        """
        创建多项式特征
        
        参数:
            df: 输入DataFrame
            columns: 要生成多项式特征的列
            degree: 多项式阶数
        
        返回:
            包含多项式特征的DataFrame
        """
        # 选择指定列
        selected_cols = [col for col in columns if col in df.columns]
        if not selected_cols:
            return df
        
        X = df[selected_cols].fillna(0)
        
        # 生成多项式特征
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly.fit_transform(X)
        
        # 获取特征名称
        feature_names = poly.get_feature_names_out(selected_cols)
        
        # 只保留新生成的特征（排除原始特征）
        new_features = feature_names[len(selected_cols):]
        X_new = X_poly[:, len(selected_cols):]
        
        # 添加到结果DataFrame
        for i, name in enumerate(new_features):
            df[name] = X_new[:, i]
            self.generated_features.append(name)
        
        return df
    
    def create_log_features(self, df: pd.DataFrame, 
                           columns: List[str],
                           return_dict: bool = False) -> Any:
        """
        创建对数特征
        """
        new_features = {}
        
        for col in columns:
            if col not in df.columns:
                continue
            
            feature_name = f'log_{col}'
            series = pd.to_numeric(df[col], errors='coerce')
            positive_values = series.clip(lower=1e-10)
            log_values = np.log(positive_values)
            
            new_features[feature_name] = log_values.replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
            self.generated_features.append(feature_name)
        
        if return_dict:
            return new_features
            
        if new_features:
            return pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)
        return df
    
    def create_sqrt_features(self, df: pd.DataFrame, 
                             columns: List[str],
                             return_dict: bool = False) -> Any:
        """
        创建平方根特征
        """
        new_features = {}
        
        for col in columns:
            if col not in df.columns:
                continue
            
            feature_name = f'sqrt_{col}'
            series = pd.to_numeric(df[col], errors='coerce')
            non_negative = series.clip(lower=0)
            new_features[feature_name] = np.sqrt(non_negative).fillna(0).astype(np.float32)
            
            self.generated_features.append(feature_name)
        
        if return_dict:
            return new_features
            
        if new_features:
            return pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)
        return df
    
    def create_rank_features(self, df: pd.DataFrame, 
                             columns: List[str],
                             window: int = 252,
                             return_dict: bool = False) -> Any:
        """
        创建排名特征
        """
        new_features = {}
        
        for col in columns:
            if col not in df.columns:
                continue
            
            feature_name = f'rank_{col}'
            series = pd.to_numeric(df[col], errors='coerce')
            new_features[feature_name] = series.rolling(window=window, min_periods=window//2).rank(pct=True).astype(np.float32)
            self.generated_features.append(feature_name)
        
        if return_dict:
            return new_features
            
        if new_features:
            return pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)
        return df
    
    def create_quantile_features(self, df: pd.DataFrame, 
                                 columns: List[str],
                                 window: int = 252,
                                 n_quantiles: int = 5,
                                 return_dict: bool = False) -> Any:
        """
        创建分位数特征
        """
        new_features = {}
        
        for col in columns:
            if col not in df.columns:
                continue
            
            feature_name = f'quantile_{col}'
            series = pd.to_numeric(df[col], errors='coerce')
            rolled_rank = series.rolling(window=window, min_periods=window//4).rank(pct=True)
            
            new_features[feature_name] = (rolled_rank * n_quantiles).fillna(0).astype(np.float32).clip(0, n_quantiles-1)
            self.generated_features.append(feature_name)
        
        if return_dict:
            return new_features
            
        if new_features:
            return pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)
        return df
    
    def create_interaction_features(self, df: pd.DataFrame,
                                    technical_cols: List[str],
                                    fundamental_cols: List[str],
                                    return_dict: bool = False) -> Any:
        """
        创建交互特征
        """
        new_features = {}
        
        important_tech = [col for col in technical_cols if col in df.columns][:10]
        important_fund = [col for col in fundamental_cols if col in df.columns][:10]
        
        # 预加载
        cached_tech = {c: pd.to_numeric(df[c], errors='coerce') for c in important_tech}
        cached_fund = {c: pd.to_numeric(df[c], errors='coerce') for c in important_fund}

        for tech_col in important_tech:
            for fund_col in important_fund:
                feature_name = f'{tech_col}_x_{fund_col}'
                interaction = cached_tech[tech_col] * cached_fund[fund_col]
                new_features[feature_name] = interaction.replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
                self.generated_features.append(feature_name)
        
        if return_dict:
            return new_features
            
        if new_features:
            return pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)
        return df
    
    def create_momentum_features(self, df: pd.DataFrame,
                                columns: List[str],
                                windows: List[int] = [5, 10, 20],
                                return_dict: bool = False) -> Any:
        """
        创建动量特征
        """
        new_features = {}
        
        for col in columns:
            if col not in df.columns:
                continue
            
            series = pd.to_numeric(df[col], errors='coerce')
            for window in windows:
                feature_name = f'{col}_momentum_{window}'
                momentum = series.pct_change(window)
                new_features[feature_name] = momentum.replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
                self.generated_features.append(feature_name)
        
        if return_dict:
            return new_features
            
        if new_features:
            return pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)
        return df
    
    def create_volatility_features(self, df: pd.DataFrame,
                                  columns: List[str],
                                  windows: List[int] = [5, 10, 20],
                                  return_dict: bool = False) -> Any:
        """
        创建波动率特征
        """
        new_features = {}
        
        for col in columns:
            if col not in df.columns:
                continue
            
            series = pd.to_numeric(df[col], errors='coerce').replace([np.inf, -np.inf], np.nan)
            for window in windows:
                feature_name = f'{col}_volatility_{window}'
                volatility = series.rolling(window).std()
                new_features[feature_name] = volatility.fillna(0).astype(np.float32)
                self.generated_features.append(feature_name)
        
        if return_dict:
            return new_features
            
        if new_features:
            return pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame,
                                   categorical_cols: List[str] = None) -> pd.DataFrame:
        """
        编码分类特征（行业、板块等）
        使用全局映射确保不同股票之间的编码一致性
        
        参数:
            df: 输入DataFrame
            categorical_cols: 分类列名列表，如果为None则自动检测
        
        返回:
            包含编码后分类特征的DataFrame
        """
        new_features = {}
        
        # 1. 尝试从数据库补充分类信息 (如 industry)
        if 'industry' not in df.columns and 'code' in df.columns:
            try:
                from core.data.baostock_fetcher import BaostockFetcher
                fetcher = BaostockFetcher()
                db_industry = fetcher._get_stock_industry_from_db()
                fetcher.close()
                
                if not db_industry.empty:
                    # 仅保留 code 和 industry
                    db_industry = db_industry[['code', 'industry']].drop_duplicates('code')
                    # 合并到主 DataFrame (基于 code)
                    df = df.merge(db_industry, on='code', how='left')
            except Exception as e:
                # 记录但不中断，可能因为没有 code 列或数据库连接失败
                pass
        
        if categorical_cols is None:
            # 自动检测分类列
            categorical_cols = [col for col in df.columns if col in ['sector', 'industry']]
        
        # 全局固定的行业列表（A股 Baostock 行业分类，顺序固定确保编码一致）
        # 编码规则: 0=Unknown, 1=制造业, 2=金融业, ... 与股票数量无关
        GLOBAL_INDUSTRY_MAPPING = {
            '农、林、牧、渔业': 1,
            '采矿业': 2,
            '制造业': 3,
            '电力、热力、燃气及水生产和供应业': 4,
            '建筑业': 5,
            '批发和零售业': 6,
            '交通运输、仓储和邮政业': 7,
            '住宿和餐饮业': 8,
            '信息传输、软件和信息技术服务业': 9,
            '金融业': 10,
            '房地产业': 11,
            '租赁和商务服务业': 12,
            '科学研究和技术服务业': 13,
            '水利、环境和公共设施管理业': 14,
            '居民服务、修理和其他服务业': 15,
            '教育': 16,
            '卫生和社会工作': 17,
            '文化、体育和娱乐业': 18,
            '公共管理、社会保障和社会组织': 19,
            'Unknown': 0,
        }
        # One-Hot 列表与映射保持一致（排除 Unknown）
        GLOBAL_TOP_INDUSTRIES = [k for k in GLOBAL_INDUSTRY_MAPPING if k != 'Unknown']

        for col in categorical_cols:
            if col not in df.columns:
                continue
                
            try:
                col_data = df[col].fillna('Unknown').astype(str)

                # 使用全局固定映射，不依赖当前数据中出现的类别
                # 修复：动态映射在单股处理时只有1个类别，导致所有股票 encoded=1
                encoded = col_data.map(GLOBAL_INDUSTRY_MAPPING).fillna(0).astype(int)

                feature_name = f'{col}_encoded'
                new_features[feature_name] = encoded
                self.generated_features.append(feature_name)

                # One-Hot 编码（全局固定列，确保所有股票生成相同的列名）
                for cat in GLOBAL_TOP_INDUSTRIES:
                    safe_cat_name = str(cat).replace(' ', '_').replace('&', 'and').replace('-', '_').replace('、', '_').replace('，', '_')
                    oh_feature_name = f'{col}_{safe_cat_name}'
                    new_features[oh_feature_name] = (col_data == cat).astype(int)
                    self.generated_features.append(oh_feature_name)

            except Exception as e:
                pass
        
        # 一次性添加所有新列
        if new_features:
            df = pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)
        
        return df
    
    def apply_all_transformations(self, df: pd.DataFrame,
                                  config: Optional[Dict] = None,
                                  verbose: bool = False) -> pd.DataFrame:
        """
        应用所有特征工程变换 (Batch 优化版本)
        """
        self.generated_features = [] 
        if config is None:
            config = {
                'ratio': True, 'product': True, 'difference': True,
                'log': True, 'sqrt': True, 'rank': True,
                'interaction': True, 'categorical': True,
            }
        
        # 收集所有新生成的特征，最后一次性合并，避免多次 DataFrame 拷贝
        collected_features = {}
        stats = {}
        initial_count = len(df.columns)

        # 1. 编码分类特征 (这步可能修改原 DataFrame，先执行)
        if config.get('categorical'):
            pre_count = len(self.generated_features)
            df = self.encode_categorical_features(df)
            stats['分类特征编码'] = len(self.generated_features) - pre_count
        
        # 识别技术指标和基本面因子
        def fuzzy_match(col_name, keywords):
            # 改进：不再暴力替换所有下划线，而是按词匹配
            # 避免 'pe' 匹配到 'is_suspended'
            parts = col_name.lower().split('_')
            for kw in keywords:
                kw_l = kw.lower().replace('_', '')
                # 如果关键词包含在任何一部分中，或者整列名（去下划线）包含关键词
                # 但排除掉一些极短的关键词在长单词中间的匹配
                if len(kw_l) <= 2:
                    # 对于 pe, pb 等短关键词，要求必须是独立的单词部分
                    if any(kw_l == p for p in parts):
                        return True
                else:
                    if kw_l in col_name.lower().replace('_', ''):
                        return True
            return False

        # 技术指标关键词：仅匹配纯技术指标，排除成交量/金额类（避免与高级时序因子混淆）
        # 注意：不包含 'amount'/'vol'/'turnover'，这些属于高级时序因子，量纲与技术指标不同，
        # 混入差分/乘积特征会产生无经济意义的组合（如 adx_14_sub_amount_per_volume）
        tech_keywords = ['rsi', 'macd', 'kdj', 'adx', 'atr', 'cci', 'mfi', 'obv', 'willr', 'bias', 'psy', 'boll', 'natr', 'cmo', 'roc', 'trix', 'aroon']
        # 排除高级时序因子（amount_per_volume, price_volatility 等）和状态列
        tech_exclude = ['amount_per_volume', 'amount_change_rate', 'volume_change_rate', 'volume_volatility',
                        'price_volume_corr', 'price_volatility', 'price_skewness', 'price_kurtosis',
                        'hl_range', 'oc_ratio', 'high_position', 'low_position', 'return_', 'momentum_',
                        'downside_risk', 'drawdown', 'sharpe', 'is_', 'days_to_', 'market_']
        tech_indicators = [col for col in df.columns 
                           if fuzzy_match(col, tech_keywords)
                           and not any(col.startswith(ex) or ex in col for ex in tech_exclude)
                           and not TrainingConfig.should_skip_transform(col)][:100]

        fund_keywords = ['pe', 'pb', 'roe', 'roa', 'margin', 'growth', 'yield', 'beta', 'market_cap', 'marketcap', 
                         'peg', 'sue', 'eav', 'revenue', 'share', 'ttm', 'yoy', 'ratio', 'equity', 'asset', 'profit']
        fundamental_factors = [col for col in df.columns if fuzzy_match(col, fund_keywords) 
                              and not any(x in col.lower() for x in ['slope', 'sharpe'])
                              and not TrainingConfig.should_skip_transform(col)][:60]
        
        # 2. 批量生成各类特征并存入 collected_features
        # 注意：使用确定性的固定索引选取，而非 random.sample，
        # 确保不同股票（列集合可能不同）生成相同名称的特征，避免 target_features 对齐时出现大量缺失。

        if config.get('ratio') and len(fundamental_factors) > 1:
            pre_count = len(self.generated_features)
            sorted_factors = sorted(fundamental_factors)
            # 均匀间隔取 10 个，保证列名确定
            step = max(1, len(sorted_factors) // 10)
            selected_factors = sorted_factors[::step][:10]
            collected_features.update(self.create_ratio_features(df, selected_factors[:5], selected_factors[5:10], return_dict=True))
            stats['比率特征'] = len(self.generated_features) - pre_count

        if config.get('product') and len(tech_indicators) > 1:
            pre_count = len(self.generated_features)
            # 乘积特征：选取趋势强度类指标（ADX、ATR 等），与动量类组合有经济意义
            trend_kws = ['adx', 'atr', 'natr', 'cci', 'mfi']
            trend_cols = [col for col in tech_indicators if any(kw in col.lower() for kw in trend_kws)]
            if len(trend_cols) < 2:
                trend_cols = sorted(tech_indicators)[:4]
            else:
                trend_cols = sorted(trend_cols)[:4]
            pairs = [(trend_cols[i], trend_cols[j]) for i in range(len(trend_cols)) for j in range(i+1, len(trend_cols))]
            collected_features.update(self.create_product_features(df, pairs, return_dict=True))
            stats['乘积特征'] = len(self.generated_features) - pre_count
        
        if config.get('difference') and len(tech_indicators) > 1:
            pre_count = len(self.generated_features)
            # 差分特征：只在同类量纲的指标间做差分，避免无意义的跨量纲组合
            # 优先选取动量/趋势类指标（RSI、ROC、CMO、WILLR 等都在 0-100 或 -100~100 范围）
            momentum_trend_kws = ['rsi', 'roc', 'cmo', 'willr', 'bias', 'aroon', 'trix']
            momentum_cols = [col for col in tech_indicators if any(kw in col.lower() for kw in momentum_trend_kws)]
            # 如果动量类不足，补充其他技术指标
            if len(momentum_cols) < 4:
                momentum_cols = sorted(tech_indicators)[:6]
            else:
                momentum_cols = sorted(momentum_cols)[:6]
            diff_pairs = [(momentum_cols[i], momentum_cols[j]) for i in range(len(momentum_cols)) for j in range(i+1, len(momentum_cols))]
            collected_features.update(self.create_difference_features(df, diff_pairs, return_dict=True))
            stats['差分特征'] = len(self.generated_features) - pre_count

        if config.get('log'):
            pre_count = len(self.generated_features)
            log_kws = ['market_cap', 'revenue', 'equity', 'asset', 'profit', 'cash']
            log_cols = [col for col in fundamental_factors if any(kw in col.lower() for kw in log_kws)]
            # 兼容技术指标
            log_cols += [col for col in tech_indicators if 'vol' in col.lower() or 'amount' in col.lower()]
            if log_cols:
                sorted_cols = sorted(list(set(log_cols)))
                selected_cols = sorted_cols[:8]
                collected_features.update(self.create_log_features(df, selected_cols, return_dict=True))
            stats['对数变换'] = len(self.generated_features) - pre_count

        if config.get('sqrt'):
            pre_count = len(self.generated_features)
            sqrt_cols = [col for col in tech_indicators if any(kw in col.lower() for kw in ['volatility', 'atr', 'vol', 'amount'])]
            if sqrt_cols:
                sorted_cols = sorted(list(set(sqrt_cols)))
                selected_cols = sorted_cols[:5]
                collected_features.update(self.create_sqrt_features(df, selected_cols, return_dict=True))
            stats['平方根变换'] = len(self.generated_features) - pre_count

        if config.get('rank'):
            pre_count = len(self.generated_features)
            sorted_factors = sorted(fundamental_factors)
            selected_factors = sorted_factors[:15]
            collected_features.update(self.create_rank_features(df, selected_factors, return_dict=True))
            stats['滚动排名'] = len(self.generated_features) - pre_count

        if config.get('interaction') and len(tech_indicators) > 0 and len(fundamental_factors) > 0:
            pre_count = len(self.generated_features)
            sorted_tech = sorted(tech_indicators)
            sorted_fund = sorted(fundamental_factors)
            selected_tech = sorted_tech[:6]
            selected_fund = sorted_fund[:5]
            collected_features.update(self.create_interaction_features(df, selected_tech, selected_fund, return_dict=True))
            stats['交互特征'] = len(self.generated_features) - pre_count

        # === 新增：长短线强制交叉特征 (Long-Short Divergence) ===
        pre_count = len(self.generated_features)
        cross_features = {}
        # 1. 高位量能背离: vroc_36 / bias_36
        if 'vroc_36' in df.columns and 'bias_36' in df.columns:
            bias_val = df['bias_36'].astype(np.float32).replace(0, 1e-4) # 防止除零
            cross_features['vroc36_div_bias36'] = (df['vroc_36'].astype(np.float32) / bias_val).replace([np.inf, -np.inf], 0).fillna(0)
            self.generated_features.append('vroc36_div_bias36')
            
        # 2. 短线高位滞涨: return_5d / vol_std_60
        if 'return_5d' in df.columns and 'vol_std_60' in df.columns:
            vol_val = df['vol_std_60'].astype(np.float32).replace(0, 1e-4)
            cross_features['return5d_div_volstd60'] = (df['return_5d'].astype(np.float32) / vol_val).replace([np.inf, -np.inf], 0).fillna(0)
            self.generated_features.append('return5d_div_volstd60')

        # 3. 相对动能衰竭: momentum_5d / bias_36
        if 'momentum_5d' in df.columns and 'bias_36' in df.columns:
            bias_val = df['bias_36'].astype(np.float32).replace(0, 1e-4)
            cross_features['momentum5d_div_bias36'] = (df['momentum_5d'].astype(np.float32) / bias_val).replace([np.inf, -np.inf], 0).fillna(0)
            self.generated_features.append('momentum5d_div_bias36')
            
        if cross_features:
            collected_features.update(cross_features)
            stats['长短线背离交叉特征'] = len(self.generated_features) - pre_count

        # 3. 最终合并 (仅一次)
        if collected_features:
            new_df = pd.DataFrame(collected_features, index=df.index)
            # 确保类型为 float32
            new_df = new_df.astype(np.float32)
            df = pd.concat([df, new_df], axis=1)

        if verbose:
            print("\n" + "-"*40)
            print("特征工程报告 (Batch 优化版):")
            for name, count in stats.items():
                if count > 0: print(f"  - {name}: +{count} 个")
            print(f"  总计: 原始 {initial_count} -> 现计 {len(df.columns)} (新增 {len(self.generated_features)})")
            print("-"*40 + "\n")
        
        return df
    
    def get_generated_features(self) -> List[str]:
        """获取生成的特征名称列表"""
        return self.generated_features
    
    def reset(self):
        """重置生成的特征列表"""
        self.generated_features = []


# 使用示例
if __name__ == '__main__':
    # 创建示例数据
    df = pd.DataFrame({
        'rsi_6': np.random.rand(100) * 100,
        'macd': np.random.randn(100),
        'pe_ratio': np.random.rand(100) * 50,
        'roe': np.random.rand(100) * 0.3,
        'market_cap': np.random.rand(100) * 1e10,
        'hammer': np.random.choice([0, 1], 100).astype(float),
        'doji': np.random.choice([0, 1], 100).astype(float),
    })

    
    # 创建特征工程器
    engineer = FeatureEngineer()
    
    # 应用所有变换
    df_enhanced = engineer.apply_all_transformations(df)
    
    print(f"\n原始特征数: {len(df.columns)}")
    print(f"增强后特征数: {len(df_enhanced.columns)}")
    print(f"\n新增特征: {engineer.get_generated_features()[:10]}")

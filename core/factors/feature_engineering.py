"""
特征工程增强模块
提供交叉因子、衍生因子和特征变换功能
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any
from sklearn.preprocessing import PolynomialFeatures


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
        
        for col in categorical_cols:
            if col not in df.columns:
                continue
                
            try:
                # 动态生成映射表 (基于列中的现有类别)
                # 这比硬编码更灵活，能适应 Baostock 返回的所有行业
                unique_categories = df[col].dropna().unique()
                mapping = {cat: i + 1 for i, cat in enumerate(sorted(unique_categories))}
                mapping['Unknown'] = 0
                
                # 执行编码
                col_data = df[col].fillna('Unknown').astype(str)
                encoded = col_data.map(mapping).fillna(0).astype(int)
                
                feature_name = f'{col}_encoded'
                new_features[feature_name] = encoded
                self.generated_features.append(feature_name)
                
                # 如果分类列在 top 10，则进行 One-Hot 编码
                top_cats = df[col].value_counts().head(10).index.tolist()
                for cat in top_cats:
                    if pd.isna(cat) or cat == 'Unknown': continue
                    # 清理分类名称用于列名
                    safe_cat_name = str(cat).replace(' ', '_').replace('&', 'and').replace('-', '_')
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
            clean_col = col_name.lower().replace('_', '')
            for kw in keywords:
                if kw.lower().replace('_', '') in clean_col:
                    return True
            return False

        tech_keywords = ['rsi', 'macd', 'kdj', 'adx', 'atr', 'cci', 'mfi', 'obv', 'willr', 'bias', 'psy', 'boll', 'ma', 'ema', 'vol', 'amount', 'turnover']
        tech_indicators = [col for col in df.columns if fuzzy_match(col, tech_keywords)][:100]

        fund_keywords = ['pe', 'pb', 'roe', 'roa', 'margin', 'growth', 'yield', 'beta', 'market_cap', 'marketcap', 
                         'peg', 'sue', 'eav', 'revenue', 'share', 'ttm', 'yoy', 'ratio', 'equity', 'asset', 'profit']
        fundamental_factors = [col for col in df.columns if fuzzy_match(col, fund_keywords) 
                              and not any(x in col.lower() for x in ['slope', 'sharpe'])][:60]
        
        # 2. 批量生成各类特征并存入 collected_features
        import random
        random.seed(42)

        if config.get('ratio') and len(fundamental_factors) > 1:
            pre_count = len(self.generated_features)
            sorted_factors = sorted(fundamental_factors)
            selected_factors = random.sample(sorted_factors, min(10, len(sorted_factors)))
            collected_features.update(self.create_ratio_features(df, selected_factors[:5], selected_factors[5:10], return_dict=True))
            stats['比率特征'] = len(self.generated_features) - pre_count

        if config.get('product') and len(tech_indicators) > 1:
            pre_count = len(self.generated_features)
            sorted_tech = sorted(tech_indicators)
            selected_tech = random.sample(sorted_tech, min(4, len(sorted_tech)))
            pairs = [(selected_tech[i], selected_tech[j]) for i in range(len(selected_tech)) for j in range(i+1, len(selected_tech))]
            collected_features.update(self.create_product_features(df, pairs, return_dict=True))
            stats['乘积特征'] = len(self.generated_features) - pre_count
        
        if config.get('difference') and len(tech_indicators) > 1:
            pre_count = len(self.generated_features)
            sorted_tech = sorted(tech_indicators)
            selected_tech = random.sample(sorted_tech, min(6, len(sorted_tech)))
            diff_pairs = [(selected_tech[i], selected_tech[j]) for i in range(len(selected_tech)) for j in range(i+1, len(selected_tech))]
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
                selected_cols = random.sample(sorted_cols, min(8, len(sorted_cols)))
                collected_features.update(self.create_log_features(df, selected_cols, return_dict=True))
            stats['对数变换'] = len(self.generated_features) - pre_count

        if config.get('sqrt'):
            pre_count = len(self.generated_features)
            sqrt_cols = [col for col in tech_indicators if any(kw in col.lower() for kw in ['volatility', 'atr', 'vol', 'amount'])]
            if sqrt_cols:
                sorted_cols = sorted(list(set(sqrt_cols)))
                selected_cols = random.sample(sorted_cols, min(5, len(sorted_cols)))
                collected_features.update(self.create_sqrt_features(df, selected_cols, return_dict=True))
            stats['平方根变换'] = len(self.generated_features) - pre_count

        if config.get('rank'):
            pre_count = len(self.generated_features)
            sorted_factors = sorted(fundamental_factors)
            selected_factors = random.sample(sorted_factors, min(15, len(sorted_factors)))
            collected_features.update(self.create_rank_features(df, selected_factors, return_dict=True))
            stats['滚动排名'] = len(self.generated_features) - pre_count

        if config.get('interaction') and len(tech_indicators) > 0 and len(fundamental_factors) > 0:
            pre_count = len(self.generated_features)
            sorted_tech = sorted(tech_indicators)
            sorted_fund = sorted(fundamental_factors)
            selected_tech = random.sample(sorted_tech, min(6, len(sorted_tech)))
            selected_fund = random.sample(sorted_fund, min(5, len(sorted_fund)))
            collected_features.update(self.create_interaction_features(df, selected_tech, selected_fund, return_dict=True))
            stats['交互特征'] = len(self.generated_features) - pre_count

        # 3. 最终合并 (仅一次)
        if collected_features:
            new_df = pd.DataFrame(collected_features, index=df.index)
            # 确保类型为 float32
            new_df = new_df.astype(np.float32, copy=False)
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

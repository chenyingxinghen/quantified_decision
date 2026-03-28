"""
特征工程增强模块
提供交叉因子、衍生因子和特征变换功能
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from sklearn.preprocessing import PolynomialFeatures


class FeatureEngineer:
    """特征工程器 - 生成交叉因子和衍生因子"""
    
    def __init__(self):
        """初始化特征工程器"""
        self.generated_features = []
    
    def create_ratio_features(self, df: pd.DataFrame, 
                             numerator_cols: List[str], 
                             denominator_cols: List[str]) -> pd.DataFrame:
        """
        创建比率特征
        
        参数:
            df: 输入DataFrame
            numerator_cols: 分子列名列表
            denominator_cols: 分母列名列表
        
        返回:
            包含比率特征的DataFrame
        """
        new_features = {}
        
        for num_col in numerator_cols:
            if num_col not in df.columns:
                continue
            
            for den_col in denominator_cols:
                if den_col not in df.columns or num_col == den_col:
                    continue
                
                feature_name = f'{num_col}_div_{den_col}'
                
                try:
                    # 转换为数值类型
                    numerator = pd.to_numeric(df[num_col], errors='coerce')
                    denominator = pd.to_numeric(df[den_col], errors='coerce')
                    
                    # 避免除以零和NaN
                    denominator = denominator.replace(0, np.nan)
                    
                    # 执行除法
                    ratio = numerator / denominator
                    
                    # 清理无穷大和NaN
                    ratio = ratio.replace([np.inf, -np.inf], np.nan)
                    ratio = ratio.fillna(0)
                    
                    new_features[feature_name] = ratio
                    self.generated_features.append(feature_name)
                except:
                    # 如果出错，跳过这个特征
                    continue
        
        # 一次性添加所有新列
        if new_features:
            df = pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)
        
        return df
    
    def create_product_features(self, df: pd.DataFrame, 
                               col_pairs: List[tuple]) -> pd.DataFrame:
        """
        创建乘积特征
        
        参数:
            df: 输入DataFrame
            col_pairs: 列对列表 [(col1, col2), ...]
        
        返回:
            包含乘积特征的DataFrame
        """
        new_features = {}
        
        for col1, col2 in col_pairs:
            if col1 not in df.columns or col2 not in df.columns:
                continue
            
            feature_name = f'{col1}_mul_{col2}'
            
            try:
                # 转换为数值类型
                val1 = pd.to_numeric(df[col1], errors='coerce')
                val2 = pd.to_numeric(df[col2], errors='coerce')
                
                # 计算乘积
                product = val1 * val2
                
                # 清理无穷大和NaN
                product = product.replace([np.inf, -np.inf], np.nan)
                product = product.fillna(0)
                
                new_features[feature_name] = product
                self.generated_features.append(feature_name)
            except:
                continue
        
        # 一次性添加所有新列
        if new_features:
            df = pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)
        
        return df
    
    def create_difference_features(self, df: pd.DataFrame, 
                                  col_pairs: List[tuple]) -> pd.DataFrame:
        """
        创建差值特征
        
        参数:
            df: 输入DataFrame
            col_pairs: 列对列表 [(col1, col2), ...]
        
        返回:
            包含差值特征的DataFrame
        """
        new_features = {}
        
        for col1, col2 in col_pairs:
            if col1 not in df.columns or col2 not in df.columns:
                continue
            
            feature_name = f'{col1}_sub_{col2}'
            
            try:
                # 转换为数值类型
                val1 = pd.to_numeric(df[col1], errors='coerce')
                val2 = pd.to_numeric(df[col2], errors='coerce')
                
                # 计算差值
                diff = val1 - val2
                
                # 清理无穷大和NaN
                diff = diff.replace([np.inf, -np.inf], np.nan)
                diff = diff.fillna(0)
                
                new_features[feature_name] = diff
                self.generated_features.append(feature_name)
            except:
                continue
        
        # 一次性添加所有新列
        if new_features:
            df = pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)
        
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
                           columns: List[str]) -> pd.DataFrame:
        """
        创建对数特征
        
        参数:
            df: 输入DataFrame
            columns: 要生成对数特征的列
        
        返回:
            包含对数特征的DataFrame
        """
        new_features = {}
        
        for col in columns:
            if col not in df.columns:
                continue
            
            feature_name = f'log_{col}'
            
            # 只对正值取对数
            positive_values = df[col].clip(lower=1e-10)
            log_values = np.log(positive_values)
            
            # 处理无穷大和NaN
            log_values = log_values.replace([np.inf, -np.inf], np.nan)
            log_values = log_values.fillna(0)
            
            new_features[feature_name] = log_values
            self.generated_features.append(feature_name)
        
        # 一次性添加所有新列
        if new_features:
            df = pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)
        
        return df
    
    def create_sqrt_features(self, df: pd.DataFrame, 
                            columns: List[str]) -> pd.DataFrame:
        """
        创建平方根特征
        
        参数:
            df: 输入DataFrame
            columns: 要生成平方根特征的列
        
        返回:
            包含平方根特征的DataFrame
        """
        new_features = {}
        
        for col in columns:
            if col not in df.columns:
                continue
            
            feature_name = f'sqrt_{col}'
            
            # 只对非负值取平方根
            non_negative = df[col].clip(lower=0)
            new_features[feature_name] = np.sqrt(non_negative)
            
            self.generated_features.append(feature_name)
        
        # 一次性添加所有新列
        if new_features:
            df = pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)
        
        return df
    
    def create_rank_features(self, df: pd.DataFrame, 
                            columns: List[str],
                            window: int = 252) -> pd.DataFrame:
        """
        创建排名特征（滚动窗口排名，防止数据泄露）
        
        参数:
            df: 输入DataFrame
            columns: 要生成排名特征的列
            window: 滚动窗口大小（默认252天）
        
        返回:
            包含滚动排名特征的DataFrame
        """
        new_features = {}
        
        for col in columns:
            if col not in df.columns:
                continue
            
            feature_name = f'rank_{col}'
            
            # 使用 pandas 原生的 rolling().rank(pct=True) 替代 apply(lambda)
            # 性能提升约 50-100 倍且能够正确处理 ties
            series = pd.to_numeric(df[col], errors='coerce')
            new_features[feature_name] = series.rolling(window=window, min_periods=window//2).rank(pct=True)
            
            self.generated_features.append(feature_name)
        
        # 一次性添加所有新列
        if new_features:
            df = pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)
        
        return df
    
    def create_quantile_features(self, df: pd.DataFrame, 
                                columns: List[str],
                                window: int = 252,
                                n_quantiles: int = 5) -> pd.DataFrame:
        """
        创建分位数特征（滚动窗口分位，防止数据泄露）
        
        参数:
            df: 输入DataFrame
            columns: 要生成分位数特征的列
            window: 滚动窗口大小（默认252天）
            n_quantiles: 分位数数量
        
        返回:
            包含滚动分位数特征的DataFrame
        """
        new_features = {}
        
        for col in columns:
            if col not in df.columns:
                continue
            
            feature_name = f'quantile_{col}'
            
            # 复用高效的 rolling rank 逻辑
            series = pd.to_numeric(df[col], errors='coerce')
            rolled_rank = series.rolling(window=window, min_periods=window//4).rank(pct=True)
            
            # 离散化为分位数
            new_features[feature_name] = (rolled_rank * n_quantiles).fillna(0).astype(int).clip(0, n_quantiles-1)
            self.generated_features.append(feature_name)
        
        # 一次性添加所有新列
        if new_features:
            df = pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame,
                                   technical_cols: List[str],
                                   fundamental_cols: List[str]) -> pd.DataFrame:
        """
        创建技术指标与基本面因子的交互特征
        
        参数:
            df: 输入DataFrame
            technical_cols: 技术指标列
            fundamental_cols: 基本面因子列
        
        返回:
            包含交互特征的DataFrame
        """
        new_features = {}
        
        # 选择重要的技术指标和基本面因子进行交互
        important_tech = [col for col in technical_cols if col in df.columns][:10]
        important_fund = [col for col in fundamental_cols if col in df.columns][:10]
        
        for tech_col in important_tech:
            for fund_col in important_fund:
                feature_name = f'{tech_col}_x_{fund_col}'
                
                try:
                    # 转换为数值类型
                    tech_val = pd.to_numeric(df[tech_col], errors='coerce')
                    fund_val = pd.to_numeric(df[fund_col], errors='coerce')
                    
                    # 计算交互
                    interaction = tech_val * fund_val
                    
                    # 清理无穷大和NaN
                    interaction = interaction.replace([np.inf, -np.inf], np.nan)
                    interaction = interaction.fillna(0)
                    
                    new_features[feature_name] = interaction
                    self.generated_features.append(feature_name)
                except:
                    continue
        
        # 一次性添加所有新列
        if new_features:
            df = pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)
        
        return df
    
    def create_momentum_features(self, df: pd.DataFrame,
                                columns: List[str],
                                windows: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """
        创建动量特征（变化率）
        
        参数:
            df: 输入DataFrame
            columns: 要生成动量特征的列
            windows: 时间窗口列表
        
        返回:
            包含动量特征的DataFrame
        """
        new_features = {}
        
        for col in columns:
            if col not in df.columns:
                continue
            
            for window in windows:
                # 动量计算
                col_data = pd.to_numeric(df[col], errors='coerce')
                momentum = col_data.pct_change(window)
                
                # 处理无穷大和NaN
                momentum = momentum.replace([np.inf, -np.inf], np.nan)
                momentum = momentum.fillna(0)
                
                feature_name = f'{col}_momentum_{window}'
                new_features[feature_name] = momentum
                self.generated_features.append(feature_name)
        
        # 一次性添加所有新列
        if new_features:
            df = pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)
        
        return df
    
    def create_volatility_features(self, df: pd.DataFrame,
                                  columns: List[str],
                                  windows: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """
        创建波动率特征（滚动标准差）
        
        参数:
            df: 输入DataFrame
            columns: 要生成波动率特征的列
            windows: 时间窗口列表
        
        返回:
            包含波动率特征的DataFrame
        """
        new_features = {}
        
        for col in columns:
            if col not in df.columns:
                continue
            
            for window in windows:
                feature_name = f'{col}_volatility_{window}'
                col_data = pd.to_numeric(df[col], errors='coerce')
                # 排除无穷大以避免std()中的RuntimeWarning
                col_data = col_data.replace([np.inf, -np.inf], np.nan)
                volatility = col_data.rolling(window).std()
                volatility = volatility.fillna(0)
                
                new_features[feature_name] = volatility
                self.generated_features.append(feature_name)
        
        # 一次性添加所有新列
        if new_features:
            df = pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)
        
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
        应用所有特征工程变换，根据 verbose 控制输出
        """
        self.generated_features = [] # 每次转换重置结果列表，确保实例内逻辑干净
        if config is None:
            config = {
                'ratio': True, 'product': True, 'difference': True,
                'log': True, 'sqrt': True, 'rank': True,
                'interaction': True, 'categorical': True,
            }
        
        result = df
        stats = {}
        initial_count = len(df.columns)

        # 1. 编码分类特征
        if config.get('categorical'):
            pre_count = len(self.generated_features)
            result = self.encode_categorical_features(result)
            stats['分类特征编码'] = len(self.generated_features) - pre_count
        
        # 识别技术指标和基本面因子 - 优化匹配逻辑以兼容多种命名风格 (如 marketCap vs market_cap)
        def fuzzy_match(col_name, keywords):
            # 将列名和关键词处理成统一格式（小写且无下划线）进行匹配
            clean_col = col_name.lower().replace('_', '')
            for kw in keywords:
                if kw.lower().replace('_', '') in clean_col:
                    return True
            return False

        tech_keywords = ['rsi', 'macd', 'kdj', 'adx', 'atr', 'cci', 'mfi', 'obv', 'willr', 'bias', 'psy', 'boll', 'ma', 'ema', 'vol', 'amount', 'turnover']
        tech_indicators = [col for col in result.columns if fuzzy_match(col, tech_keywords)][:100]

        fund_keywords = ['pe', 'pb', 'roe', 'roa', 'margin', 'growth', 'yield', 'beta', 'market_cap', 'marketcap', 
                         'peg', 'sue', 'eav', 'revenue', 'share', 'ttm', 'yoy', 'ratio', 'equity', 'asset', 'profit']
        fundamental_factors = [col for col in result.columns if fuzzy_match(col, fund_keywords) 
                              and not any(x in col.lower() for x in ['slope', 'sharpe'])][:60]
        
        # 2. 应用变换
        if config.get('ratio') and len(fundamental_factors) > 1:
            pre_count = len(self.generated_features)
            # 比率特征：从基本面因子中随机选择 10 个，分为两组（每组 5 个）进行比率组合 (5*5=25)
            import random
            # 设置随机种子以保证可复现性
            random.seed(42)
            selected_factors = random.sample(fundamental_factors, min(10, len(fundamental_factors)))
            numerator_factors = selected_factors[:5]
            denominator_factors = selected_factors[5:10]
            result = self.create_ratio_features(result, numerator_factors, denominator_factors)
            stats['比率特征 (Fund/Fund)'] = len(self.generated_features) - pre_count
        
        if config.get('product') and len(tech_indicators) > 1:
            pre_count = len(self.generated_features)
            # 乘积特征：从技术指标中随机选择 6 个，生成两两组合 (4*3/2=6)
            import random
            random.seed(42)
            selected_tech = random.sample(tech_indicators, min(4, len(tech_indicators)))
            important_pairs = [(selected_tech[i], selected_tech[j]) 
                              for i in range(len(selected_tech)) 
                              for j in range(i+1, len(selected_tech))]
            result = self.create_product_features(result, important_pairs)
            stats['乘积特征 (Tech*Tech)'] = len(self.generated_features) - pre_count
        
        if config.get('difference') and len(tech_indicators) > 1:
            pre_count = len(self.generated_features)
            # 差分特征：从技术指标中随机选择 8 个，生成两两差值组合 (6*5/2=15)
            import random
            random.seed(42)
            selected_tech = random.sample(tech_indicators, min(6, len(tech_indicators)))
            diff_pairs = [(selected_tech[i], selected_tech[j]) 
                         for i in range(len(selected_tech)) 
                         for j in range(i+1, len(selected_tech))]
            result = self.create_difference_features(result, diff_pairs)
            stats['差分特征 (Tech-Tech)'] = len(self.generated_features) - pre_count
        
        if config.get('log'):
            pre_count = len(self.generated_features)
            # 对数特征：涵盖规模类因子（市值、营收、资产等）及成交量
            log_kws = ['market_cap', 'revenue', 'equity', 'asset', 'profit', 'cash']
            log_cols = [col for col in fundamental_factors if any(kw in col.lower() for kw in log_kws)]
            log_cols += [col for col in tech_indicators if 'vol' in col.lower() or 'amount' in col.lower()]
                    
            if log_cols:
                # 从符合条件的列中随机选择 8 个进行对数变换
                import random
                random.seed(42)
                selected_cols = random.sample(log_cols, min(8, len(log_cols)))
                result = self.create_log_features(result, selected_cols)
            stats['对数变换 (Log)'] = len(self.generated_features) - pre_count
        
        if config.get('sqrt'):
            pre_count = len(self.generated_features)
            # 平方根特征：包含波动率和成交量相关指标
            sqrt_cols = [col for col in tech_indicators if any(kw in col.lower() for kw in ['volatility', 'atr', 'vol', 'amount'])]
            if sqrt_cols:
                # 从符合条件的列中随机选择 5 个进行平方根变换
                import random
                random.seed(42)
                selected_cols = random.sample(sqrt_cols, min(5, len(sqrt_cols)))
                result = self.create_sqrt_features(result, selected_cols)
            stats['平方根变换 (Sqrt)'] = len(self.generated_features) - pre_count
        
        if config.get('rank'):
            pre_count = len(self.generated_features)
            # 排名特征：从基本面因子中随机选择 12 个进行滚动排名
            import random
            random.seed(42)
            selected_factors = random.sample(fundamental_factors, min(15, len(fundamental_factors)))
            result = self.create_rank_features(result, selected_factors)
            stats['滚动排名 (Rolling Rank)'] = len(self.generated_features) - pre_count
        
        if config.get('interaction') and len(tech_indicators) > 0 and len(fundamental_factors) > 0:
            pre_count = len(self.generated_features)
            # 交互特征：从技术指标和基本面因子中各随机选择 6 个和 6 个进行交互
            import random
            random.seed(42)
            selected_tech = random.sample(tech_indicators, min(6, len(tech_indicators)))
            selected_fund = random.sample(fundamental_factors, min(5, len(fundamental_factors)))
            result = self.create_interaction_features(result, selected_tech, selected_fund)
            stats['交互特征 (Tech*Fund)'] = len(self.generated_features) - pre_count

        # 仅在 verbose 为 True 时输出统计报告
        if verbose:
            print("\n" + "-"*40)
            print("特征工程报告:")
            print(f"  识别到: {len(tech_indicators)} 个技术指标, {len(fundamental_factors)} 个基本面因子")
            for name, count in stats.items():
                if count > 0:
                    print(f"  - {name}: +{count} 个")
            print(f"  总计: 原始 {initial_count} -> 现计 {len(result.columns)} (新增 {len(self.generated_features)})")
            print("-"*40 + "\n")
        
        return result
    
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

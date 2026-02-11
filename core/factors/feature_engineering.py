"""
特征工程增强模块
提供交叉因子、衍生因子和特征变换功能
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from sklearn.preprocessing import PolynomialFeatures
import psutil


class FeatureEngineer:
    """特征工程器 - 生成交叉因子和衍生因子（单例模式）"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        """实现单例模式"""
        if cls._instance is None:
            cls._instance = super(FeatureEngineer, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """初始化特征工程器（仅初始化一次）"""
        if not FeatureEngineer._initialized:
            self.generated_features = []
            FeatureEngineer._initialized = True
    
    @classmethod
    def reset(cls):
        """重置单例实例"""
        cls._instance = None
        cls._initialized = False
    
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
                            columns: List[str]) -> pd.DataFrame:
        """
        创建排名特征（横截面排名）
        
        参数:
            df: 输入DataFrame
            columns: 要生成排名特征的列
        
        返回:
            包含排名特征的DataFrame
        """
        new_features = {}
        
        for col in columns:
            if col not in df.columns:
                continue
            
            feature_name = f'rank_{col}'
            new_features[feature_name] = df[col].rank(pct=True)
            
            self.generated_features.append(feature_name)
        
        # 一次性添加所有新列
        if new_features:
            df = pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)
        
        return df
    
    def create_quantile_features(self, df: pd.DataFrame, 
                                columns: List[str],
                                n_quantiles: int = 5) -> pd.DataFrame:
        """
        创建分位数特征
        
        参数:
            df: 输入DataFrame
            columns: 要生成分位数特征的列
            n_quantiles: 分位数数量
        
        返回:
            包含分位数特征的DataFrame
        """
        new_features = {}
        
        for col in columns:
            if col not in df.columns:
                continue
            
            feature_name = f'quantile_{col}'
            new_features[feature_name] = pd.qcut(df[col], q=n_quantiles, 
                                          labels=False, duplicates='drop')
            
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
        
        参数:
            df: 输入DataFrame
            categorical_cols: 分类列名列表，如果为None则自动检测
        
        返回:
            包含编码后分类特征的DataFrame
        """
        new_features = {}
        
        if categorical_cols is None:
            # 自动检测分类列
            categorical_cols = [col for col in df.columns if col in ['sector', 'industry']]
        
        for col in categorical_cols:
            if col not in df.columns:
                continue
            
            # 使用 LabelEncoder 进行编码
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            
            # 处理缺失值
            df_col = df[col].fillna('Unknown')
            
            try:
                new_features[f'{col}_encoded'] = le.fit_transform(df_col)
                self.generated_features.append(f'{col}_encoded')
                
                # 创建 one-hot 编码（只保留前10个最常见的类别以避免维度爆炸）
                top_categories = df[col].value_counts().head(10).index
                for category in top_categories:
                    feature_name = f'{col}_{category}'
                    new_features[feature_name] = (df[col] == category).astype(int)
                    self.generated_features.append(feature_name)
                
            except Exception as e:
                print(f"编码 {col} 失败: {e}")
        
        # 一次性添加所有新列
        if new_features:
            df = pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)
        
        return df
    
    def apply_all_transformations(self, df: pd.DataFrame,
                                  config: Optional[Dict] = None) -> pd.DataFrame:
        """
        应用所有特征工程变换
        
        参数:
            df: 输入DataFrame
            config: 配置字典，指定要应用的变换
        
        返回:
            包含所有新特征的DataFrame
        """
        # 重置生成的特征列表，防止累积
        self.reset()
        
        if config is None:
            config = {
                'ratio': True,
                'product': True,
                'difference': True,
                'log': True,
                'sqrt': True,
                'rank': True,
                'interaction': True,
                'categorical': True,
            }
        
        # Check memory before proceeding
        available_memory = psutil.virtual_memory().available / (1024**3)  # GB
        estimated_size = (df.shape[0] * df.shape[1] * 8) / (1024**3)  # GB
        
        # if estimated_size > available_memory * 0.5:
        #     print(f"警告: 数据框过大 ({df.shape[1]} 列)，跳过特征工程以节省内存")
        #     return df
        
        result = df
        
        # 编码分类特征（行业、板块）
        if config.get('categorical'):
            print("编码分类特征...")
            result = self.encode_categorical_features(result)
        
        # 识别技术指标和基本面因子 - 限制列数
        tech_indicators = [col for col in result.columns if any(
            indicator in col.lower() for indicator in 
            ['rsi', 'macd', 'kdj', 'adx', 'atr', 'cci', 'mfi', 'obv', 
             'candle', 'pattern', 'star', 'hammer', 'doji', 'engulfing', 'harami', 'piercing']
        )][:40]  # 增加上限以容纳K线形态因子

        
        fundamental_factors = [col for col in result.columns if any(
            factor in col.lower() for factor in 
            ['pe_', 'pb_', 'roe', 'roa', 'margin', 'growth', 'yield', 'beta']
        )][:20]  # 限制为20列
        
        print(f"识别到 {len(tech_indicators)} 个技术指标")
        print(f"识别到 {len(fundamental_factors)} 个基本面因子")
        
        # 应用各种变换
        if config.get('ratio') and len(fundamental_factors) > 0:
            print("生成比率特征...")
            result = self.create_ratio_features(
                result, 
                fundamental_factors[:5], 
                fundamental_factors[:5]
            )
        
        if config.get('product') and len(tech_indicators) > 0:
            print("生成乘积特征...")
            important_pairs = [
                (tech_indicators[i], tech_indicators[j])
                for i in range(min(3, len(tech_indicators)))
                for j in range(i+1, min(5, len(tech_indicators)))
            ]
            result = self.create_product_features(result, important_pairs)
        
        if config.get('difference') and len(tech_indicators) > 0:
            print("生成差分特征...")
            diff_pairs = [
                (tech_indicators[i], tech_indicators[j])
                for i in range(min(3, len(tech_indicators)))
                for j in range(i+1, min(5, len(tech_indicators)))
            ]
            result = self.create_difference_features(result, diff_pairs)
        
        if config.get('log') and len(fundamental_factors) > 0:
            print("生成对数特征...")
            log_cols = [col for col in fundamental_factors if 'market_cap' in col or 'volume' in col]
            if log_cols:
                result = self.create_log_features(result, log_cols[:5])
        
        if config.get('sqrt') and len(tech_indicators) > 0:
            print("生成平方根特征...")
            sqrt_cols = [col for col in tech_indicators if 'volatility' in col or 'atr' in col]
            if sqrt_cols:
                result = self.create_sqrt_features(result, sqrt_cols[:5])
        
        if config.get('rank') and len(fundamental_factors) > 0:
            print("生成排名特征...")
            result = self.create_rank_features(result, fundamental_factors[:5])
        
        if config.get('interaction') and len(tech_indicators) > 0 and len(fundamental_factors) > 0:
            print("生成交互特征...")
            result = self.create_interaction_features(
                result, 
                tech_indicators[:5], 
                fundamental_factors[:5]
            )
        
        print(f"\n特征工程完成，生成了 {len(self.generated_features)} 个新特征")
        
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

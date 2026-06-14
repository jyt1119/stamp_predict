
#feature_engine.py：
"""
特征工程模块 - 构造技术指标、价格特征、财务特征等
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional


class FeatureEngineer:
    """特征工程器，计算各类用于模型输入的特征"""
    
    # ==================== 价格相关特征 ====================
    
    @staticmethod
    def calc_returns(df: pd.DataFrame) -> pd.DataFrame:
        """计算多周期收益率"""
        features = pd.DataFrame(index=df.index)
        for period in [1, 3, 5, 10, 20]:
            features[f'return_{period}d'] = df['close'].pct_change(period)
        return features
    
    @staticmethod
    def calc_ma_features(df: pd.DataFrame) -> pd.DataFrame:
        """计算移动平均线相关特征"""
        features = pd.DataFrame(index=df.index)
        for window in [5, 10, 20, 60]:
            ma = df['close'].rolling(window=window).mean()
            features[f'ma_{window}_dev'] = (df['close'] - ma) / (ma + 1e-10)
            features[f'ma_{window}_slope'] = ma.pct_change(5)
        return features
    
    @staticmethod
    def calc_price_position(df: pd.DataFrame) -> pd.DataFrame:
        """计算价格位置特征"""
        features = pd.DataFrame(index=df.index)
        
        # 日内价格位置
        features['intraday_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
        
        # 振幅
        features['amplitude'] = (df['high'] - df['low']) / (df['close'].shift(1) + 1e-10)
        
        # 跳空缺口
        features['gap'] = (df['open'] - df['close'].shift(1)) / (df['close'].shift(1) + 1e-10)
        
        # 上影线/下影线
        features['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (df['close'] + 1e-10)
        features['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / (df['close'] + 1e-10)
        
        return features
    
    @staticmethod
    def calc_volatility(df: pd.DataFrame) -> pd.DataFrame:
        """计算波动率特征"""
        features = pd.DataFrame(index=df.index)
        returns = df['close'].pct_change()
        
        for window in [5, 10, 20, 60]:
            features[f'volatility_{window}d'] = returns.rolling(window=window).std()
            features[f'vol_ratio_{window}d'] = features[f'volatility_{window}d'] / (
                features[f'volatility_{window}d'].shift(window) + 1e-10
            )
        
        return features
    
    # ==================== 技术指标 ====================
    
    @staticmethod
    def calc_macd(df: pd.DataFrame) -> pd.DataFrame:
        """计算MACD指标"""
        features = pd.DataFrame(index=df.index)
        
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        histogram = macd_line - signal_line
        
        features['macd'] = macd_line
        features['macd_signal'] = signal_line
        features['macd_hist'] = histogram
        features['macd_divergence'] = macd_line - signal_line
        
        return features
    
    @staticmethod
    def calc_rsi(df: pd.DataFrame) -> pd.DataFrame:
        """计算RSI指标"""
        features = pd.DataFrame(index=df.index)
        
        for window in [6, 14, 24]:
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / (loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            features[f'rsi_{window}'] = rsi
        
        return features
    
    @staticmethod
    def calc_bollinger(df: pd.DataFrame) -> pd.DataFrame:
        """计算布林带指标"""
        features = pd.DataFrame(index=df.index)
        
        ma_20 = df['close'].rolling(window=20).mean()
        std_20 = df['close'].rolling(window=20).std()
        
        upper = ma_20 + 2 * std_20
        lower = ma_20 - 2 * std_20
        
        features['bb_position'] = (df['close'] - lower) / (upper - lower + 1e-10)
        features['bb_width'] = (upper - lower) / (ma_20 + 1e-10)
        features['bb_squeeze'] = features['bb_width'] / features['bb_width'].rolling(20).mean()
        
        return features
    
    @staticmethod
    def calc_atr(df: pd.DataFrame) -> pd.DataFrame:
        """计算ATR指标"""
        features = pd.DataFrame(index=df.index)
        
        high, low, close = df['high'], df['low'], df['close']
        
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        for window in [14, 20]:
            features[f'atr_{window}'] = true_range.rolling(window=window).mean()
            features[f'atr_{window}_ratio'] = features[f'atr_{window}'] / (close + 1e-10)
        
        return features
    
    @staticmethod
    def calc_kdj(df: pd.DataFrame) -> pd.DataFrame:
        """计算KDJ指标"""
        features = pd.DataFrame(index=df.index)
        
        low_min = df['low'].rolling(window=9).min()
        high_max = df['high'].rolling(window=9).max()
        
        rsv = (df['close'] - low_min) / (high_max - low_min + 1e-10) * 100
        
        k = rsv.ewm(com=2, adjust=False).mean()
        d = k.ewm(com=2, adjust=False).mean()
        j = 3 * k - 2 * d
        
        features['kdj_k'] = k
        features['kdj_d'] = d
        features['kdj_j'] = j
        
        return features
    
    # ==================== 成交量特征 ====================
    
    @staticmethod
    def calc_volume_features(df: pd.DataFrame) -> pd.DataFrame:
        """计算成交量相关特征"""
        features = pd.DataFrame(index=df.index)
        volume = df['vol']
        
        # 成交量移动平均
        for window in [5, 10, 20, 60]:
            vol_ma = volume.rolling(window=window).mean()
            features[f'vol_ratio_{window}'] = volume / (vol_ma + 1e-10)
            features[f'vol_change_{window}'] = volume.pct_change(window)
        
        # 量价关系
        features['vol_price_corr'] = volume.rolling(20).corr(df['close'].pct_change())
        
        # OBV指标
        obv = (np.sign(df['close'].diff()) * volume).fillna(0).cumsum()
        features['obv'] = obv
        features['obv_ma5'] = obv.rolling(5).mean()
        features['obv_change'] = obv.pct_change(10)
        
        # 换手率相关（如果数据中有amount字段）
        if 'amount' in df.columns:
            features['amount_ma5'] = df['amount'].rolling(5).mean()
            features['amount_ratio'] = df['amount'] / (features['amount_ma5'] + 1e-10)
        
        return features
    
    # ==================== 截面特征 ====================
    
    @staticmethod
    def calc_cross_sectional(df: pd.DataFrame, all_stocks_data: dict = None) -> pd.DataFrame:
        """
        计算截面特征（需要全市场数据，暂不实现）
        例如：行业排名、市值分位数等
        """
        features = pd.DataFrame(index=df.index)
        return features
    
    # ==================== 财务特征 ====================
    
    @staticmethod
    def calc_financial_features(metric_df: Optional[pd.DataFrame]) -> pd.DataFrame:
        """计算财务指标特征"""
        if metric_df is None or len(metric_df) == 0:
            return pd.DataFrame(index=[])
        df = metric_df.copy()
        # 确保有 trade_date 列为 datetime
        if 'trade_date' in df.columns:
            df['trade_date'] = pd.to_datetime(df['trade_date'])
        # 对数市值、估值、周转等
        out = pd.DataFrame(index=df.index)
        out['log_total_mv'] = np.log1p(df.get('total_mv', pd.Series(0)))
        out['pe_ttm'] = df.get('pe_ttm')
        out['pb'] = df.get('pb')
        out['ps_ttm'] = df.get('ps_ttm')
        out['dv_ttm'] = df.get('dv_ttm')
        out['turnover_rate'] = df.get('turnover_rate')
        out['volume_ratio'] = df.get('volume_ratio')
        # 简单比率
        out['circ_ratio'] = df.get('circ_mv', 1) / (df.get('total_mv', 1) + 1e-9)
        # 前向填充缺失
        out = out.fillna(method='ffill').fillna(0.0)
        return out
    
    # ==================== 资金流向特征 ====================
    
    @staticmethod
    def calc_moneyflow_features(moneyflow_df: Optional[pd.DataFrame]) -> pd.DataFrame:
        """计算资金流向特征"""
        if moneyflow_df is None or len(moneyflow_df) == 0:
            return pd.DataFrame()
        
        features = pd.DataFrame(index=moneyflow_df.index)
        
        # 净流入
        if 'net_mf_amount' in moneyflow_df.columns:
            features['net_mf_amount'] = moneyflow_df['net_mf_amount']
            features['net_mf_ma5'] = features['net_mf_amount'].rolling(5).mean()
        
        # 大单净流入（大单+特大单）
        large_buy = moneyflow_df.get('buy_lg_amount', 0) + moneyflow_df.get('buy_elg_amount', 0)
        large_sell = moneyflow_df.get('sell_lg_amount', 0) + moneyflow_df.get('sell_elg_amount', 0)
        features['large_order_net'] = large_buy - large_sell
        features['large_order_net_ma5'] = features['large_order_net'].rolling(5).mean()
        
        # 主力净流入占比
        total_buy = (
            moneyflow_df.get('buy_sm_amount', 0) +
            moneyflow_df.get('buy_md_amount', 0) +
            moneyflow_df.get('buy_lg_amount', 0) +
            moneyflow_df.get('buy_elg_amount', 0)
        )
        features['main_force_ratio'] = features['large_order_net'] / (total_buy + 1e-10)
        
        return features
    
    # ==================== 综合特征工程 ====================
    
    def engineer_features(self, 
                          df: pd.DataFrame,
                          metric_df: Optional[pd.DataFrame] = None,
                          moneyflow_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        综合特征工程，生成所有特征
        
        Args:
            df: 量价数据DataFrame
            metric_df: 基本面数据DataFrame（可选）
            moneyflow_df: 资金流向数据DataFrame（可选）
            
        Returns:
            包含所有特征的DataFrame
        """
        # 收集所有特征
        feature_list = [
            self.calc_returns(df),
            self.calc_ma_features(df),
            self.calc_price_position(df),
            self.calc_volatility(df),
            self.calc_macd(df),
            self.calc_rsi(df),
            self.calc_bollinger(df),
            self.calc_atr(df),
            self.calc_kdj(df),
            self.calc_volume_features(df),
        ]
        
        # 添加可选特征
        if metric_df is not None and len(metric_df) > 0:
            # metric_df 可能是单只股票的时间序列：按日期与 price df 对齐（向后对齐，使用最近已知基本面）
            mdf = metric_df.copy()
            mdf['trade_date'] = pd.to_datetime(mdf['trade_date'])
            price_dates = pd.DataFrame({'trade_date': pd.to_datetime(df['trade_date'])})
            merged = pd.merge_asof(price_dates.sort_values('trade_date'),
                           mdf.sort_values('trade_date'),
                           on='trade_date',
                           direction='backward')
            merged = merged.fillna(method='ffill').fillna(0.0)
            fin_feats = self.calc_financial_features(merged)
            # fin_feats 与 price df index 对齐后加入特征列表（注意索引对应）
            fin_feats.index = df.index
            feature_list.append(fin_feats)
        
        if moneyflow_df is not None and len(moneyflow_df) > 0:
            flow_df_aligned = moneyflow_df.set_index('trade_date').reindex(df.index)
            feature_list.append(self.calc_moneyflow_features(flow_df_aligned))
        
        # 合并所有特征
        all_features = pd.concat(feature_list, axis=1)
        
        # 去除全为NaN的列
        all_features = all_features.dropna(axis=1, how='all')
        
                # 对大数值特征取对数，缩小量级差异
        for col in all_features.columns:
            col_max = all_features[col].abs().max()
            if col_max > 1e5:
                sign = np.sign(all_features[col])
                all_features[col] = sign * np.log1p(all_features[col].abs())
                
        return all_features
    
    def get_feature_names(self, df: pd.DataFrame,
                          metric_df: Optional[pd.DataFrame] = None,
                          moneyflow_df: Optional[pd.DataFrame] = None) -> list:
        """获取特征名称列表"""
        sample_features = self.engineer_features(
            df.head(self.config.SEQUENCE_LENGTH + 10),
            metric_df,
            moneyflow_df
        )
        return sample_features.columns.tolist()
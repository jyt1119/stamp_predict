"""
回测系统 - 模拟历史交易
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class Backtest:
    """回测引擎"""
    
    def __init__(self, config):
        """
        初始化回测引擎
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.initial_capital = config.INITIAL_CAPITAL
        self.commission = config.COMMISSION_RATE
        self.max_positions = config.MAX_POSITIONS
        self.daily_trades = config.DAILY_TRADES
        
        # 状态
        self.cash = self.initial_capital
        self.positions = {}  # {stock_code: {'shares': int, 'avg_cost': float}}
        
        # 记录
        self.daily_records = []
        self.trade_records = []
    
    def reset(self):
        """重置回测状态"""
        self.cash = self.initial_capital
        self.positions = {}
        self.daily_records = []
        self.trade_records = []
    
    def get_portfolio_value(self, prices: Dict[str, float]) -> float:
        """计算当前组合总价值"""
        position_value = 0.0
        for code, pos in self.positions.items():
            if code in prices:
                position_value += pos['shares'] * prices[code]
        return position_value + self.cash
    
    def execute_trade(self, code: str, action: str, price: float, 
                      shares: int, date) -> bool:
        """
        执行单笔交易
        
        Returns:
            是否成功执行
        """
        if shares <= 0 or price <= 0:
            return False
        
        if action == 'buy':
            cost = price * shares * (1 + self.commission)
            if cost > self.cash:
                # 调整到可买入的最大数量
                max_shares = int(self.cash / (price * (1 + self.commission)))
                if max_shares <= 0:
                    return False
                shares = max_shares
                cost = price * shares * (1 + self.commission)
            
            self.cash -= cost
            if code in self.positions:
                old = self.positions[code]
                total_cost = old['avg_cost'] * old['shares'] + cost
                new_shares = old['shares'] + shares
                self.positions[code] = {
                    'shares': new_shares,
                    'avg_cost': total_cost / new_shares
                }
            else:
                self.positions[code] = {
                    'shares': shares,
                    'avg_cost': price
                }
        
        elif action == 'sell':
            if code not in self.positions:
                return False
            
            sell_shares = min(shares, self.positions[code]['shares'])
            if sell_shares <= 0:
                return False
            
            proceeds = price * sell_shares * (1 - self.commission)
            self.cash += proceeds
            
            remaining = self.positions[code]['shares'] - sell_shares
            if remaining == 0:
                del self.positions[code]
            else:
                self.positions[code]['shares'] = remaining
        
        self.trade_records.append({
            'date': date,
            'code': code,
            'action': action,
            'price': price,
            'shares': shares,
            'cash': self.cash
        })
        
        return True
    
    def run(self, 
            predictions_df: pd.DataFrame,
            price_data: Dict[str, pd.DataFrame],
            start_date, end_date) -> Dict:
        """
        运行回测
        
        Args:
            predictions_df: 预测DataFrame，包含date, stock_code, prediction
            price_data: 价格数据 {stock_code: DataFrame with trade_date, close}
            start_date: 起始日期
            end_date: 结束日期
            
        Returns:
            回测结果指标字典
        """
        self.reset()
        
        # 获取回测日期范围
        dates = sorted(predictions_df['date'].unique())
        dates = [d for d in dates if pd.to_datetime(start_date) <= d <= pd.to_datetime(end_date)]
        
        if len(dates) == 0:
            print("警告：回测日期范围内无数据")
            return {}
        
        print(f"回测日期: {dates[0]} ~ {dates[-1]}, 共 {len(dates)} 个交易日")
        
        for date in dates:
            # 获取当日预测
            day_pred = predictions_df[predictions_df['date'] == date].copy()
            
            if len(day_pred) == 0:
                continue
            
            # 按预测分数排序
            day_pred = day_pred.sort_values('prediction', ascending=False)
            
            # 获取当日收盘价
            day_prices = {}
            for code in day_pred['stock_code']:
                if code in price_data:
                    code_prices = price_data[code]
                    if date in code_prices.index:
                        day_prices[code] = code_prices.loc[date, 'close']
                    else:
                        # 尝试找到最近的交易日价格
                        prev_prices = code_prices[code_prices.index <= date]
                        if len(prev_prices) > 0:
                            day_prices[code] = prev_prices['close'].iloc[-1]
            
            # 策略执行
            if len(self.positions) == 0:
                # 第一天：等权建仓
                top_n = day_pred.head(self.max_positions)
                n_stocks = len(top_n)
                if n_stocks > 0:
                    alloc = self.cash * (1 - self.config.CASH_RESERVE) / n_stocks
                    for _, row in top_n.iterrows():
                        code = row['stock_code']
                        if code in day_prices:
                            price = day_prices[code]
                            shares = int(alloc / (price * (1 + self.commission)))
                            if shares > 0:
                                self.execute_trade(code, 'buy', price, shares, date)
            else:
                # 卖出持仓中评分最低的
                position_scores = []
                for code in self.positions:
                    pred_row = day_pred[day_pred['stock_code'] == code]
                    score = pred_row['prediction'].iloc[0] if len(pred_row) > 0 else -np.inf
                    position_scores.append((code, score))
                
                position_scores.sort(key=lambda x: x[1])
                
                # 卖出最低分的k只
                for i in range(min(self.daily_trades, len(position_scores))):
                    code, score = position_scores[i]
                    if code in day_prices and code in self.positions:
                        self.execute_trade(
                            code, 'sell', day_prices[code],
                            self.positions[code]['shares'], date
                        )
                
                # 买入新的高分股票
                # 买入新的高分股票
                current_codes = set(self.positions.keys())
                buy_candidates = day_pred[~day_pred['stock_code'].isin(current_codes)].copy()
                buy_candidates = buy_candidates[buy_candidates['stock_code'].isin(day_prices.keys())]
                
                if len(buy_candidates) > 0:
                    k = min(self.max_positions - len(self.positions), len(buy_candidates))
                    if k > 0:
                        candidates = buy_candidates.head(k).copy()
                        
                        # 计算近20日波动率
                        vols = {}
                        for code in candidates['stock_code']:
                            if code in price_data:
                                hist = price_data[code][price_data[code].index <= date].tail(20)
                                if len(hist) >= 5:
                                    ret = hist['close'].pct_change().dropna()
                                    vols[code] = ret.std() * np.sqrt(252)
                                else:
                                    vols[code] = 0.5
                            else:
                                vols[code] = 0.5
                        
                        candidates['vol'] = candidates['stock_code'].map(vols).clip(0.01, 10)
                        candidates['risk_weight'] = 1.0 / candidates['vol']
                        candidates['risk_weight'] = candidates['risk_weight'].clip(upper=100)
                        
                        portfolio_val = self.get_portfolio_value(day_prices)
                        cash_alloc = self.cash * (1 - self.config.CASH_RESERVE)
                        w = candidates['risk_weight'] / candidates['risk_weight'].sum()
                        candidates['alloc_value'] = w * cash_alloc
                        max_per_stock = portfolio_val * 0.05
                        candidates['alloc_value'] = candidates['alloc_value'].clip(upper=max_per_stock)
                        
                        for _, row in candidates.iterrows():
                            code = row['stock_code']
                            price = day_prices[code]
                            shares = int(row['alloc_value'] / (price * (1 + self.commission)))
                            if shares > 0:
                                self.execute_trade(code, 'buy', price, shares, date)

             # 止损：跌破成本10%则卖出
            stop_loss_pct = 0.10
            for code in list(self.positions.keys()):
                if code in day_prices:
                    today_price = day_prices[code]
                    if today_price < self.positions[code]['avg_cost'] * (1 - stop_loss_pct):
                        self.execute_trade(code, 'sell', today_price, self.positions[code]['shares'], date)

            # 记录每日组合价值
            portfolio_value = self.get_portfolio_value(day_prices)
            self.daily_records.append({
                'date': date,
                'value': portfolio_value,
                'cash': self.cash,
                'n_positions': len(self.positions)
            })
        
        # 计算绩效指标
        results = self._calculate_metrics()
        return results
    
    def _calculate_metrics(self) -> Dict:
        """计算回测绩效指标"""
        if len(self.daily_records) < 2:
            return {}
        
        df = pd.DataFrame(self.daily_records)
        df['return'] = df['value'].pct_change()
        df = df.iloc[1:]  # 去掉第一天
        
        # 基本指标
        final_value = df['value'].iloc[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        # 年化收益率
        n_days = len(df)
        annual_return = (1 + total_return) ** (252 / n_days) - 1 if n_days > 0 else 0
        
        # 年化波动率
        annual_vol = df['return'].std() * np.sqrt(252)
        
        # 夏普比率
        rf_daily = 0.02 / 252  # 无风险利率
        excess = df['return'] - rf_daily
        sharpe = np.sqrt(252) * excess.mean() / (excess.std() + 1e-10)
        
        # 最大回撤
        df['cummax'] = df['value'].cummax()
        df['drawdown'] = (df['value'] - df['cummax']) / df['cummax']
        max_dd = df['drawdown'].min()
        
        # Calmar比率
        calmar = annual_return / abs(max_dd) if max_dd != 0 else 0
        
        # 胜率
        win_rate = (df['return'] > 0).mean()
        
        # 盈亏比
        positive = df[df['return'] > 0]['return']
        negative = df[df['return'] < 0]['return']
        profit_loss_ratio = abs(positive.mean() / negative.mean()) if len(negative) > 0 else np.inf
        
        return {
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'calmar_ratio': calmar,
            'win_rate': win_rate,
            'profit_loss_ratio': profit_loss_ratio,
            'n_trading_days': n_days,
            'total_trades': len(self.trade_records),
            'daily_df': df,
            'trade_df': pd.DataFrame(self.trade_records) if self.trade_records else pd.DataFrame()
        }
    
    def print_summary(self, results: Dict):
        """打印回测结果摘要"""
        print("\n" + "=" * 60)
        print("回测结果摘要")
        print("=" * 60)
        print(f"初始资金:     ¥{self.initial_capital:,.0f}")
        print(f"最终资金:     ¥{results['final_value']:,.0f}")
        print(f"总收益率:     {results['total_return']:.2%}")
        print(f"年化收益率:   {results['annual_return']:.2%}")
        print(f"年化波动率:   {results['annual_volatility']:.2%}")
        print(f"夏普比率:     {results['sharpe_ratio']:.2f}")
        print(f"最大回撤:     {results['max_drawdown']:.2%}")
        print(f"Calmar比率:   {results['calmar_ratio']:.2f}")
        print(f"日胜率:       {results['win_rate']:.2%}")
        print(f"盈亏比:       {results['profit_loss_ratio']:.2f}")
        print(f"交易天数:     {results['n_trading_days']}")
        print(f"总交易次数:   {results['total_trades']}")
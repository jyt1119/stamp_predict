"""
工具函数 - 绘图、指标计算、报告生成
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from scipy.stats import spearmanr
from typing import Dict, Optional
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def calculate_ic_stats(predictions_df: pd.DataFrame) -> Dict:
    """
    计算IC和ICIR统计
    
    Args:
        predictions_df: 包含date, prediction, true_return的DataFrame
        
    Returns:
        IC统计字典
    """
    # 按日期计算IC
    ic_values = []
    for date in predictions_df['date'].unique():
        day_data = predictions_df[predictions_df['date'] == date]
        if len(day_data) >= 10:
            ic = spearmanr(day_data['prediction'], day_data['true_return'])[0]
            if not np.isnan(ic):
                ic_values.append(ic)
    
    if len(ic_values) == 0:
        return {}
    
    ic_series = pd.Series(ic_values)
    
    ic_mean = ic_series.mean()
    ic_std = ic_series.std()
    icir = ic_mean / (ic_std + 1e-10)
    ic_positive_ratio = (ic_series > 0).mean()
    
    return {
        'IC_mean': ic_mean,
        'IC_std': ic_std,
        'ICIR': icir,
        'IC_positive_ratio': ic_positive_ratio,
        'IC_values': ic_series.values
    }


def plot_portfolio_curve(daily_df: pd.DataFrame, 
                         benchmark_df: Optional[pd.DataFrame] = None,
                         save_path: str = 'portfolio_value.png'):
    """绘制组合价值曲线"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # 净值曲线
    ax1 = axes[0]
    normalized_value = daily_df['value'] / daily_df['value'].iloc[0]
    ax1.plot(daily_df['date'], normalized_value, label='策略净值', linewidth=1.5, color='blue')
    
    if benchmark_df is not None and len(benchmark_df) > 0:
        benchmark_norm = benchmark_df['close'] / benchmark_df['close'].iloc[0]
        ax1.plot(benchmark_df['trade_date'], benchmark_norm, 
                label='沪深300', linewidth=1.5, color='orange', alpha=0.7)
    
    ax1.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax1.set_title('组合净值曲线')
    ax1.set_ylabel('净值')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 日收益率
    ax2 = axes[1]
    ax2.bar(daily_df['date'], daily_df['return'] * 100, 
            width=1, alpha=0.5, color='blue')
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_title('日收益率 (%)')
    ax2.set_ylabel('收益率 (%)')
    ax2.grid(True, alpha=0.3)
    
    # 回撤曲线
    ax3 = axes[2]
    ax3.fill_between(daily_df['date'], daily_df['drawdown'] * 100, 0, 
                     alpha=0.3, color='red')
    ax3.plot(daily_df['date'], daily_df['drawdown'] * 100, 
             color='red', linewidth=1)
    ax3.set_title('回撤曲线 (%)')
    ax3.set_xlabel('日期')
    ax3.set_ylabel('回撤 (%)')
    ax3.grid(True, alpha=0.3)
    
    # 格式化x轴
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"组合曲线图已保存到 {save_path}")


def plot_ic_distribution(ic_values: np.ndarray, save_path: str = 'ic_distribution.png'):
    """绘制IC分布图"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # IC时间序列
    axes[0].plot(ic_values, linewidth=1, color='green')
    axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[0].axhline(y=np.mean(ic_values), color='red', linestyle='--', 
                    alpha=0.7, label=f'Mean: {np.mean(ic_values):.4f}')
    axes[0].set_title('IC时序图')
    axes[0].set_ylabel('Rank IC')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # IC分布直方图
    axes[1].hist(ic_values, bins=30, edgecolor='black', alpha=0.7, color='green')
    axes[1].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    axes[1].axvline(x=np.mean(ic_values), color='red', linestyle='--', 
                    alpha=0.7)
    axes[1].set_title(f'IC分布 (ICIR: {np.mean(ic_values)/np.std(ic_values):.4f})')
    axes[1].set_xlabel('Rank IC')
    axes[1].set_ylabel('频次')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"IC分布图已保存到 {save_path}")


def load_benchmark(config, start_date, end_date):
    """加载基准指数数据"""
    benchmark_path = Path(config.MARKET_DIR) / "000300.SH.csv"
    
    if not benchmark_path.exists():
        print(f"警告：未找到基准指数文件 {benchmark_path}")
        return None
    
    df = pd.read_csv(benchmark_path)
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    
    mask = (df['trade_date'] >= pd.to_datetime(start_date)) & \
           (df['trade_date'] <= pd.to_datetime(end_date))
    
    return df[mask].sort_values('trade_date')
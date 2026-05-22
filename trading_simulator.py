"""
模拟交易辅助脚本 - 比赛期间每日生成交易信号
"""
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from config import Config
from data_processor import DataProcessor
from feature_engine import FeatureEngineer
from model import AlphaNet


class TradingAssistant:
    """模拟交易助手"""
    
    def __init__(self, config, model_path='outputs/best_model.pth'):
        self.config = config
        self.device = torch.device(config.DEVICE)
        
        # 加载数据
        self.processor = DataProcessor(config)
        self.processor.load_basic_info()
        self.processor.load_trade_calendar()
        
        # 加载模型
        self.model = self._load_model(model_path)
        
        # 特征工程器
        self.engineer = FeatureEngineer()
    
    def _load_model(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        
        model = AlphaNet(
            input_dim=checkpoint.get('input_dim', 82),
            d_model=checkpoint.get('d_model', 128),
            n_heads=8, n_layers=4,
            d_ff=512, dropout=0.1
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        return model
    
    def update_data(self):
        """更新最新数据"""
        today = datetime.now().strftime('%Y-%m-%d')
        # 加载最近3个月数据
        start = (datetime.now() - pd.Timedelta(days=90)).strftime('%Y-%m-%d')
        self.processor.load_daily_data(start, today)
    
    def generate_signals(self, trade_date):
        """生成交易信号"""
        print(f"为 {trade_date} 生成信号...")
        
        signals = []
        
        for code in self.processor.daily_data:
            df = self.processor.daily_data[code]
            df = df[df['trade_date'] <= pd.to_datetime(trade_date)]
            
            if len(df) < self.config.SEQUENCE_LENGTH + 10:
                continue
            
            # 取最近N天
            recent = df.tail(self.config.SEQUENCE_LENGTH + 5)
            
            # 特征工程
            features_df = self.engineer.engineer_features(recent)
            features = features_df.iloc[-self.config.SEQUENCE_LENGTH:].values
            
            if len(features) != self.config.SEQUENCE_LENGTH:
                continue
            
            # 预测
            with torch.no_grad():
                x = torch.FloatTensor(features).unsqueeze(0).to(self.device)
                pred = self.model(x).cpu().item()
            
            signals.append({'stock_code': code, 'prediction': pred})
        
        signals_df = pd.DataFrame(signals)
        signals_df = signals_df.sort_values('prediction', ascending=False)
        
        # 过滤
        valid = self.processor.basic_info[
            ~self.processor.basic_info['name'].str.contains('ST', na=False)
        ]['ts_code']
        signals_df = signals_df[signals_df['stock_code'].isin(valid)]
        
        return signals_df
    
    def print_report(self, signals_df, current_positions=None):
        """打印交易报告"""
        if current_positions is None:
            current_positions = []
        
        print("\n" + "=" * 60)
        print("每日交易报告")
        print("=" * 60)
        
        print(f"\n信号总数: {len(signals_df)}")
        
        print(f"\nTop 20 推荐买入:")
        print("-" * 40)
        for i, (_, row) in enumerate(signals_df.head(20).iterrows(), 1):
            name = self.processor.basic_info[
                self.processor.basic_info['ts_code'] == row['stock_code']
            ]['name'].values
            name = name[0] if len(name) > 0 else 'Unknown'
            marker = '★' if row['stock_code'] in current_positions else ' '
            print(f"  {marker} {i:2d}. {row['stock_code']} {name:8s}  {row['prediction']:+.4f}")
        
        if current_positions:
            print(f"\n当前持仓 ({len(current_positions)}只):")
            for code in current_positions:
                pos_row = signals_df[signals_df['stock_code'] == code]
                if len(pos_row) > 0:
                    rank = signals_df['stock_code'].tolist().index(code) + 1
                    score = pos_row['prediction'].iloc[0]
                    print(f"  {code}  排名: {rank}/{len(signals_df)}  分数: {score:+.4f}")


def main():
    config = Config()
    assistant = TradingAssistant(config)
    assistant.update_data()
    
    # 获取最新交易日
    latest_date = max(assistant.processor.daily_data[list(assistant.processor.daily_data.keys())[0]]['trade_date'])
    
    signals = assistant.generate_signals(latest_date)
    
    # 从文件读取当前持仓（如果有的话）
    positions = []
    pos_file = Path('current_positions.txt')
    if pos_file.exists():
        with open(pos_file) as f:
            positions = [line.strip() for line in f if line.strip()]
    
    assistant.print_report(signals, positions)
    
    # 保存信号
    signals.to_csv(f'signals_{latest_date.strftime("%Y%m%d")}.csv', index=False)


if __name__ == "__main__":
    main()
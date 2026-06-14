"""
模拟交易辅助脚本 - 比赛期间每日生成交易信号
用法：python trading_simulator.py
"""
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import copy
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
        
        # 加载模型
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model_input_dim = checkpoint.get('input_dim', 65)
        
        self.model = AlphaNet(
            input_dim=self.model_input_dim,
            d_model=checkpoint.get('d_model', config.HIDDEN_DIM),
            n_heads=config.NUM_HEADS,
            n_layers=config.NUM_LAYERS,
            d_ff=config.HIDDEN_DIM * 4,
            dropout=config.DROPOUT,
            sequence_length=config.SEQUENCE_LENGTH
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        print(f"模型已加载 (epoch {checkpoint['epoch']+1}, IC={checkpoint.get('val_ic', 'N/A'):.4f})")
        
        # 加载股票基本信息
        self.processor = DataProcessor(config)
        self.processor.load_basic_info()
        self.processor.load_trade_calendar()
        
        # 加载最近半年数据
        self.update_data()
        
        self.engineer = FeatureEngineer()
    
    def update_data(self):
        """加载最近半年的量价和基本面数据"""
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
        
        print(f"加载数据: {start_date} ~ {end_date}")
        
        # 临时关闭选股限制，加载全部股票
        temp_config = copy.copy(self.config)
        temp_config.MAX_STOCKS = None
        
        self.processor.config = temp_config
        self.processor.load_daily_data(start_date, end_date)
        self.processor.load_metric_data(start_date, end_date)
        
        # 恢复config
        self.processor.config = self.config
        
        print(f"已加载 {len(self.processor.daily_data)} 只股票")
    
    def generate_signals(self, trade_date):
        """生成某日交易信号"""
        print(f"为 {trade_date} 生成信号...")
        
        signals = []
        valid_codes = set(self.processor.basic_info['ts_code'].values)
        total = len(self.processor.daily_data)
        
        for i, code in enumerate(self.processor.daily_data):
            if i % 500 == 0:
                print(f"  处理进度: {i}/{total}")
            
            if code not in valid_codes:
                continue
            
            df = self.processor.daily_data[code]
            df = df[df['trade_date'] <= pd.to_datetime(trade_date)]
            
            if len(df) < self.config.SEQUENCE_LENGTH + 10:
                continue
            
            recent = df.tail(self.config.SEQUENCE_LENGTH + 10)
            
            # 加载该股票的基本面数据
            metric_df = self.processor.metric_data.get(code)
            
            try:
                features_df = self.engineer.engineer_features(recent, metric_df, None)
            except Exception:
                continue
            
            features = features_df.iloc[-self.config.SEQUENCE_LENGTH:].values.astype(np.float32)
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 统一特征维度（防止有无基本面数据导致维度不一致）
            if features.shape[1] < self.model_input_dim:
                pad = np.zeros((features.shape[0], self.model_input_dim - features.shape[1]), dtype=np.float32)
                features = np.concatenate([features, pad], axis=1)
            elif features.shape[1] > self.model_input_dim:
                features = features[:, :self.model_input_dim]
            
            if len(features) != self.config.SEQUENCE_LENGTH:
                continue
            
            with torch.no_grad():
                x = torch.FloatTensor(features).unsqueeze(0).to(self.device)
                pred = torch.sigmoid(self.model(x)).cpu().item()
            
            signals.append({'stock_code': code, 'prediction': pred})
        
        signals_df = pd.DataFrame(signals)
        if len(signals_df) == 0:
            print("  警告：未生成任何信号")
            return signals_df
        
        signals_df = signals_df.sort_values('prediction', ascending=False)
        return signals_df
    
    def print_report(self, signals_df, current_positions=None):
        """打印交易报告"""
        if current_positions is None:
            current_positions = []
        
        print("\n" + "=" * 60)
        print("每日交易报告")
        print("=" * 60)
        print(f"信号总数: {len(signals_df)}")
        
        print(f"\nTop 20 推荐买入:")
        print("-" * 40)
        for i, (_, row) in enumerate(signals_df.head(20).iterrows(), 1):
            name_info = self.processor.basic_info[
                self.processor.basic_info['ts_code'] == row['stock_code']
            ]
            name = name_info['name'].values[0] if len(name_info) > 0 else 'Unknown'
            marker = '★' if row['stock_code'] in current_positions else ' '
            print(f"  {marker} {i:2d}. {row['stock_code']} {name:8s}  {row['prediction']:.4f}")
        
        if current_positions:
            print(f"\n当前持仓建议:")
            for code in current_positions:
                pos_row = signals_df[signals_df['stock_code'] == code]
                if len(pos_row) > 0:
                    rank = list(signals_df['stock_code']).index(code) + 1
                    score = pos_row['prediction'].iloc[0]
                    action = "持有" if score > 0.55 else "考虑卖出"
                    print(f"  {code} 排名:{rank}/{len(signals_df)} 分数:{score:.4f} → {action}")
                else:
                    print(f"  {code} 无信号 → 建议卖出")


def main():
    config = Config()
    
    # 检查模型是否存在
    model_path = Path('outputs/best_model.pth')
    if not model_path.exists():
        print("错误：未找到模型文件 outputs/best_model.pth")
        print("请先运行 main.py 训练模型")
        return
    
    assistant = TradingAssistant(config, str(model_path))
    
    # 获取最新交易日
    if len(assistant.processor.daily_data) == 0:
        print("错误：未加载到任何数据")
        return
    
    sample_code = list(assistant.processor.daily_data.keys())[0]
    latest_date = assistant.processor.daily_data[sample_code]['trade_date'].max()
    print(f"\n最新数据日期: {pd.Timestamp(latest_date).strftime('%Y-%m-%d')}")
    
    # 生成信号
    signals = assistant.generate_signals(latest_date)
    
    if len(signals) == 0:
        print("未生成任何交易信号")
        return
    
    # 读取当前持仓
    positions = []
    pos_file = Path('current_positions.txt')
    if pos_file.exists():
        with open(pos_file, 'r') as f:
            positions = [line.strip() for line in f if line.strip()]
    
    # 打印报告
    assistant.print_report(signals, positions)
    
    # 保存信号
    output_file = f'signals_{pd.Timestamp(latest_date).strftime("%Y%m%d")}.csv'
    signals.to_csv(output_file, index=False)
    print(f"\n信号已保存到 {output_file}")


if __name__ == "__main__":
    main()

#config.py：
"""
配置文件 - 所有超参数和路径设置
"""
import os
import torch
from pathlib import Path

class Config:
    """全局配置类"""
    
    # ==================== 路径配置 ====================
    BASE_DIR = Path(__file__).parent
    DATA_DIR = Path(r"D:\Software\科大云盘\Data\科大云盘\Stamp_new")
    DAILY_DIR = DATA_DIR / "daily"
    BASIC_FILE = DATA_DIR / "basic.csv"
    TRADE_CAL_FILE = DATA_DIR / "trade_cal.csv"
    METRIC_DIR = DATA_DIR / "metric"
    MONEYFLOW_DIR = DATA_DIR / "moneyflow"
    STOCK_ST_DIR = DATA_DIR / "stock_st"
    MARKET_DIR = DATA_DIR / "market"
    OUTPUT_DIR = BASE_DIR / "outputs"
    
    # ==================== 时间配置 ====================
    TRAIN_START = "2019-01-01"
    TRAIN_END = "2024-12-31"
    VAL_START = "2025-01-01"
    VAL_END = "2025-12-31"
    
    # ==================== 模型配置 ====================
    SEQUENCE_LENGTH = 60          # 输入序列长度（交易日）
    PREDICT_HORIZON = 5           # 预测未来N日收益
    BATCH_SIZE = 256             # 批大小
    HIDDEN_DIM = 256              # 隐藏层维度
    NUM_LAYERS = 2                # Transformer层数
    NUM_HEADS = 4                 # 多头注意力头数
    DROPOUT = 0.1                 # Dropout率
    LEARNING_RATE = 1e-3          # 学习率
    WEIGHT_DECAY = 1e-5           # 权重衰减
    NUM_EPOCHS = 50               # 最大训练轮数
    EARLY_STOPPING = 10           # 早停耐心值
    GRAD_CLIP = 1.0               # 梯度裁剪阈值
    
    # ==================== 交易策略配置 ====================
    INITIAL_CAPITAL = 1_000_000   # 初始资金
    MAX_POSITIONS = 20            # 最大持仓股票数
    DAILY_TRADES = 3              # 每日调仓数量
    COMMISSION_RATE = 0.0003      # 手续费率（万三）
    SLIPPAGE = 0.001              # 滑点
    CASH_RESERVE = 0.05           # 现金保留比例
    
    # ==================== 股票池配置 ====================
    EXCLUDE_MARKETS = ["北交所"]   # 排除的市场
    EXCLUDE_ST = True              # 是否排除ST股票
    MAX_STOCKS = 2000
    
    # ==================== 设备配置 ====================
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 0                # 数据加载线程数
    SEED = 42                      # 随机种子
    
    # ==================== 数据配置 ====================
    MAX_SAMPLES_PER_STOCK = 200   # 每只股票最多200个窗口
    SAMPLE_SIZE = 5000            # 标准化采样数

    # ==================== 对比模型配置 ====================
    MODEL_TYPE = 'lstm'  # 默认，run_comparison会覆盖
    
    # LSTM配置
    LSTM_HIDDEN_SIZE = 128
    LSTM_NUM_LAYERS = 2
    LSTM_DROPOUT = 0.2
    
    # MLP配置
    MLP_HIDDEN_DIMS = [1024, 512, 256, 128]
    MLP_DROPOUT = 0.3

    # ==================== 输出目录初始化 ====================
    def __init__(self):
        """创建必要的输出目录"""
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
    
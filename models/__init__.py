
#models/__init__.py:
"""
仅用于对比实验（LSTM、MLP）
"""
from .lstm import StockLSTM
from .mlp import StockMLP


def get_model(config, input_dim=None):
    """根据配置返回对比模型"""
    model_type = config.MODEL_TYPE
    
    if model_type == 'lstm':
        return StockLSTM(config, input_dim=input_dim)
    
    elif model_type == 'mlp':
        return StockMLP(config, input_dim=input_dim)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
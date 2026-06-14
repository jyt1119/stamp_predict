
# models/mlp.py
"""
MLP模型用于股票预测（基线模型）
"""
import torch
import torch.nn as nn

class StockMLP(nn.Module):
    """股票预测MLP（基线模型）"""
    
    def __init__(self, config, input_dim=None):
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        
        # 如果知道输入维度，直接初始化
        if input_dim is not None:
            self._build_layers(input_dim)
        else:
            self.fc_layers = None
            self._initialized = False
    
    def _build_layers(self, input_dim):
        """构建全连接层"""
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in self.config.MLP_HIDDEN_DIMS:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(self.config.MLP_DROPOUT))
            layers.append(nn.BatchNorm1d(hidden_dim))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.fc_layers = nn.Sequential(*layers)
        self._initialized = True
    
    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        batch_size = x.size(0)
        
        # 展平序列
        x = x.reshape(batch_size, -1)  # (batch_size, seq_len * input_dim)
        
        if self.fc_layers is None:
            self._build_layers(x.size(-1))
        
        x = self.fc_layers(x)  # (batch_size, 1)
        return x
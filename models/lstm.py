
#models/lstm.py:
"""
LSTM模型用于股票预测
"""
import torch
import torch.nn as nn

class StockLSTM(nn.Module):
    """股票预测LSTM"""
    
    def __init__(self, config, input_dim=None):
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        
        # 如果知道输入维度，直接初始化
        if input_dim is not None:
            self._build_lstm(input_dim)
        else:
            self.lstm = None
            self.attention = None
            self.fc = None
            self._initialized = False
    
    def _build_lstm(self, input_dim: int):
        """构建LSTM层"""
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.config.LSTM_HIDDEN_SIZE,
            num_layers=self.config.LSTM_NUM_LAYERS,
            dropout=self.config.LSTM_DROPOUT if self.config.LSTM_NUM_LAYERS > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(self.config.LSTM_HIDDEN_SIZE * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(self.config.LSTM_HIDDEN_SIZE * 2, 128),
            nn.GELU(),
            nn.Dropout(self.config.LSTM_DROPOUT),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(self.config.LSTM_DROPOUT),
            nn.Linear(64, 1)
        )
        
        self._initialized = True
    
    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        if not self._initialized:
            self._build_lstm(x.size(-1))
            self.lstm = self.lstm.to(x.device)
            self.attention = self.attention.to(x.device)
            self.fc = self.fc.to(x.device)
        
        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_size*2)
        
        # 注意力权重
        attn_weights = self.attention(lstm_out)  # (batch_size, seq_len, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        
        # 加权求和
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (batch_size, hidden_size*2)
        
        # 输出层
        x = self.fc(context)  # (batch_size, 1)
    
        return x
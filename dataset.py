#dataset.py：
"""
数据集构建模块 - 滑动窗口采样、标准化、DataLoader创建
内存优化版：不预先存储所有样本，按需读取；标准化参数在线估计避免大数组拼接
"""
import gc
import bisect
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class StockDataset(Dataset):
    """股票时间序列数据集（内存优化版）"""

    def __init__(self, data_processor, feature_engineer,
                 start_date: str, end_date: str,
                 sequence_length: int, predict_horizon: int, desc: str = ""):
        self.data_processor = data_processor
        self.feature_engineer = feature_engineer
        self.sequence_length = sequence_length
        self.predict_horizon = predict_horizon
        
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        
        self.samples = []  # 直接存样本，不用复杂索引
        self._build_samples(desc)

    def _build_samples(self, desc: str):
        """直接构建样本列表，确保无数据泄漏"""
        self.samples = []
        min_length = self.sequence_length + self.predict_horizon + 1
        stocks = list(self.data_processor.daily_data.keys())
        
        max_stocks = getattr(self.data_processor.config, 'MAX_STOCKS', None)
        if max_stocks:
            stocks = stocks[:int(max_stocks)]
        
        max_per_stock = getattr(self.data_processor.config, 'MAX_SAMPLES_PER_STOCK', None)
        
        skipped = 0
        for stock_code in tqdm(stocks, desc=f"构建{desc}样本"):
            daily_df = self.data_processor.daily_data[stock_code]
            if len(daily_df) < min_length:
                skipped += 1
                continue
            
            # 用日期筛选，而不是整数位置
            mask = (daily_df['trade_date'] >= self.start_date) & \
                   (daily_df['trade_date'] <= self.end_date)
            period_df = daily_df[mask].reset_index(drop=True)
            
            if len(period_df) < min_length:
                skipped += 1
                continue
            
            n_total = len(period_df) - self.sequence_length - self.predict_horizon + 1
            if max_per_stock:
                n_windows = min(n_total, int(max_per_stock))
            else:
                n_windows = n_total
            
            # 计算特征（整个period只算一次）
            metric_df = self.data_processor.metric_data.get(stock_code)
            try:
                all_features = self.feature_engineer.engineer_features(period_df, metric_df, None)
            except Exception:
                skipped += 1
                continue
            
            # 滑动窗口采样，直接存特征切片
            for i in range(n_windows):
                feat_slice = all_features.iloc[i:i + self.sequence_length]
                if feat_slice.isnull().any().any():
                    continue
                
                features = feat_slice.values.astype(np.float32)
                features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
                
                cur_close = period_df['close'].iloc[i + self.sequence_length - 1]
                future_close = period_df['close'].iloc[i + self.sequence_length + self.predict_horizon - 1]
                label = np.float32((future_close - cur_close) / (cur_close + 1e-10))
                
                sample_date = period_df['trade_date'].iloc[i + self.sequence_length - 1]
                
                self.samples.append({
                    'features': features,
                    'label': label,
                    'stock_code': stock_code,
                    'date': sample_date
                })
        
        print(f"  跳过股票数: {skipped}")
        print(f"  样本总数: {len(self.samples)}")
        
        if len(self.samples) > 0:
            labels = np.array([s['label'] for s in self.samples])
            print(f"  标签均值: {labels.mean():.6f}, 标准差: {labels.std():.6f}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            'features': torch.from_numpy(s['features']),
            'label': torch.tensor(s['label'], dtype=torch.float32),
            'stock_code': s['stock_code'],
            'date': s['date']
        }


class NormalizedDataset(Dataset):
    """带标准化的数据集包装器（内存优化版）
    通过在线/Welford 方法估计均值与方差，避免一次性拼接大量样本。
    """

    def __init__(self, base_dataset: StockDataset, scaler_stats: Tuple = None,
                 sample_size: int = 5000):
        self.base_dataset = base_dataset
        if scaler_stats is None:
            self._compute_stats(sample_size)
        else:
            self.feature_mean, self.feature_std = scaler_stats

    def _compute_stats(self, sample_size: int):
        """在线增量计算均值和方差（将时间步当作独立样本进行统计）"""
        print("估计标准化参数（采样法、在线计算）...")
        n_total = len(self.base_dataset)
        if n_total == 0:
            self.feature_mean = np.zeros(0, dtype=np.float32)
            self.feature_std = np.ones(0, dtype=np.float32)
            return

        n_sample = min(sample_size, n_total)
        indices = np.random.choice(n_total, n_sample, replace=False)

        count = 0
        mean = None
        M2 = None

        for i in tqdm(indices, desc="采样计算统计量"):
            sample = self.base_dataset[i]
            feats = sample['features'].numpy().astype(np.float64)  # (seq_len, feat_dim)
            if feats.size == 0:
                continue
            # 对NaN/inf做安全处理，避免后续均值/方差为NaN
            feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
            bcount = feats.shape[0]
            bmean = feats.mean(axis=0)
            bM2 = np.sum((feats - bmean) ** 2, axis=0)

            if mean is None:
                mean = bmean
                M2 = bM2
                count = bcount
            else:
                delta = bmean - mean
                new_count = count + bcount
                mean = mean + delta * (bcount / new_count)
                M2 = M2 + bM2 + (delta ** 2) * (count * bcount / new_count)
                count = new_count

            del sample, feats, bmean, bM2
            gc.collect()

        if count == 0:
            self.feature_mean = np.zeros(0, dtype=np.float32)
            self.feature_std = np.ones(0, dtype=np.float32)
            return

        self.feature_mean = mean.astype(np.float32)
        self.feature_std = np.sqrt(M2 / count).astype(np.float32) + 1e-8

        print(f"  特征均值范围: [{self.feature_mean.min():.6f}, {self.feature_mean.max():.6f}]")
        print(f"  特征标准差范围: [{self.feature_std.min():.6f}, {self.feature_std.max():.6f}]")

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        feats = sample['features'].numpy()
        feats = (feats - self.feature_mean) / self.feature_std
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        sample['features'] = torch.from_numpy(feats)
        return sample


def collate_fn(batch):
    """自定义批处理函数，返回 (features, labels)"""
    features = torch.stack([item['features'] for item in batch])  # (B, seq_len, feat_dim)
    labels = torch.stack([item['label'] for item in batch]).unsqueeze(-1)  # (B, 1)
    return features, labels


def create_dataloaders(config, data_processor, feature_engineer, sample_size: int = 5000):
    """创建训练和验证 DataLoader"""
    print("\n创建数据集...")

    train_dataset = StockDataset(
        data_processor=data_processor,
        feature_engineer=feature_engineer,
        start_date=config.TRAIN_START,
        end_date=config.TRAIN_END,
        sequence_length=config.SEQUENCE_LENGTH,
        predict_horizon=config.PREDICT_HORIZON,
        desc="训练"
    )

    # 用采样拟合标准化器（默认采样数较小，避免内存压力）
    normalized_train = NormalizedDataset(train_dataset, scaler_stats=None, sample_size=sample_size)

    val_dataset = StockDataset(
        data_processor=data_processor,
        feature_engineer=feature_engineer,
        start_date=config.VAL_START,
        end_date=config.VAL_END,
        sequence_length=config.SEQUENCE_LENGTH,
        predict_horizon=config.PREDICT_HORIZON,
        desc="验证"
    )
    normalized_val = NormalizedDataset(val_dataset, scaler_stats=(normalized_train.feature_mean, normalized_train.feature_std))

    train_loader = DataLoader(
        normalized_train,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=True if config.NUM_WORKERS > 0 else False
    )

    val_loader = DataLoader(
        normalized_val,
        batch_size=max(1, config.BATCH_SIZE // 2),
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn
    )

    print(f"  训练批次: {len(train_loader)}")
    print(f"  验证批次: {len(val_loader)}")

    return (train_loader, val_loader,
            normalized_train.feature_mean, normalized_train.feature_std,
            normalized_train, normalized_val)

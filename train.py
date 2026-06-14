
#train.py:
"""
模型训练模块
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error


class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.best_val_ic = -float('inf')
    
    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
            return False


class Trainer:
    """模型训练器"""
    
    def __init__(self, model, config, device):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # 损失函数
        self.criterion_cls = nn.BCEWithLogitsLoss()
        self.criterion_reg = nn.SmoothL1Loss()
        self.cls_weight = 0.6
        
        # 优化器
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        # 学习率调度器
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        # 早停
        self.early_stopping = EarlyStopping(patience=config.EARLY_STOPPING)
        
        # 记录
        self.train_losses = []
        self.val_losses = []
        self.val_ics = []
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.best_val_ic = -float('inf')

        # 输出目录
        self.output_dir = Path(config.OUTPUT_DIR)
    
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        n_batches = len(train_loader)
        
        pbar = tqdm(train_loader, desc="训练", leave=False)
        for features, labels in pbar:
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(features)
            pred = outputs.squeeze(1)
            target = labels.squeeze(1)
            target_dir = (target > 0).float()
            loss_cls = self.criterion_cls(pred, target_dir)
            loss_reg = self.criterion_reg(pred, target)
            loss = self.cls_weight * loss_cls + (1 - self.cls_weight) * loss_reg
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRAD_CLIP)
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        return total_loss / n_batches
    
    @torch.no_grad()
    def validate(self, val_loader):
        """验证"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for features, labels in tqdm(val_loader, desc="验证", leave=False):
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(features)
            pred = outputs.squeeze(1)
            target = labels.squeeze(1)
            target_dir = (target > 0).float()
            loss_cls = self.criterion_cls(pred, target_dir)
            loss_reg = self.criterion_reg(pred, target)
            loss = self.cls_weight * loss_cls + (1 - self.cls_weight) * loss_reg
            total_loss += loss.item()
            
            all_preds.extend(torch.sigmoid(pred).cpu().numpy().flatten())
            all_labels.extend(target.cpu().numpy().flatten())
        
        # 计算指标
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Rank IC
        ic = spearmanr(all_preds, all_labels)[0]
        
        # Pearson相关系数
        pearson = pearsonr(all_preds, all_labels)[0]
        
        # MSE和MAE
        mse = mean_squared_error(all_labels, all_preds)
        mae = mean_absolute_error(all_labels, all_preds)
        
        # 方向胜率
        pred_direction = np.where(all_preds > 0.5, 1, -1)
        true_direction = np.where(all_labels > 0, 1, -1)
        
        valid = true_direction != 0
        if valid.sum() > 0:
            direction_acc = np.mean(pred_direction[valid] == true_direction[valid])
        else:
            direction_acc = 0
        
        metrics = {
            'loss': total_loss / len(val_loader),
            'ic': ic,
            'pearson': pearson,
            'mse': mse,
            'mae': mae,
            'direction_acc': direction_acc
        }
        
        return metrics
    
    def train(self, train_loader, val_loader, num_epochs):
        """完整训练流程"""
        print("\n" + "=" * 60)
        print("开始训练")
        print("=" * 60)
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 40)
            
            # 训练
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # 验证
            val_metrics = self.validate(val_loader)
            self.val_losses.append(val_metrics['loss'])
            self.val_ics.append(val_metrics['ic'])
            
            # 学习率调整
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 打印指标
            print(f"  Train Loss:      {train_loss:.6f}")
            print(f"  Val Loss:        {val_metrics['loss']:.6f}")
            print(f"  Rank IC:         {val_metrics['ic']:.4f}")
            print(f"  Pearson CC:      {val_metrics['pearson']:.4f}")
            print(f"  Direction Acc:   {val_metrics['direction_acc']:.4f}")
            print(f"  Learning Rate:   {current_lr:.2e}")
            
            # 保存最佳模型
            if val_metrics['ic'] > self.best_val_ic:
                self.best_val_ic = val_metrics['ic']
                self.best_epoch = epoch + 1
                self._save_checkpoint(epoch, val_metrics)
                print(f"  ✓ 新的最佳模型 (IC: {val_metrics['ic']:.4f})")
            
            # 早停检查
            if self.early_stopping(val_metrics['loss']):
                print(f"\n早停触发！在第 {epoch + 1} 轮停止训练")
                break
        
        print(f"\n训练完成！最佳模型在第 {self.best_epoch} 轮, Val IC: {self.best_val_ic:.4f}")
        self._plot_curves()
        
        return self.train_losses, self.val_losses
    
    def _save_checkpoint(self, epoch, val_metrics):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'input_dim': self.model.input_dim,
            'd_model': self.model.d_model,
            'val_loss': val_metrics['loss'],
            'val_ic': val_metrics['ic'],
            'val_pearson': val_metrics['pearson'],
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }
        torch.save(checkpoint, self.output_dir / 'best_model.pth')
    
    def _plot_curves(self):
        """绘制训练曲线"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 损失曲线
        axes[0].plot(self.train_losses, label='Train Loss', linewidth=1.5)
        axes[0].plot(self.val_losses, label='Val Loss', linewidth=1.5)
        axes[0].axvline(x=self.best_epoch - 1, color='r', linestyle='--', alpha=0.5, label='Best Model')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # IC曲线
        axes[1].plot(self.val_ics, label='Rank IC', color='green', linewidth=1.5)
        axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Rank IC')
        axes[1].set_title('Validation Rank IC')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"训练曲线已保存到 {self.output_dir / 'training_curves.png'}")
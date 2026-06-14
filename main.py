
# main.py:
"""
主程序 - 完整训练和回测流程
"""
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from config import Config
from data_processor import DataProcessor
from feature_engine import FeatureEngineer
from dataset import StockDataset, NormalizedDataset, create_dataloaders, collate_fn
from model import AlphaNet
from train import Trainer
from backtest import Backtest
from utils import (
    calculate_ic_stats, plot_portfolio_curve, 
    plot_ic_distribution, load_benchmark
)
import torch.nn as nn

def generate_predictions(model, dataset, device, config):
    """生成预测结果DataFrame"""
    model.eval()
    predictions_list = []
    
    print("生成预测...")
    
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="预测"):
            sample = dataset[i]
            features = sample['features'].unsqueeze(0).to(device)
            pred = model(features).cpu().item()
            
            predictions_list.append({
                'date': sample['date'],
                'stock_code': sample['stock_code'],
                'prediction': pred,
                'true_return': sample['label'].item()
            })
    
    df = pd.DataFrame(predictions_list)
    
    # 计算IC统计
    ic_stats = calculate_ic_stats(df)
    if ic_stats:
        print(f"\n验证集IC统计:")
        print(f"  IC均值:  {ic_stats['IC_mean']:.4f}")
        print(f"  IC标准差: {ic_stats['IC_std']:.4f}")
        print(f"  ICIR:     {ic_stats['ICIR']:.4f}")
        print(f"  IC胜率:   {ic_stats['IC_positive_ratio']:.2%}")
    
    return df, ic_stats


def main():
    """主函数"""
    # ==================== 初始化 ====================
    config = Config()
    
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    
    device = torch.device(config.DEVICE)
    print(f"设备: {device}")
    
    output_dir = Path(config.OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)
    
    # ==================== 步骤1: 数据处理 ====================
    print("\n" + "=" * 60)
    print("步骤 1/5: 数据处理")
    print("=" * 60)
    
    processor = DataProcessor(config)
    processor.process_all()
    
    # ==================== 步骤2: 特征工程 + 数据集 ====================
    print("\n" + "=" * 60)
    print("步骤 2/5: 构建数据集")
    print("=" * 60)
    
    engineer = FeatureEngineer()
    
    # 测试特征维度
    sample_code = list(processor.daily_data.keys())[0]
    sample_df = processor.daily_data[sample_code].head(100)
    sample_feat = engineer.engineer_features(sample_df)
    print(f"特征维度: {sample_feat.shape}")
    
    # 创建DataLoader
    train_loader, val_loader, feat_mean, feat_std, norm_train, norm_val = \
        create_dataloaders(config, processor, engineer, 
                          sample_size=config.SAMPLE_SIZE)
    
    input_dim = next(iter(train_loader))[0].shape[-1]
    print(f"输入特征数: {input_dim}")
    
    # ==================== 步骤3: 模型训练 ====================
    print("\n" + "=" * 60)
    print("步骤 3/5: 模型训练")
    print("=" * 60)
    
    model = AlphaNet(
        input_dim=input_dim,
        d_model=config.HIDDEN_DIM,
        n_heads=config.NUM_HEADS,
        n_layers=config.NUM_LAYERS,
        d_ff=config.HIDDEN_DIM * 4,
        dropout=config.DROPOUT,
        sequence_length=config.SEQUENCE_LENGTH
    )
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数: {n_params:,}")
    
    trainer = Trainer(model, config, device)
    trainer.train(train_loader, val_loader, config.NUM_EPOCHS)
    
    # 加载最佳模型
    best_path = output_dir / 'best_model.pth'
    if best_path.exists():
        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"加载最佳模型 (epoch {checkpoint['epoch']+1})")
    
    # ==================== 步骤4: 生成预测 ====================
    print("\n" + "=" * 60)
    print("步骤 4/5: 生成预测")
    print("=" * 60)
    
    pred_df, ic_stats = generate_predictions(model, norm_val, device, config)
    pred_df.to_csv(output_dir / 'predictions.csv', index=False)
    
    # 绘制IC分布
    if ic_stats and 'IC_values' in ic_stats:
        plot_ic_distribution(ic_stats['IC_values'], 
                           str(output_dir / 'ic_distribution.png'))
    
    # ==================== 步骤5: 回测 ====================
    print("\n" + "=" * 60)
    print("步骤 5/5: 回测评估")
    print("=" * 60)
    
    # 准备价格数据
    price_data = {}
    for code, df in processor.daily_data.items():
        df_copy = df.copy()
        df_copy['trade_date'] = pd.to_datetime(df_copy['trade_date'])
        price_data[code] = df_copy.set_index('trade_date')[['close']]
    
    # 运行回测
    backtest = Backtest(config)
    results = backtest.run(
        pred_df, price_data,
        pd.to_datetime(config.VAL_START),
        pd.to_datetime(config.VAL_END)
    )
    
    backtest.print_summary(results)
    
    # 加载基准并绘图
    benchmark = load_benchmark(config, config.VAL_START, config.VAL_END)
    
    if 'daily_df' in results:
        plot_portfolio_curve(
            results['daily_df'], benchmark,
            str(output_dir / 'portfolio_value.png')
        )
    
    # ==================== 保存最终结果 ====================
    summary = {
        '特征数量': input_dim,
        '模型参数': n_params,
    }
    if results:
        summary.update({
            '初始资金': results['initial_capital'],
            '最终资金': results['final_value'],
            '总收益率': f"{results['total_return']:.4%}",
            '年化收益率': f"{results['annual_return']:.4%}",
            '夏普比率': f"{results['sharpe_ratio']:.4f}",
            '最大回撤': f"{results['max_drawdown']:.4%}",
            '胜率': f"{results['win_rate']:.4%}",
        })
    
    summary_df = pd.DataFrame([summary]).T
    summary_df.columns = ['值']
    summary_df.to_csv(output_dir / 'summary.csv')
    
    print("\n" + "=" * 60)
    print("全部流程完成！")
    print("=" * 60)
    print(f"\n输出文件位于: {output_dir}/")
    print("  - best_model.pth       : 最佳模型权重")
    print("  - training_curves.png  : 训练曲线")
    print("  - predictions.csv      : 预测结果")
    print("  - portfolio_value.png  : 回测曲线")
    print("  - summary.csv          : 结果汇总")


if __name__ == "__main__":
    main()
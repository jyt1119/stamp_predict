
#run_comparison.py:
"""
对比实验脚本 - 独立运行LSTM和MLP对比模型
"""
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from config import Config
from data_processor import DataProcessor
from feature_engine import FeatureEngineer
from dataset import create_dataloaders
from backtest import Backtest
from utils import calculate_ic_stats, plot_portfolio_curve, load_benchmark
from models import get_model

import random

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def generate_predictions(model, dataset, device):
    """生成预测结果DataFrame"""
    model.eval()
    predictions_list = []
    
    print("生成预测...")
    
    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            features = sample['features'].unsqueeze(0).to(device)
            out = model(features)
            if out.dim() > 0:
                pred = out.squeeze().cpu().item()
            else:
                pred = out.cpu().item()
            
            predictions_list.append({
                'date': sample['date'],
                'stock_code': sample['stock_code'],
                'prediction': pred,
                'true_return': sample['label'].item()
            })
    
    df = pd.DataFrame(predictions_list)
    ic_stats = calculate_ic_stats(df)
    
    if ic_stats:
        print(f"\nIC统计:")
        print(f"  IC均值:  {ic_stats['IC_mean']:.4f}")
        print(f"  ICIR:     {ic_stats['ICIR']:.4f}")
        print(f"  IC胜率:   {ic_stats['IC_positive_ratio']:.2%}")
    
    return df, ic_stats


def run_model_experiment(config, model_name, train_loader, val_loader, 
                          norm_val, price_data, benchmark, device):
    """运行单个模型的完整实验"""
    
    print(f"\n{'='*60}")
    print(f"运行模型: {model_name}")
    print(f"{'='*60}")
    
    # 为每个模型创建独立输出目录
    import copy
    config_copy = copy.copy(config)
    model_output_dir = Path(config.OUTPUT_DIR) / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)
    config_copy.OUTPUT_DIR = str(model_output_dir)
    
    # 获取输入维度
    input_dim = next(iter(train_loader))[0].shape[-1]
    
    # 创建模型
    model = get_model(config_copy, input_dim=input_dim)
    model = model.to(device)
    
    # 计算参数量
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params:,}")
    
    # 训练模型
    from train import Trainer
    trainer = Trainer(model, config_copy, device)
    
    num_epochs = min(config_copy.NUM_EPOCHS, 30)
    trainer.train(train_loader, val_loader, num_epochs)
    
    # 加载最佳模型
    best_path = model_output_dir / 'best_model.pth'
    
    if best_path.exists():
        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # 生成预测
    pred_df, ic_stats = generate_predictions(model, norm_val, device)
    
    # 保存预测结果
    pred_df.to_csv(model_output_dir / 'predictions.csv', index=False)
    
    # 回测
    backtest = Backtest(config_copy)
    results = backtest.run(
        pred_df, price_data,
        pd.to_datetime(config_copy.VAL_START),
        pd.to_datetime(config_copy.VAL_END)
    )
    
    return {
        'model_name': model_name,
        'params': n_params,
        'ic_stats': ic_stats,
        'backtest': results,
        'pred_df': pred_df
    }

def main():
    # 初始化配置
    config = Config()
    device = torch.device(config.DEVICE)
    print(f"设备: {device}")
    
    # 创建输出目录
    output_dir = Path(config.OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)
    
    # ========== 数据处理（只做一次）==========
    print("\n" + "="*60)
    print("加载数据...")
    print("="*60)
    
    processor = DataProcessor(config)
    processor.process_all()
    
    engineer = FeatureEngineer()
    
    train_loader, val_loader, feat_mean, feat_std, norm_train, norm_val = \
        create_dataloaders(config, processor, engineer, sample_size=config.SAMPLE_SIZE)
    
    input_dim = next(iter(train_loader))[0].shape[-1]
    print(f"输入特征数: {input_dim}")
    
    # 准备价格数据
    price_data = {}
    for code, df in processor.daily_data.items():
        df_copy = df.copy()
        df_copy['trade_date'] = pd.to_datetime(df_copy['trade_date'])
        price_data[code] = df_copy.set_index('trade_date')[['close']]
    
    # 加载基准
    benchmark = load_benchmark(config, config.VAL_START, config.VAL_END)
    
    # ========== 运行对比模型 ==========
    # 选择要运行的模型
    models_to_run = ['lstm', 'mlp'] 
    
    results = []
    for model_name in models_to_run:
        try:
            result = run_model_experiment(
                config, model_name,
                train_loader, val_loader, norm_val,
                price_data, benchmark, device
            )
            results.append(result)
        except Exception as e:
            print(f"模型 {model_name} 运行失败: {e}")
            import traceback
            traceback.print_exc()
    
    # ========== 打印对比结果 ==========
    print("\n" + "="*60)
    print("对比实验结果汇总")
    print("="*60)
    
    comparison = []
    for r in results:
        ic = r['ic_stats'] or {}
        bt = r['backtest'] or {}
        comparison.append({
            '模型': r['model_name'],
            '参数量': r['params'],
            'IC均值': f"{ic.get('IC_mean', 0):.4f}",
            'ICIR': f"{ic.get('ICIR', 0):.2f}",
            '年化收益': f"{bt.get('annual_return', 0):.2%}",
            '夏普比率': f"{bt.get('sharpe_ratio', 0):.2f}",
            '最大回撤': f"{bt.get('max_drawdown', 0):.2%}",
        })
    
    comp_df = pd.DataFrame(comparison)
    print(comp_df.to_string(index=False))
    
    # 保存对比结果
    comp_df.to_csv(output_dir / 'model_comparison.csv', index=False)
    print(f"\n对比结果已保存到 {output_dir / 'model_comparison.csv'}")


if __name__ == "__main__":
    main()
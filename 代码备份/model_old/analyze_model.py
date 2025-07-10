#!/usr/bin/env python3
"""
模型性能分析和改进建议脚本
分析LSTM模型的训练结果并提供改进建议
"""

import sys
import os
sys.path.append('/home/phil/Files/lstmPaper_v250618/model')

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from data_processor import CosmicRayDataProcessor, split_time_series
from lstm_model import CosmicRayLSTM, ModelTrainer, calculate_metrics

# 设置matplotlib显示
plt.rcParams['figure.figsize'] = (12, 8)
plt.style.use('seaborn-v0_8')

def analyze_model_performance():
    """分析模型性能"""
    print("=== 模型性能分析 ===")
    print("\n从你的训练结果来看:")
    
    print("\n1. 训练集性能 (很好):")
    print("   - R² = 0.894 (89.4%的方差被解释)")
    print("   - 相关系数 = 0.948 (非常强的正相关)")
    print("   - RMSE = 1.92 (相对误差较小)")
    
    print("\n2. 验证集性能 (中等):")
    print("   - R² = 0.214 (仅21.4%的方差被解释)")
    print("   - 相关系数 = 0.523 (中等程度相关)")
    print("   - RMSE = 2.09 (误差增加)")
    
    print("\n3. 测试集性能 (差):")
    print("   - R² = -14.08 (负值说明模型表现比简单平均还差)")
    print("   - 相关系数 = -0.249 (弱负相关)")
    print("   - RMSE = 5.41 (误差很大)")
    
    print("\n🔍 问题诊断:")
    print("   ❌ 严重过拟合: 训练集和测试集性能差距巨大")
    print("   ❌ 泛化能力差: 模型无法很好地预测未见过的数据")
    print("   ❌ 时间分布不均: 可能存在时间相关的系统性偏差")

def suggest_improvements():
    """提供改进建议"""
    print("\n=== 改进建议 ===")
    
    print("\n💡 策略1: 解决过拟合")
    print("   1. 增加Dropout比例 (0.2 → 0.4)")
    print("   2. 减少模型复杂度 (隐藏层 128 → 64)")
    print("   3. 添加L1/L2正则化")
    print("   4. 增加早停patience")
    
    print("\n💡 策略2: 改进数据处理")
    print("   1. 使用滑动窗口交叉验证")
    print("   2. 特征工程: 添加移动平均、趋势等")
    print("   3. 数据增强: 添加噪声、时间扰动")
    print("   4. 检查数据质量和异常值")
    
    print("\n💡 策略3: 调整网络架构")
    print("   1. 尝试单层LSTM")
    print("   2. 添加注意力机制")
    print("   3. 使用GRU替代LSTM")
    print("   4. 尝试Transformer架构")
    
    print("\n💡 策略4: 优化训练策略")
    print("   1. 降低学习率 (0.001 → 0.0001)")
    print("   2. 使用余弦退火调度")
    print("   3. 增加训练轮数")
    print("   4. 调整批大小")

def create_improved_config():
    """创建改进的配置"""
    print("\n=== 改进配置建议 ===")
    
    configs = {
        "保守改进": {
            'sequence_length': 365,
            'hidden_size': 64,      # 减少复杂度
            'num_layers': 1,        # 单层LSTM
            'dropout': 0.4,         # 增加dropout
            'batch_size': 16,       # 小批量
            'epochs': 200,          # 更多轮次
            'learning_rate': 0.0001, # 更小学习率
            'patience': 25,         # 更多耐心
        },
        
        "激进改进": {
            'sequence_length': 180,  # 减少序列长度
            'hidden_size': 32,       # 更小网络
            'num_layers': 1,
            'dropout': 0.5,          # 更多dropout
            'batch_size': 8,
            'epochs': 300,
            'learning_rate': 0.00005,
            'patience': 30,
        }
    }
    
    for name, config in configs.items():
        print(f"\n{name}配置:")
        for key, value in config.items():
            print(f"   {key}: {value}")

def plot_data_analysis():
    """分析数据分布"""
    print("\n=== 数据分析 ===")
    
    # 数据路径
    solar_data_path = '/home/phil/Files/lstmPaper_v250618/data/solar_physics_data_1980_extend_smoothed.csv'
    cosmic_ray_path = '/home/phil/Files/lstmPaper_v250618/data/raw_data/ams/helium.csv'
    
    try:
        # 加载数据
        processor = CosmicRayDataProcessor(solar_data_path, cosmic_ray_path)
        processor.load_data()
        
        # 分析宇宙线数据分布
        cosmic_flux = processor.cosmic_data['helium_flux m^-2sr^-1s^-1GV^-1']
        
        plt.figure(figsize=(15, 10))
        
        # 时间序列图
        plt.subplot(2, 2, 1)
        plt.plot(processor.cosmic_data['date YYYY-MM-DD'], cosmic_flux)
        plt.title('Cosmic Ray Flux Time Series')
        plt.xlabel('Date')
        plt.ylabel('Flux')
        plt.xticks(rotation=45)
        
        # 直方图
        plt.subplot(2, 2, 2)
        plt.hist(cosmic_flux, bins=50, alpha=0.7)
        plt.title('Cosmic Ray Flux Distribution')
        plt.xlabel('Flux')
        plt.ylabel('Frequency')
        
        # 统计信息
        plt.subplot(2, 2, 3)
        stats_text = f"""
        统计信息:
        Mean: {cosmic_flux.mean():.2f}
        Std: {cosmic_flux.std():.2f}
        Min: {cosmic_flux.min():.2f}
        Max: {cosmic_flux.max():.2f}
        Skewness: {cosmic_flux.skew():.2f}
        """
        plt.text(0.1, 0.5, stats_text, fontsize=12, transform=plt.gca().transAxes)
        plt.axis('off')
        
        # 趋势分析
        plt.subplot(2, 2, 4)
        # 计算30天移动平均
        cosmic_flux_ma = cosmic_flux.rolling(window=30).mean()
        plt.plot(processor.cosmic_data['date YYYY-MM-DD'], cosmic_flux, alpha=0.3, label='Original')
        plt.plot(processor.cosmic_data['date YYYY-MM-DD'], cosmic_flux_ma, label='30-day MA')
        plt.title('Trend Analysis')
        plt.xlabel('Date')
        plt.ylabel('Flux')
        plt.legend()
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        save_path = '/home/phil/Files/lstmPaper_v250618/model/results/data_analysis.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"数据分析图已保存: {save_path}")
        
        # 检查数据异常
        q1 = cosmic_flux.quantile(0.25)
        q3 = cosmic_flux.quantile(0.75)
        iqr = q3 - q1
        outliers = cosmic_flux[(cosmic_flux < q1 - 1.5*iqr) | (cosmic_flux > q3 + 1.5*iqr)]
        
        print(f"\n数据质量检查:")
        print(f"   总样本数: {len(cosmic_flux)}")
        print(f"   异常值数量: {len(outliers)} ({len(outliers)/len(cosmic_flux)*100:.1f}%)")
        print(f"   数据变异系数: {cosmic_flux.std()/cosmic_flux.mean()*100:.1f}%")
        
    except Exception as e:
        print(f"数据分析失败: {e}")

def create_improved_model_script():
    """创建改进的模型脚本"""
    print("\n=== 创建改进模型 ===")
    
    script_content = '''#!/usr/bin/env python3
"""
改进的LSTM模型训练脚本
基于第一次训练的经验进行优化
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import sys

sys.path.append('/home/phil/Files/lstmPaper_v250618/model')

from data_processor import CosmicRayDataProcessor, split_time_series
from lstm_model import CosmicRayLSTM, ModelTrainer, calculate_metrics

def create_data_loaders(X_train, y_train, X_val, y_val, batch_size=16):
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def main_improved():
    """改进的主训练流程"""
    print("=== 改进的宇宙线LSTM模型 ===")
    
    # 改进的超参数
    config = {
        'sequence_length': 365,
        'hidden_size': 64,        # 减少复杂度
        'num_layers': 1,          # 单层LSTM
        'dropout': 0.4,           # 增加正则化
        'batch_size': 16,         # 小批量
        'epochs': 200,            # 更多轮次
        'learning_rate': 0.0001,  # 更小学习率
        'patience': 25,           # 更多耐心
    }
    
    print("改进配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # 数据处理 (与原来相同)
    solar_data_path = '/home/phil/Files/lstmPaper_v250618/data/solar_physics_data_1980_extend_smoothed.csv'
    cosmic_ray_path = '/home/phil/Files/lstmPaper_v250618/data/raw_data/ams/helium.csv'
    
    processor = CosmicRayDataProcessor(
        solar_data_path=solar_data_path,
        cosmic_ray_path=cosmic_ray_path,
        sequence_length=config['sequence_length'],
        target_rigidity=2.97
    )
    
    X, y, dates = processor.prepare_data()
    
    # 时间序列分割
    (X_train, y_train, dates_train), (X_val, y_val, dates_val), (X_test, y_test, dates_test) = \\
        split_time_series(X, y, dates, train_ratio=0.7, val_ratio=0.15)
    
    # 创建数据加载器
    train_loader, val_loader = create_data_loaders(
        X_train, y_train, X_val, y_val, 
        batch_size=config['batch_size']
    )
    
    # 创建改进的模型
    model = CosmicRayLSTM(
        input_size=X.shape[2],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = ModelTrainer(model, device=device)
    
    # 训练
    output_dir = '/home/phil/Files/lstmPaper_v250618/model/results'
    model_save_path = os.path.join(output_dir, 'improved_model.pth')
    
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config['epochs'],
        learning_rate=config['learning_rate'],
        patience=config['patience'],
        save_path=model_save_path
    )
    
    # 评估
    checkpoint = torch.load(model_save_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 预测和评估
    y_test_pred = trainer.predict(X_test)
    y_test_true = processor.inverse_transform_y(y_test.numpy())
    y_test_pred = processor.inverse_transform_y(y_test_pred)
    
    test_metrics = calculate_metrics(y_test_true, y_test_pred)
    print("\\n改进模型测试集性能:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.6f}")
    
    return model, processor, trainer, test_metrics

if __name__ == "__main__":
    model, processor, trainer, metrics = main_improved()
'''
    
    save_path = '/home/phil/Files/lstmPaper_v250618/model/train_improved_lstm.py'
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print(f"改进模型脚本已创建: {save_path}")

def main():
    """主分析函数"""
    print("🔍 LSTM宇宙线预测模型 - 性能分析与改进建议")
    print("="*60)
    
    # 分析当前性能
    analyze_model_performance()
    
    # 提供改进建议
    suggest_improvements()
    
    # 配置建议
    create_improved_config()
    
    # 数据分析
    plot_data_analysis()
    
    # 创建改进脚本
    create_improved_model_script()
    
    print("\n" + "="*60)
    print("📋 下一步行动计划:")
    print("1. 运行改进的模型: python train_improved_lstm.py")
    print("2. 比较新旧模型性能")
    print("3. 根据结果进一步调优")
    print("4. 考虑更高级的技术 (注意力机制, Transformer等)")

if __name__ == "__main__":
    main()
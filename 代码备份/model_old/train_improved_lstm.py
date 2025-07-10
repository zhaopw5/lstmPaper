#!/usr/bin/env python3
"""
改进的LSTM训练脚本 - 解决过拟合问题
基于第一次训练结果的改进版本
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import sys

# 添加项目路径
sys.path.append('/home/phil/Files/lstmPaper_v250618/model')

from data_processor import CosmicRayDataProcessor, split_time_series
from lstm_model import CosmicRayLSTM, ModelTrainer, calculate_metrics

def create_data_loaders(X_train, y_train, X_val, y_val, batch_size=16):
    """创建数据加载器"""
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def plot_predictions_simple(y_true, y_pred, dates, title="Prediction Results", save_path=None):
    """简化的预测结果绘图函数"""
    plt.figure(figsize=(15, 8))
    
    # 转换日期
    date_objects = [datetime.strptime(date, '%Y-%m-%d') for date in dates]
    
    plt.subplot(2, 1, 1)
    plt.plot(date_objects, y_true, label='True Values', alpha=0.7, linewidth=1)
    plt.plot(date_objects, y_pred, label='Predictions', alpha=0.7, linewidth=1)
    plt.xlabel('Date')
    plt.ylabel('Cosmic Ray Flux')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # 散点图
    plt.subplot(2, 1, 2)
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('True vs Predicted')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def main_improved():
    """改进的主训练流程"""
    print("=== 改进的宇宙线LSTM模型 ===")
    print("目标: 解决过拟合问题\n")
    
    # 数据路径
    solar_data_path = '/home/phil/Files/lstmPaper_v250618/data/solar_physics_data_1980_extend_smoothed.csv'
    cosmic_ray_path = '/home/phil/Files/lstmPaper_v250618/data/raw_data/ams/helium.csv'
    
    # 创建输出目录
    output_dir = '/home/phil/Files/lstmPaper_v250618/model/results'
    os.makedirs(output_dir, exist_ok=True)
    
    # 改进的超参数配置
    config = {
        'sequence_length': 365,
        'hidden_size': 64,        # 减少复杂度: 128 → 64
        'num_layers': 1,          # 减少层数: 2 → 1  
        'dropout': 0.4,           # 增加dropout: 0.2 → 0.4
        'batch_size': 16,         # 减少批大小: 32 → 16
        'epochs': 200,            # 增加训练轮数: 100 → 200
        'learning_rate': 0.0001,  # 降低学习率: 0.001 → 0.0001
        'patience': 25,           # 增加耐心: 15 → 25
    }
    
    print("改进配置 (vs 原配置):")
    original_config = {
        'hidden_size': 128, 'num_layers': 2, 'dropout': 0.2,
        'batch_size': 32, 'epochs': 100, 'learning_rate': 0.001, 'patience': 15
    }
    
    for key in config.keys():
        if key in original_config:
            print(f"  {key}: {original_config[key]} → {config[key]}")
        else:
            print(f"  {key}: {config[key]}")
    print()
    
    # 步骤1: 数据处理
    print("步骤1: 数据处理")
    processor = CosmicRayDataProcessor(
        solar_data_path=solar_data_path,
        cosmic_ray_path=cosmic_ray_path,
        sequence_length=config['sequence_length'],
        target_rigidity=2.97
    )
    
    X, y, dates = processor.prepare_data()
    print(f"数据准备完成: {len(X)} 个样本\n")
    
    # 步骤2: 数据分割
    print("步骤2: 数据分割")
    (X_train, y_train, dates_train), (X_val, y_val, dates_val), (X_test, y_test, dates_test) = \
        split_time_series(X, y, dates, train_ratio=0.7, val_ratio=0.15)
    print()
    
    # 步骤3: 创建数据加载器
    print("步骤3: 创建数据加载器")
    train_loader, val_loader = create_data_loaders(
        X_train, y_train, X_val, y_val, 
        batch_size=config['batch_size']
    )
    print(f"批次配置: 训练={len(train_loader)}, 验证={len(val_loader)}\n")
    
    # 步骤4: 创建改进的模型
    print("步骤4: 创建改进的LSTM模型")
    model = CosmicRayLSTM(
        input_size=X.shape[2],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = ModelTrainer(model, device=device)
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    print(f"使用设备: {device}\n")
    
    # 步骤5: 训练改进模型
    print("步骤5: 开始训练改进模型")
    model_save_path = os.path.join(output_dir, 'improved_model.pth')
    
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config['epochs'],
        learning_rate=config['learning_rate'],
        patience=config['patience'],
        save_path=model_save_path
    )
    print()
    
    # 步骤6: 模型评估
    print("步骤6: 模型评估")
    
    # 加载最佳模型
    checkpoint = torch.load(model_save_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 预测
    print("正在进行预测...")
    
    # 测试集预测
    y_test_pred = trainer.predict(X_test)
    y_test_true = processor.inverse_transform_y(y_test.numpy())
    y_test_pred = processor.inverse_transform_y(y_test_pred)
    
    # 验证集预测  
    y_val_pred = trainer.predict(X_val)
    y_val_true = processor.inverse_transform_y(y_val.numpy())
    y_val_pred = processor.inverse_transform_y(y_val_pred)
    
    # 训练集预测
    y_train_pred = trainer.predict(X_train)
    y_train_true = processor.inverse_transform_y(y_train.numpy())
    y_train_pred = processor.inverse_transform_y(y_train_pred)
    
    # 计算评估指标
    print("\n=== 改进模型性能评估 ===")
    
    train_metrics = calculate_metrics(y_train_true, y_train_pred)
    print("训练集性能:")
    for metric, value in train_metrics.items():
        print(f"  {metric}: {value:.6f}")
    
    val_metrics = calculate_metrics(y_val_true, y_val_pred)
    print("\n验证集性能:")
    for metric, value in val_metrics.items():
        print(f"  {metric}: {value:.6f}")
    
    test_metrics = calculate_metrics(y_test_true, y_test_pred)
    print("\n测试集性能:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.6f}")
    
    # 性能比较
    print("\n=== 性能对比 (vs 原模型) ===")
    print("原模型测试集性能:")
    print("  R²: -14.083278")
    print("  Correlation: -0.248674") 
    print("  RMSE: 5.406580")
    
    print(f"\n改进模型测试集性能:")
    print(f"  R²: {test_metrics['R²']:.6f}")
    print(f"  Correlation: {test_metrics['Correlation']:.6f}")
    print(f"  RMSE: {test_metrics['RMSE']:.6f}")
    
    # 改进程度
    if test_metrics['R²'] > -14.083278:
        print(f"\n✅ R²改进: {test_metrics['R²'] - (-14.083278):.6f}")
    if abs(test_metrics['Correlation']) > 0.248674:
        print(f"✅ 相关性改进: {abs(test_metrics['Correlation']) - 0.248674:.6f}")
    if test_metrics['RMSE'] < 5.406580:
        print(f"✅ RMSE改进: {5.406580 - test_metrics['RMSE']:.6f}")
    
    # 步骤7: 可视化结果
    print("\n步骤7: 保存预测结果")
    
    # 测试集结果
    plot_predictions_simple(y_test_true, y_test_pred, dates_test, 
                           "Improved Model - Test Set Predictions",
                           os.path.join(output_dir, 'improved_test_predictions.png'))
    
    print(f"\n所有结果已保存到: {output_dir}")
    print("=== 改进模型训练完成! ===")
    
    return model, processor, trainer, test_metrics

if __name__ == "__main__":
    # 运行改进的训练
    print("🚀 开始训练改进的LSTM模型...")
    model, processor, trainer, metrics = main_improved()
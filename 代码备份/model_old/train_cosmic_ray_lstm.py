import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import sys

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 添加项目路径
sys.path.append('/home/phil/Files/lstmPaper_v250618/model')

from data_processor import CosmicRayDataProcessor, split_time_series
from lstm_model import CosmicRayLSTM, ModelTrainer, calculate_metrics

def create_data_loaders(X_train, y_train, X_val, y_val, batch_size=32):
    """
    创建数据加载器
    
    Args:
        X_train, y_train: 训练数据
        X_val, y_val: 验证数据
        batch_size: 批大小
        
    Returns:
        训练和验证数据加载器
    """
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def plot_predictions(y_true, y_pred, dates, title="Prediction Results", save_path=None):
    """
    绘制预测结果对比图
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        dates: 日期列表
        title: 图标题
        save_path: 保存路径
    """
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

def main():
    """
    主训练和评估流程
    """
    print("=== 宇宙线通量LSTM预测模型 ===")
    print("初学者友好版本\n")
    
    # 数据路径
    solar_data_path = '/home/phil/Files/lstmPaper_v250618/data/solar_physics_data_1980_extend_smoothed.csv'
    cosmic_ray_path = '/home/phil/Files/lstmPaper_v250618/data/raw_data/ams/helium.csv'
    
    # 创建输出目录
    output_dir = '/home/phil/Files/lstmPaper_v250618/model/results'
    os.makedirs(output_dir, exist_ok=True)
    
    # 超参数设置
    config = {
        'sequence_length': 365,  # 输入序列长度（天）
        'hidden_size': 128,      # LSTM隐藏层大小
        'num_layers': 2,         # LSTM层数
        'dropout': 0.2,          # Dropout比例
        'batch_size': 32,        # 批大小
        'epochs': 100,           # 训练轮数
        'learning_rate': 0.001,  # 学习率
        'patience': 15,          # 早停耐心值
    }
    
    print("配置参数:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # 步骤1: 数据处理
    print("步骤1: 数据处理")
    processor = CosmicRayDataProcessor(
        solar_data_path=solar_data_path,
        cosmic_ray_path=cosmic_ray_path,
        sequence_length=config['sequence_length'],
        target_rigidity=2.97
    )
    
    # 准备数据
    X, y, dates = processor.prepare_data()
    print(f"总共准备了 {len(X)} 个样本\n")
    
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
    print(f"训练批次数: {len(train_loader)}")
    print(f"验证批次数: {len(val_loader)}\n")
    
    # 步骤4: 创建模型
    print("步骤4: 创建LSTM模型")
    input_size = X.shape[2]  # 特征数量（5个太阳参数）
    print(f"输入特征数: {input_size}")
    
    model = CosmicRayLSTM(
        input_size=input_size,
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    )
    
    # 检查GPU可用性
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建训练器
    trainer = ModelTrainer(model, device=device)
    print()
    
    # 步骤5: 模型训练
    print("步骤5: 开始训练模型")
    model_save_path = os.path.join(output_dir, 'best_model.pth')
    
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config['epochs'],
        learning_rate=config['learning_rate'],
        patience=config['patience'],
        save_path=model_save_path
    )
    print()
    
    # 步骤6: 绘制训练历史
    print("步骤6: 保存训练历史")
    history_plot_path = os.path.join(output_dir, 'training_history.png')
    trainer.plot_training_history(save_path=history_plot_path)
    print()
    
    # 步骤7: 模型评估
    print("步骤7: 模型评估")
    
    # 加载最佳模型
    checkpoint = torch.load(model_save_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 在各数据集上进行预测
    print("正在进行预测...")
    
    # 训练集预测
    y_train_pred = trainer.predict(X_train)
    y_train_true = processor.inverse_transform_y(y_train.numpy())
    y_train_pred = processor.inverse_transform_y(y_train_pred)
    
    # 验证集预测
    y_val_pred = trainer.predict(X_val)
    y_val_true = processor.inverse_transform_y(y_val.numpy())
    y_val_pred = processor.inverse_transform_y(y_val_pred)
    
    # 测试集预测
    y_test_pred = trainer.predict(X_test)
    y_test_true = processor.inverse_transform_y(y_test.numpy())
    y_test_pred = processor.inverse_transform_y(y_test_pred)
    
    # 计算评估指标
    print("\n=== 模型性能评估 ===")
    
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
    
    # 绘制预测结果
    print("\n步骤8: 可视化预测结果")
    
    # 训练集结果
    plot_predictions(y_train_true, y_train_pred, dates_train, 
                    "Training Set Predictions", 
                    os.path.join(output_dir, 'train_predictions.png'))
    
    # 验证集结果
    plot_predictions(y_val_true, y_val_pred, dates_val, 
                    "Validation Set Predictions",
                    os.path.join(output_dir, 'val_predictions.png'))
    
    # 测试集结果
    plot_predictions(y_test_true, y_test_pred, dates_test, 
                    "Test Set Predictions",
                    os.path.join(output_dir, 'test_predictions.png'))
    
    print(f"\n所有结果已保存到: {output_dir}")
    print("=== 训练完成! ===")
    
    return model, processor, trainer, test_metrics

if __name__ == "__main__":
    # 运行主程序
    model, processor, trainer, metrics = main()
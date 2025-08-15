import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from datetime import datetime, timedelta
from pathlib import Path

from data_processor import load_and_check_data, create_sequences, normalize_data
from models import CosmicRayDataset, lstm_model, train_model
from predictions import create_prediction_dates, predict_cosmic_ray_extended
from visualization import plot_comprehensive_results
from evaluation import calculate_metrics_and_stats, save_complete_results, save_metrics_to_txt

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)


def main():
    """主函数"""
    print("=== 完整LSTM宇宙线预测模型（包含训练和扩展预测）===\n")
    
    # 1. 加载数据
    solar_data, cosmic_data = load_and_check_data()

    # 2. 创建序列, 大写 X 表示“特征矩阵”(是二维数组)
    X, y, dates = create_sequences(solar_data, cosmic_data, sequence_length=365)

    if len(X) == 0:
        print("错误: 没有成功创建任何训练样例！")
        return
    
    # 3. 划分数据集 (时间顺序)
    train_ratio = 0.85
    split_idx = int(len(X) * train_ratio)
    
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    dates_train, dates_test = dates[:split_idx], dates[split_idx:]
    
    print(f"\n数据划分:")
    print(f"  训练集: {len(X_train)} 样例")
    print(f"  测试集: {len(X_test)} 样例")
    print(f"  训练时间范围: {dates_train[0]} 到 {dates_train[-1]}")
    print(f"  测试时间范围: {dates_test[0]} 到 {dates_test[-1]}")
    
    # 4. 归一化
    X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_X, scaler_y = normalize_data(
        X_train, X_test, y_train, y_test)
    
    # 5. 创建数据加载器
    train_dataset = CosmicRayDataset(X_train_scaled, y_train_scaled)
    test_dataset = CosmicRayDataset(X_test_scaled, y_test_scaled)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # 6. 创建LSTM模型
    input_size = 7  # 7个特征（6太阳参数+1宇宙线流强）
    hidden_size = 64  # 适中的隐藏层大小
    num_layers = 2
    output_size = 1
    
    model = lstm_model(input_size, hidden_size, num_layers, output_size, dropout=0.05)
    
    print(f"\n模型配置:")
    print('model is : ', model)
    print(f"  输入特征数: {input_size}")
    print(f"  隐藏层大小: {hidden_size}")
    print(f"  LSTM层数: {num_layers}")
    print(f"  总参数数: {sum(p.numel() for p in model.parameters()):,}")
    # save model summary to a text file
    with open('model_summary.txt', 'w') as f:
        f.write(str(model))
        f.write(f"\n  输入特征数: {input_size}\n")
        f.write(f"  隐藏层大小: {hidden_size}\n")
        f.write(f"  LSTM层数: {num_layers}\n")
        f.write(f"  总参数数: {sum(p.numel() for p in model.parameters()):,}\n")
    print(f"\n模型结构已保存到: {Path('model_summary.txt').resolve()}\n")
    
    # 7. 训练模型
    train_losses, val_losses = train_model(model, train_loader, test_loader, num_epochs=1000)

    # 7.1 保存训练历史，便于独立绘图
    history_df = pd.DataFrame({
        'epoch': np.arange(1, len(train_losses)+1),
        'train_loss': train_losses,
        'val_loss': val_losses
    })
    history_out = Path('training_history.csv').resolve()
    history_df.to_csv(history_out, index=False)
    print(f"训练历史已保存到: {history_out}")

    # 8. 评估测试集
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    test_predictions = []
    test_actuals = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            test_predictions.extend(outputs.cpu().numpy())
            test_actuals.extend(y_batch.numpy())
    # 反归一化
    test_predictions = scaler_y.inverse_transform(np.array(test_predictions).reshape(-1, 1)).flatten()
    test_actuals = scaler_y.inverse_transform(np.array(test_actuals).reshape(-1, 1)).flatten()
    # 8.5 计算训练集预测和反归一化（新增）
    train_predictions = []
    train_actuals = []
    with torch.no_grad():
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            train_predictions.extend(outputs.cpu().numpy())
            train_actuals.extend(y_batch.numpy())
    train_predictions = scaler_y.inverse_transform(np.array(train_predictions).reshape(-1, 1)).flatten()
    train_actuals = scaler_y.inverse_transform(np.array(train_actuals).reshape(-1, 1)).flatten()
    # 保存主要性能指标到txt文件
    save_metrics_to_txt(train_actuals, train_predictions, test_actuals, test_predictions, filename='model_metrics.txt')
    print("\n模型主要性能指标已自动保存到 'model_metrics.txt'\n")
    # 9. 扩展预测（2011-2025）
    print(f"\n=== 开始扩展预测 ===")
    start_date = datetime(2011, 5, 20)
    end_date = datetime(2025, 12, 31)
    prediction_dates = create_prediction_dates(start_date, end_date)
    
    print(f"将预测从 {start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')}")
    print(f"总共 {len(prediction_dates)} 个日期")
    
    extended_pred_dates, extended_predictions = predict_cosmic_ray_extended(
        model, solar_data, cosmic_data, prediction_dates, scaler_X, scaler_y
    )
    
    # 10. 计算详细指标
    calculate_metrics_and_stats(cosmic_data, test_actuals, test_predictions, 
                               extended_pred_dates, extended_predictions)
    
    # 11. 绘图已独立为脚本
    print("绘图已拆分为独立脚本: model3_osf/plot_after_training.py（无需重新训练）")

    # 12. 保存结果
    save_complete_results(dates_test, test_actuals, test_predictions,
                         extended_pred_dates, extended_predictions)

    # 13. 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'model_config': {
            'input_size': input_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'output_size': output_size
        }
    }, str(Path('complete_lstm_model.pth').resolve()))
    
    print("\n=== 完整LSTM模型训练和预测完成！===")
    print(f"模型已保存为: {Path('complete_lstm_model.pth').resolve()}")

if __name__ == "__main__":
    main()
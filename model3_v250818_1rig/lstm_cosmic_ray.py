import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datetime import datetime, timedelta
from pathlib import Path

from data_processor import load_and_check_data, create_sequences, normalize_data, SOLAR_PARAMETERS, HELIUM_FLUX_COL

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)


class CosmicRayDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class lstm_model(nn.Module):
    """
    lstm_model 是一个用于时间序列回归预测的神经网络模型，核心结构如下：

    1. LSTM层：{num_layers}层堆叠，每层{hidden_size}个隐藏单元
    2. 全连接层：将LSTM对365天历史数据的最终抽象表示映射到宇宙线强度预测值
    - LSTM的最后一个隐藏状态包含了对整个时间序列的压缩表示
    - 先降维到{hidden_size//2}，经过激活函数和Dropout防止过拟合，最后输出1个预测值

    输入：x，形状为(batch_size, 365, 7) - 365天的7个参数（6个太阳参数+1个宇宙线参数）
    输出：预测值，形状为(batch_size, 1) - 宇宙线强度
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.05):
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=dropout)
        
        # 对LSTM输出的最后一天隐藏状态做归一化，
        # 虽然输入数据已经做了归一化，但LSTM输出的隐藏状态在训练过程中分布可能会发生变化，
        # LayerNorm可以动态调整，进一步提升模型鲁棒性。
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # fc:全连接层(Fully Connected layer), 
        # Sequential是一个容器，
        # 用来把多个神经网络层（比如线性层、激活函数、Dropout等）按顺序组合在一起，
        # 形成一个“流水线”式的网络结构
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2), # 第一个线性层：降维
            # nn.BatchNorm1d(hidden_size // 2), # 对全连接层的输出做归一化(按需使用)
            # nn.LayerNorm(hidden_size//2), # 对全连接层的输出做归一化(按需使用)
            nn.ReLU(), # 先用最常用的ReLU，等后续需要调优时再尝试其他激活函数
            # nn.LeakyReLU(0.01),  # 或者 nn.GELU()等别的激活函数
            nn.Dropout(dropout),  # 防止过拟合
            nn.Linear(hidden_size // 2, output_size) # 第二个线性层：输出一个数（预测值）
        )
    
    # 定义数据如何流动
    # 训练和预测时，PyTorch会自动调用 forward，
    # 只需要定义好它，模型就知道怎么处理输入数据了。
    def forward(self, x):
        # x = self.input_norm(x) # x:(batch, seq_len, feat)
        lstm_out, _ = self.lstm(x) # 1. 先经过LSTM层
        h = lstm_out[:, -1, :] # 2. 取最后一天的隐藏状态
        h = self.layer_norm(h) # 3. 做归一化
        output = self.fc(h) # 4. 经过全连接层，输出预测值
        return output


def train_model(model, train_loader, val_loader, num_epochs=1000):
    """
    使用MSE损失函数和Adam优化器
    支持学习率调度（ReduceLROnPlateau）和早停机制（Early Stopping）
    梯度裁剪防止梯度爆炸
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    ################# 可调超参数 ########################################################################################
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0) 
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-6
    # )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-4,
        steps_per_epoch=len(train_loader), epochs=num_epochs
    )

    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_path = Path('best_corrected_model.pth').resolve()
    
    print(f"开始训练模型，使用设备: {device}")
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
            
            # 梯度裁剪防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            # OneCycleLR 需要按batch更新
            scheduler.step()
            train_loss += loss.item()
        
        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs.squeeze(), y_batch)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # 移除对val_loss的scheduler.step调用（OneCycleLR不需要）
        
        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), str(best_path))
            print(f"保存当前最佳模型到: {best_path}")
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        
        if patience_counter >= 30:  # 增加早停patience
            print(f"早停在第 {epoch+1} 轮")
            break
    
    # 加载最佳模型
    model.load_state_dict(torch.load(str(best_path)))
    
    return train_losses, val_losses


def create_prediction_dates(start_date, end_date):
    """创建预测日期序列（返回pandas Timestamp列表，确保与数据键类型一致）"""
    return list(pd.date_range(start=pd.to_datetime(start_date), end=pd.to_datetime(end_date), freq='D'))


#############################################################
######--------------->>> 递归预测 <<<------------------######
#############################################################
def predict_cosmic_ray_extended(model, solar_data, cosmic_data, prediction_dates, scaler_X, scaler_y, sequence_length=365):
    """
    递归式长期预测：
    - 观测期内，输入序列氦通量用真实数据
    - 观测期外，输入序列氦通量用已预测值递归推进
    """
    print(f"\n=== 扩展预测（递归式）：{len(prediction_dates)} 个日期 ===")
    features = SOLAR_PARAMETERS + [HELIUM_FLUX_COL]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    predictions = []
    valid_dates = []
    # cosmic_data按日期排序，方便查找
    cosmic_data = cosmic_data.sort_values('date YYYY-MM-DD').reset_index(drop=True)
    # 构造氦通量时间序列（初始为观测数据）
    helium_flux_series = cosmic_data[['date YYYY-MM-DD', HELIUM_FLUX_COL]].copy()
    helium_flux_dict = dict(zip(helium_flux_series['date YYYY-MM-DD'], helium_flux_series[HELIUM_FLUX_COL]))
    # 递归预测
    for i, target_date in enumerate(prediction_dates):
        if (i + 1) % 1000 == 0:
            print(f"已处理 {i + 1}/{len(prediction_dates)} 个日期")
        # 计算需要的日期范围
        start_date = target_date - timedelta(days=sequence_length)
        end_date = target_date - timedelta(days=1)
        expected_dates = pd.date_range(start=start_date, end=end_date, freq='D')
        input_rows = []
        for date_i in expected_dates:
            solar_mask = solar_data['date'] == date_i
            # 氦通量优先用观测值，否则用已预测值
            if date_i in helium_flux_dict:
                helium_flux = helium_flux_dict[date_i]
            else:
                # 预测值还没生成，跳过
                helium_flux = None
            if solar_mask.sum() == 1 and helium_flux is not None:
                solar_row = solar_data[solar_mask][SOLAR_PARAMETERS].iloc[0].values
                input_rows.append(np.concatenate([solar_row, [helium_flux]]))
            else:
                break
        if len(input_rows) == sequence_length:
            solar_sequence_flat = np.array(input_rows).reshape(-1, len(features))
            solar_sequence_scaled_flat = scaler_X.transform(solar_sequence_flat)
            solar_sequence_scaled = solar_sequence_scaled_flat.reshape(1, sequence_length, len(features))
            X_tensor = torch.FloatTensor(solar_sequence_scaled).to(device)
            with torch.no_grad():
                pred_scaled = model(X_tensor).cpu().numpy()
                pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()[0]
            predictions.append(pred)
            valid_dates.append(target_date)
            # 如果该日期没有观测值，则将预测值加入序列，供后续递归
            if target_date not in helium_flux_dict:
                helium_flux_dict[target_date] = pred
    print(f"成功预测了 {len(predictions)} 个数据点（递归式扩展）")
    return valid_dates, predictions


def calculate_metrics_and_stats(cosmic_data, test_actuals, test_predictions, 
                               extended_pred_dates=None, extended_predictions=None):
    """计算详细的评估指标和统计信息"""
    print(f"\n=== 详细评估结果 ===")
    
    # 测试集指标
    mse = mean_squared_error(test_actuals, test_predictions)
    mae = mean_absolute_error(test_actuals, test_predictions)
    r2 = r2_score(test_actuals, test_predictions)
    mape = np.mean(np.abs((test_actuals - test_predictions) / test_actuals)) * 100
    mre = np.mean((test_actuals - test_predictions) / test_actuals) * 100
    
    print(f"测试集性能:")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  R²: {r2:.6f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  MRE: {mre:.2f}%")
    
    # 如果有扩展预测，计算重叠期间的指标
    if extended_pred_dates is not None and extended_predictions is not None:
        obs_dates = cosmic_data['date YYYY-MM-DD']
        obs_flux = cosmic_data[HELIUM_FLUX_COL]
        
        # 找到重叠的日期
        aligned_obs = []
        aligned_pred = []
        
        for i, pred_date in enumerate(extended_pred_dates):
            matching_obs = cosmic_data[cosmic_data['date YYYY-MM-DD'] == pred_date]
            if not matching_obs.empty:
                aligned_obs.append(matching_obs[HELIUM_FLUX_COL].iloc[0])
                aligned_pred.append(extended_predictions[i])
        
        if len(aligned_obs) > 0:
            overlap_mse = mean_squared_error(aligned_obs, aligned_pred)
            overlap_mae = mean_absolute_error(aligned_obs, aligned_pred)
            overlap_r2 = r2_score(aligned_obs, aligned_pred)
            overlap_mape = np.mean(np.abs((np.array(aligned_obs) - np.array(aligned_pred)) / np.array(aligned_obs))) * 100
            overlap_mre = np.mean((np.array(aligned_obs) - np.array(aligned_pred)) / np.array(aligned_obs)) * 100
            
            print(f"\n完整预测重叠期间性能:")
            print(f"  对比数据点数: {len(aligned_obs)}")
            print(f"  MSE: {overlap_mse:.6f}")
            print(f"  MAE: {overlap_mae:.6f}")
            print(f"  R²: {overlap_r2:.6f}")
            print(f"  MAPE: {overlap_mape:.2f}%")
            print(f"  MRE: {overlap_mre:.2f}%")
        
        # 统计信息
        print(f"\n=== 预测统计信息 ===")
        print(f"观测数据统计:")
        print(f"  平均值: {obs_flux.mean():.2f}")
        print(f"  标准差: {obs_flux.std():.2f}")
        print(f"  最小值: {obs_flux.min():.2f}")
        print(f"  最大值: {obs_flux.max():.2f}")
        
        print(f"\n扩展预测统计:")
        print(f"  预测数据点数: {len(extended_predictions)}")
        print(f"  平均值: {np.mean(extended_predictions):.2f}")
        print(f"  标准差: {np.std(extended_predictions):.2f}")
        print(f"  最小值: {np.min(extended_predictions):.2f}")
        print(f"  最大值: {np.max(extended_predictions):.2f}")
        
        # 未来预测统计
        future_start = obs_dates.iloc[-1]
        future_predictions = [p for i, p in enumerate(extended_predictions) 
                            if extended_pred_dates[i] > future_start]
        
        if len(future_predictions) > 0:
            print(f"\n未来预测统计 (2019年后):")
            print(f"  未来预测点数: {len(future_predictions)}")
            print(f"  平均值: {np.mean(future_predictions):.2f}")
            print(f"  标准差: {np.std(future_predictions):.2f}")
            print(f"  最小值: {np.min(future_predictions):.2f}")
            print(f"  最大值: {np.max(future_predictions):.2f}")


def save_complete_results(test_dates, test_actuals, test_predictions, 
                         extended_pred_dates=None, extended_predictions=None):
    """保存完整的预测结果"""
    test_results = pd.DataFrame({
        'date': test_dates,
        'actual_flux': test_actuals,
        'predicted_flux': test_predictions,
        'absolute_error': np.abs(test_actuals - test_predictions),
        'relative_error': np.abs(test_actuals - test_predictions) / test_actuals * 100
    })
    test_out = Path('test_set_predictions.csv').resolve()
    test_results.to_csv(test_out, index=False)
    print(f"\n测试集结果已保存到: {test_out}")

    if extended_pred_dates is not None and extended_predictions is not None:
        extended_results = pd.DataFrame({
            'date': extended_pred_dates,
            'predicted_flux': extended_predictions
        })
        ext_out = Path('cosmic_ray_predictions_extended.csv').resolve()
        extended_results.to_csv(ext_out, index=False)
        print(f"扩展预测结果已保存到: {ext_out}")


def save_metrics_to_txt(train_actuals, train_predictions, test_actuals, test_predictions, filename='model_metrics.txt'):
    """保存训练集和测试集的主要性能指标到txt文件"""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import numpy as np
    
    # 训练集指标
    train_mse = mean_squared_error(train_actuals, train_predictions)
    train_mae = mean_absolute_error(train_actuals, train_predictions)
    train_r2 = r2_score(train_actuals, train_predictions)
    train_mre = np.mean((train_actuals - train_predictions) / train_actuals) * 100
    train_mape = np.mean(np.abs((train_actuals - train_predictions) / train_actuals)) * 100
    
    # 测试集指标
    test_mse = mean_squared_error(test_actuals, test_predictions)
    test_mae = mean_absolute_error(test_actuals, test_predictions)
    test_r2 = r2_score(test_actuals, test_predictions)
    test_mre = np.mean((test_actuals - test_predictions) / test_actuals) * 100
    test_mape = np.mean(np.abs((test_actuals - test_predictions) / test_actuals)) * 100
    
    out_path = Path(filename).resolve()
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('模型性能指标\n')
        f.write('--------------------------\n')
        f.write(f'训练集样本数: {len(train_actuals)}\n')
        f.write(f'训练集 MSE: {train_mse:.6f}\n')
        f.write(f'训练集 MAE: {train_mae:.6f}\n')
        f.write(f'训练集 R²: {train_r2:.6f}\n')
        f.write(f'训练集 MAPE: {train_mape:.2f}%\n')
        f.write(f'训练集 MRE: {train_mre:.2f}%\n')
        f.write('--------------------------\n')
        f.write(f'测试集样本数: {len(test_actuals)}\n')
        f.write(f'测试集 MSE: {test_mse:.6f}\n')
        f.write(f'测试集 MAE: {test_mae:.6f}\n')
        f.write(f'测试集 R²: {test_r2:.6f}\n')
        f.write(f'测试集 MAPE: {test_mape:.2f}%\n')
        f.write(f'测试集 MRE: {test_mre:.2f}%\n')
        f.write('--------------------------\n')
        f.write('说明: MRE为平均相对误差(可为负)，MAPE为平均绝对百分比误差。\n')
        f.write('所有指标均为自动保存，无需手动复制。\n')
    print(f"模型性能指标已保存到: {out_path}")


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
    input_size = len(SOLAR_PARAMETERS) + 1  # 动态特征数（太阳参数 + 氦通量）
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
    # 9. 扩展预测（动态至太阳数据末日）
    print(f"\n=== 开始扩展预测 ===")
    start_date = pd.to_datetime(cosmic_data['date YYYY-MM-DD'].min())
    end_date = pd.to_datetime(solar_data['date'].max())
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
    print("绘图已拆分为独立脚本: plot_after_training.py（无需重新训练）")

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
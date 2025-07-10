import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

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

class CorrectedLSTMModel(nn.Module):
    """修正版LSTM模型 - 简单有效的架构"""
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(CorrectedLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
    
    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])
        return output

def debug_data_alignment():
    """调试数据对齐问题"""
    print("=== 数据对齐调试分析 ===\n")
    
    # 加载数据
    solar_data = pd.read_csv('/home/phil/Files/lstmPaper_v250618/data/solar_physics_data_1980_extend_smoothed.csv')
    solar_data['date'] = pd.to_datetime(solar_data['date'])
    solar_data = solar_data.sort_values('date').reset_index(drop=True)
    
    cosmic_data = pd.read_csv('/home/phil/Files/lstmPaper_v250618/data/raw_data/ams/helium.csv')
    cosmic_data['date YYYY-MM-DD'] = pd.to_datetime(cosmic_data['date YYYY-MM-DD'])
    cosmic_data = cosmic_data[cosmic_data['rigidity_min GV'] == 2.97].copy()
    cosmic_data = cosmic_data.sort_values('date YYYY-MM-DD').reset_index(drop=True)
    
    print(f"太阳数据范围: {solar_data['date'].min()} 到 {solar_data['date'].max()}")
    print(f"宇宙线数据范围: {cosmic_data['date YYYY-MM-DD'].min()} 到 {cosmic_data['date YYYY-MM-DD'].max()}")
    
    # 检查数据频率
    solar_dates = pd.to_datetime(solar_data['date'])
    cosmic_dates = pd.to_datetime(cosmic_data['date YYYY-MM-DD'])
    
    print(f"\n太阳数据总点数: {len(solar_data)}")
    print(f"宇宙线数据总点数: {len(cosmic_data)}")
    
    # 检查是否每天都有数据
    solar_date_range = pd.date_range(start=solar_dates.min(), end=solar_dates.max(), freq='D')
    cosmic_date_range = pd.date_range(start=cosmic_dates.min(), end=cosmic_dates.max(), freq='D')
    
    missing_solar_dates = set(solar_date_range) - set(solar_dates)
    missing_cosmic_dates = set(cosmic_date_range) - set(cosmic_dates)
    
    print(f"\n太阳数据缺失天数: {len(missing_solar_dates)}")
    print(f"宇宙线数据缺失天数: {len(missing_cosmic_dates)}")
    
    if len(missing_solar_dates) > 0:
        print(f"太阳数据缺失的前5个日期: {sorted(list(missing_solar_dates))[:5]}")
    if len(missing_cosmic_dates) > 0:
        print(f"宇宙线数据缺失的前5个日期: {sorted(list(missing_cosmic_dates))[:5]}")
    
    # 检查重复日期
    solar_duplicates = solar_data[solar_data.duplicated('date', keep=False)]
    cosmic_duplicates = cosmic_data[cosmic_data.duplicated('date YYYY-MM-DD', keep=False)]
    
    print(f"\n太阳数据重复日期数: {len(solar_duplicates)}")
    print(f"宇宙线数据重复日期数: {len(cosmic_duplicates)}")
    
    return solar_data, cosmic_data

def load_and_check_data():
    """仔细加载和检查数据"""
    print("=== 仔细加载和检查数据 ===\n")
    
    # 加载太阳参数数据
    solar_data = pd.read_csv('/home/phil/Files/lstmPaper_v250618/data/solar_physics_data_1980_extend_smoothed.csv')
    solar_data['date'] = pd.to_datetime(solar_data['date'])
    solar_data = solar_data.sort_values('date').reset_index(drop=True)
    
    # 去除重复日期，保留第一个
    solar_data = solar_data.drop_duplicates('date', keep='first').reset_index(drop=True)
    
    # 加载宇宙线数据
    cosmic_data = pd.read_csv('/home/phil/Files/lstmPaper_v250618/data/raw_data/ams/helium.csv')
    cosmic_data['date YYYY-MM-DD'] = pd.to_datetime(cosmic_data['date YYYY-MM-DD'])
    cosmic_data = cosmic_data[cosmic_data['rigidity_min GV'] == 2.97].copy()
    cosmic_data = cosmic_data.sort_values('date YYYY-MM-DD').reset_index(drop=True)
    
    # 去除重复日期，保留第一个
    cosmic_data = cosmic_data.drop_duplicates('date YYYY-MM-DD', keep='first').reset_index(drop=True)
    
    print(f"太阳数据: {len(solar_data)} 天 ({solar_data['date'].min()} - {solar_data['date'].max()})")
    print(f"宇宙线数据: {len(cosmic_data)} 天 ({cosmic_data['date YYYY-MM-DD'].min()} - {cosmic_data['date YYYY-MM-DD'].max()})")
    
    # 检查数据完整性
    solar_date_range = pd.date_range(start=solar_data['date'].min(), end=solar_data['date'].max(), freq='D')
    actual_solar_dates = set(solar_data['date'])
    missing_solar = set(solar_date_range) - actual_solar_dates
    
    cosmic_date_range = pd.date_range(start=cosmic_data['date YYYY-MM-DD'].min(), end=cosmic_data['date YYYY-MM-DD'].max(), freq='D')
    actual_cosmic_dates = set(cosmic_data['date YYYY-MM-DD'])
    missing_cosmic = set(cosmic_date_range) - actual_cosmic_dates
    
    print(f"太阳数据缺失: {len(missing_solar)} 天")
    print(f"宇宙线数据缺失: {len(missing_cosmic)} 天")
    
    return solar_data, cosmic_data

def create_sequences_carefully(solar_data, cosmic_data, sequence_length=365):
    """非常仔细地创建序列，确保对齐正确"""
    print(f"\n=== 仔细创建 {sequence_length} 天序列 ===")
    
    solar_features = ['HMF', 'wind_speed', 'HCS_tilt', 'polarity', 'SSN']
    
    X = []
    y = []
    dates = []
    successful_alignments = 0
    failed_alignments = 0
    
    for idx, row in cosmic_data.iterrows():
        target_date = row['date YYYY-MM-DD']
        target_flux = row['helium_flux m^-2sr^-1s^-1GV^-1']
        
        # 计算需要的365天: [target_date-365, target_date-364, ..., target_date-1]
        start_date = target_date - timedelta(days=sequence_length)  # 365天前
        end_date = target_date - timedelta(days=1)  # 前一天
        
        # 生成期望的日期序列
        expected_dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # 提取对应的太阳数据
        solar_sequence = []
        dates_found = []
        
        for expected_date in expected_dates:
            # 在太阳数据中查找这个日期
            mask = solar_data['date'] == expected_date
            if mask.sum() == 1:  # 找到唯一匹配
                solar_row = solar_data[mask][solar_features].iloc[0].values
                solar_sequence.append(solar_row)
                dates_found.append(expected_date)
            elif mask.sum() > 1:  # 重复日期
                if idx < 5:  # 只对前5个样例打印调试信息
                    print(f"警告: 日期 {expected_date} 在太阳数据中重复")
                solar_row = solar_data[mask][solar_features].iloc[0].values
                solar_sequence.append(solar_row)
                dates_found.append(expected_date)
            else:  # 未找到
                break
        
        # 检查是否获得了完整的365天数据
        if len(solar_sequence) == sequence_length:
            X.append(np.array(solar_sequence))
            y.append(target_flux)
            dates.append(target_date)
            successful_alignments += 1
            
            # 对前几个样例进行详细检查
            if idx < 3:
                print(f"\n样例 {idx+1}:")
                print(f"  目标日期: {target_date}")
                print(f"  目标通量: {target_flux:.2f}")
                print(f"  太阳数据范围: {dates_found[0]} 到 {dates_found[-1]}")
                print(f"  太阳数据形状: {np.array(solar_sequence).shape}")
        else:
            failed_alignments += 1
            if failed_alignments < 5:  # 只打印前5个失败样例
                print(f"失败样例 {failed_alignments}: 只找到 {len(solar_sequence)} 天数据，需要 {sequence_length} 天")
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"\n=== 序列创建结果 ===")
    print(f"成功对齐: {successful_alignments} 个样例")
    print(f"失败对齐: {failed_alignments} 个样例")
    print(f"最终数据形状:")
    print(f"  X: {X.shape} (样例数, 时间步数, 特征数)")
    print(f"  y: {y.shape}")
    print(f"  特征顺序: {solar_features}")
    
    # 数据质量检查
    print(f"\n=== 数据质量检查 ===")
    print(f"X 中的 NaN 数量: {np.isnan(X).sum()}")
    print(f"y 中的 NaN 数量: {np.isnan(y).sum()}")
    
    if X.shape[0] > 0:
        print(f"\nX 统计 (所有特征):")
        for i, feature in enumerate(solar_features):
            feature_data = X[:, :, i].flatten()
            print(f"  {feature}: 均值={np.mean(feature_data):.4f}, 标准差={np.std(feature_data):.4f}, 范围=[{np.min(feature_data):.2f}, {np.max(feature_data):.2f}]")
        
        print(f"\ny 统计:")
        print(f"  均值={np.mean(y):.4f}, 标准差={np.std(y):.4f}, 范围=[{np.min(y):.2f}, {np.max(y):.2f}]")
    
    return X, y, dates

def normalize_data_carefully(X_train, X_test, y_train, y_test):
    """仔细进行数据归一化"""
    print(f"\n=== 仔细进行数据归一化 ===")
    
    # 打印归一化前的统计
    print(f"归一化前:")
    print(f"  X_train 形状: {X_train.shape}")
    print(f"  y_train 统计: 均值={np.mean(y_train):.4f}, 标准差={np.std(y_train):.4f}")
    
    # 对输入数据进行归一化 (重塑为二维进行归一化)
    n_samples_train, n_timesteps, n_features = X_train.shape
    n_samples_test = X_test.shape[0]
    
    # 重塑为二维
    X_train_2d = X_train.reshape(-1, n_features)
    X_test_2d = X_test.reshape(-1, n_features)
    
    # 使用StandardScaler
    scaler_X = StandardScaler()
    X_train_scaled_2d = scaler_X.fit_transform(X_train_2d)
    X_test_scaled_2d = scaler_X.transform(X_test_2d)
    
    # 重塑回三维
    X_train_scaled = X_train_scaled_2d.reshape(n_samples_train, n_timesteps, n_features)
    X_test_scaled = X_test_scaled_2d.reshape(n_samples_test, n_timesteps, n_features)
    
    # 对输出数据进行归一化
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    
    # 打印归一化后的统计
    print(f"归一化后:")
    print(f"  X_train_scaled 统计: 均值={np.mean(X_train_scaled):.4f}, 标准差={np.std(X_train_scaled):.4f}")
    print(f"  y_train_scaled 统计: 均值={np.mean(y_train_scaled):.4f}, 标准差={np.std(y_train_scaled):.4f}")
    
    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_X, scaler_y

def train_corrected_model(model, train_loader, val_loader, num_epochs=100):
    """训练修正版模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5, verbose=True)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"开始训练修正模型，使用设备: {device}")
    
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
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
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
        
        scheduler.step(val_loss)
        
        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_corrected_model.pth')
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        
        if patience_counter >= 20:
            print(f"早停在第 {epoch+1} 轮")
            break
    
    # 加载最佳模型
    model.load_state_dict(torch.load('best_corrected_model.pth'))
    
    return train_losses, val_losses

def create_prediction_dates(start_date, end_date):
    """创建预测日期序列"""
    dates = []
    current_date = start_date
    while current_date <= end_date:
        dates.append(current_date)
        current_date += timedelta(days=1)
    return dates

def predict_cosmic_ray_extended(model, solar_data, prediction_dates, scaler_X, scaler_y, sequence_length=365):
    """预测宇宙线通量（包含训练期间和未来）"""
    print(f"\n=== 扩展预测：{len(prediction_dates)} 个日期 ===")
    
    solar_features = ['HMF', 'wind_speed', 'HCS_tilt', 'polarity', 'SSN']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    predictions = []
    valid_dates = []
    
    for i, target_date in enumerate(prediction_dates):
        if (i + 1) % 1000 == 0:
            print(f"已处理 {i + 1}/{len(prediction_dates)} 个日期")
            
        # 计算需要的太阳参数日期范围
        start_date = target_date - timedelta(days=sequence_length)
        end_date = target_date - timedelta(days=1)
        
        # 生成期望的日期序列
        expected_dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # 提取对应的太阳数据
        solar_sequence = []
        
        for expected_date in expected_dates:
            mask = solar_data['date'] == expected_date
            if mask.sum() == 1:
                solar_row = solar_data[mask][solar_features].iloc[0].values
                solar_sequence.append(solar_row)
            else:
                break
        
        # 确保有足够的数据点
        if len(solar_sequence) == sequence_length:
            # 标准化输入数据
            solar_sequence_flat = np.array(solar_sequence).reshape(-1, len(solar_features))
            solar_sequence_scaled_flat = scaler_X.transform(solar_sequence_flat)
            solar_sequence_scaled = solar_sequence_scaled_flat.reshape(1, sequence_length, len(solar_features))
            
            # 转换为tensor并预测
            X_tensor = torch.FloatTensor(solar_sequence_scaled).to(device)
            
            with torch.no_grad():
                pred_scaled = model(X_tensor).cpu().numpy()
                pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()[0]
            
            predictions.append(pred)
            valid_dates.append(target_date)
    
    print(f"成功预测了 {len(predictions)} 个数据点")
    return valid_dates, predictions

def plot_comprehensive_results(cosmic_data, train_losses, val_losses, test_predictions, test_actuals, test_dates, 
                              extended_pred_dates=None, extended_predictions=None):
    """绘制综合结果（训练结果 + 扩展预测）"""
    print("正在绘制综合结果图...")
    
    # 创建大图
    fig = plt.figure(figsize=(20, 15))
    
    # 子图1: 训练损失
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(train_losses, label='训练损失', color='blue', alpha=0.8)
    ax1.plot(val_losses, label='验证损失', color='red', alpha=0.8)
    ax1.set_xlabel('训练轮数')
    ax1.set_ylabel('损失值')
    ax1.set_title('训练过程')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 子图2: 测试集预测效果
    ax2 = plt.subplot(3, 3, 2)
    ax2.scatter(test_actuals, test_predictions, alpha=0.6, color='green')
    ax2.plot([test_actuals.min(), test_actuals.max()], [test_actuals.min(), test_actuals.max()], 'r--', lw=2)
    r2 = r2_score(test_actuals, test_predictions)
    ax2.set_xlabel('实际值')
    ax2.set_ylabel('预测值')
    ax2.set_title(f'测试集预测效果 (R²={r2:.4f})')
    ax2.grid(True, alpha=0.3)
    
    # 子图3: 测试集时间序列
    ax3 = plt.subplot(3, 3, 3)
    ax3.plot(test_dates, test_actuals, label='实际值', alpha=0.8, linewidth=2, color='blue')
    ax3.plot(test_dates, test_predictions, label='预测值', alpha=0.8, linewidth=2, color='red')
    ax3.set_xlabel('日期')
    ax3.set_ylabel('氦通量')
    ax3.set_title('测试集时间序列预测')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='x', rotation=45)
    
    # 子图4: 残差分析
    residuals = test_actuals - test_predictions
    ax4 = plt.subplot(3, 3, 4)
    ax4.scatter(test_predictions, residuals, alpha=0.6, color='purple')
    ax4.axhline(y=0, color='r', linestyle='--')
    ax4.set_xlabel('预测值')
    ax4.set_ylabel('残差')
    ax4.set_title('残差分析')
    ax4.grid(True, alpha=0.3)
    
    # 子图5: 残差分布
    ax5 = plt.subplot(3, 3, 5)
    ax5.hist(residuals, bins=20, alpha=0.7, color='orange')
    ax5.set_xlabel('残差')
    ax5.set_ylabel('频次')
    ax5.set_title('残差分布')
    ax5.grid(True, alpha=0.3)
    
    # 如果有扩展预测数据，绘制完整时间序列
    if extended_pred_dates is not None and extended_predictions is not None:
        # 子图6: 完整时间序列（2011-2032）
        ax6 = plt.subplot(3, 2, 3)
        obs_dates = cosmic_data['date YYYY-MM-DD']
        obs_flux = cosmic_data['helium_flux m^-2sr^-1s^-1GV^-1']
        
        ax6.plot(obs_dates, obs_flux, 'b-', label='实际观测数据', linewidth=1.5, alpha=0.8)
        ax6.plot(extended_pred_dates, extended_predictions, 'r-', label='LSTM预测数据', linewidth=1.5, alpha=0.8)
        ax6.axvline(x=obs_dates.iloc[-1], color='green', linestyle='--', alpha=0.7, label='观测数据结束')
        ax6.set_title('宇宙线氦通量预测：2011-2032年完整时间序列', fontsize=12, fontweight='bold')
        ax6.set_xlabel('日期')
        ax6.set_ylabel('氦通量 (m⁻²sr⁻¹s⁻¹GV⁻¹)')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        ax6.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax6.xaxis.set_major_locator(mdates.YearLocator())
        
        # 子图7: 未来预测部分（2019-2032）
        ax7 = plt.subplot(3, 2, 4)
        future_start = obs_dates.iloc[-1]
        future_mask = [d > future_start for d in extended_pred_dates]
        future_pred_dates = [d for i, d in enumerate(extended_pred_dates) if future_mask[i]]
        future_predictions = [p for i, p in enumerate(extended_predictions) if future_mask[i]]
        
        # 显示最后一段观测数据作为连接
        recent_obs = obs_dates.iloc[-100:]  # 最后100个观测点
        recent_flux = obs_flux.iloc[-100:]
        
        ax7.plot(recent_obs, recent_flux, 'b-', label='最近观测数据', linewidth=2, alpha=0.8)
        if future_pred_dates:
            ax7.plot(future_pred_dates, future_predictions, 'r-', 
                    label='未来预测', linewidth=2, alpha=0.8)
        ax7.axvline(x=obs_dates.iloc[-1], color='green', linestyle='--', alpha=0.7, label='预测起始点')
        ax7.set_title('未来宇宙线通量预测 (2019-2032)', fontsize=12, fontweight='bold')
        ax7.set_xlabel('日期')
        ax7.set_ylabel('氦通量 (m⁻²sr⁻¹s⁻¹GV⁻¹)')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        ax7.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax7.xaxis.set_major_locator(mdates.YearLocator())
        
        # 子图8: 预测统计对比
        ax8 = plt.subplot(3, 3, 8)
        
        # 观测期间的预测数据
        overlap_start = obs_dates.iloc[0]
        overlap_end = obs_dates.iloc[-1]
        overlap_mask = [(d >= overlap_start) and (d <= overlap_end) for d in extended_pred_dates]
        overlap_predictions = [p for i, p in enumerate(extended_predictions) if overlap_mask[i]]
        
        # 统计数据
        obs_stats = [obs_flux.mean(), obs_flux.std(), obs_flux.min(), obs_flux.max()]
        pred_stats = [np.mean(overlap_predictions), np.std(overlap_predictions), 
                     np.min(overlap_predictions), np.max(overlap_predictions)]
        future_stats = [np.mean(future_predictions), np.std(future_predictions),
                       np.min(future_predictions), np.max(future_predictions)]
        
        x_pos = np.arange(4)
        width = 0.25
        
        ax8.bar(x_pos - width, obs_stats, width, label='观测数据', alpha=0.7)
        ax8.bar(x_pos, pred_stats, width, label='预测数据(训练期)', alpha=0.7)
        ax8.bar(x_pos + width, future_stats, width, label='预测数据(未来)', alpha=0.7)
        
        ax8.set_xlabel('统计指标')
        ax8.set_ylabel('数值')
        ax8.set_title('数据统计对比')
        ax8.set_xticks(x_pos)
        ax8.set_xticklabels(['均值', '标准差', '最小值', '最大值'])
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        
        # 子图9: 误差随时间变化
        ax9 = plt.subplot(3, 3, 9)
        if len(overlap_predictions) > 0:
            # 计算重叠期间的误差
            aligned_obs = []
            aligned_pred = []
            aligned_dates = []
            
            for i, pred_date in enumerate(extended_pred_dates):
                if overlap_mask[i]:
                    matching_obs = cosmic_data[cosmic_data['date YYYY-MM-DD'] == pred_date]
                    if not matching_obs.empty:
                        aligned_obs.append(matching_obs['helium_flux m^-2sr^-1s^-1GV^-1'].iloc[0])
                        aligned_pred.append(extended_predictions[i])
                        aligned_dates.append(pred_date)
            
            if len(aligned_obs) > 0:
                abs_errors = np.abs(np.array(aligned_obs) - np.array(aligned_pred))
                ax9.plot(aligned_dates, abs_errors, alpha=0.7, color='red')
                ax9.set_xlabel('日期')
                ax9.set_ylabel('绝对误差')
                ax9.set_title('预测误差随时间变化')
                ax9.grid(True, alpha=0.3)
                ax9.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('完整LSTM预测结果.png', dpi=300, bbox_inches='tight')
    plt.show()

def calculate_metrics_and_stats(cosmic_data, test_actuals, test_predictions, 
                               extended_pred_dates=None, extended_predictions=None):
    """计算详细的评估指标和统计信息"""
    print(f"\n=== 详细评估结果 ===")
    
    # 测试集指标
    mse = mean_squared_error(test_actuals, test_predictions)
    mae = mean_absolute_error(test_actuals, test_predictions)
    r2 = r2_score(test_actuals, test_predictions)
    mape = np.mean(np.abs((test_actuals - test_predictions) / test_actuals)) * 100
    
    print(f"测试集性能:")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  R²: {r2:.6f}")
    print(f"  MAPE: {mape:.2f}%")
    
    # 如果有扩展预测，计算重叠期间的指标
    if extended_pred_dates is not None and extended_predictions is not None:
        obs_dates = cosmic_data['date YYYY-MM-DD']
        obs_flux = cosmic_data['helium_flux m^-2sr^-1s^-1GV^-1']
        
        # 找到重叠的日期
        aligned_obs = []
        aligned_pred = []
        
        for i, pred_date in enumerate(extended_pred_dates):
            matching_obs = cosmic_data[cosmic_data['date YYYY-MM-DD'] == pred_date]
            if not matching_obs.empty:
                aligned_obs.append(matching_obs['helium_flux m^-2sr^-1s^-1GV^-1'].iloc[0])
                aligned_pred.append(extended_predictions[i])
        
        if len(aligned_obs) > 0:
            overlap_mse = mean_squared_error(aligned_obs, aligned_pred)
            overlap_mae = mean_absolute_error(aligned_obs, aligned_pred)
            overlap_r2 = r2_score(aligned_obs, aligned_pred)
            overlap_mape = np.mean(np.abs((np.array(aligned_obs) - np.array(aligned_pred)) / np.array(aligned_obs))) * 100
            
            print(f"\n完整预测重叠期间性能:")
            print(f"  对比数据点数: {len(aligned_obs)}")
            print(f"  MSE: {overlap_mse:.6f}")
            print(f"  MAE: {overlap_mae:.6f}")
            print(f"  R²: {overlap_r2:.6f}")
            print(f"  MAPE: {overlap_mape:.2f}%")
        
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
    # 保存测试集结果
    test_results = pd.DataFrame({
        'date': test_dates,
        'actual_flux': test_actuals,
        'predicted_flux': test_predictions,
        'absolute_error': np.abs(test_actuals - test_predictions),
        'relative_error': np.abs(test_actuals - test_predictions) / test_actuals * 100
    })
    test_results.to_csv('测试集预测结果.csv', index=False)
    print(f"\n测试集结果已保存到 '测试集预测结果.csv'")
    
    # 保存扩展预测结果
    if extended_pred_dates is not None and extended_predictions is not None:
        extended_results = pd.DataFrame({
            'date': extended_pred_dates,
            'predicted_flux': extended_predictions
        })
        extended_results.to_csv('宇宙线预测结果_2011_2032.csv', index=False)
        print(f"扩展预测结果已保存到 '宇宙线预测结果_2011_2032.csv'")

def main():
    """主函数"""
    print("=== 完整LSTM宇宙线预测模型（包含训练和扩展预测）===\n")
    
    # 1. 数据调试（可选）
    print("正在进行数据调试检查...")
    debug_data_alignment()
    
    # 2. 仔细加载数据
    solar_data, cosmic_data = load_and_check_data()
    
    # 3. 仔细创建序列
    X, y, dates = create_sequences_carefully(solar_data, cosmic_data, sequence_length=365)
    
    if len(X) == 0:
        print("错误: 没有成功创建任何训练样例！")
        return
    
    # 4. 划分数据集 (时间顺序)
    train_ratio = 0.8
    split_idx = int(len(X) * train_ratio)
    
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    dates_train, dates_test = dates[:split_idx], dates[split_idx:]
    
    print(f"\n数据划分:")
    print(f"  训练集: {len(X_train)} 样例")
    print(f"  测试集: {len(X_test)} 样例")
    print(f"  训练时间范围: {dates_train[0]} 到 {dates_train[-1]}")
    print(f"  测试时间范围: {dates_test[0]} 到 {dates_test[-1]}")
    
    # 5. 仔细归一化
    X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_X, scaler_y = normalize_data_carefully(
        X_train, X_test, y_train, y_test)
    
    # 6. 创建数据加载器
    train_dataset = CosmicRayDataset(X_train_scaled, y_train_scaled)
    test_dataset = CosmicRayDataset(X_test_scaled, y_test_scaled)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 7. 创建LSTM模型
    input_size = X.shape[2]  # 5个特征
    hidden_size = 64  # 适中的隐藏层大小
    num_layers = 2
    output_size = 1
    
    model = CorrectedLSTMModel(input_size, hidden_size, num_layers, output_size, dropout=0.2)
    
    print(f"\n模型配置:")
    print(f"  输入特征数: {input_size}")
    print(f"  隐藏层大小: {hidden_size}")
    print(f"  LSTM层数: {num_layers}")
    print(f"  总参数数: {sum(p.numel() for p in model.parameters()):,}")
    
    # 8. 训练模型
    train_losses, val_losses = train_corrected_model(model, train_loader, test_loader, num_epochs=100)
    
    # 9. 评估测试集
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
    
    # 10. 扩展预测（2011-2032）
    print(f"\n=== 开始扩展预测 ===")
    start_date = datetime(2011, 5, 20)
    end_date = datetime(2032, 12, 31)
    prediction_dates = create_prediction_dates(start_date, end_date)
    
    print(f"将预测从 {start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')}")
    print(f"总共 {len(prediction_dates)} 个日期")
    
    extended_pred_dates, extended_predictions = predict_cosmic_ray_extended(
        model, solar_data, prediction_dates, scaler_X, scaler_y
    )
    
    # 11. 计算详细指标
    calculate_metrics_and_stats(cosmic_data, test_actuals, test_predictions, 
                               extended_pred_dates, extended_predictions)
    
    # 12. 绘制综合结果
    plot_comprehensive_results(cosmic_data, train_losses, val_losses, 
                              test_predictions, test_actuals, dates_test,
                              extended_pred_dates, extended_predictions)
    
    # 13. 保存结果
    save_complete_results(dates_test, test_actuals, test_predictions,
                         extended_pred_dates, extended_predictions)
    
    # 14. 保存模型
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
    }, 'complete_lstm_model.pth')
    
    print("\n=== 完整LSTM模型训练和预测完成！===")
    print("模型已保存为 'complete_lstm_model.pth'")

if __name__ == "__main__":
    main()
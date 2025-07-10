import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子确保结果可重现
torch.manual_seed(42)
np.random.seed(42)

class CosmicRayDataset(Dataset):
    """自定义数据集类"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class ImprovedLSTMModel(nn.Module):
    """改进的LSTM模型"""
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.3):
        super(ImprovedLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 双向LSTM
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=dropout, bidirectional=True)
        
        # 第二层LSTM
        self.lstm2 = nn.LSTM(hidden_size * 2, hidden_size, 1, 
                            batch_first=True, dropout=dropout)
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=4, dropout=dropout)
        
        # 改进的全连接层
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, output_size)
        )
        
        # 残差连接
        self.residual = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # 第一层双向LSTM
        lstm1_out, _ = self.lstm1(x)
        
        # 第二层LSTM
        lstm2_out, _ = self.lstm2(lstm1_out)
        
        # 注意力机制
        lstm_out_transposed = lstm2_out.transpose(0, 1)  # (seq_len, batch, hidden_size)
        attn_out, attn_weights = self.attention(lstm_out_transposed, lstm_out_transposed, lstm_out_transposed)
        attn_out = attn_out.transpose(0, 1)  # (batch, seq_len, hidden_size)
        
        # 全局平均池化和最大池化
        avg_pool = torch.mean(attn_out, dim=1)
        max_pool, _ = torch.max(attn_out, dim=1)
        
        # 结合池化结果
        combined = avg_pool + max_pool
        
        # 全连接层
        main_output = self.fc(combined)
        
        # 残差连接（使用输入的平均值）
        residual_input = torch.mean(x, dim=1)  # (batch, input_size)
        residual_output = self.residual(residual_input)
        
        # 最终输出
        output = main_output + 0.1 * residual_output
        
        return output

def create_enhanced_features(solar_data):
    """创建增强特征"""
    print("正在创建增强特征...")
    enhanced_data = solar_data.copy()
    
    # 基础特征
    base_features = ['HMF', 'wind_speed', 'HCS_tilt', 'SSN']
    
    # 1. 移动平均特征（多个时间窗口）
    for col in base_features:
        enhanced_data[f'{col}_ma3'] = enhanced_data[col].rolling(window=3, center=True).mean()
        enhanced_data[f'{col}_ma7'] = enhanced_data[col].rolling(window=7, center=True).mean()
        enhanced_data[f'{col}_ma27'] = enhanced_data[col].rolling(window=27, center=True).mean()  # 太阳自转周期
        enhanced_data[f'{col}_ma81'] = enhanced_data[col].rolling(window=81, center=True).mean()  # 3个太阳自转周期
    
    # 2. 变化率特征
    for col in base_features:
        enhanced_data[f'{col}_diff1'] = enhanced_data[col].diff(1)
        enhanced_data[f'{col}_diff7'] = enhanced_data[col].diff(7)
        enhanced_data[f'{col}_pct_change'] = enhanced_data[col].pct_change()
    
    # 3. 统计特征（滚动窗口）
    for col in base_features:
        enhanced_data[f'{col}_std7'] = enhanced_data[col].rolling(window=7).std()
        enhanced_data[f'{col}_std27'] = enhanced_data[col].rolling(window=27).std()
        enhanced_data[f'{col}_min27'] = enhanced_data[col].rolling(window=27).min()
        enhanced_data[f'{col}_max27'] = enhanced_data[col].rolling(window=27).max()
    
    # 4. 交互特征
    enhanced_data['HMF_wind_interaction'] = enhanced_data['HMF'] * enhanced_data['wind_speed']
    enhanced_data['HCS_polarity_interaction'] = enhanced_data['HCS_tilt'] * enhanced_data['polarity']
    enhanced_data['SSN_HMF_ratio'] = enhanced_data['SSN'] / (enhanced_data['HMF'] + 1e-8)
    
    # 5. 周期性特征
    enhanced_data['day_of_year'] = enhanced_data['date'].dt.dayofyear
    enhanced_data['solar_cycle_phase'] = np.sin(2 * np.pi * enhanced_data['day_of_year'] / 365.25)
    enhanced_data['solar_cycle_phase_cos'] = np.cos(2 * np.pi * enhanced_data['day_of_year'] / 365.25)
    
    # 6. 填充缺失值
    enhanced_data = enhanced_data.fillna(method='bfill').fillna(method='ffill')
    
    print(f"特征数量从 {len(solar_data.columns)} 增加到 {len(enhanced_data.columns)}")
    
    return enhanced_data

def select_best_features(enhanced_data, cosmic_data, top_k=20):
    """选择最重要的特征"""
    print("正在进行特征选择...")
    
    from sklearn.feature_selection import mutual_info_regression
    from scipy.stats import pearsonr
    
    # 准备数据进行特征选择
    feature_cols = [col for col in enhanced_data.columns if col != 'date']
    
    # 创建简单的对齐数据用于特征选择
    aligned_data = []
    aligned_targets = []
    
    for _, cosmic_row in cosmic_data.iterrows():
        target_date = cosmic_row['date YYYY-MM-DD']
        target_flux = cosmic_row['helium_flux m^-2sr^-1s^-1GV^-1']
        
        # 取目标日期前30天的平均值作为特征（简化版）
        start_date = target_date - timedelta(days=30)
        
        mask = (enhanced_data['date'] >= start_date) & (enhanced_data['date'] < target_date)
        if mask.sum() > 0:
            feature_values = enhanced_data[mask][feature_cols].mean().values
            if not np.isnan(feature_values).any():
                aligned_data.append(feature_values)
                aligned_targets.append(target_flux)
    
    if len(aligned_data) > 100:  # 确保有足够的数据
        X_for_selection = np.array(aligned_data)
        y_for_selection = np.array(aligned_targets)
        
        # 计算相关性分数
        correlations = []
        for i, col in enumerate(feature_cols):
            try:
                corr, _ = pearsonr(X_for_selection[:, i], y_for_selection)
                correlations.append((col, abs(corr)))
            except:
                correlations.append((col, 0))
        
        # 按相关性排序
        correlations.sort(key=lambda x: x[1], reverse=True)
        selected_features = [item[0] for item in correlations[:top_k]]
        
        print(f"选择了前 {len(selected_features)} 个最重要的特征")
        print("前10个特征及其相关性:")
        for i, (feature, corr) in enumerate(correlations[:10]):
            print(f"  {i+1}. {feature}: {corr:.4f}")
        
        return selected_features
    else:
        # 如果数据不够，使用原始特征
        basic_features = ['HMF', 'wind_speed', 'HCS_tilt', 'polarity', 'SSN']
        enhanced_features = [col for col in feature_cols if any(base in col for base in basic_features)]
        return enhanced_features[:top_k]

def load_and_preprocess_data():
    """加载和预处理数据"""
    print("正在加载数据...")
    
    # 加载太阳参数数据
    solar_data = pd.read_csv('/home/phil/Files/lstmPaper_v250618/data/solar_physics_data_1980_extend_smoothed.csv')
    solar_data['date'] = pd.to_datetime(solar_data['date'])
    solar_data = solar_data.sort_values('date').reset_index(drop=True)
    
    # 加载宇宙线数据
    cosmic_data = pd.read_csv('/home/phil/Files/lstmPaper_v250618/data/raw_data/ams/helium.csv')
    cosmic_data['date YYYY-MM-DD'] = pd.to_datetime(cosmic_data['date YYYY-MM-DD'])
    
    # 筛选特定刚度的数据
    cosmic_data = cosmic_data[cosmic_data['rigidity_min GV'] == 2.97].copy()
    cosmic_data = cosmic_data.sort_values('date YYYY-MM-DD').reset_index(drop=True)
    
    print(f"太阳参数数据范围: {solar_data['date'].min()} 到 {solar_data['date'].max()}")
    print(f"宇宙线数据范围: {cosmic_data['date YYYY-MM-DD'].min()} 到 {cosmic_data['date YYYY-MM-DD'].max()}")
    print(f"宇宙线数据点数: {len(cosmic_data)}")
    
    # 创建增强特征
    enhanced_solar_data = create_enhanced_features(solar_data)
    
    # 特征选择
    selected_features = select_best_features(enhanced_solar_data, cosmic_data, top_k=25)
    
    return enhanced_solar_data, cosmic_data, selected_features

def create_sequences_improved(solar_data, cosmic_data, selected_features, sequence_length=365):
    """创建改进的训练序列"""
    print("正在创建训练序列...")
    
    X = []  # 输入序列
    y = []  # 输出序列
    dates = []  # 对应的日期
    
    for _, row in cosmic_data.iterrows():
        target_date = row['date YYYY-MM-DD']
        target_flux = row['helium_flux m^-2sr^-1s^-1GV^-1']
        
        # 计算需要的太阳参数日期范围
        start_date = target_date - timedelta(days=sequence_length)
        end_date = target_date - timedelta(days=1)
        
        # 提取对应时间段的太阳参数
        mask = (solar_data['date'] >= start_date) & (solar_data['date'] <= end_date)
        solar_sequence = solar_data[mask][selected_features].values
        
        # 确保有足够的数据点
        if len(solar_sequence) == sequence_length:
            X.append(solar_sequence)
            y.append(target_flux)
            dates.append(target_date)
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"成功创建 {len(X)} 个训练样本")
    print(f"输入形状: {X.shape} (样本数, 时间步长, 特征数)")
    print(f"输出形状: {y.shape}")
    print(f"使用特征: {selected_features}")
    
    return X, y, dates

def split_data_improved(X, y, dates, train_ratio=0.7, val_ratio=0.15):
    """改进的数据划分（训练/验证/测试）"""
    train_end = int(len(X) * train_ratio)
    val_end = int(len(X) * (train_ratio + val_ratio))
    
    X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
    y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]
    dates_train, dates_val, dates_test = dates[:train_end], dates[train_end:val_end], dates[val_end:]
    
    print(f"训练集大小: {len(X_train)}")
    print(f"验证集大小: {len(X_val)}")
    print(f"测试集大小: {len(X_test)}")
    print(f"训练集时间范围: {dates_train[0]} 到 {dates_train[-1]}")
    print(f"验证集时间范围: {dates_val[0]} 到 {dates_val[-1]}")
    print(f"测试集时间范围: {dates_test[0]} 到 {dates_test[-1]}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, dates_train, dates_val, dates_test

def normalize_data_improved(X_train, X_val, X_test, y_train, y_val, y_test):
    """改进的数据标准化"""
    print("正在进行数据标准化...")
    
    # 使用RobustScaler，对异常值更鲁棒
    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    X_val_flat = X_val.reshape(-1, X_val.shape[-1])
    X_test_flat = X_test.reshape(-1, X_test.shape[-1])
    
    scaler_X = RobustScaler()
    X_train_scaled_flat = scaler_X.fit_transform(X_train_flat)
    X_val_scaled_flat = scaler_X.transform(X_val_flat)
    X_test_scaled_flat = scaler_X.transform(X_test_flat)
    
    X_train_scaled = X_train_scaled_flat.reshape(X_train.shape)
    X_val_scaled = X_val_scaled_flat.reshape(X_val.shape)
    X_test_scaled = X_test_scaled_flat.reshape(X_test.shape)
    
    # 标准化输出数据
    scaler_y = RobustScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    
    return (X_train_scaled, X_val_scaled, X_test_scaled, 
            y_train_scaled, y_val_scaled, y_test_scaled, 
            scaler_X, scaler_y)

def train_model_improved(model, train_loader, val_loader, num_epochs=150, learning_rate=0.001):
    """改进的模型训练"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # 使用更复杂的损失函数
    def combined_loss(outputs, targets):
        mse_loss = nn.MSELoss()(outputs, targets)
        mae_loss = nn.L1Loss()(outputs, targets)
        return 0.7 * mse_loss + 0.3 * mae_loss
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=15, factor=0.5, verbose=True)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"开始训练改进模型，使用设备: {device}")
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = combined_loss(outputs.squeeze(), y_batch)
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
                loss = combined_loss(outputs.squeeze(), y_batch)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        # 早停机制
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), 'best_improved_model.pth')
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        
        # 早停
        if patience_counter >= 25:
            print(f"早停触发，在第{epoch+1}轮停止训练")
            break
    
    # 加载最佳模型
    model.load_state_dict(torch.load('best_improved_model.pth'))
    
    return train_losses, val_losses

def evaluate_model_improved(model, test_loader, scaler_y):
    """改进的模型评估"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(y_batch.numpy())
    
    # 反标准化
    predictions = scaler_y.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    actuals = scaler_y.inverse_transform(np.array(actuals).reshape(-1, 1)).flatten()
    
    # 计算多种评估指标
    mse = mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)
    
    # 计算MAPE
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
    
    print(f"改进模型测试集评估结果:")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"R²: {r2:.6f}")
    print(f"MAPE: {mape:.2f}%")
    
    return predictions, actuals

def plot_improved_results(train_losses, val_losses, predictions, actuals, dates_test):
    """绘制改进后的结果"""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # 训练损失曲线
    axes[0, 0].plot(train_losses, label='训练损失', color='blue')
    axes[0, 0].plot(val_losses, label='验证损失', color='red')
    axes[0, 0].set_xlabel('训练轮数')
    axes[0, 0].set_ylabel('损失值')
    axes[0, 0].set_title('改进模型训练过程')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 预测 vs 实际值散点图
    axes[0, 1].scatter(actuals, predictions, alpha=0.6, color='green')
    axes[0, 1].plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'r--', lw=2)
    axes[0, 1].set_xlabel('实际值')
    axes[0, 1].set_ylabel('预测值')
    axes[0, 1].set_title('预测值 vs 实际值')
    axes[0, 1].grid(True)
    
    # 添加R²值
    r2 = r2_score(actuals, predictions)
    axes[0, 1].text(0.05, 0.95, f'R² = {r2:.4f}', transform=axes[0, 1].transAxes, 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 时间序列预测结果
    axes[0, 2].plot(dates_test, actuals, label='实际值', alpha=0.8, color='blue')
    axes[0, 2].plot(dates_test, predictions, label='预测值', alpha=0.8, color='red')
    axes[0, 2].set_xlabel('日期')
    axes[0, 2].set_ylabel('氦通量')
    axes[0, 2].set_title('时间序列预测结果')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    axes[0, 2].tick_params(axis='x', rotation=45)
    
    # 残差图
    residuals = actuals - predictions
    axes[1, 0].scatter(predictions, residuals, alpha=0.6, color='purple')
    axes[1, 0].axhline(y=0, color='r', linestyle='--')
    axes[1, 0].set_xlabel('预测值')
    axes[1, 0].set_ylabel('残差')
    axes[1, 0].set_title('残差分布图')
    axes[1, 0].grid(True)
    
    # 残差直方图
    axes[1, 1].hist(residuals, bins=30, alpha=0.7, color='orange')
    axes[1, 1].set_xlabel('残差')
    axes[1, 1].set_ylabel('频次')
    axes[1, 1].set_title('残差分布直方图')
    axes[1, 1].grid(True)
    
    # 误差随时间变化
    abs_errors = np.abs(residuals)
    axes[1, 2].plot(dates_test, abs_errors, alpha=0.7, color='red')
    axes[1, 2].set_xlabel('日期')
    axes[1, 2].set_ylabel('绝对误差')
    axes[1, 2].set_title('预测误差随时间变化')
    axes[1, 2].grid(True)
    axes[1, 2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('改进模型预测结果.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """主函数"""
    print("=== 改进版宇宙线通量LSTM预测模型 ===\n")
    
    # 1. 加载和预处理数据
    solar_data, cosmic_data, selected_features = load_and_preprocess_data()
    
    # 2. 创建序列
    X, y, dates = create_sequences_improved(solar_data, cosmic_data, selected_features)
    
    # 3. 划分数据集
    X_train, X_val, X_test, y_train, y_val, y_test, dates_train, dates_val, dates_test = split_data_improved(X, y, dates)
    
    # 4. 数据标准化
    (X_train_scaled, X_val_scaled, X_test_scaled, 
     y_train_scaled, y_val_scaled, y_test_scaled, 
     scaler_X, scaler_y) = normalize_data_improved(X_train, X_val, X_test, y_train, y_val, y_test)
    
    # 5. 创建数据加载器
    train_dataset = CosmicRayDataset(X_train_scaled, y_train_scaled)
    val_dataset = CosmicRayDataset(X_val_scaled, y_val_scaled)
    test_dataset = CosmicRayDataset(X_test_scaled, y_test_scaled)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 6. 创建改进的模型
    input_size = X.shape[2]  # 特征数量
    hidden_size = 128  # 增加隐藏层大小
    num_layers = 2
    output_size = 1
    
    model = ImprovedLSTMModel(input_size, hidden_size, num_layers, output_size, dropout=0.3)
    
    print(f"\n改进模型参数:")
    print(f"输入特征数: {input_size}")
    print(f"隐藏层大小: {hidden_size}")
    print(f"LSTM层数: {num_layers}")
    print(f"输出大小: {output_size}")
    
    # 计算模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数数: {total_params:,}")
    
    # 7. 训练模型
    train_losses, val_losses = train_model_improved(model, train_loader, val_loader, num_epochs=150)
    
    # 8. 评估模型
    predictions, actuals = evaluate_model_improved(model, test_loader, scaler_y)
    
    # 9. 绘制结果
    plot_improved_results(train_losses, val_losses, predictions, actuals, dates_test)
    
    # 10. 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'selected_features': selected_features,
        'model_config': {
            'input_size': input_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'output_size': output_size
        }
    }, 'improved_cosmic_ray_lstm_model.pth')
    
    print("\n改进模型已保存为 'improved_cosmic_ray_lstm_model.pth'")
    print("训练完成！")

if __name__ == "__main__":
    main()
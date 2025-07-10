import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

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

class LSTMModel(nn.Module):
    """LSTM模型类"""
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
    
    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM前向传播
        out, _ = self.lstm(x, (h0, c0))
        
        # 取最后一个时间步的输出
        out = self.fc(out[:, -1, :])
        return out

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
    
    # 筛选特定刚度的数据 (rigidity_min GV == 2.97)
    cosmic_data = cosmic_data[cosmic_data['rigidity_min GV'] == 2.97].copy()
    cosmic_data = cosmic_data.sort_values('date YYYY-MM-DD').reset_index(drop=True)
    
    print(f"太阳参数数据范围: {solar_data['date'].min()} 到 {solar_data['date'].max()}")
    print(f"宇宙线数据范围: {cosmic_data['date YYYY-MM-DD'].min()} 到 {cosmic_data['date YYYY-MM-DD'].max()}")
    print(f"宇宙线数据点数: {len(cosmic_data)}")
    
    return solar_data, cosmic_data

def create_sequences(solar_data, cosmic_data, sequence_length=365):
    """创建训练序列"""
    print("正在创建训练序列...")
    
    # 太阳参数列名
    solar_features = ['HMF', 'wind_speed', 'HCS_tilt', 'polarity', 'SSN']
    
    X = []  # 输入序列
    y = []  # 输出序列
    dates = []  # 对应的日期
    
    for _, row in cosmic_data.iterrows():
        target_date = row['date YYYY-MM-DD']
        target_flux = row['helium_flux m^-2sr^-1s^-1GV^-1']
        
        # 计算需要的太阳参数日期范围（target_date前365天）
        start_date = target_date - timedelta(days=sequence_length)
        end_date = target_date - timedelta(days=1)
        
        # 提取对应时间段的太阳参数
        mask = (solar_data['date'] >= start_date) & (solar_data['date'] <= end_date)
        solar_sequence = solar_data[mask][solar_features].values
        
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
    
    return X, y, dates

def split_data(X, y, dates, train_ratio=0.8):
    """按时间顺序划分训练集和测试集"""
    split_idx = int(len(X) * train_ratio)
    
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    dates_train, dates_test = dates[:split_idx], dates[split_idx:]
    
    print(f"训练集大小: {len(X_train)}")
    print(f"测试集大小: {len(X_test)}")
    print(f"训练集时间范围: {dates_train[0]} 到 {dates_train[-1]}")
    print(f"测试集时间范围: {dates_test[0]} 到 {dates_test[-1]}")
    
    return X_train, X_test, y_train, y_test, dates_train, dates_test

def normalize_data(X_train, X_test, y_train, y_test):
    """数据标准化"""
    print("正在进行数据标准化...")
    
    # 标准化输入数据
    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    X_test_flat = X_test.reshape(-1, X_test.shape[-1])
    
    scaler_X = StandardScaler()
    X_train_scaled_flat = scaler_X.fit_transform(X_train_flat)
    X_test_scaled_flat = scaler_X.transform(X_test_flat)
    
    X_train_scaled = X_train_scaled_flat.reshape(X_train.shape)
    X_test_scaled = X_test_scaled_flat.reshape(X_test.shape)
    
    # 标准化输出数据
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    
    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_X, scaler_y

def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=0.001):
    """训练模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
    
    train_losses = []
    val_losses = []
    
    print(f"开始训练模型，使用设备: {device}")
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.squeeze(), y_batch)
            loss.backward()
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
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
    
    return train_losses, val_losses

def evaluate_model(model, test_loader, scaler_y):
    """评估模型"""
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
    
    # 计算评估指标
    mse = mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)
    
    print(f"测试集评估结果:")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"R²: {r2:.6f}")
    
    return predictions, actuals

def plot_results(train_losses, val_losses, predictions, actuals, dates_test):
    """绘制结果"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 训练损失曲线
    axes[0, 0].plot(train_losses, label='训练损失')
    axes[0, 0].plot(val_losses, label='验证损失')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('训练过程损失曲线')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 预测 vs 实际值散点图
    axes[0, 1].scatter(actuals, predictions, alpha=0.6)
    axes[0, 1].plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'r--', lw=2)
    axes[0, 1].set_xlabel('实际值')
    axes[0, 1].set_ylabel('预测值')
    axes[0, 1].set_title('预测值 vs 实际值')
    axes[0, 1].grid(True)
    
    # 时间序列预测结果
    axes[1, 0].plot(dates_test, actuals, label='实际值', alpha=0.7)
    axes[1, 0].plot(dates_test, predictions, label='预测值', alpha=0.7)
    axes[1, 0].set_xlabel('日期')
    axes[1, 0].set_ylabel('氦通量')
    axes[1, 0].set_title('时间序列预测结果')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 残差图
    residuals = actuals - predictions
    axes[1, 1].scatter(predictions, residuals, alpha=0.6)
    axes[1, 1].axhline(y=0, color='r', linestyle='--')
    axes[1, 1].set_xlabel('预测值')
    axes[1, 1].set_ylabel('残差')
    axes[1, 1].set_title('残差图')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('cosmic_ray_prediction_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """主函数"""
    print("=== 宇宙线通量LSTM预测模型 ===\n")
    
    # 1. 加载和预处理数据
    solar_data, cosmic_data = load_and_preprocess_data()
    
    # 2. 创建序列
    X, y, dates = create_sequences(solar_data, cosmic_data)
    
    # 3. 划分数据集
    X_train, X_test, y_train, y_test, dates_train, dates_test = split_data(X, y, dates)
    
    # 4. 数据标准化
    X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_X, scaler_y = normalize_data(
        X_train, X_test, y_train, y_test)
    
    # 5. 创建数据加载器
    train_dataset = CosmicRayDataset(X_train_scaled, y_train_scaled)
    test_dataset = CosmicRayDataset(X_test_scaled, y_test_scaled)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 6. 创建模型
    input_size = X.shape[2]  # 特征数量（5个太阳参数）
    hidden_size = 64
    num_layers = 2
    output_size = 1
    
    model = LSTMModel(input_size, hidden_size, num_layers, output_size)
    
    print(f"\n模型参数:")
    print(f"输入特征数: {input_size}")
    print(f"隐藏层大小: {hidden_size}")
    print(f"LSTM层数: {num_layers}")
    print(f"输出大小: {output_size}")
    
    # 7. 训练模型
    train_losses, val_losses = train_model(model, train_loader, test_loader, num_epochs=100)
    
    # 8. 评估模型
    predictions, actuals = evaluate_model(model, test_loader, scaler_y)
    
    # 9. 绘制结果
    plot_results(train_losses, val_losses, predictions, actuals, dates_test)
    
    # 10. 保存模型
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
    }, 'cosmic_ray_lstm_model.pth')
    
    print("\n模型已保存为 'cosmic_ray_lstm_model.pth'")
    print("训练完成！")

if __name__ == "__main__":
    main()
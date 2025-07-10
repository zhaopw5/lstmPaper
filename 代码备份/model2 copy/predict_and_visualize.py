import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from matplotlib import rcParams

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class LSTMModel(nn.Module):
    """LSTM模型类（与训练时相同）"""
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def load_model_and_scalers():
    """加载训练好的模型和标准化器"""
    print("正在加载训练好的模型...")
    
    checkpoint = torch.load('improved_cosmic_ray_lstm_model.pth', map_location='cpu')
    
    model_config = checkpoint['model_config']
    model = LSTMModel(
        model_config['input_size'],
        model_config['hidden_size'], 
        model_config['num_layers'],
        model_config['output_size']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    scaler_X = checkpoint['scaler_X']
    scaler_y = checkpoint['scaler_y']
    
    model.eval()
    print("模型加载完成！")
    
    return model, scaler_X, scaler_y

def load_data():
    """加载原始数据"""
    print("正在加载原始数据...")
    
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
    
    return solar_data, cosmic_data

def create_prediction_dates(start_date, end_date):
    """创建预测日期序列"""
    dates = []
    current_date = start_date
    while current_date <= end_date:
        dates.append(current_date)
        current_date += timedelta(days=1)
    return dates

def predict_cosmic_ray(model, solar_data, prediction_dates, scaler_X, scaler_y, sequence_length=365):
    """预测宇宙线通量"""
    print(f"正在预测 {len(prediction_dates)} 个日期的宇宙线通量...")
    
    solar_features = ['HMF', 'wind_speed', 'HCS_tilt', 'polarity', 'SSN']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    predictions = []
    valid_dates = []
    
    for i, target_date in enumerate(prediction_dates):
        if (i + 1) % 500 == 0:
            print(f"已处理 {i + 1}/{len(prediction_dates)} 个日期")
            
        # 计算需要的太阳参数日期范围
        start_date = target_date - timedelta(days=sequence_length)
        end_date = target_date - timedelta(days=1)
        
        # 提取对应时间段的太阳参数
        mask = (solar_data['date'] >= start_date) & (solar_data['date'] <= end_date)
        solar_sequence = solar_data[mask][solar_features].values
        
        # 确保有足够的数据点
        if len(solar_sequence) == sequence_length:
            # 标准化输入数据
            solar_sequence_flat = solar_sequence.reshape(-1, len(solar_features))
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

def plot_comprehensive_results(cosmic_data, pred_dates, predictions):
    """绘制综合预测结果"""
    print("正在绘制结果图...")
    
    # 创建图形
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    
    # 实际观测数据
    obs_dates = cosmic_data['date YYYY-MM-DD']
    obs_flux = cosmic_data['helium_flux m^-2sr^-1s^-1GV^-1']
    
    # 图1：完整时间序列（2011-2032）
    axes[0].plot(obs_dates, obs_flux, 'b-', label='实际观测数据', linewidth=1.5, alpha=0.8)
    axes[0].plot(pred_dates, predictions, 'r-', label='LSTM预测数据', linewidth=1.5, alpha=0.8)
    axes[0].axvline(x=obs_dates.iloc[-1], color='green', linestyle='--', alpha=0.7, label='观测数据结束')
    axes[0].set_title('宇宙线氦通量预测：2011-2032年完整时间序列', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('日期')
    axes[0].set_ylabel('氦通量 (m⁻²sr⁻¹s⁻¹GV⁻¹)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    axes[0].xaxis.set_major_locator(mdates.YearLocator())
    
    # 图2：重叠期间的对比（训练效果验证）
    overlap_start = obs_dates.iloc[0]
    overlap_end = obs_dates.iloc[-1]
    
    # 找到预测数据中的重叠部分
    overlap_mask = [(d >= overlap_start) and (d <= overlap_end) for d in pred_dates]
    overlap_pred_dates = [d for i, d in enumerate(pred_dates) if overlap_mask[i]]
    overlap_predictions = [p for i, p in enumerate(predictions) if overlap_mask[i]]
    
    axes[1].plot(obs_dates, obs_flux, 'b-', label='实际观测', linewidth=2, alpha=0.8)
    if overlap_pred_dates:
        axes[1].plot(overlap_pred_dates, overlap_predictions, 'r--', 
                    label='LSTM预测', linewidth=2, alpha=0.8)
    axes[1].set_title('观测期间预测效果对比 (2011-2019)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('日期')
    axes[1].set_ylabel('氦通量 (m⁻²sr⁻¹s⁻¹GV⁻¹)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    
    # 图3：未来预测部分（2019-2032）
    future_start = obs_dates.iloc[-1]
    future_mask = [d > future_start for d in pred_dates]
    future_pred_dates = [d for i, d in enumerate(pred_dates) if future_mask[i]]
    future_predictions = [p for i, p in enumerate(predictions) if future_mask[i]]
    
    # 显示最后一段观测数据作为连接
    recent_obs = obs_dates.iloc[-100:]  # 最后100个观测点
    recent_flux = obs_flux.iloc[-100:]
    
    axes[2].plot(recent_obs, recent_flux, 'b-', label='最近观测数据', linewidth=2, alpha=0.8)
    if future_pred_dates:
        axes[2].plot(future_pred_dates, future_predictions, 'r-', 
                    label='未来预测', linewidth=2, alpha=0.8)
    axes[2].axvline(x=obs_dates.iloc[-1], color='green', linestyle='--', alpha=0.7, label='预测起始点')
    axes[2].set_title('未来宇宙线通量预测 (2019-2032)', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('日期')
    axes[2].set_ylabel('氦通量 (m⁻²sr⁻¹s⁻¹GV⁻¹)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axes[2].xaxis.set_major_locator(mdates.YearLocator())
    
    # 调整布局
    plt.tight_layout()
    plt.savefig('宇宙线预测结果_2011_2032.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印一些统计信息
    print("\n=== 预测统计信息 ===")
    print(f"观测数据范围: {obs_dates.iloc[0].strftime('%Y-%m-%d')} 到 {obs_dates.iloc[-1].strftime('%Y-%m-%d')}")
    print(f"预测数据范围: {pred_dates[0].strftime('%Y-%m-%d')} 到 {pred_dates[-1].strftime('%Y-%m-%d')}")
    print(f"观测数据点数: {len(obs_flux)}")
    print(f"预测数据点数: {len(predictions)}")
    
    print(f"\n观测数据统计:")
    print(f"  平均值: {obs_flux.mean():.2f}")
    print(f"  标准差: {obs_flux.std():.2f}")
    print(f"  最小值: {obs_flux.min():.2f}")
    print(f"  最大值: {obs_flux.max():.2f}")
    
    print(f"\n预测数据统计:")
    print(f"  平均值: {np.mean(predictions):.2f}")
    print(f"  标准差: {np.std(predictions):.2f}")
    print(f"  最小值: {np.min(predictions):.2f}")
    print(f"  最大值: {np.max(predictions):.2f}")
    
    if len([d for d in pred_dates if d > obs_dates.iloc[-1]]) > 0:
        future_predictions = [p for i, p in enumerate(predictions) if pred_dates[i] > obs_dates.iloc[-1]]
        print(f"\n未来预测统计 (2019年后):")
        print(f"  平均值: {np.mean(future_predictions):.2f}")
        print(f"  标准差: {np.std(future_predictions):.2f}")
        print(f"  最小值: {np.min(future_predictions):.2f}")
        print(f"  最大值: {np.max(future_predictions):.2f}")

def calculate_overlap_metrics(cosmic_data, pred_dates, predictions):
    """计算重叠期间的预测精度"""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    obs_dates = cosmic_data['date YYYY-MM-DD']
    obs_flux = cosmic_data['helium_flux m^-2sr^-1s^-1GV^-1']
    
    # 找到重叠的日期
    aligned_obs = []
    aligned_pred = []
    
    for i, pred_date in enumerate(pred_dates):
        # 在观测数据中查找对应日期
        matching_obs = cosmic_data[cosmic_data['date YYYY-MM-DD'] == pred_date]
        if not matching_obs.empty:
            aligned_obs.append(matching_obs['helium_flux m^-2sr^-1s^-1GV^-1'].iloc[0])
            aligned_pred.append(predictions[i])
    
    if len(aligned_obs) > 0:
        mse = mean_squared_error(aligned_obs, aligned_pred)
        mae = mean_absolute_error(aligned_obs, aligned_pred)
        r2 = r2_score(aligned_obs, aligned_pred)
        
        print(f"\n=== 重叠期间预测精度 ===")
        print(f"对比数据点数: {len(aligned_obs)}")
        print(f"MSE: {mse:.6f}")
        print(f"MAE: {mae:.6f}")
        print(f"R²: {r2:.6f}")
        
        return aligned_obs, aligned_pred
    else:
        print("没有找到重叠的日期数据")
        return [], []

def save_predictions(pred_dates, predictions):
    """保存预测结果"""
    results_df = pd.DataFrame({
        'date': pred_dates,
        'predicted_helium_flux': predictions
    })
    
    results_df.to_csv('宇宙线预测结果_2011_2032.csv', index=False)
    print(f"\n预测结果已保存到 '宇宙线预测结果_2011_2032.csv'")

def main():
    """主函数"""
    print("=== 宇宙线通量预测可视化 ===\n")
    
    # 1. 加载模型和标准化器
    model, scaler_X, scaler_y = load_model_and_scalers()
    
    # 2. 加载原始数据
    solar_data, cosmic_data = load_data()
    
    # 3. 创建预测日期序列（从2011-05-20到2032-12-31）
    start_date = datetime(2011, 5, 20)
    end_date = datetime(2032, 12, 31)
    prediction_dates = create_prediction_dates(start_date, end_date)
    
    print(f"将预测从 {start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')}")
    print(f"总共 {len(prediction_dates)} 个日期")
    
    # 4. 进行预测
    pred_dates, predictions = predict_cosmic_ray(
        model, solar_data, prediction_dates, scaler_X, scaler_y
    )
    
    # 5. 计算重叠期间的精度
    aligned_obs, aligned_pred = calculate_overlap_metrics(cosmic_data, pred_dates, predictions)
    
    # 6. 绘制结果
    plot_comprehensive_results(cosmic_data, pred_dates, predictions)
    
    # 7. 保存预测结果
    save_predictions(pred_dates, predictions)
    
    print("\n预测和可视化完成！")

if __name__ == "__main__":
    main()
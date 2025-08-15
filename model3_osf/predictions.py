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
import logging


def create_prediction_dates(start_date, end_date):
    """创建预测日期序列"""
    dates = []
    current_date = start_date
    while current_date <= end_date:
        dates.append(current_date)
        current_date += timedelta(days=1)
    return dates


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
    features = ['HMF', 'wind_speed', 'HCS_tilt', 'polarity', 'SSN', 'daily_OSF', 'helium_flux m^-2sr^-1s^-1GV^-1']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    predictions = []
    valid_dates = []
    # cosmic_data按日期排序，方便查找
    cosmic_data = cosmic_data.sort_values('date YYYY-MM-DD').reset_index(drop=True)
    # 构造氦通量时间序列（初始为观测数据）
    helium_flux_series = cosmic_data[['date YYYY-MM-DD', 'helium_flux m^-2sr^-1s^-1GV^-1']].copy()
    helium_flux_dict = dict(zip(helium_flux_series['date YYYY-MM-DD'], helium_flux_series['helium_flux m^-2sr^-1s^-1GV^-1']))
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
                solar_row = solar_data[solar_mask][['HMF', 'wind_speed', 'HCS_tilt', 'polarity', 'SSN', 'daily_OSF']].iloc[0].values
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

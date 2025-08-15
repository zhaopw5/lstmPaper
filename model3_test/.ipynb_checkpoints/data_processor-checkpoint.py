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



def load_and_check_data():
    """读数据，检查缺失/重复日期，插值补全"""

    # Load solar parameter data
    solar_data = pd.read_csv(
        '/home/phil/Files/lstmPaper/data/outputs/cycle_analysis/solar_physics_data_1985_2025_cycle_prediction_smoothed.csv'
    )
    solar_data['date'] = pd.to_datetime(solar_data['date'])

    # Load cosmic ray flux data
    cosmic_data = pd.read_csv(
        '/home/phil/Files/lstmPaper/data/raw_data/ams/helium.csv'
    )
    cosmic_data['date YYYY-MM-DD'] = pd.to_datetime(cosmic_data['date YYYY-MM-DD'])
    cosmic_data = cosmic_data[cosmic_data['rigidity_min GV'] == 2.97].copy()

    # --- Debug alignment information ---
    print("=== Data Alignment Debug ===")
    # Print ranges
    print(
        f"Solar data range: {solar_data['date'].min()} to {solar_data['date'].max()}"
    )
    print(
        f"Cosmic data range: {cosmic_data['date YYYY-MM-DD'].min()} to {cosmic_data['date YYYY-MM-DD'].max()}"
    )
    
    # Check total counts
    print(f"Total solar days: {len(solar_data)}")
    print(f"Total cosmic days before interpolation: {len(cosmic_data)}")

    # 检查数据中是否有缺失日期
    solar_dates = pd.to_datetime(solar_data['date'])
    cosmic_dates = pd.to_datetime(cosmic_data['date YYYY-MM-DD'])
    full_solar = pd.date_range(start=solar_dates.min(), end=solar_dates.max(), freq='D')
    full_cosmic = pd.date_range(start=cosmic_dates.min(), end=cosmic_dates.max(), freq='D')
    missing_solar = set(full_solar) - set(solar_dates)
    missing_cosmic = set(full_cosmic) - set(cosmic_dates)
    print(f"Missing solar days: {len(missing_solar)}")
    if missing_solar:
        first5 = sorted(list(missing_solar))[:5]
        print(f"First 5 missing solar dates: {first5}")
    print(f"Missing cosmic days: {len(missing_cosmic)}")
    if missing_cosmic:
        first5_cos = sorted(list(missing_cosmic))[:5]
        print(f"First 5 missing cosmic dates: {first5_cos}")

    # 检查数据中是否有重复的日期
    solar_dups = solar_data[solar_data.duplicated('date', keep=False)]
    cosmic_dups = cosmic_data[cosmic_data.duplicated('date YYYY-MM-DD', keep=False)]
    print(f"Duplicate solar dates: {len(solar_dups)}")
    print(f"Duplicate cosmic dates: {len(cosmic_dups)}")

    ##########################################################################
    # --- 插值补全 ---
    print("Interpolating cosmic data...")
    full_range = pd.date_range(start=cosmic_dates.min(), end=cosmic_dates.max(), freq='D')
    # 生成完整的日期范围
    cosmic_data = cosmic_data.set_index('date YYYY-MM-DD').reindex(full_range)
    # 线性插值填充缺失值
    cosmic_data['helium_flux m^-2sr^-1s^-1GV^-1'] = (
        cosmic_data['helium_flux m^-2sr^-1s^-1GV^-1'].interpolate(method='linear')
    )
    # 把插值后 DataFrame 的索引（即日期）还原成普通列，并把列名改回原来的名字
    cosmic_data = (cosmic_data
        .reset_index()
        .rename(columns={'index': 'date YYYY-MM-DD'})
    )
    print(f"Total cosmic days after interpolation: {len(cosmic_data)}")

    # Final summary
    print("Final data summary:")
    print(
        f"Solar data: {len(solar_data)} days ({solar_data['date'].min()} - {solar_data['date'].max()})"
    )
    print(
        f"Cosmic data: {len(cosmic_data)} days "
        f"({cosmic_data['date YYYY-MM-DD'].min()} - {cosmic_data['date YYYY-MM-DD'].max()})"
    )

    return solar_data, cosmic_data


# 重要的函数构造：每一组：输入-输出
def create_sequences(solar_data, cosmic_data, sequence_length=365):
    """
    每个样本输入：过去365天的[太阳参数*5 + helium_flux*1]（共6个特征），输出：第366天的helium_flux
    """

    print(f"\n=== 创建 {sequence_length} 天序列（太阳参数+宇宙线流强） ===")
    features = ['HMF', 'wind_speed', 'HCS_tilt', 'polarity', 'SSN', 'helium_flux m^-2sr^-1s^-1GV^-1']
    X = []
    y = []
    dates = []
    successful_alignments = 0
    failed_alignments = 0
    # cosmic_data按日期排序，保证滑窗正确
    cosmic_data = cosmic_data.sort_values('date YYYY-MM-DD').reset_index(drop=True)
    for idx in range(1):#len(cosmic_data) - sequence_length):
        # 输入窗口的起止日期
        input_start = cosmic_data.loc[idx, 'date YYYY-MM-DD']
        input_end = cosmic_data.loc[idx + sequence_length - 1, 'date YYYY-MM-DD']
        output_date = cosmic_data.loc[idx + sequence_length, 'date YYYY-MM-DD']
        print(f"\n处理样例: 输入 {input_start} 到 {input_end}, 输出 {output_date}")
        # 构造输入序列
        input_rows = []
        for i in range(sequence_length):
            date_i = cosmic_data.loc[idx + i, 'date YYYY-MM-DD']
            print(f"  日期 {date_i} 的数据:")
            # 查找太阳参数
            solar_mask = solar_data['date'] == date_i
            print(f" --------------- solar_mask : {solar_mask} -------------------- ")
            if solar_mask.sum() == 1:
                solar_row = solar_data[solar_mask][['HMF', 'wind_speed', 'HCS_tilt', 'polarity', 'SSN']].iloc[0].values
                print(f"  solar_row: {solar_row}")
                helium_flux = cosmic_data.loc[idx + i, 'helium_flux m^-2sr^-1s^-1GV^-1']
                print(f"  helium_flux: {helium_flux:.2f}")
                input_rows.append(np.concatenate([solar_row, [helium_flux]]))
                print(f"  input_rows: {input_rows}")
            else:
                break
        if len(input_rows) == sequence_length:
            X.append(np.array(input_rows))
            y.append(cosmic_data.loc[idx + sequence_length, 'helium_flux m^-2sr^-1s^-1GV^-1'])
            print(f'X is {X}')
            print(f'y is {y}')
            dates.append(output_date)
            successful_alignments += 1
            if successful_alignments <= 3:
                print(f"\n样例 {successful_alignments}: 输入 {input_start} 到 {input_end}, 输出 {output_date}")
                print(f"  输入形状: {np.array(input_rows).shape}")
                print(f"  输出: {y[-1]:.2f}")
        else:
            failed_alignments += 1
            if failed_alignments <= 3:
                print(f"失败样例 {failed_alignments}: 只找到 {len(input_rows)} 天数据，需要 {sequence_length} 天")
        # break
    X = np.array(X)
    y = np.array(y)
    print(f"\n=== 序列创建结果 ===")
    print(f"成功对齐: {successful_alignments} 个样例")
    print(f"失败对齐: {failed_alignments} 个样例")
    print(f"最终数据形状:")
    print(f"  X: {X.shape} (样例数, 时间步数, 特征数)")
    print(f"  y: {y.shape}")
    print(f"  特征顺序: {features}")
    print(f"\n=== 数据质量检查 ===")
    print(f"X 中的 NaN 数量: {np.isnan(X).sum()}")
    print(f"y 中的 NaN 数量: {np.isnan(y).sum()}")
    if X.shape[0] > 0:
        print(f"\nX 统计 (所有特征):")
        for i, feature in enumerate(features):
            feature_data = X[:, :, i].flatten()
            print(f"  {feature}: 均值={np.mean(feature_data):.4f}, 标准差={np.std(feature_data):.4f}, 范围=[{np.min(feature_data):.2f}, {np.max(feature_data):.2f}]")
        print(f"\ny 统计:")
        print(f"  均值={np.mean(y):.4f}, 标准差={np.std(y):.4f}, 范围=[{np.min(y):.2f}, {np.max(y):.2f}]")
    return X, y, dates


def normalize_data(X_train, X_test, y_train, y_test):
    """数据归一化"""
    print(f"\n=== 数据归一化 ===")
    
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


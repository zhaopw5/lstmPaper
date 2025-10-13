from config import SEQUENCE_LENGTH, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, SOLAR_DATA_PATH, COSMIC_DATA_PATH
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
from pathlib import Path


# data to use
SOLAR_PARAMETERS = ['HMF', 'wind_speed', 'HCS_tilt', 'polarity', 'SSN', 'daily_OSF']
HELIUM_FLUX_COL = 'helium_flux m^-2sr^-1s^-1GV^-1'

# rigidity bins in GV
RIGIDITY_BIN_EDGES = [1.71, 1.92, 2.15, 2.4, 2.67, 2.97, 3.29, 3.64, 4.02, 
                      4.43, 4.88, 5.37, 5.9, 6.47, 7.09, 7.76, 8.48, 9.26, 10.1]
# rigidity_min GV
RIGIDITY_VALUES = RIGIDITY_BIN_EDGES[:-1]


def load_and_check_data():
    """读数据，检查缺失/重复日期，插值补全"""
    # 太阳数据
    solar_data = pd.read_csv(SOLAR_DATA_PATH)
    solar_data['date'] = pd.to_datetime(solar_data['date'])

    # 宇宙线数据
    cosmic_data = pd.read_csv(COSMIC_DATA_PATH)
    cosmic_data['date YYYY-MM-DD'] = pd.to_datetime(cosmic_data['date YYYY-MM-DD'])
    
    # 筛选所需的刚度数据并重组
    cosmic_multi_rigidity = []
    for rigidity in RIGIDITY_VALUES:
        rigidity_data = cosmic_data[cosmic_data['rigidity_min GV'] == rigidity].copy()
        if len(rigidity_data) > 0:
            rigidity_data = rigidity_data[['date YYYY-MM-DD', HELIUM_FLUX_COL]].copy()
            rigidity_data = rigidity_data.rename(columns={HELIUM_FLUX_COL: f'helium_{rigidity}GV'})
            cosmic_multi_rigidity.append(rigidity_data)
        else:
            print(f"警告: 刚度 {rigidity} GV 没有数据")
    
    # 合并所有刚度数据
    if cosmic_multi_rigidity:
        cosmic_data = cosmic_multi_rigidity[0]
        for i in range(1, len(cosmic_multi_rigidity)):
            cosmic_data = cosmic_data.merge(cosmic_multi_rigidity[i], on='date YYYY-MM-DD', how='outer')
    else:
        raise ValueError("没有找到任何刚度数据")

    # 更新氦通量列名列表
    helium_flux_cols = [f'helium_{rigidity}GV' for rigidity in RIGIDITY_VALUES if f'helium_{rigidity}GV' in cosmic_data.columns]
    print(f"成功加载 {len(helium_flux_cols)} 个刚度的数据: {[col.split('_')[-1] for col in helium_flux_cols]}")

    # --- Debug alignment information ---
    print("=== Data Alignment Debug ===")
    # Print ranges
    print(f"Solar data range: {solar_data['date'].min()} to {solar_data['date'].max()}")
    print(f"Cosmic data range: {cosmic_data['date YYYY-MM-DD'].min()} to {cosmic_data['date YYYY-MM-DD'].max()}")
    
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
    # --- 分别插值补全太阳数据和宇宙线数据 ---
    print("=== 分别插值补全数据 ===")
    
    # 插值补全太阳数据 - 使用太阳数据自己的时间范围
    print("Interpolating solar data...")
    solar_full_range = pd.date_range(start=solar_dates.min(), end=solar_dates.max(), freq='D')
    print(f"Solar interpolation range: {solar_full_range[0]} to {solar_full_range[-1]} ({len(solar_full_range)} days)")
    
    solar_data = solar_data.set_index('date').reindex(solar_full_range)
    for col in SOLAR_PARAMETERS:
        if col in solar_data.columns:
            solar_data[col] = solar_data[col].interpolate(method='linear')
    solar_data = solar_data.reset_index().rename(columns={'index': 'date'})
    print(f"Total solar days after interpolation: {len(solar_data)}")
    
    # 插值补全宇宙线数据 - 使用宇宙线数据自己的时间范围
    print("Interpolating cosmic data...")
    cosmic_full_range = pd.date_range(start=cosmic_dates.min(), end=cosmic_dates.max(), freq='D')
    print(f"Cosmic interpolation range: {cosmic_full_range[0]} to {cosmic_full_range[-1]} ({len(cosmic_full_range)} days)")
    
    cosmic_data = cosmic_data.set_index('date YYYY-MM-DD').reindex(cosmic_full_range)
    for col in helium_flux_cols:
        if col in cosmic_data.columns:
            cosmic_data[col] = cosmic_data[col].interpolate(method='linear')
    cosmic_data = cosmic_data.reset_index().rename(columns={'index': 'date YYYY-MM-DD'})
    print(f"Total cosmic days after interpolation: {len(cosmic_data)}")
    
    print("太阳和宇宙线数据已分别按各自时间范围线性插值补全完成。")
    
    # 保存插值后的数据
    cosmic_data.to_csv("interpolated_cosmic_data.csv", index=False)
    solar_data.to_csv("interpolated_solar_data.csv", index=False)

    # Final summary
    print("\n=== Final data summary ===")
    print(f"Solar data: {len(solar_data)} days ({solar_data['date'].min()} - {solar_data['date'].max()})")
    print(f"Cosmic data: {len(cosmic_data)} days ({cosmic_data['date YYYY-MM-DD'].min()} - {cosmic_data['date YYYY-MM-DD'].max()})")

    return solar_data, cosmic_data


def create_sequences(solar_data, cosmic_data, sequence_length=SEQUENCE_LENGTH):
    """
    每个样本输入：过去SEQUENCE_LENGTH天的[太阳参数*len(SOLAR_PARAMETERS) + helium_flux*len(RIGIDITY_VALUES)]，
    输出：第SEQUENCE_LENGTH+1天的所有刚度的helium_flux
    """

    print(f"\n=== 创建 {sequence_length} 天序列（太阳参数+多刚度宇宙线流强） ===")
    helium_flux_cols = [f'helium_{rigidity}GV' for rigidity in RIGIDITY_VALUES if f'helium_{rigidity}GV' in cosmic_data.columns]
    features = SOLAR_PARAMETERS + helium_flux_cols
    print(f"输入特征: {features}")
    
    X = []
    y = []
    dates = []
    successful_alignments = 0
    failed_alignments = 0
    
    for idx in range(len(cosmic_data) - sequence_length):
        # 输入窗口的起止日期
        input_start = cosmic_data.loc[idx, 'date YYYY-MM-DD']
        input_end = cosmic_data.loc[idx + sequence_length - 1, 'date YYYY-MM-DD']
        output_date = cosmic_data.loc[idx + sequence_length, 'date YYYY-MM-DD']
        
        # 构造输入序列
        input_rows = []
        for i in range(sequence_length):
            date_i = cosmic_data.loc[idx + i, 'date YYYY-MM-DD']
            # 查找太阳参数
            solar_mask = solar_data['date'] == date_i
            if solar_mask.sum() == 1:
                solar_row = solar_data[solar_mask][SOLAR_PARAMETERS].iloc[0].values
                # 获取所有刚度的氦通量数据
                helium_flux_row = cosmic_data.loc[idx + i, helium_flux_cols].values
                input_rows.append(np.concatenate([solar_row, helium_flux_row]))
            else:
                break
        
        if len(input_rows) == sequence_length:
            X.append(np.array(input_rows))
            # 输出是所有刚度的氦通量
            y.append(cosmic_data.loc[idx + sequence_length, helium_flux_cols].values)
            dates.append(output_date)
            successful_alignments += 1
            if successful_alignments <= 3:
                print(f"\n样例 {successful_alignments}: 输入 {input_start} 到 {input_end}, 输出 {output_date}")
                print(f"  输入形状: {np.array(input_rows).shape}")
                print(f"  输出形状: {len(helium_flux_cols)} 个刚度")
        else:
            failed_alignments += 1
            if failed_alignments <= 3:
                print(f"失败样例 {failed_alignments}: 只找到 {len(input_rows)} 天数据，需要 {sequence_length} 天")
    
    X = np.array(X)
    y = np.array(y)
    print(f"\n=== 序列创建结果 ===")
    print(f"成功对齐: {successful_alignments} 个样例")
    print(f"失败对齐: {failed_alignments} 个样例")
    print(f"最终数据形状:")
    print(f"  X: {X.shape} (样例数, 时间步数, 特征数)")
    print(f"  y: {y.shape} (样例数, 刚度数)")
    print(f"  特征顺序: {features}")
    print(f"  刚度输出顺序: {helium_flux_cols}")
    
    print(f"\n=== 数据质量检查 ===")
    if X.shape[0] > 0:
        # 转换为numpy数组并检查NaN
        X_array = np.array(X, dtype=np.float64)
        y_array = np.array(y, dtype=np.float64)
        
        print(f"X 中的 NaN 数量: {np.isnan(X_array).sum()}")
        print(f"y 中的 NaN 数量: {np.isnan(y_array).sum()}")
        
        print(f"\nX 统计 (所有特征):")
        for i, feature in enumerate(features):
            feature_data = X_array[:, :, i].flatten()
            if not np.all(np.isnan(feature_data)):
                print(f"  {feature}: 均值={np.nanmean(feature_data):.4f}, 标准差={np.nanstd(feature_data):.4f}, 范围=[{np.nanmin(feature_data):.2f}, {np.nanmax(feature_data):.2f}]")
            else:
                print(f"  {feature}: 全部为NaN")
        
        print(f"\ny 统计 (所有刚度):")
        for i, rigidity in enumerate(helium_flux_cols):
            rigidity_data = y_array[:, i].flatten()
            if not np.all(np.isnan(rigidity_data)):
                print(f"  {rigidity}: 均值={np.nanmean(rigidity_data):.4f}, 标准差={np.nanstd(rigidity_data):.4f}, 范围=[{np.nanmin(rigidity_data):.2f}, {np.nanmax(rigidity_data):.2f}]")
            else:
                print(f"  {rigidity}: 全部为NaN")
    else:
        print("没有数据可检查")
    
    return X, y, dates


def split_data_three_way(X, y, dates, 
                        train_ratio=TRAIN_RATIO, 
                        val_ratio=VAL_RATIO, 
                        test_ratio=TEST_RATIO):
    """标准三分法数据划分（使用配置中的比例）"""
    print(f"=== 标准三分法数据划分 ===")
    
    n_samples = len(X)
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    n_test = n_samples - n_train - n_val  # 确保所有样本都被使用
    
    print(f"总样本数: {n_samples}")
    print(f"  训练集: {n_train} 样例 ({train_ratio*100:.1f}%)")
    print(f"  验证集: {n_val} 样例 ({val_ratio*100:.1f}%)")
    print(f"  测试集: {n_test} 样例 ({(n_test/n_samples)*100:.1f}%)")
    
    # 按时间顺序划分
    X_train = X[:n_train]
    X_val = X[n_train:n_train+n_val]
    X_test = X[n_train+n_val:]
    
    y_train = y[:n_train]
    y_val = y[n_train:n_train+n_val]
    y_test = y[n_train+n_val:]
    
    dates_train = dates[:n_train]
    dates_val = dates[n_train:n_train+n_val]
    dates_test = dates[n_train+n_val:]
    
    print(f"  训练时间范围: {dates_train[0]} 到 {dates_train[-1]}")
    print(f"  验证时间范围: {dates_val[0]} 到 {dates_val[-1]}")
    print(f"  测试时间范围: {dates_test[0]} 到 {dates_test[-1]}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, dates_train, dates_val, dates_test


def normalize_data_three_way(X_train, X_val, X_test, y_train, y_val, y_test):
    """三分法数据归一化 - 只用训练集拟合归一化器"""
    print(f"\n=== 三分法数据归一化 ===")
    
    # 打印归一化前的统计
    print(f"归一化前:")
    print(f"  X_train 形状: {X_train.shape}")
    print(f"  X_val 形状: {X_val.shape}")
    print(f"  X_test 形状: {X_test.shape}")
    print(f"  y_train 形状: {y_train.shape}")
    
    # 确保数据类型正确并计算统计量
    y_train_array = np.asarray(y_train, dtype=np.float64)
    y_mean = np.mean(y_train_array, axis=0)
    y_std = np.std(y_train_array, axis=0)
    print(f"  y_train 统计: 均值={y_mean[:3]}..., 标准差={y_std[:3]}...")  # 只显示前3个以节省空间    
    # 对输入数据进行归一化 (重塑为二维进行归一化)
    n_samples_train, n_timesteps, n_features = X_train.shape
    n_samples_val = X_val.shape[0]
    n_samples_test = X_test.shape[0]
    
    # 重塑为二维
    X_train_2d = X_train.reshape(-1, n_features)
    X_val_2d = X_val.reshape(-1, n_features)
    X_test_2d = X_test.reshape(-1, n_features)
    
    # 使用StandardScaler - 只用训练集拟合
    scaler_X = StandardScaler()
    X_train_scaled_2d = scaler_X.fit_transform(X_train_2d)
    X_val_scaled_2d = scaler_X.transform(X_val_2d)
    X_test_scaled_2d = scaler_X.transform(X_test_2d)
    
    # 重塑回三维
    X_train_scaled = X_train_scaled_2d.reshape(n_samples_train, n_timesteps, n_features)
    X_val_scaled = X_val_scaled_2d.reshape(n_samples_val, n_timesteps, n_features)
    X_test_scaled = X_test_scaled_2d.reshape(n_samples_test, n_timesteps, n_features)
    
    # 对输出数据进行归一化 - 处理多维输出，只用训练集拟合
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_val_scaled = scaler_y.transform(y_val)
    y_test_scaled = scaler_y.transform(y_test)
    
    # 打印归一化后的统计
    print(f"归一化后:")
    print(f"  X_train_scaled 统计: 均值={np.mean(X_train_scaled):.4f}, 标准差={np.std(X_train_scaled):.4f}")
    print(f"  X_val_scaled 统计: 均值={np.mean(X_val_scaled):.4f}, 标准差={np.std(X_val_scaled):.4f}")
    print(f"  X_test_scaled 统计: 均值={np.mean(X_test_scaled):.4f}, 标准差={np.std(X_test_scaled):.4f}")
    print(f"  y_train_scaled 形状: {y_train_scaled.shape}")
    
    # 计算归一化后的统计量
    y_scaled_mean = np.mean(y_train_scaled, axis=0)
    y_scaled_std = np.std(y_train_scaled, axis=0)
    print(f"  y_train_scaled 统计: 均值={y_scaled_mean[:3]}..., 标准差={y_scaled_std[:3]}...")  # 只显示前3个
    
    return (X_train_scaled, X_val_scaled, X_test_scaled, 
            y_train_scaled, y_val_scaled, y_test_scaled, 
            scaler_X, scaler_y)


def main():
    """
    主函数，用于独立运行数据处理流程并进行调试。
    """
    print("--- [开始] 独立运行 data_processor.py ---")

    # 1. 加载和预处理数据
    solar_data, cosmic_data = load_and_check_data()

    # 2. 创建时序序列
    X, y, dates = create_sequences(solar_data, cosmic_data)

    if X.shape[0] == 0:
        print("未能创建任何序列，程序终止。")
        return

    # 3. 划分数据集
    (X_train, X_val, X_test, 
     y_train, y_val, y_test, 
     dates_train, dates_val, dates_test) = split_data_three_way(X, y, dates)

    # 4. 归一化数据
    (X_train_scaled, X_val_scaled, X_test_scaled, 
     y_train_scaled, y_val_scaled, y_test_scaled, 
     scaler_X, scaler_y) = normalize_data_three_way(
        X_train, X_val, X_test, y_train, y_val, y_test
    )

    print("\n--- [完成] 数据处理流程 ---")
    print("最终生成的数据集形状:")
    print(f"  X_train_scaled: {X_train_scaled.shape}")
    print(f"  y_train_scaled: {y_train_scaled.shape}")
    print(f"  X_val_scaled:   {X_val_scaled.shape}")
    print(f"  y_val_scaled:   {y_val_scaled.shape}")
    print(f"  X_test_scaled:  {X_test_scaled.shape}")
    print(f"  y_test_scaled:  {y_test_scaled.shape}")


if __name__ == "__main__":
    main()


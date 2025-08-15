#!/usr/bin/env python3
"""
太阳物理数据提取与整合脚本
从OMNI、SILSO和WSO和OSF数据源中提取特定参数并整合
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
from pathlib import Path

def load_omni_data(file_path):
    """
    加载OMNI数据并提取所需列
    """
    print("正在加载OMNI数据...")
    df = pd.read_csv(file_path)
    
    # 确保日期列是datetime格式
    df['date'] = pd.to_datetime(df['date'])
    
    # 提取所需列
    omni_data = df[['date', 'b_magnitude', 'plasma_speed']].copy()
    
    # 重命名列
    omni_data.rename(columns={
        'b_magnitude': 'HMF',
        'plasma_speed': 'wind_speed'
    }, inplace=True)
    
    print(f"OMNI数据加载完成，共{len(omni_data)}行")
    return omni_data

def load_silso_data(file_path):
    """
    加载SILSO太阳黑子数数据
    """
    print("正在加载SILSO数据...")
    df = pd.read_csv(file_path)
    
    # 确保日期列是datetime格式
    df['date'] = pd.to_datetime(df['date'])
    
    # 提取所需列
    silso_data = df[['date', 'sunspot_num']].copy()
    
    # 重命名列
    silso_data.rename(columns={
        'sunspot_num': 'SSN'
    }, inplace=True)
    
    print(f"SILSO数据加载完成，共{len(silso_data)}行")
    return silso_data

def load_wso_polar_data(file_path):
    """
    加载WSO极性日数据
    """
    print("正在加载WSO极性数据...")
    df = pd.read_csv(file_path)
    
    # 确保日期列是datetime格式
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    elif 'Date' in df.columns:
        df['date'] = pd.to_datetime(df['Date'])
    else:
        # 尝试第一列作为日期
        df['date'] = pd.to_datetime(df.iloc[:, 0])
    
    # 查找极性列（可能的列名）
    polarity_col = None
    for col in df.columns:
        if col.lower() == 'polarity_raw':
            polarity_col = col
            break
    
    if polarity_col is None:
        # 如果没有直接的极性列，尝试从北极和南极数据计算
        north_col = [col for col in df.columns if 'north' in col.lower() and 'gauss' in col.lower()]
        south_col = [col for col in df.columns if 'south' in col.lower() and 'gauss' in col.lower()]
        
        if north_col and south_col:
            # 计算极性
            df['polarity'] = np.where(df[north_col[0]] * df[south_col[0]] < 0, 1, -1)
        else:
            print("警告：无法找到极性数据列")
            df['polarity'] = np.nan
    else:
        df['polarity'] = df[polarity_col]
    
    # 提取所需列
    polar_data = df[['date', 'polarity']].copy()
    
    # 去除无效日期
    polar_data = polar_data.dropna(subset=['date'])
    
    print(f"WSO极性数据加载完成，共{len(polar_data)}行")
    return polar_data

def load_wso_tilt_data(file_path):
    """
    加载WSO倾斜角日数据
    """
    print("正在加载WSO倾斜角数据...")
    df = pd.read_csv(file_path)
    
    # 确保日期列是datetime格式
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    elif 'Date' in df.columns:
        df['date'] = pd.to_datetime(df['Date'])
    else:
        # 尝试第一列作为日期
        df['date'] = pd.to_datetime(df.iloc[:, 0])
    
    # 查找倾斜角列（可能的列名）
    tilt_col = None
    possible_names = ['tilt', 'hcs_tilt', 'tilts', 'angle', 'tilt_angle', 'R_av', 'r_av']
    
    for col in df.columns:
        for name in possible_names:
            if name.lower() in col.lower():
                tilt_col = col
                break
        if tilt_col:
            break
    
    if tilt_col is None:
        print(f"警告：无法找到倾斜角数据列。可用列名：{list(df.columns)}")
        df['HCS_tilt'] = np.nan
    else:
        df['HCS_tilt'] = df[tilt_col]
    
    # 提取所需列
    tilt_data = df[['date', 'HCS_tilt']].copy()
    
    # 去除无效日期
    tilt_data = tilt_data.dropna(subset=['date'])
    
    print(f"WSO倾斜角数据加载完成，共{len(tilt_data)}行")
    return tilt_data

def merge_all_data(omni_data, silso_data, polar_data, tilt_data, start_date='1980-01-01'):
    """
    合并所有数据源，自动确定结束日期为所有数据源的最后共有日期
    """
    print("\n正在合并数据...")
    
    # 设置起始日期
    start_date = pd.Timestamp(start_date)
    
    # 找到所有数据源中最晚的开始日期和最早的结束日期
    data_sources = [
        ('OMNI', omni_data),
        ('SILSO', silso_data), 
        ('WSO Polar', polar_data),
        ('WSO Tilt', tilt_data)
    ]
    
    # 打印各数据源的日期范围
    print("\n各数据源日期范围:")
    latest_start = start_date
    earliest_end = None
    
    for name, data in data_sources:
        if len(data) > 0:
            data_start = data['date'].min()
            data_end = data['date'].max()
            print(f"{name}: {data_start.strftime('%Y-%m-%d')} 至 {data_end.strftime('%Y-%m-%d')}")
            
            # 更新最晚开始日期
            if data_start > latest_start:
                latest_start = data_start
                
            # 更新最早结束日期
            if earliest_end is None or data_end < earliest_end:
                earliest_end = data_end
    
    # 确定实际的时间范围
    actual_start = max(start_date, latest_start)
    actual_end = earliest_end
    
    print(f"\n实际合并时间范围: {actual_start.strftime('%Y-%m-%d')} 至 {actual_end.strftime('%Y-%m-%d')}")
    
    # 筛选日期范围内的数据
    omni_data = omni_data[(omni_data['date'] >= actual_start) & (omni_data['date'] <= actual_end)]
    silso_data = silso_data[(silso_data['date'] >= actual_start) & (silso_data['date'] <= actual_end)]
    polar_data = polar_data[(polar_data['date'] >= actual_start) & (polar_data['date'] <= actual_end)]
    tilt_data = tilt_data[(tilt_data['date'] >= actual_start) & (tilt_data['date'] <= actual_end)]
    
    # 由于所有数据都已经是日数据，直接合并
    # 首先合并OMNI和SILSO数据
    merged_data = pd.merge(omni_data, silso_data, on='date', how='outer')
    
    # 合并极性数据
    merged_data = pd.merge(merged_data, polar_data, on='date', how='left')
    
    # 合并倾斜角数据
    merged_data = pd.merge(merged_data, tilt_data, on='date', how='left')
    
    # 按日期排序
    merged_data = merged_data.sort_values('date')
    
    # 重新排列列的顺序
    merged_data = merged_data[['date', 'HMF', 'wind_speed', 'SSN', 'polarity', 'HCS_tilt']]
    
    print(f"数据合并完成，共{len(merged_data)}行")
    print(f"日期范围：{merged_data['date'].min()} 至 {merged_data['date'].max()}")
    
    return merged_data

def main():
    """主函数"""
    print("=== 太阳物理数据合并工具 ===")
    
    # 创建输出目录
    output_dir = Path("./outputs/cycle_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {output_dir}")
    
    # 设置数据路径（根据实际情况修改）
    base_path = Path("./raw_data")
    
    # 数据文件路径
    omni_file = base_path / "omni" / "omni_daily_1980_2025.csv"
    silso_file = base_path / "silso" / "SN_d_tot_V2.0.csv"
    polar_file = base_path / "wso" / "polar_daily.csv"  # 使用日数据
    tilt_file = base_path / "wso" / "tilts_daily.csv"   # 使用日数据
    
    # 检查文件是否存在
    files_to_check = [
        ("OMNI", omni_file),
        ("SILSO", silso_file),
        ("WSO Polar", polar_file),
        ("WSO Tilt", tilt_file)
    ]
    
    for name, file_path in files_to_check:
        if not file_path.exists():
            print(f"警告：{name}数据文件不存在：{file_path}")
            return
    
    # 加载各个数据源
    omni_data = load_omni_data(omni_file)
    silso_data = load_silso_data(silso_file)
    polar_data = load_wso_polar_data(polar_file)
    tilt_data = load_wso_tilt_data(tilt_file)
    
    # 合并所有数据
    merged_data = merge_all_data(omni_data, silso_data, polar_data, tilt_data)
    
    
    # 截取1985年之后的数据
    start_filter_date = pd.Timestamp('1985-01-01')
    merged_data = merged_data[merged_data['date'] >= start_filter_date]
    print(f"截取1985年后数据，剩余{len(merged_data)}行")
    print(f"过滤后日期范围：{merged_data['date'].min()} 至 {merged_data['date'].max()}")


    # # 保存合并后的数据到指定目录
    # output_file = output_dir / "solar_physics_data_1985_2025.csv"
    # merged_data.to_csv(output_file, index=False)
    # print(f"\n合并后的数据已保存到: {output_file}")
    
    # 插值补全缺失值，使数据连续
    merged_data.interpolate(method='linear', limit_direction='both', inplace=True)

    # 保存插值后的连续数据
    output_file_continuous = output_dir / "solar_physics_data_1985_2025_original.csv"
    merged_data.to_csv(output_file_continuous, index=False)
    print(f"插值后的连续数据已保存到: {output_file_continuous}")
    
    # # 对除了日期和极性列外的其他列进行移动平均
    # cols_to_smooth = ['HMF', 'wind_speed', 'SSN', 'HCS_tilt']
    # merged_data[cols_to_smooth] = merged_data[cols_to_smooth].rolling(window=7, min_periods=1, center=True).mean()
    # print("对数据进行了7天的移动平均平滑处理")

    # 保存插值后的连续数据
    output_file_ma = output_dir / "solar_physics_data_1985_2025.csv"
    merged_data.to_csv(output_file_ma, index=False)
    print(f"插值后的连续数据已保存到: {output_file_ma}")

    # 生成数据摘要报告
    # generate_data_summary_report(merged_data, output_dir / "data_summary_report.txt")
    
    print("\n=== 数据合并完成 ===")
    print(f"所有输出文件已保存到: {output_dir}")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
解析OMNI每日平均数据文件并转换为CSV格式
基于omni2.text文档的格式说明
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def parse_omni_daily_data():
    """解析OMNI daily数据文件"""
    
    print("=" * 60)
    print("解析OMNI每日平均数据...")
    print("=" * 60)
    
    # 定义列名（基于omni2.text文档）
    column_names = [
        'year',           # 1: Year
        'day',            # 2: Decimal Day
        'hour',           # 3: Hour (应该都是0，因为是日平均)
        'bartels_num',    # 4: Bartels rotation number
        'imf_sc_id',      # 5: ID for IMF spacecraft
        'sw_sc_id',       # 6: ID for SW plasma spacecraft
        'imf_pts',        # 7: # of points in IMF averages
        'sw_pts',         # 8: # of points in plasma averages
        'b_magnitude',    # 9: Field Magnitude Average |B|, nT
        'b_mag_vector',   # 10: Magnitude of Average Field Vector, nT
        'b_lat_gse',      # 11: Lat.Angle of Aver. Field Vector, Degrees
        'b_lon_gse',      # 12: Long.Angle of Aver.Field Vector, Degrees
        'bx_gse',         # 13: Bx GSE, nT
        'by_gse',         # 14: By GSE, nT
        'bz_gse',         # 15: Bz GSE, nT
        'by_gsm',         # 16: By GSM, nT
        'bz_gsm',         # 17: Bz GSM, nT
        'sigma_b_mag',    # 18: sigma|B|, nT
        'sigma_b_vec',    # 19: sigma B, nT
        'sigma_bx',       # 20: sigma Bx, nT
        'sigma_by',       # 21: sigma By, nT
        'sigma_bz',       # 22: sigma Bz, nT
        'proton_temp',    # 23: Proton temperature, K
        'proton_density', # 24: Proton Density, N/cm^3
        'plasma_speed',   # 25: Plasma (Flow) speed, km/s
        'flow_lon_angle', # 26: Plasma Flow Long. Angle, Degrees
        'flow_lat_angle', # 27: Plasma Flow Lat. Angle, Degrees
        'na_np_ratio',    # 28: Na/Np ratio
        'flow_pressure',  # 29: Flow Pressure, nPa
        'sigma_temp',     # 30: sigma T, K
        'sigma_density',  # 31: sigma N, N/cm^3
        'sigma_speed',    # 32: sigma V, km/s
        'sigma_phi_v',    # 33: sigma phi V, Degrees
        'sigma_theta_v',  # 34: sigma theta V, Degrees
        'sigma_na_np',    # 35: sigma-Na/Np
        'electric_field', # 36: Electric field, mV/m
        'plasma_beta',    # 37: Plasma beta
        'alfven_mach',    # 38: Alfven mach number
        'kp_index',       # 39: Kp index
        'sunspot_num',    # 40: R - Sunspot number
        'dst_index',      # 41: DST Index, nT
        'ae_index',       # 42: AE-index, nT
        'proton_flux_1',  # 43: Proton flux >1 Mev
        'proton_flux_2',  # 44: Proton flux >2 Mev
        'proton_flux_4',  # 45: Proton flux >4 Mev
        'proton_flux_10', # 46: Proton flux >10 Mev
        'proton_flux_30', # 47: Proton flux >30 Mev
        'proton_flux_60', # 48: Proton flux >60 Mev
        'flag',           # 49: Flag
        'ap_index',       # 50: ap-index, nT
        'f107_index',     # 51: f10.7_index
        'pc_n_index',     # 52: PC(N) index
        'al_index',       # 53: AL-index, nT
        'au_index',       # 54: AU-index, nT
        'magnetosonic_mach' # 55: Magnetosonic mach number
    ]
    
    # 读取数据文件
    data_file = 'omni_01_av.dat'
    
    # 读取所有行
    with open(data_file, 'r') as f:
        lines = f.readlines()
    print(f"读取了 {len(lines)} 行数据")

    # 解析数据
    data_rows = []
    for line_num, line in enumerate(lines):
        values = line.strip().split()
        if len(values) == 0:
            continue
        if len(values) != len(column_names):
            print(f"字段数不一致：第 {line_num+1} 行，字段数 {len(values)}，内容: {values}")
            raise Exception(f"数据格式错误：第 {line_num+1} 行字段数为 {len(values)}，应为 {len(column_names)}")
        year = int(values[0])
        if year < 1980 or year > 2025:
            continue
        data_rows.append(values)
    
    print(f"筛选出 {len(data_rows)} 行有效数据（1980-2025年）")
    
    # 创建DataFrame
    df = pd.DataFrame(data_rows, columns=column_names)
    
    # 数据类型转换
    numeric_columns = [col for col in column_names if col not in ['year', 'day', 'hour']]
    
    # 转换为数值类型
    for col in ['year', 'day', 'hour']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 创建日期列
    df['date'] = pd.to_datetime(df['year'].astype(int).astype(str) + '-01-01') + pd.to_timedelta(df['day'] - 1, unit='D')
    
    # 重新排列列的顺序，将日期放在最前面
    cols = ['date'] + [col for col in df.columns if col != 'date']
    df = df[cols]
    
    # 处理缺失值标记
    fill_values = {
        'b_magnitude': 999.9,
        'plasma_speed': 9999.0,
        'proton_density': 999.9,
        'proton_temp': 9999999.0,
        'bx_gse': 999.9,
        'by_gse': 999.9,
        'bz_gse': 999.9,
        'sunspot_num': 999,
        'dst_index': 99999,
        'ae_index': 9999,
        'kp_index': 99
    }
    
    # 将填充值替换为NaN
    for col, fill_val in fill_values.items():
        if col in df.columns:
            df[col] = df[col].replace(fill_val, np.nan)
    
    # 保存为CSV
    output_file = 'omni_daily_1980_2025_new.csv'
    df.to_csv(output_file, index=False)
    
    print(f"\n数据保存到: {output_file}")
    print(f"数据形状: {df.shape}")
    print(f"时间范围: {df['date'].min()} 到 {df['date'].max()}")
    
    # 显示前几行数据
    print("\n前5行数据:")
    print(df[['date', 'year', 'day', 'b_magnitude', 'plasma_speed', 'proton_density', 'sunspot_num']].head())
    
    # 数据质量检查
    print("\n主要字段的缺失值统计:")
    key_fields = ['b_magnitude', 'plasma_speed', 'proton_density', 'sunspot_num', 'bx_gse', 'by_gse', 'bz_gse']
    for field in key_fields:
        if field in df.columns:
            missing = df[field].isna().sum()
            total = len(df)
            pct = (missing / total) * 100
            print(f"  {field}: {missing}/{total} ({pct:.1f}%)")
    
    return df

if __name__ == "__main__":
    try:
        print("开始执行OMNI数据解析...")
        df = parse_omni_daily_data()
        print("\nOMNI每日数据解析完成！")
    except Exception as e:
        print(f"数据解析失败: {e}")
        import traceback
        traceback.print_exc()

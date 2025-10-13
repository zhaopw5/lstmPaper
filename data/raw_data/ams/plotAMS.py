# 要画的数据：
# /home/phil/Files/lstmPaper/data/raw_data/ams/p-table-BR_new.csv
'''
bartels_rotation_number,rigidity_min GV,rigidity_max GV,proton_flux m^-2sr^-1s^-1GV^-1 m^-2sr^-1s^-1GV^-1,proton_flux_error_statistical m^-2sr^-1s^-1GV^-1,proton_flux_error_timedependent m^-2sr^-1s^-1GV^-1,proton_flux_error_systematic_total m^-2sr^-1s^-1GV^-1,center_date
2426,1.0,1.92,844.3,0.5,1.8,16.2,2011-05-28
2426,1.92,2.97,482.0,0.3,0.8,5.8,2011-05-28
2426,2.97,4.02,262.8,0.2,0.5,2.8,2011-05-28
2426,4.02,4.88,162.2,0.1,0.4,1.7,2011-05-28
2426,4.88,5.9,107.8,0.1,0.3,1.1,2011-05-28
2426,5.9,7.09,70.61,0.05,0.17,0.75,2011-05-28
2426,7.09,8.48,46.02,0.03,0.11,0.49,2011-05-28
2426,8.48,11.0,26.86,0.02,0.05,0.29,2011-05-28
2426,11.0,16.6,11.39,0.01,0.02,0.13,2011-05-28
2426,16.6,22.8,4.306,0.004,0.007,0.05,2011-05-28
...
'''

import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

data_path = "/home/phil/Files/lstmPaper/data/raw_data/ams/p-table-BR_new.csv"
output_dir = "/home/phil/Files/lstmPaper/data/raw_data/ams/plots"
os.makedirs(output_dir, exist_ok=True)

# 读取数据
df = pd.read_csv(data_path)
df['center_date'] = pd.to_datetime(df['center_date'])
df.sort_values(by='center_date', inplace=True)
df['proton_flux m^-2sr^-1s^-1GV^-1'] = df['proton_flux m^-2sr^-1s^-1GV^-1'].astype(float)

# 每一个br周期对应的起始时间保存在：/home/phil/Files/lstmPaper/data/raw_data/bartels_rotation_number.csv
# 内容格式如下：
'''
bartels_rotation_number,start_date,end_date
2426,2011-05-15,2011-06-10
2427,2011-06-11,2011-07-07
2428,2011-07-08,2011-08-03
2429,2011-08-04,2011-08-30
2430,2011-08-31,2011-09-26
2431,2011-09-27,2011-10-23
2432,2011-10-24,2011-11-19
2433,2011-11-20,2011-12-16
2434,2011-12-17,2012-01-12
...
'''

# 可调参数：前后多少个BR周期
br_range = 4  # 可以修改这个值来调整时间范围

# 查找日期2021 11 04所在的br周期，并画出该周期以及前后2个br周期的质子通量随时间变化的曲线
br_data_path = "/home/phil/Files/lstmPaper/data/raw_data/bartels_rotation_number.csv"
br_df = pd.read_csv(br_data_path)
br_df['start_date'] = pd.to_datetime(br_df['start_date'])
br_df['end_date'] = pd.to_datetime(br_df['end_date'])

target_date = pd.to_datetime("2021-11-04")
target_br = br_df[(br_df['start_date'] <= target_date) & (br_df['end_date'] >= target_date)]
if target_br.empty:
    raise ValueError("目标日期不在任何Bartels周期内")
target_br_number = target_br['bartels_rotation_number'].values[0]
br_numbers_to_plot = [target_br_number + i for i in range(-br_range, br_range + 1)]
br_periods = br_df[br_df['bartels_rotation_number'].isin(br_numbers_to_plot)]
if br_periods.empty:
    raise ValueError("没有找到对应的Bartels周期数据")
br_periods = br_periods.sort_values(by='bartels_rotation_number')
print("要画的br周期：", br_periods)

# 计算误差：统计误差和时间相关系统误差的平方和的平方根
df['total_error'] = np.sqrt(df['proton_flux_error_statistical m^-2sr^-1s^-1GV^-1']**2 + 
                           df['proton_flux_error_timedependent m^-2sr^-1s^-1GV^-1']**2)

# 找到所有刚度区间
rigidity_groups = df.groupby(['rigidity_min GV', 'rigidity_max GV'])

for (rig_min, rig_max), group in rigidity_groups:
    fig, ax = plt.subplots(figsize=(12, 6))
    color = 'blue'  # 统一颜色为蓝色
    for i, (index, row) in enumerate(br_periods.iterrows()):
        br_number = row['bartels_rotation_number']
        start_date = row['start_date']
        end_date = row['end_date']
        # 只选当前刚度区间的数据
        br_data = group[(group['center_date'] >= start_date) & (group['center_date'] <= end_date)]
        if br_data.empty:
            print(f"警告：Bartels周期 {br_number} 内刚度区间 {rig_min}-{rig_max} GV 没有数据，跳过")
            continue
        # 画数据点和误差棒
        ax.errorbar(br_data['center_date'], br_data['proton_flux m^-2sr^-1s^-1GV^-1'], 
                   yerr=br_data['total_error'], marker='o', linestyle='-', color=color, capsize=3)
        # 画虚线
        ax.axvline(x=start_date, color=color, linestyle='--', linewidth=1)
        # 使用相对坐标标注时间在虚线旁边
        ax.text(start_date, 0.1, start_date.strftime('%Y-%m-%d'), color=color, ha='left', va='bottom', 
               fontsize=9, transform=ax.get_xaxis_transform())

    ax.set_xlabel('date')
    ax.set_ylabel('proton flux (m$^{-2}$sr$^{-1}$s$^{-1}$GV$^{-1}$)')
    ax.set_title(f'Proton Flux for Rigidity {rig_min}-{rig_max} GV')
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.tight_layout()
    output_file = os.path.join(output_dir, f'proton_flux_{rig_min}_{rig_max}_GV_br_{br_numbers_to_plot[0]}_to_{br_numbers_to_plot[-1]}.png')
    plt.savefig(output_file)
    print(f"刚度区间 {rig_min}-{rig_max} GV 图已保存到 {output_file}")

import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


# Read CSV file
data = pd.read_csv('tilts.csv')

# Convert 'Start Date' including hour info to datetime
data['start_date'] = pd.to_datetime(data['Start Date'].str.replace('h', ''), format="%Y-%m-%d %H")

# Calculate center date between adjacent start dates
data['next_start'] = data['start_date'].shift(-1)
data['date'] = data['start_date'] + (data['next_start'] - data['start_date']) / 2

# For the last row, use the same interval as previous
last_interval = data['date'].iloc[-2] - data['start_date'].iloc[-2]
data.loc[data.index[-1], 'date'] = data['start_date'].iloc[-1] + last_interval

print("Using midpoint dates between adjacent Carrington Rotations")
print(f"Example: Start1 {data['start_date'].iloc[0]} Start2 {data['start_date'].iloc[1]} -> Center {data['date'].iloc[0]}")

# # save data:
# data.to_csv('tilts_update.csv', index=False)


# 文件路径
output_file = '/home/phil/Files/lstmPaper/data/raw_data/wso/tilts_daily.csv'
plot_file = '/home/phil/Files/lstmPaper/data/raw_data/wso/tilts_comparison.png'

print("=== 开始处理tilts数据 ===")

# 1. 使用已加载的数据
print("使用已加载的tilts数据")
original_df = data.copy()

# 转换日期列为datetime格式
original_df['start_date'] = pd.to_datetime(original_df['start_date'])
original_df['next_start'] = pd.to_datetime(original_df['next_start'])
original_df['date'] = pd.to_datetime(original_df['date'])

print(f"数据加载完成，共 {len(original_df)} 个Carr Rotation周期")
print(f"时间范围: {original_df['start_date'].min()} 到 {original_df['next_start'].max()}")

# 2. 准备插值参数
start_date = original_df['start_date'].min().date()
end_date = original_df['next_start'].dropna().max().date()

print(f"插值时间范围: {start_date} 到 {end_date}")

# 创建日期序列
daily_dates = pd.date_range(start=start_date, end=end_date, freq='D')
print(f"将生成 {len(daily_dates)} 天的日分辨率数据")

# 3. 执行线性插值
# 准备插值数据 - 使用Carr Rotation的中心日期作为插值点
carr_dates = original_df['date'].dropna()
carr_timestamps = carr_dates.astype('int64') // 10**9  # 转换为时间戳

# 创建日分辨率时间戳
daily_timestamps = daily_dates.astype('int64') // 10**9

# 要插值的列
columns_to_interpolate = ['R_av', 'R_n', 'R_s', 'L_av', 'L_n', 'L_s']

# 存储插值结果
daily_data = {'date': daily_dates}

# 对每列进行插值
for col in columns_to_interpolate:
    print(f"正在插值列: {col}")
    
    # 获取有效数据
    valid_data = original_df[col].dropna()
    valid_dates = original_df.loc[valid_data.index, 'date'].dropna()
    valid_timestamps = valid_dates.astype('int64') // 10**9
    
    if len(valid_data) < 2:
        print(f"警告: {col} 列的有效数据点少于2个，跳过插值")
        daily_data[col] = np.nan
        continue
    
    # 创建插值函数
    interp_func = interp1d(
        valid_timestamps, 
        valid_data, 
        kind='linear',
        bounds_error=False,
        fill_value='extrapolate'
    )
    
    # 执行插值
    daily_values = interp_func(daily_timestamps)
    daily_data[col] = daily_values

# 创建结果DataFrame
daily_df = pd.DataFrame(daily_data)
print(f"插值完成，生成了 {len(daily_df)} 天的数据")

# 4. 保存日分辨率数据
print(f"正在保存数据到: {output_file}")

# 添加一些有用的列
daily_df['year'] = daily_df['date'].dt.year
daily_df['month'] = daily_df['date'].dt.month
daily_df['day'] = daily_df['date'].dt.day
daily_df['doy'] = daily_df['date'].dt.dayofyear  # day of year

# 保存为CSV
daily_df.to_csv(output_file, index=False, float_format='%.2f')
print(f"数据保存完成")

# 5. 绘制对比图 - 只绘制2010-2019年的R_av
print("正在绘制对比图...")

# 筛选时间段：2010-2019
start_plot = pd.to_datetime('2010-01-01')
end_plot = pd.to_datetime('2019-12-31')

# 筛选原始数据
original_filtered = original_df[(original_df['date'] >= start_plot) & (original_df['date'] <= end_plot)]

# 筛选日分辨率数据
daily_filtered = daily_df[(daily_df['date'] >= start_plot) & (daily_df['date'] <= end_plot)]

fig, ax = plt.subplots(1, 1, figsize=(12, 6))
fig.suptitle('Tilts Data: Carr Rotation vs Daily Resolution (2010-2019)', fontsize=16)

# 只绘制R_av
col = 'R_av'
title = 'Right Average Tilt'

# 绘制原始数据点
ax.scatter(original_filtered['date'], original_filtered[col], 
          color='red', s=10, alpha=0.7, label='Carr Rotation', zorder=2)

# 绘制插值结果
ax.scatter(daily_filtered['date'], daily_filtered[col], 
       color='blue', s=1, alpha=0.8, label='Daily Interpolated', zorder=1)

ax.set_title(title)
ax.set_ylabel('Tilt (degrees)')
ax.set_xlabel('Date')
ax.grid(True, alpha=0.3)
ax.legend()

# 设置x轴日期格式
ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
print(f"对比图保存到: {plot_file}")
plt.show()

# 6. 显示统计信息
print("\n=== 数据统计信息 ===")
print(f"原始数据点数: {len(original_df)}")
print(f"日分辨率数据点数: {len(daily_df)}")
print(f"时间跨度: {(daily_df['date'].max() - daily_df['date'].min()).days + 1} 天")

print("\n=== 2010-2019年R_av数据统计 ===")
if 'R_av' in daily_filtered.columns:
    min_val = daily_filtered['R_av'].min()
    max_val = daily_filtered['R_av'].max()
    mean_val = daily_filtered['R_av'].mean()
    print(f"R_av (2010-2019): [{min_val:.2f}, {max_val:.2f}], 平均值: {mean_val:.2f}")

print("\n=== 各列数据范围 ===")
for col in ['R_av', 'R_n', 'R_s', 'L_av', 'L_n', 'L_s']:
    if col in daily_df.columns:
        min_val = daily_df[col].min()
        max_val = daily_df[col].max()
        mean_val = daily_df[col].mean()
        print(f"{col}: [{min_val:.2f}, {max_val:.2f}], 平均值: {mean_val:.2f}")

print(f"\n处理完成！日分辨率数据已保存到: {output_file}")
print("=== 处理结束 ===")
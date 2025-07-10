# 处理polar.csv文件并计算极性（+1/-1格式）
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 配置变量
USE_FILTERED = True  # True使用过滤数据，False使用原始数据
START_YEAR = 1980
END_YEAR = 2024

# 文件路径
input_file = 'polar.csv'
output_file = 'polar_daily.csv'

print("开始读取文件...")
data = pd.read_csv(input_file)
print("文件读取完成，总共", len(data), "行数据")

# 处理日期列
print("正在处理日期...")
# 将日期字符串转换为标准格式
date_list = []
for i in range(len(data)):
    date_str = str(data['Date_Time'][i])
    if 'XXX' not in date_str:
        # 替换格式: 1976:05:31_21h:07m:13s -> 1976-05-31
        date_str = date_str.replace('h:', ':')
        date_str = date_str.replace('m:', ':')
        date_str = date_str.replace('s', '')
        try:
            date = pd.to_datetime(date_str, format='%Y:%m:%d_%H:%M:%S')
            date_list.append(date)
        except:
            date_list.append(None)
    else:
        date_list.append(None)

data['date'] = date_list

# 移除无效日期的行
data_clean = data[data['date'].notna()]
data_clean = data_clean.reset_index(drop=True)
print("移除无效日期后，剩余", len(data_clean), "行数据")

# 处理数字列，移除字母后缀
print("正在处理数字列...")
north_values = []
south_values = []

# 根据配置选择列名
if USE_FILTERED:
    north_col = 'North_Filtered_Gauss'
    south_col = 'South_Filtered_Gauss'
    north_suffix = 'Nf'
    south_suffix = 'Sf'
    print("使用过滤后的数据")
else:
    north_col = 'North_Pole_Field_Gauss'
    south_col = 'South_Pole_Field_Gauss'
    north_suffix = 'N'
    south_suffix = 'S'
    print("使用原始数据")

for i in range(len(data_clean)):
    # 处理北极数据
    north_str = str(data_clean[north_col].iloc[i])
    if 'XXX' in north_str:
        north_values.append(np.nan)
    else:
        north_clean = north_str.replace(north_suffix, '')
        try:
            north_values.append(float(north_clean))
        except:
            north_values.append(np.nan)
    
    # 处理南极数据
    south_str = str(data_clean[south_col].iloc[i])
    if 'XXX' in south_str:
        south_values.append(np.nan)
    else:
        south_clean = south_str.replace(south_suffix, '')
        try:
            south_values.append(float(south_clean))
        except:
            south_values.append(np.nan)

data_clean['north_field'] = north_values
data_clean['south_field'] = south_values

# 按日期排序
data_clean = data_clean.sort_values('date')
data_clean = data_clean.reset_index(drop=True)

# 移除无效数据的行
valid_data = data_clean
# 只保留北极和南极数据都有效的行
valid_indices = []
for i in range(len(valid_data)):
    north_val = valid_data['north_field'].iloc[i]
    south_val = valid_data['south_field'].iloc[i]
    if not pd.isna(north_val) and not pd.isna(south_val):
        valid_indices.append(i)

valid_data = valid_data.iloc[valid_indices]
valid_data = valid_data.reset_index(drop=True)
print("有效数据行数:", len(valid_data))

# 筛选时间范围
print("筛选时间范围从", START_YEAR, "到", END_YEAR)
filtered_data = []
for i in range(len(valid_data)):
    year = valid_data['date'].iloc[i].year
    if year >= START_YEAR and year <= END_YEAR:
        filtered_data.append(i)

valid_data = valid_data.iloc[filtered_data]
valid_data = valid_data.reset_index(drop=True)
print("时间筛选后的数据行数:", len(valid_data))

# 创建每日时间序列
start_date = valid_data['date'].min().date()
end_date = valid_data['date'].max().date()
print("数据时间范围:", start_date, "到", end_date)

# 生成完整的日期列表（中午12点）
all_dates = pd.date_range(start=start_date, end=end_date, freq='D') + pd.Timedelta(hours=12)
print("生成每日时间序列（中午12点），总共", len(all_dates), "天")

# 准备插值
print("开始插值...")
# 设置日期为索引
valid_data = valid_data.set_index('date')

# 重采样为每日数据并插值（基准时间为中午12点）
north_daily = valid_data['north_field'].resample('D', origin='start_day', offset='12H').mean()
south_daily = valid_data['south_field'].resample('D', origin='start_day', offset='12H').mean()

# 线性插值填充缺失值
north_daily = north_daily.interpolate(method='linear')
south_daily = south_daily.interpolate(method='linear')

# 确保完整时间范围覆盖
north_daily = north_daily.reindex(all_dates)
south_daily = south_daily.reindex(all_dates)

# 再次插值填充边界
north_daily = north_daily.interpolate(method='linear')
south_daily = south_daily.interpolate(method='linear')

# Calculate polarity (+1 or -1)
print("Calculating polarity...")
polarity = []
for i in range(len(north_daily)):
    north_val = north_daily.iloc[i]
    south_val = south_daily.iloc[i]
    
    if pd.isna(north_val) or pd.isna(south_val):
        polarity.append(np.nan)
    else:
        # If north is positive and south is negative, polarity is +1
        # If north is negative and south is positive, polarity is -1
        if north_val > 0 and south_val < 0:
            polarity.append(1)
        elif north_val < 0 and south_val > 0:
            polarity.append(-1)
        else:
            # Decide based on field strength magnitude
            if abs(north_val) > abs(south_val):
                polarity.append(1 if north_val > 0 else -1)
            else:
                polarity.append(-1 if south_val > 0 else 1)

# Smooth polarity transitions using sigmoid
print("Smoothing polarity transitions...")
def smooth_polarity_transitions(polarity_list, window_size=100, steepness=0.8):
    """Apply sigmoid smoothing to polarity transitions"""
    smoothed = polarity_list.copy()
    
    # Find transition points
    for i in range(1, len(polarity_list)):
        if (not pd.isna(polarity_list[i]) and not pd.isna(polarity_list[i-1]) and 
            polarity_list[i] != polarity_list[i-1]):
            
            # Found a transition, apply sigmoid smoothing
            start_idx = max(0, i - window_size//2)
            end_idx = min(len(polarity_list), i + window_size//2)
            
            # Create sigmoid transition
            transition_length = end_idx - start_idx
            x = np.linspace(-6, 6, transition_length)
            sigmoid = 1 / (1 + np.exp(-steepness * x))
            
            # Scale sigmoid to match polarity values
            from_val = polarity_list[i-1]
            to_val = polarity_list[i]
            scaled_sigmoid = from_val + (to_val - from_val) * sigmoid
            
            # Apply smoothing
            for j, idx in enumerate(range(start_idx, end_idx)):
                if not pd.isna(polarity_list[idx]):
                    smoothed[idx] = scaled_sigmoid[j]
    
    return smoothed

# Apply smoothing to polarity
smoothed_polarity = smooth_polarity_transitions(polarity)

# Create result dataframe
result = pd.DataFrame()
result['date'] = all_dates.date  # 只保留日期部分，不包含时分秒
result['north_field'] = north_daily.values
result['south_field'] = south_daily.values
result['polarity_raw'] = polarity
result['polarity'] = smoothed_polarity

# Save result
print("Saving result to", output_file)
result.to_csv(output_file, index=False)

print("Processing completed!")
print("Output data shape:", result.shape)
print("Polarity statistics:")
print("Positive polarity (+1):", sum(result['polarity'] == 1), "days")
print("Negative polarity (-1):", sum(result['polarity'] == -1), "days")
print("Missing values:", sum(result['polarity'].isna()), "days")

# Generate plots for checking
print("Starting to plot...")
plt.figure(figsize=(15, 8))

# Convert dates for plotting
plot_dates = pd.to_datetime(result['date'])

# First subplot: North and South pole field strength
plt.subplot(2, 1, 1)
plt.plot(plot_dates, result['north_field'], 'b-', label='North Field', linewidth=1)
plt.plot(plot_dates, result['south_field'], 'r-', label='South Field', linewidth=1)
plt.ylabel('Field Strength (Gauss)')
plt.title('Solar Polar Magnetic Field Strength')
plt.legend()
plt.grid(True, alpha=0.3)

# Second subplot: Polarity
plt.subplot(2, 1, 2)
plt.plot(plot_dates, result['polarity_raw'], 'r-', label='Raw Polarity', linewidth=1, alpha=0.7)
plt.plot(plot_dates, result['polarity'], 'k-', label='Smoothed Polarity', linewidth=2)
plt.ylabel('Polarity')
plt.xlabel('Date')
plt.title('Solar Magnetic Field Polarity (+1/-1)')
plt.ylim(-1.5, 1.5)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('polar_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("Plot saved as polar_analysis.png")
print("First 5 rows of data:")
print(result.head())
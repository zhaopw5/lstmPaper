# pandas resample函数演示脚本
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("=== Pandas resample函数演示 ===\n")

# 1. 创建示例数据 - 每小时的时间序列
print("1. 创建示例数据（每小时数据）")
dates = pd.date_range('2023-01-01', periods=168, freq='h')  # 7天，每小时一个数据点
values = np.random.randn(168) + 10  # 随机数据，均值为10
df = pd.DataFrame({'datetime': dates, 'value': values})
df = df.set_index('datetime')

print(f"原始数据形状: {df.shape}")
print("前5行数据:")
print(df.head())
print()

# 2. 基本的resample操作 - 转换为每日数据
print("2. 基本resample - 每小时数据转为每日数据（取平均值）")
daily_mean = df.resample('D').mean()
print(f"每日平均值数据形状: {daily_mean.shape}")
print("每日平均值:")
print(daily_mean)
print()

# 3. 不同的聚合函数
print("3. 使用不同的聚合函数")
daily_stats = pd.DataFrame({
    'mean': df.resample('D').mean()['value'],
    'max': df.resample('D').max()['value'],
    'min': df.resample('D').min()['value'],
    'sum': df.resample('D').sum()['value'],
    'count': df.resample('D').count()['value']
})
print(daily_stats)
print()

# 4. 不同的频率
print("4. 不同的重采样频率")
print("每2小时（取平均值）:")
two_hourly = df.resample('2h').mean()
print(f"形状: {two_hourly.shape}")
print(two_hourly.head())
print()

print("每12小时（取平均值）:")
twelve_hourly = df.resample('12h').mean()
print(f"形状: {twelve_hourly.shape}")
print(twelve_hourly.head())
print()

# 5. 演示offset参数 - 类似你代码中的例子
print("5. 演示offset参数（类似太阳数据中午12点的例子）")
print("默认resample('D') - 从午夜0点开始:")
default_daily = df.resample('D').mean()
print("时间索引:")
print(default_daily.index)
print()

print("使用offset='12H' - 从中午12点开始:")
noon_daily = df.resample('D', offset='12h').mean()
print("时间索引:")
print(noon_daily.index)
print()

# 6. 处理缺失数据的插值
print("6. 处理缺失数据和插值")
# 创建有缺失值的数据
df_with_gaps = df.copy()
df_with_gaps.iloc[10:15] = np.nan  # 人为创建缺失值
df_with_gaps.iloc[50:55] = np.nan

print("有缺失值的数据重采样:")
daily_with_gaps = df_with_gaps.resample('D').mean()
print("重采样后（有NaN）:")
print(daily_with_gaps)
print()

print("使用插值填充缺失值:")
daily_interpolated = daily_with_gaps.interpolate(method='linear')
print("插值后:")
print(daily_interpolated)
print()

# 7. 可视化对比
print("7. 生成可视化图表...")
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 10), sharex=True)

# 设置字体大小和粗细
plt.rcParams.update({'font.size': 14, 'font.weight': 'bold'})

# 获取x轴范围
x_min = df.index.min()
x_max = df.index.max()

# 原始小时数据
ax1.plot(df.index, df['value'], 'b-', linewidth=0.5, alpha=0.7, label='Original Hourly Data')
ax1.set_title('Original Data (Hourly)', fontsize=16, fontweight='bold')
ax1.set_ylabel('Value', fontsize=14, fontweight='bold')
ax1.legend(fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(x_min, x_max)

# 每日数据（午夜0点）
ax2.plot(default_daily.index, default_daily['value'], 'r-o', linewidth=2, label='Daily Data (Midnight)')
ax2.set_title('Resampled to Daily Data (Midnight Reference)', fontsize=16, fontweight='bold')
ax2.set_ylabel('Value', fontsize=14, fontweight='bold')
ax2.legend(fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(x_min, x_max)

# 每日数据（中午12点）
ax3.plot(noon_daily.index, noon_daily['value'], 'g-s', linewidth=2, label='Daily Data (Noon)')
ax3.set_title('Resampled to Daily Data (Noon Reference)', fontsize=16, fontweight='bold')
ax3.set_ylabel('Value', fontsize=14, fontweight='bold')
ax3.set_xlabel('Date', fontsize=14, fontweight='bold')
ax3.legend(fontsize=12)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(x_min, x_max)

plt.tight_layout()
plt.savefig('resample_demo.png', dpi=300, bbox_inches='tight')
plt.show()

print("可视化图表已保存为 resample_demo.png")

# 8. 常用的频率字符串
print("\n8. 常用的resample频率字符串:")
freq_examples = {
    'h': '每小时',
    'D': '每天',
    'W': '每周',
    'M': '每月',
    'Q': '每季度',
    'A': '每年',
    '2h': '每2小时',
    '15T': '每15分钟（T表示分钟）',
    '30S': '每30秒'
}

for freq, desc in freq_examples.items():
    print(f"'{freq}': {desc}")

print("\n演示完成！")

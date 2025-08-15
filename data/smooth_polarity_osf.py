import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def smooth_polarity_with_sigmoid(
    polarity, 
    transition_days=200, 
    steepness=0.1
):
    """
    用更平滑的sigmoid对极性跳变点做平滑处理
    :param polarity: 原始极性序列(np.ndarray)
    :param transition_days: 平滑窗口长度，越大越平滑
    :param steepness: sigmoid陡峭度，越小越平滑
    :return: 平滑后的极性序列(np.ndarray)
    """
    smoothed = polarity.astype(float).copy()
    jump_points = []
    # 检测跳变点（符号变化）
    for i in range(1, len(polarity)):
        if np.sign(polarity[i]) != np.sign(polarity[i-1]):
            jump_points.append(i)
    # 对每个跳变点应用sigmoid平滑
    for jump_idx in jump_points:
        # 以跳变点为中心的窗口
        half_win = transition_days // 2
        start = max(0, jump_idx - half_win)
        end = min(len(polarity), jump_idx + half_win)
        window = end - start
        before = polarity[jump_idx-1]
        after = polarity[jump_idx]
        # x以跳变点为中心对称
        x = np.arange(window) - (window // 2)
        sigmoid = 1 / (1 + np.exp(-steepness * x))
        # S型插值
        transition = before + (after - before) * sigmoid
        # 只在start:end区间内更新平滑值（避免多个跳变点间覆盖）
        smoothed[start:end] = transition
    return smoothed

# 读取数据
output_dir = '/home/phil/Files/lstmPaper/data/outputs/cycle_analysis/'

df = pd.read_csv(f'{output_dir}solar_physics_data_1985_2025_cycle_prediction_osf.csv')

# 进行sigmoid极性平滑
smoothed_polarity = smooth_polarity_with_sigmoid(
    df['polarity'].values, 
    transition_days=900, 
    steepness=0.02
)

# 将平滑结果直接覆盖原polarity列
df['polarity'] = smoothed_polarity

# 保存
df.to_csv(f'{output_dir}solar_physics_data_1985_2025_cycle_prediction_smoothed_osf.csv', index=False)

print("极性平滑处理完成！")

# 绘图部分
# 指定画图顺序
parameter_cols = ['HMF', 'wind_speed', 'HCS_tilt', 'polarity', 'SSN', 'daily_OSF']
parameter_cols = [col for col in parameter_cols if col in df.columns]
n = len(parameter_cols)
fig, axes = plt.subplots(n, 1, figsize=(16, 15), sharex=True)
plt.rcParams.update({'font.size': 15, 'font.weight': 'bold'})
# 确保日期为datetime类型
if not np.issubdtype(df['date'].dtype, np.datetime64):
    df['date'] = pd.to_datetime(df['date'])
# 分割日期
split_date = pd.Timestamp('2019-10-29')
for ax, col in zip(axes, parameter_cols):
    # 分别画两段数据
    before_mask = df['date'] < split_date
    after_mask = df['date'] >= split_date
    ax.plot(df.loc[before_mask, 'date'], df.loc[before_mask, col], color='green', label=f'Observation')
    ax.plot(df.loc[after_mask, 'date'], df.loc[after_mask, col], color='orange', label=f'Theory Prediction')
    ax.set_ylabel(col, fontsize=12, fontweight='bold')
    ax.axhline(0, color='gray', linestyle=':', linewidth=1)
    ax.set_xlim(pd.Timestamp('2010-05-20'), pd.Timestamp('2032-01-01'))
    # 标出 NaN 值
    nan_dates = df.loc[df[col].isna(), 'date']
    if not nan_dates.empty:
        ymin, ymax = ax.get_ylim()
        ax.vlines(nan_dates, ymin, ymax, color='red', alpha=0.1, linewidth=1)
    ax.legend()
    ax.tick_params(axis='both', labelsize=12)
plt.xlabel('Date', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}all_parameters.png', dpi=500)
# plt.show()
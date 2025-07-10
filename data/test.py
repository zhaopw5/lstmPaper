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
df = pd.read_csv('/home/phil/Files/lstmPaper/data/outputs/cycle_analysis/solar_physics_data_1985_2025_cycle_prediction.csv')

# 进行sigmoid极性平滑
smoothed_polarity = smooth_polarity_with_sigmoid(
    df['polarity'].values, 
    transition_days=900, 
    steepness=0.02
)

# 将平滑结果加回DataFrame
df['polarity_smoothed'] = smoothed_polarity

# 保存
df.to_csv('solar_physics_data_1985_2025_cycle_prediction_smoothed.csv', index=False)

print("极性平滑处理完成！")

# ==================
# 绘图部分
# ==================

# 时间列转为datetime
if not np.issubdtype(df['date'].dtype, np.datetime64):
    df['date'] = pd.to_datetime(df['date'])

plt.figure(figsize=(16, 6))
plt.plot(df['date'], df['polarity'], label='Original Polarity', alpha=0.5, linestyle='--', color='red')
plt.plot(df['date'], df['polarity_smoothed'], label='Smoothed Polarity (Sigmoid)', color='blue', linewidth=2)
plt.axhline(0, color='gray', linestyle=':', linewidth=1)
plt.xlabel('Date')
plt.ylabel('Polarity')
plt.title('Solar Polarity Smoothing with Sigmoid')
plt.legend()
plt.tight_layout()
plt.grid(True, alpha=0.3)
plt.savefig('solar_polarity_sigmoid_smoothing.png', dpi=300)
plt.show()
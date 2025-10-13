# Helium Flux Completion Project (pr2he)

## 概述

用于补全 AMS-02 实验中 Helium 通量的缺失数据。通过利用 Proton 与 Helium 通量之间的线性关系，结合机器学习（ML）模型预测的连续 Proton 数据，实现对 Helium 通量缺失时段的重建。

## 方法原理

### 步骤

对于 AMS 观测值缺失的时间段，采用以下方法补全：

1. **时间分段拟合**：由于不同时间段 Proton-Helium 的线性关系不同，将整个观测周期划分为多个时间段，分别线性拟合。

2. **缺失类型区分**：
   - **NODATA 类型**：普通数据缺失，使用 ML 模型预测的 Proton 通量，结合线性拟合关系计算对应的 Helium 通量
   - **SEP 类型**（Solar Energetic Particle）：在该时段内其他刚度区间有观测数据，但当前刚度区间缺失。采用时间线性插值方法补齐，而非 ML 预测


```
输入数据
├── AMS 观测数据 (Proton & Helium)[PROTON_CSV,HELIUM_CSV]
├── ML 模型预测数据 (Proton)[ML_CSV]
└── 总误差数据（最大误差） [ML_ERR_CSV]
         ↓
时间分段 & 刚度分bin
         ↓
对每个 (时间段, 刚度区间) 观测值组合：
├── 提取观测数据点
├── 建立 Proton-Helium 线性拟合
├── 识别缺失日期类型 (SEP/NODATA)
├── SEP: 线性插值补全
└── NODATA: ML Proton → 拟合关系 → Helium

         ↓
输出完整 Helium 数据集
```

## 输入数据

### 数据文件路径

```python
PROTON_CSV = '/home/zpw/Files/lstmPaper/data/raw_data/ams/proton.csv'
HELIUM_CSV = '/home/zpw/Files/lstmPaper/data/raw_data/ams/helium.csv'
ML_CSV     = '/home/zpw/Files/AMS_NeutronMonitors/AMS/paperPlots/data/lightning_logs/version_3/2011-01-01-2024-07-31_pred_ams_updated.csv'
ML_ERR_CSV = '/home/zpw/Files/AMS_NeutronMonitors/AMS/paperPlots/hist/maximum_error_data.csv'
```

### 数据格式说明

- **PROTON_CSV / HELIUM_CSV**: AMS-02 观测数据
  - 列: `date YYYY-MM-DD`, `rigidity_min GV`, `rigidity_max GV`, `flux`, `statistical_error`, `timedependent_error`
  
- **ML_CSV**: 机器学习模型预测的 Proton 通量
  - 列: `date`, 各刚度区间列（列名为数字字符串，如 '5', '6', ...）
  
- **ML_ERR_CSV**: ML 模型的总误差
  - 列: `rig_min`, `rig_max`, `std_dev`（相对标准差）

## 时间分段

分析周期划分为 10 个时间段，覆盖 2011-05-20 至 2019-10-29：

```python
TIME_SEGMENTS = [
    ('2011-05-20', '2011-12-31'),  # 太阳活动上升期
    ('2012-01-01', '2012-12-31'),
    ('2013-01-01', '2013-12-31'),
    ('2014-01-01', '2014-06-30'),  # 太阳活动极大期前半段
    ('2014-07-01', '2015-05-01'),  # 太阳活动极大期后半段
    ('2015-05-01', '2015-12-31'),  # 下降期开始
    ('2016-01-01', '2016-12-31'),
    ('2017-01-01', '2017-12-31'),
    ('2018-01-01', '2018-12-31'),
    ('2019-01-01', '2019-12-29'),  # 太阳活动极小期
]
```

**分段依据**：太阳活动周期的不同阶段 Proton-Helium 线性关系不同。

## 刚度区间

使用 26 个刚度区间，范围从 1.71 GV 到 100 GV：

```python
rig_bin = [1.71, 1.92, 2.15, 2.4, 2.67, 2.97, 3.29, 3.64, 4.02, 4.43, 
           4.88, 5.37, 5.9, 6.47, 7.09, 7.76, 8.48, 9.26, 10.1, 11, 
           13, 16.6, 22.8, 33.5, 48.5, 69.7, 100]
```

## 输出结果

### 1. CSV 文件

**文件名**: `helium_completed_2011-05-20_2019-10-29.csv`

**位置**: `./results_root/`

**列结构**:
- `date YYYY-MM-DD`: 日期
- `rigidity_min GV`: 刚度区间下限
- `rigidity_max GV`: 刚度区间上限
- `helium_flux m^-2sr^-1s^-1GV^-1`: Helium 通量
- `SEPorNODATA`: 数据来源标签
  - `OBSERVED`: 原始 AMS 观测数据
  - `SEP`: 线性插值补全（SEP 事件期间）
  - `NODATA`: ML 预测补全（普通缺失）

### 2. 图表

**目录**: `./results_root/`

**命名规则**: `plot_{start_date}_{end_date}_bin{n}_{rig_min}_{rig_max}_GV.png`

**图表布局** (2×2):

```
┌─────────────────────┬─────────────────────┐
│  ① Proton 时间序列   │  ② He vs P 拟合图    │
│  (AMS vs ML+误差棒) │                     │
├─────────────────────┼─────────────────────┤
│  ③ Helium 观测时序   │  ④ Helium 完整时序   │
│  (仅 AMS 观测点)    │  (观测+插值+预测)    │
└─────────────────────┴─────────────────────┘
```

**子图说明**:
- **左上①**: Proton 通量时间序列，蓝色点为 AMS 观测，红色点为 ML 预测（带误差棒）
- **右上②**: Helium vs Proton 散点图及线性拟合直线
- **左下③**: 仅显示 AMS 观测的 Helium 通量（带误差棒）
- **右下④**: 完整的 Helium 时间序列
  - 蓝色圆点: 观测数据
  - 紫色三角: SEP 插值数据
  - 红色方块: ML 预测数据


## 一些细节


### 1. SEP 识别逻辑

若某日期在当前刚度区间缺失 Helium 数据，但在同一天的其他刚度区间（Proton 或 Helium）存在观测数据，则判定为 SEP 类型缺失。

### 2. 误差处理

- AMS 误差：统计误差与时间相关误差的平方和开方
- ML 误差：从 `ML_ERR_CSV` 读取相对误差，绝对误差 = 预测值 × 相对误差



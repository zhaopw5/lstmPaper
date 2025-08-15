# 合并下面两个csv，时间对齐：
# /home/phil/Files/lstmPaper/data/outputs/cycle_analysis/solar_physics_data_1985_2025_cycle_prediction_smoothed.csv
'''部分内容示例：
date,HMF,wind_speed,SSN,polarity,HCS_tilt
1985-01-01,6.2,701.0,0.0,-1.0,11.35
1985-01-02,5.7,650.0,0.0,-1.0,11.25
1985-01-03,5.5,551.0,0.0,-1.0,11.15
1985-01-04,5.3,452.0,0.0,-1.0,11.06
...
'''
# /home/phil/Files/lstmPaper/data/raw_data/osf/osf_data.csv
'''部分内容示例：
date,daily_OSF
2010/05/20,-0.00049274502564744
2010/05/21,-0.0008489510559834529
2010/05/22,0.00020167137457164925
2010/05/23,-0.00017751343472552825
...
'''
import pandas as pd

# 读取两个csv文件
solar_data = pd.read_csv('/home/phil/Files/lstmPaper/data/outputs/cycle_analysis/solar_physics_data_1985_2025_cycle_prediction_smoothed.csv')
osf_data = pd.read_csv('/home/phil/Files/lstmPaper/data/raw_data/osf/osf_data.csv')

# 将日期列都转换为datetime格式
solar_data['date'] = pd.to_datetime(solar_data['date'])
osf_data['date'] = pd.to_datetime(osf_data['date'])

# 按日期升序排序（merge_asof要求）
solar_data = solar_data.sort_values('date')
osf_data = osf_data.sort_values('date')

# 合并数据
merged_data = pd.merge_asof(solar_data, osf_data, on='date')

# 保存
merged_data.to_csv('/home/phil/Files/lstmPaper/data/outputs/cycle_analysis/all_solar_data.csv', index=False)

# plot osf:
import pandas as pd
import matplotlib.pyplot as plt


osf_data = pd.read_csv('/home/phil/Files/lstmPaper/data/raw_data/osf/osf_data_1981_2025.csv')
# /home/phil/Files/lstmPaper/data/raw_data/osf/osf_data.csv
'''部分内容示例：
date,daily_OSF
2010/05/20,-0.00049274502564744
2010/05/21,-0.0008489510559834529
2010/05/22,0.00020167137457164925
2010/05/23,-0.00017751343472552825
...
'''
# 将日期列都转换为datetime格式
osf_data['date'] = pd.to_datetime(osf_data['date'])

# # set start date to cut the data:
# start_date = '1996-01-01'
# end_date = '2010-12-31'
# mask = (osf_data['date'] >= start_date) & (osf_data['date'] <= end_date)
# osf_data = osf_data.loc[mask]

# 对数据做27天移动平均：
# osf_data['daily_OSF'] = osf_data['daily_OSF'].rolling(window=27).mean()
# 对大于50的值用nan替换：
osf_data.loc[osf_data['daily_OSF'] > 50, 'daily_OSF'] = None
# save the processed data:
osf_data.to_csv('/home/phil/Files/lstmPaper/data/raw_data/osf/osf_data_processed.csv', index=False)

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(osf_data['date'], osf_data['daily_OSF'], label='Daily OSF', color='tab:blue')
ax.set_xlabel('Date')
ax.set_ylabel('Daily OSF')
ax.set_title('Daily OSF Over Time')
ax.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
fig.savefig('/home/phil/Files/lstmPaper/data/raw_data/osf/osf_plot.png', dpi=300)

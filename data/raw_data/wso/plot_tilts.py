import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv('/home/phil/Files/lstmPaper/data/raw_data/wso/tilts_daily.csv')
'''
date,R_av,R_n,R_s,L_av,L_n,L_s,year,month,day,doy
1976-05-27,7.11,4.80,-9.63,14.08,12.50,-15.87,1976,5,27,148
1976-05-28,7.09,4.80,-9.58,14.11,12.58,-15.84,1976,5,28,149
1976-05-29,7.07,4.79,-9.54,14.14,12.66,-15.81,1976,5,29,150
1976-05-30,7.05,4.78,-9.50,14.17,12.73,-15.79,1976,5,30,151
1976-05-31,7.03,4.78,-9.45,14.20,12.81,-15.76,1976,5,31,152
...
'''

# plot R_av,R_n,R_s,L_av,L_n,L_s
# set a date:2019-10-29 beforeuse green, after use orange
fig, ax = plt.subplots(figsize=(12, 6))
split_date = pd.Timestamp('2019-10-29')
df['date'] = pd.to_datetime(df['date'])
before_mask = df['date'] < split_date
after_mask = df['date'] >= split_date
ax.plot(df.loc[before_mask, 'date'], df.loc[before_mask, 'R_av'], color='C1', label='R_av')
ax.plot(df.loc[before_mask, 'date'], df.loc[before_mask, 'R_n'], color='C2',  label='R_n')
# ax.plot(df.loc[before_mask, 'date'], df.loc[before_mask, 'R_s'], color='C3', linestyle=':', label='R_s')
# ax.plot(df.loc[after_mask, 'date'], df.loc[after_mask, 'R_av'], color='grey', label='R_av')
# ax.plot(df.loc[after_mask, 'date'], df.loc[after_mask, 'R_n'], color='grey', linestyle='--', label='R_n')
# ax.plot(df.loc[after_mask, 'date'], df.loc[after_mask, 'R_s'], color='grey', linestyle=':', label='R_s')
ax.plot(df.loc[before_mask, 'date'], df.loc[before_mask, 'L_av'], color='C7', label='L_av')
ax.plot(df.loc[before_mask, 'date'], df.loc[before_mask, 'L_n'], color='C8', label='L_n')
# ax.plot(df.loc[before_mask, 'date'], df.loc[before_mask, 'L_s'], color='C9', linestyle=':', label='L_s')
# ax.plot(df.loc[after_mask, 'date'], df.loc[after_mask, 'L_av'], color='grey', label='L_av')
# ax.plot(df.loc[after_mask, 'date'], df.loc[after_mask, 'L_n'], color='grey', linestyle='--', label='L_n')
# ax.plot(df.loc[after_mask, 'date'], df.loc[after_mask, 'L_s'], color='grey', linestyle=':', label='L_s')
ax.axvline(split_date, color='gray', linestyle='--', label='Split Date')
ax.set_xlim(pd.Timestamp('2010-05-20'), pd.Timestamp('2020-01-01'))
ax.set_xlabel('Date')
ax.set_ylabel('Tilt')
ax.set_title('WSO Daily Tilts')
ax.axvline(split_date, color='gray', linestyle='--', label='Split Date')
ax.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('/home/phil/Files/lstmPaper/data/raw_data/wso/wso_tilts.png', dpi=300)
plt.show()
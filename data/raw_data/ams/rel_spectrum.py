import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime

df = pd.read_csv('/home/phil/Files/lstmPaper/data/raw_data/ams/proton.csv')

'''示例：
date YYYY-MM-DD,rigidity_min GV,rigidity_max GV,proton_flux m^-2sr^-1s^-1GV^-1,proton_flux_error_statistical m^-2sr^-1s^-1GV^-1,proton_flux_error_timedependent m^-2sr^-1s^-1GV^-1,proton_flux_error_systematic_total m^-2sr^-1s^-1GV^-1
2011-05-20,1.00,1.16,9.998e+2,0.157e+2,0.100e+2,0.293e+2
2011-05-20,1.16,1.33,9.749e+2,0.075e+2,0.071e+2,0.222e+2
2011-05-20,1.33,1.51,9.144e+2,0.067e+2,0.050e+2,0.171e+2
2011-05-20,1.51,1.71,8.404e+2,0.058e+2,0.038e+2,0.135e+2
2011-05-20,1.71,1.92,7.394e+2,0.049e+2,0.031e+2,0.107e+2
2011-05-20,1.92,2.15,6.302e+2,0.041e+2,0.025e+2,0.084e+2
2011-05-20,2.15,2.40,5.489e+2,0.036e+2,0.022e+2,0.069e+2
'''

rig_bin_edges = [1,1.16,1.33,1.51,1.71,1.92,2.15,2.4,2.67,2.97,
			3.29,3.64,4.02,4.43,4.88,5.37,5.9,6.47,7.09,7.76,
			8.48,9.26,10.1,11,13,16.6,22.8,33.5,48.5,69.7,100]

# set start date and end date to cut the data:
start_date = datetime.datetime(2011,5,20)
end_date = datetime.datetime(2017,5,26)
df['date YYYY-MM-DD'] = pd.to_datetime(df['date YYYY-MM-DD'])
df = df[(df['date YYYY-MM-DD'] >= start_date) & (df['date YYYY-MM-DD'] <= end_date)]


# 画不同能段的相对flux=(flux-mean)/std:
plt.figure(figsize=(15,10))
for i in range(len(rig_bin_edges)-25):
    mask = df['rigidity_min GV'] == rig_bin_edges[i]
    subdf = df[mask]
    if not subdf.empty:
        mean_flux = subdf['proton_flux m^-2sr^-1s^-1GV^-1'].mean()
        std_flux = subdf['proton_flux m^-2sr^-1s^-1GV^-1'].std()
        if std_flux != 0:
            rel_flux = (subdf['proton_flux m^-2sr^-1s^-1GV^-1'] - mean_flux) #/ std_flux
            plt.scatter(subdf['date YYYY-MM-DD'], rel_flux, label=f'{rig_bin_edges[i]}-{rig_bin_edges[i+1]} GV')
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.title('Relative Proton Flux by Rigidity')
plt.xlabel('Date')
plt.ylabel('Relative Flux')
plt.legend()
plt.grid()
plt.tight_layout()
# save
plt.savefig('/home/phil/Files/lstmPaper/data/raw_data/ams/relative_proton_flux1.png')
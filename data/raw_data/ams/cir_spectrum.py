import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime

df = pd.read_csv('/home/phil/Files/lstmPaper/data/raw_data/ams/proton.csv')


def get_spectrum_by_date(date):
    mask = (df['date YYYY-MM-DD'] == date)
    filtered_df = df[mask]
    x = (filtered_df['rigidity_min GV'] + filtered_df['rigidity_max GV']) / 2
    y = filtered_df['proton_flux m^-2sr^-1s^-1GV^-1']
    return x, y

rig_bins = [1,1.16,1.33,1.51,1.71,1.92,2.15,2.4,2.67,2.97,
			3.29,3.64,4.02,4.43,4.88,5.37,5.9,6.47,7.09,7.76,
			8.48,9.26,10.1,11,13,16.6,22.8,33.5,48.5,69.7,100]

rig_bins = [16.6,22.8]
    
def plot_lightcurve_and_spectrum(start_date, end_date):
    date_list = pd.date_range(start=start_date, end=end_date)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'width_ratios': [1.5, 1]})
    # 左图：各能段的光变
    rig_colors = plt.cm.tab20(np.linspace(0, 1, len(rig_bins)-1))
    for i in range(len(rig_bins)-1):
        rig_min = rig_bins[i]
        mask = (df['date YYYY-MM-DD'] >= start_date) & (df['date YYYY-MM-DD'] <= end_date) & (df['rigidity_min GV'] == rig_min)
        subdf = df[mask]
        x = pd.to_datetime(subdf['date YYYY-MM-DD'])
        y = subdf['proton_flux m^-2sr^-1s^-1GV^-1']
        axes[0].plot(x, y, marker='o', label=f'{rig_min}-{rig_bins[i+1]} GV', color=rig_colors[i])
    axes[0].set_ylim(y.min(), y.max())
    axes[0].set_yscale('log')
    axes[0].set_title('Light Curve by Rigidity')
    # axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Proton Flux (m^-2 sr^-1 s^-1 GV^-1)')
    axes[0].legend(fontsize='x-small', ncol=2)
    axes[0].tick_params(axis='x', rotation=45)
    # axes[0].grid()
    # 右图：这几天的能谱
    spec_colors = plt.cm.viridis(np.linspace(0, 1, len(date_list)))
    for idx, date in enumerate(date_list):
        date_str = date.strftime('%Y-%m-%d')
        x, y = get_spectrum_by_date(date_str)
        axes[1].plot(x, y, marker='o', label=date_str, color=spec_colors[idx], linewidth=1, markersize=3)
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    axes[1].set_xlim(1e0, 1e2)
    axes[1].set_title(f'Proton Spectrum {start_date} to {end_date}')
    axes[1].set_xlabel('Rigidity (GV)')
    axes[1].set_ylabel('Proton Flux (m^-2 sr^-1 s^-1 GV^-1)')
    axes[1].legend(fontsize='x-small', ncol=2)
    axes[1].grid()
    plt.tight_layout()
    plt.savefig(f'proton_lightcurve_and_spectrum_{start_date}_to_{end_date}.png')
    plt.close()


# 示例调用
# plot_lightcurve_and_spectrum('2015-06-19', '2015-07-07')
# plot_lightcurve_and_spectrum('2013-06-19', '2013-06-30')
# plot_lightcurve_and_spectrum('2016-08-31', '2016-09-10')
# plot_lightcurve_and_spectrum('2017-05-25', '2017-06-10')
plot_lightcurve_and_spectrum('2016-05-01', '2016-05-31')



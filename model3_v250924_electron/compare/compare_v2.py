import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 全局字体与粗细
plt.rcParams.update({
    "axes.titlesize": 18,
    "axes.titleweight": "bold",
    "axes.labelsize": 18,
    "axes.labelweight": "bold",
    "xtick.labelsize": 18,
    "ytick.labelsize": 18
})
'''
daily proton :
date YYYY-MM-DD,rigidity_min GV,rigidity_max GV,proton_flux m^-2sr^-1s^-1GV^-1,proton_flux_error_statistical m^-2sr^-1s^-1GV^-1,proton_flux_error_timedependent m^-2sr^-1s^-1GV^-1,proton_flux_error_systematic_total m^-2sr^-1s^-1GV^-1
2011-05-20,1.00,1.16,9.998e+2,0.157e+2,0.100e+2,0.293e+2
2011-05-20,1.16,1.33,9.749e+2,0.075e+2,0.071e+2,0.222e+2
2011-05-20,1.33,1.51,9.144e+2,0.067e+2,0.050e+2,0.171e+2
2011-05-20,1.51,1.71,8.404e+2,0.058e+2,0.038e+2,0.135e+2
2011-05-20,1.71,1.92,7.394e+2,0.049e+2,0.031e+2,0.107e+2
2011-05-20,1.92,2.15,6.302e+2,0.041e+2,0.025e+2,0.084e+2
2011-05-20,2.15,2.40,5.489e+2,0.036e+2,0.022e+2,0.069e+2
2011-05-20,2.40,2.67,4.628e+2,0.030e+2,0.018e+2,0.056e+2
2011-05-20,2.67,2.97,3.927e+2,0.025e+2,0.015e+2,0.046e+2
2011-05-20,2.97,3.29,3.278e+2,0.021e+2,0.012e+2,0.037e+2
2011-05-20,3.29,3.64,2.749e+2,0.018e+2,0.010e+2,0.031e+2
2011-05-20,3.64,4.02,2.249e+2,0.014e+2,0.008e+2,0.025e+2
2011-05-20,4.02,4.43,1.844e+2,0.011e+2,0.007e+2,0.020e+2
2011-05-20,4.43,4.88,1.500e+2,0.009e+2,0.006e+2,0.016e+2
2011-05-20,4.88,5.37,1.218e+2,0.008e+2,0.005e+2,0.013e+2
2011-05-20,5.37,5.90,9.897e+1,0.063e+1,0.037e+1,0.108e+1
2011-05-20,5.90,6.47,7.975e+1,0.052e+1,0.030e+1,0.087e+1
2011-05-20,6.47,7.09,6.481e+1,0.042e+1,0.024e+1,0.071e+1
2011-05-20,7.09,7.76,5.183e+1,0.035e+1,0.019e+1,0.057e+1
2011-05-20,7.76,8.48,4.123e+1,0.029e+1,0.015e+1,0.046e+1
2011-05-20,8.48,9.26,3.392e+1,0.025e+1,0.013e+1,0.038e+1
2011-05-20,9.26,10.1,2.669e+1,0.021e+1,0.010e+1,0.030e+1
2011-05-20,10.1,11.0,2.129e+1,0.017e+1,0.008e+1,0.024e+1
2011-05-20,11.0,13.0,1.559e+1,0.010e+1,0.006e+1,0.018e+1
2011-05-20,13.0,16.6,9.228e+0,0.052e+0,0.034e+0,0.108e+0
2011-05-20,16.6,22.8,4.333e+0,0.023e+0,0.016e+0,0.052e+0
2011-05-20,22.8,33.5,1.661e+0,0.009e+0,0.006e+0,0.021e+0
2011-05-20,33.5,48.5,5.846e-1,0.047e-1,0.025e-1,0.073e-1
2011-05-20,48.5,69.7,2.080e-1,0.023e-1,0.010e-1,0.026e-1
2011-05-20,69.7,100.0,7.630e-2,0.117e-2,0.049e-2,0.101e-2
...
'''

'''
bartels rotation proton :
bartels_rotation_number,rigidity_min GV,rigidity_max GV,proton_flux m^-2sr^-1s^-1GV^-1,proton_flux_error_statistical m^-2sr^-1s^-1GV^-1,proton_flux_error_timedependent m^-2sr^-1s^-1GV^-1,proton_flux_error_systematic_total m^-2sr^-1s^-1GV^-1
2426,1.00,1.92,8.443E+2,0.005E+2,0.018E+2,0.162E+2
2426,1.92,2.97,4.820E+2,0.003E+2,0.008E+2,0.058E+2
2426,2.97,4.02,2.628E+2,0.002E+2,0.005E+2,0.028E+2
2426,4.02,4.88,1.622E+2,0.001E+2,0.004E+2,0.017E+2
2426,4.88,5.90,1.078E+2,0.001E+2,0.003E+2,0.011E+2
2426,5.90,7.09,7.061E+1,0.005E+1,0.017E+1,0.075E+1
2426,7.09,8.48,4.602E+1,0.003E+1,0.011E+1,0.049E+1
2426,8.48,11.00,2.686E+1,0.002E+1,0.005E+1,0.029E+1
2426,11.00,16.60,1.139E+1,0.001E+1,0.002E+1,0.013E+1
2426,16.60,22.80,4.306E+0,0.004E+0,0.007E+0,0.050E+0
2426,22.80,41.90,1.240E+0,0.001E+0,0.002E+0,0.015E+0
...
'''


# ---------- Load
# 使用相对路径，从compare文件夹向上找results文件夹
current_dir = Path(__file__).parent
results_dir = current_dir.parent / "results"
data_dir = Path("/home/phil/Files/lstmPaper/data/raw_data")

pred = pd.read_csv(results_dir / "cosmic_ray_predictions_extended.csv")
proton = pd.read_csv(data_dir / "ams" / "proton.csv")
'''表头：
date YYYY-MM-DD,rigidity_min GV,rigidity_max GV,proton_flux m^-2sr^-1s^-1GV^-1,proton_flux_error_statistical m^-2sr^-1s^-1GV^-1,proton_flux_error_timedependent m^-2sr^-1s^-1GV^-1,proton_flux_error_systematic_total m^-2sr^-1s^-1GV^-1,proton_to_proton_ratio,proton_to_proton_ratio_error_statistical,proton_to_proton_ratio_error_timedependent,proton_to_proton_ratio_error_systematic_total
'''
brn = pd.read_csv(data_dir / "bartels_rotation_number.csv")
tab = pd.read_csv(data_dir / "ams" / "p-table-BR.csv")
'''表头：
bartels_rotation_number,rigidity_min GV,rigidity_max GV,proton_flux m^-2sr^-1s^-1GV^-1,proton_flux_error_statistical m^-2sr^-1s^-1GV^-1,proton_flux_error_timedependent m^-2sr^-1s^-1GV^-1,proton_flux_error_time_independent m^-2sr^-1s^-1GV^-1,center_date
'''

# ---------- Parse
pred["date"] = pd.to_datetime(pred["date"])
pred["rigidity_GV"] = pred["rigidity"].str.replace("GV","",regex=False).astype(float)

proton = proton.rename(columns={"date YYYY-MM-DD":"date",
                                "rigidity_min GV":"rigidity_min GV",
                                "rigidity_max GV":"rigidity_max GV"})
proton["date"] = pd.to_datetime(proton["date"])

brn["start_date"] = pd.to_datetime(brn["start_date"])
brn["end_date"] = pd.to_datetime(brn["end_date"])
brn["center_date"] = brn["start_date"] + (brn["end_date"] - brn["start_date"]) / 2

tab = tab.merge(brn[["bartels_rotation_number","center_date"]],
                on="bartels_rotation_number", how="left")
# save tab:
tab.to_csv("/home/phil/Files/lstmPaper/data/raw_data/ams/p-table-BR_new.csv", index=False)



# ---------- Helpers
def select_bin_by_rigidity_range(df, r_min, r_max):
    """根据刚度范围选择数据，而不是单一刚度值"""
    return df[(df["rigidity_min GV"] >= r_min) & (df["rigidity_max GV"] <= r_max)].copy()

def weighted_average_by_rigidity_range(daily_df, r_min, r_max, flux_col):
    """
    在给定刚度范围内计算加权平均
    daily_df: 日常数据 (精细刚度bins)
    r_min, r_max: Bartels rotation数据的刚度范围
    flux_col: 通量列名
    """
    # 找到落在 [r_min, r_max] 范围内的所有精细bins
    mask = (daily_df["rigidity_min GV"] >= r_min) & (daily_df["rigidity_max GV"] <= r_max)
    subset = daily_df[mask].copy()
    
    if len(subset) == 0:
        return np.nan, np.nan  # 返回通量和误差
    
    # 计算每个bin的刚度宽度作为权重
    subset["rigidity_width"] = subset["rigidity_max GV"] - subset["rigidity_min GV"]
    total_width = subset["rigidity_width"].sum()
    
    if total_width == 0:
        return np.nan, np.nan
    
    # 加权平均通量
    weights = subset["rigidity_width"] / total_width
    weighted_flux = (subset[flux_col] * weights).sum()
    
    # 加权平均误差 (假设有统计和时间依赖误差)
    stat_col = flux_col.replace("_flux", "_flux_error_statistical")
    time_col = flux_col.replace("_flux", "_flux_error_timedependent")
    
    if stat_col in subset.columns and time_col in subset.columns:
        # 误差的加权组合
        weighted_stat_err = np.sqrt((weights * subset[stat_col]**2).sum())
        weighted_time_err = np.sqrt((weights * subset[time_col]**2).sum())
        weighted_total_err = np.sqrt(weighted_stat_err**2 + weighted_time_err**2)
    else:
        weighted_total_err = np.nan
    
    return weighted_flux, weighted_total_err

def pick_flux_col(df):
    # Try common names for proton flux
    for c in [
        "proton_flux m^-2sr^-1s^-1GV^-1",
        "proton_flux",
        "P_flux",
        "flux"
    ]:
        if c in df.columns:
            return c
    return None

# 新增函数：将pred数据按Bartels rotation和刚度范围分组求加权平均
def aggregate_pred_by_bartels_and_rigidity_range(pred_df, brn_df, tab_df):
    """
    将pred数据按照Bartels rotation周期和tab的刚度范围分组求加权平均
    """
    pred_aggregated = []
    
    # 获取tab中的刚度范围
    unique_rigidity_ranges = tab_df[['rigidity_min GV', 'rigidity_max GV']].drop_duplicates()
    
    for _, rotation in brn_df.iterrows():
        start_date = rotation['start_date']
        end_date = rotation['end_date']
        center_date = rotation['center_date']
        bartels_num = rotation['bartels_rotation_number']
        
        # 筛选在当前Bartels rotation时间范围内的pred数据
        mask = (pred_df['date'] >= start_date) & (pred_df['date'] <= end_date)
        pred_in_rotation = pred_df[mask]
        
        if len(pred_in_rotation) > 0:
            # 对每个tab的刚度范围计算加权平均
            for _, range_row in unique_rigidity_ranges.iterrows():
                r_min = range_row['rigidity_min GV']
                r_max = range_row['rigidity_max GV']
                
                # 找到落在该刚度范围内的预测数据
                pred_mask = (pred_in_rotation['rigidity_GV'] >= r_min) & (pred_in_rotation['rigidity_GV'] < r_max)
                pred_subset = pred_in_rotation[pred_mask]
                
                if len(pred_subset) > 0:
                    # 计算加权平均（按刚度bin宽度加权）
                    # 这里假设每个预测点代表一个刚度区间，权重相等
                    avg_flux = pred_subset['predicted_flux'].mean()
                    
                    pred_aggregated.append({
                        'bartels_rotation_number': bartels_num,
                        'rigidity_min_GV': r_min,
                        'rigidity_max_GV': r_max,
                        'predicted_flux_avg': avg_flux,
                        'center_date': center_date,
                        'data_points': len(pred_subset)
                    })
    
    return pd.DataFrame(pred_aggregated)

# 计算pred的Bartels rotation平均值（按tab的刚度范围）
pred_bartels_avg = aggregate_pred_by_bartels_and_rigidity_range(pred, brn, tab)

# 创建输出目录
output_dir = Path("figs")
output_dir.mkdir(exist_ok=True)

# ---------- 定义列名
flux_col_proton = pick_flux_col(proton)
flux_col_tab = pick_flux_col(tab)

# 误差列名定义
stat_proton_col = "proton_flux_error_statistical m^-2sr^-1s^-1GV^-1"
time_proton_col = "proton_flux_error_timedependent m^-2sr^-1s^-1GV^-1"

stat_tab_col = "proton_flux_error_statistical m^-2sr^-1s^-1GV^-1"
time_tab_col = "proton_flux_error_timedependent m^-2sr^-1s^-1GV^-1"

# ---------- Plot
# 按Bartels rotation数据的刚度范围进行绘图
rigidity_ranges = tab[['rigidity_min GV', 'rigidity_max GV']].drop_duplicates().sort_values('rigidity_min GV')
out_files = []

for _, range_row in rigidity_ranges.iterrows():
    r_min = range_row['rigidity_min GV']
    r_max = range_row['rigidity_max GV']
    
    # 获取对应的数据
    pred_bartels_r = pred_bartels_avg[
        (pred_bartels_avg["rigidity_min_GV"] == r_min) & 
        (pred_bartels_avg["rigidity_max_GV"] == r_max)
    ].sort_values("center_date")
    
    tab_r = tab[
        (tab["rigidity_min GV"] == r_min) & 
        (tab["rigidity_max GV"] == r_max)
    ].sort_values("center_date")
    
    # 对于daily数据，需要计算在该刚度范围内的加权平均
    daily_data_by_date = {}
    daily_errors_by_date = {}
    
    for date in proton['date'].unique():
        daily_subset = proton[proton['date'] == date]
        weighted_flux, weighted_err = weighted_average_by_rigidity_range(
            daily_subset, r_min, r_max, flux_col_proton
        )
        if not np.isnan(weighted_flux):
            daily_data_by_date[date] = weighted_flux
            daily_errors_by_date[date] = weighted_err
    
    # 转换为DataFrame格式便于绘图
    if daily_data_by_date:
        daily_dates = list(daily_data_by_date.keys())
        daily_fluxes = list(daily_data_by_date.values())
        daily_errs = [daily_errors_by_date[d] for d in daily_dates]
    else:
        daily_dates, daily_fluxes, daily_errs = [], [], []
    
    # 对于预测的daily数据，也需要在刚度范围内计算加权平均
    pred_daily_by_date = {}
    for date in pred['date'].unique():
        pred_subset = pred[pred['date'] == date]
        pred_mask = (pred_subset['rigidity_GV'] >= r_min) & (pred_subset['rigidity_GV'] < r_max)
        pred_in_range = pred_subset[pred_mask]
        if len(pred_in_range) > 0:
            avg_pred_flux = pred_in_range['predicted_flux'].mean()
            pred_daily_by_date[date] = avg_pred_flux
    
    if pred_daily_by_date:
        pred_daily_dates = list(pred_daily_by_date.keys())
        pred_daily_fluxes = list(pred_daily_by_date.values())
    else:
        pred_daily_dates, pred_daily_fluxes = [], []
    
    fig, ax = plt.subplots(figsize=(9,5))
    
    # daily (proton) - 加权平均
    if len(daily_dates) > 0:
        yerr_h = daily_errs if not all(np.isnan(daily_errs)) else None
        ax.errorbar(daily_dates, daily_fluxes, yerr=yerr_h,
                    fmt='o', markersize=4, color='tab:orange',
                    ecolor='tab:orange', elinewidth=1, alpha=0.8, capsize=2, zorder=1)    
    
    # model daily - 加权平均
    if len(pred_daily_dates) > 0:
        ax.scatter(pred_daily_dates, pred_daily_fluxes,
                marker='o', s=4, color="tab:blue", linewidth=1.6, alpha=0.8, zorder=2)

    # model Bartels rotation average
    if len(pred_bartels_r):
        ax.scatter(pred_bartels_r["center_date"], pred_bartels_r["predicted_flux_avg"],
                marker='o', s=16, color="tab:red", linewidth=1.6, alpha=0.8, zorder=4)

    # Bartels rotations (tab)
    if flux_col_tab and len(tab_r):
        yerr_t = None
        if stat_tab_col and time_tab_col in tab_r.columns:
            yerr_t = np.sqrt(tab_r[stat_tab_col]**2 + tab_r[time_tab_col]**2)
        ax.errorbar(tab_r["center_date"], tab_r[flux_col_tab], yerr=yerr_t,
                    fmt='o', markersize=4, color='tab:green',
                    ecolor='tab:green', elinewidth=1, alpha=0.8, capsize=2, zorder=3)
    
    ax.set_ylabel("Flux (m$^{-2}$ sr$^{-1}$ s$^{-1}$ GV$^{-1}$)")

    # 彩色类别文字
    ax.text(0.98, 0.85, "Prediction (daily)", color="tab:blue",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=16, fontweight="bold")
    ax.text(0.98, 0.8, "AMS daily", color="tab:orange",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=16, fontweight="bold")
    ax.text(0.98, 0.9, "AMS Bartels rotations", color="tab:green",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=16, fontweight="bold")
    ax.text(0.98, 0.05, "Prediction (Bartels avg)", color="tab:red",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=16, fontweight="bold")
    
    ax.text(0.02, 0.9, f"Rigidity [{r_min:.2f}-{r_max:.2f}] GV",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=16, fontweight="bold")
    
    out_path = f"figs/proton_compare_{r_min:.2f}-{r_max:.2f}GV.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    out_files.append(out_path)
    plt.close(fig)

    # ===== 新增：预测Bartels平均 / AMS Bartels平均 比值图 =====
    if len(pred_bartels_r) and len(tab_r) and flux_col_tab:
        merged = pred_bartels_r.merge(
            tab_r,
            left_on=["bartels_rotation_number", "rigidity_min_GV", "rigidity_max_GV"],
            right_on=["bartels_rotation_number", "rigidity_min GV", "rigidity_max GV"],
            how="inner",
            suffixes=("_pred", "_ams")
        )
        if len(merged):
            # 统一 center_date 列
            if "center_date" not in merged.columns:
                if "center_date_pred" in merged.columns and "center_date_ams" in merged.columns:
                    merged["center_date"] = merged["center_date_pred"].fillna(merged["center_date_ams"])
                elif "center_date_pred" in merged.columns:
                    merged["center_date"] = merged["center_date_pred"]
                elif "center_date_ams" in merged.columns:
                    merged["center_date"] = merged["center_date_ams"]

            # 过滤掉无效或为 0 的 AMS flux，避免除零
            valid = (merged[flux_col_tab] > 0) & np.isfinite(merged["predicted_flux_avg"])
            merged = merged[valid].copy()
            if len(merged):
                merged["ratio"] = merged["predicted_flux_avg"] / merged[flux_col_tab]

                fig2, ax2 = plt.subplots(figsize=(6, 5))
                ax2.axhline(1.0, color="gray", lw=1, ls="--", zorder=0)
                ax2.scatter(
                    merged["center_date"], merged["ratio"],
                    s=25, color="tab:red", alpha=0.9, zorder=2
                )
                ax2.set_ylabel("Ratio: Pred / AMS")
                ax2.text(0.02, 0.9, f"Rigidity [{r_min:.2f}-{r_max:.2f}] GV",
                         transform=ax2.transAxes, ha="left", va="bottom",
                         fontsize=14, fontweight="bold")
                # y 轴范围
                ymin = np.nanmin(merged["ratio"])
                ymax = np.nanmax(merged["ratio"])
                if np.isfinite(ymin) and np.isfinite(ymax):
                    ax2.set_ylim(max(0, ymin*0.95), ymax*1.05)
                
                fig2.tight_layout()
                out_ratio = f"figs/proton_compare_ratio_{r_min:.2f}-{r_max:.2f}GV.png"
                fig2.savefig(out_ratio, dpi=150)
                out_files.append(out_ratio)
                plt.close(fig2)

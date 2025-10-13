import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 全局字体与粗细
plt.rcParams.update({
    "axes.titlesize": 18,
    "axes.titleweight": "bold",
    "axes.labelsize": 18,
    "axes.labelweight": "bold",
    "xtick.labelsize": 18,
    "ytick.labelsize": 18
})

# ---------- Load
pred = pd.read_csv("/home/phil/Files/lstmPaper/model3_v250824/cosmic_ray_predictions_extended.csv")
helium = pd.read_csv("/home/phil/Files/lstmPaper/data/raw_data/ams/helium.csv")
'''表头：
date YYYY-MM-DD,rigidity_min GV,rigidity_max GV,helium_flux m^-2sr^-1s^-1GV^-1,helium_flux_error_statistical m^-2sr^-1s^-1GV^-1,helium_flux_error_timedependent m^-2sr^-1s^-1GV^-1,helium_flux_error_systematic_total m^-2sr^-1s^-1GV^-1,helium_to_proton_ratio,helium_to_proton_ratio_error_statistical,helium_to_proton_ratio_error_timedependent,helium_to_proton_ratio_error_systematic_total
'''
brn = pd.read_csv("/home/phil/Files/lstmPaper/data/raw_data/bartels_rotation_number.csv")
tab = pd.read_csv("/home/phil/Files/lstmPaper/data/raw_data/ams/table-S1-S147.csv")
'''表头：
bartels_rotation_number,rigidity_min_GV,rigidity_max_GV,helium_flux m^-2sr^-1s^-1GV^-1,helium_flux_error_statistical m^-2sr^-1s^-1GV^-1,helium_flux_error_time_dependent m^-2sr^-1s^-1GV^-1,helium_flux_error_time_independent m^-2sr^-1s^-1GV^-1,center_date
'''

# ---------- Parse
pred["date"] = pd.to_datetime(pred["date"])
pred["rigidity_GV"] = pred["rigidity"].str.replace("GV","",regex=False).astype(float)

helium = helium.rename(columns={"date YYYY-MM-DD":"date",
                                "rigidity_min GV":"rigidity_min_GV",
                                "rigidity_max GV":"rigidity_max_GV"})
helium["date"] = pd.to_datetime(helium["date"])

brn["start_date"] = pd.to_datetime(brn["start_date"])
brn["end_date"] = pd.to_datetime(brn["end_date"])
brn["center_date"] = brn["start_date"] + (brn["end_date"] - brn["start_date"]) / 2

tab = tab.merge(brn[["bartels_rotation_number","center_date"]],
                on="bartels_rotation_number", how="left")
# save tab:
tab.to_csv("/home/phil/Files/lstmPaper/data/raw_data/ams/table-S1-S147_new.csv", index=False)



# ---------- Helpers
def select_bin_by_rigidity(df, r):
    return df[(df["rigidity_min_GV"] == r)].copy()

def pick_flux_col(df):
    # Try common names
    for c in [
        "helium_flux m^-2sr^-1s^-1GV^-1",
        "helium_flux",
        "He_flux",
        "flux"
    ]:
        if c in df.columns:
            return c
    return None

# 新增函数：将pred数据按Bartels rotation分组求平均
def aggregate_pred_by_bartels(pred_df, brn_df):
    """
    将pred数据按照Bartels rotation周期分组求平均
    """
    pred_aggregated = []
    
    for _, rotation in brn_df.iterrows():
        start_date = rotation['start_date']
        end_date = rotation['end_date']
        center_date = rotation['center_date']
        bartels_num = rotation['bartels_rotation_number']
        
        # 筛选在当前Bartels rotation时间范围内的pred数据
        mask = (pred_df['date'] >= start_date) & (pred_df['date'] <= end_date)
        pred_in_rotation = pred_df[mask]
        
        if len(pred_in_rotation) > 0:
            # 按rigidity分组求平均
            for rigidity in pred_in_rotation['rigidity_GV'].unique():
                pred_rigidity = pred_in_rotation[pred_in_rotation['rigidity_GV'] == rigidity]
                
                if len(pred_rigidity) > 0:
                    avg_flux = pred_rigidity['predicted_flux'].mean()
                    pred_aggregated.append({
                        'bartels_rotation_number': bartels_num,
                        'rigidity_GV': rigidity,
                        'predicted_flux_avg': avg_flux,
                        'center_date': center_date,
                        'data_points': len(pred_rigidity)
                    })
    
    return pd.DataFrame(pred_aggregated)

# 计算pred的Bartels rotation平均值
pred_bartels_avg = aggregate_pred_by_bartels(pred, brn)

# ---------- 定义列名
flux_col_helium = pick_flux_col(helium)
flux_col_tab = pick_flux_col(tab)

# 误差列名定义
stat_helium_col = "helium_flux_error_statistical m^-2sr^-1s^-1GV^-1"
time_helium_col = "helium_flux_error_timedependent m^-2sr^-1s^-1GV^-1"

stat_tab_col = "helium_flux_error_statistical m^-2sr^-1s^-1GV^-1"
time_tab_col = "helium_flux_error_time_dependent m^-2sr^-1s^-1GV^-1"

# ---------- Plot
rigidities = sorted(pred["rigidity_GV"].unique())
out_files = []

for r in rigidities:
    pred_r = pred[pred["rigidity_GV"] == r].sort_values("date")
    pred_bartels_r = pred_bartels_avg[pred_bartels_avg["rigidity_GV"] == r].sort_values("center_date")
    helium_r = select_bin_by_rigidity(helium, r).sort_values("date")
    tab_r = select_bin_by_rigidity(tab, r).sort_values("center_date")
    
    fig, ax = plt.subplots(figsize=(9,5))
    
    # daily (helium)
    if flux_col_helium and len(helium_r):
        yerr_h = None
        if stat_helium_col and time_helium_col:
            yerr_h = np.sqrt(helium_r[stat_helium_col]**2 + helium_r[time_helium_col]**2)
        ax.errorbar(helium_r["date"], helium_r[flux_col_helium], yerr=yerr_h,
                    fmt='o', markersize=4, color='tab:orange',
                    ecolor='tab:orange', elinewidth=1, alpha=0.8, capsize=2, zorder=1)    
    
    # model daily
    ax.scatter(pred_r["date"], pred_r["predicted_flux"],
            marker='o', s=4, color="tab:blue", linewidth=1.6, alpha=0.8, zorder=2)

    # model Bartels rotation average (新增红色)
    if len(pred_bartels_r):
        ax.scatter(pred_bartels_r["center_date"], pred_bartels_r["predicted_flux_avg"],
                marker='o', s=16, color="tab:red", linewidth=1.6, alpha=0.8, zorder=4)

    # Bartels rotations (tab)
    if flux_col_tab and len(tab_r):
        yerr_t = None
        if stat_tab_col and time_tab_col:
            yerr_t = np.sqrt(tab_r[stat_tab_col]**2 + tab_r[time_tab_col]**2)
        ax.errorbar(tab_r["center_date"], tab_r[flux_col_tab], yerr=yerr_t,
                    fmt='o', markersize=4, color='tab:green',
                    ecolor='tab:green', elinewidth=1, alpha=0.8, capsize=2, zorder=3)
    
    ax.set_ylabel("Flux (m$^{-2}$ sr$^{-1}$ s$^{-1}$ GV$^{-1}$)")

    # 彩色类别文字（新增红色标签）
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
    
    ax.text(0.02, 0.9, f"Rigidity min = {r:.2f} GV",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=16, fontweight="bold")
    
    out_path = f"figs/helium_compare_{r:.2f}GV.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    out_files.append(out_path)
    plt.close(fig)

    # ===== 新增：预测Bartels平均 / AMS Bartels平均 比值图 =====
    if len(pred_bartels_r) and len(tab_r) and flux_col_tab:
        merged = pred_bartels_r.merge(
            tab_r,
            left_on=["bartels_rotation_number", "rigidity_GV"],
            right_on=["bartels_rotation_number", "rigidity_min_GV"],
            how="inner",
            suffixes=("_pred", "_ams")
        )
        if len(merged):
            # 统一 center_date 列
            if "center_date" not in merged.columns:
                if "center_date_pred" in merged.columns and "center_date_ams" in merged.columns:
                    # 一般两列应完全相同，这里做一个简单一致性检查（可选）
                    try:
                        delta_max = (merged["center_date_pred"] - merged["center_date_ams"]).abs().max()
                        # 若差异很大，可在此打印提示（当前不打印以保持脚本安静）
                    except Exception:
                        pass
                    merged["center_date"] = merged["center_date_pred"].fillna(merged["center_date_ams"])
                elif "center_date_pred" in merged.columns:
                    merged["center_date"] = merged["center_date_pred"]
                elif "center_date_ams" in merged.columns:
                    merged["center_date"] = merged["center_date_ams"]

            # AMS Bartels 不确定度
            if stat_tab_col in merged.columns and time_tab_col in merged.columns:
                merged["_ams_err"] = np.sqrt(
                    merged[stat_tab_col]**2 + merged[time_tab_col]**2
                )
            else:
                merged["_ams_err"] = np.nan

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
                ax2.text(0.02, 0.9, f"Rigidity min = {r:.2f} GV",
                         transform=ax2.transAxes, ha="left", va="bottom",
                         fontsize=14, fontweight="bold")
                # y 轴范围 - 简化，不考虑误差棒
                ymin = np.nanmin(merged["ratio"])
                ymax = np.nanmax(merged["ratio"])
                if np.isfinite(ymin) and np.isfinite(ymax):
                    ax2.set_ylim(max(0, ymin*0.95), ymax*1.05)
                
                fig2.tight_layout()
                out_ratio = f"figs/helium_compare_ratio_{r:.2f}GV.png"
                fig2.savefig(out_ratio, dpi=150)
                out_files.append(out_ratio)
                plt.close(fig2)

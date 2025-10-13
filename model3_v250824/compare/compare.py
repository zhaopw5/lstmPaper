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

# 直接使用已知误差列名（统计 + 时间相关），不存在则置 None
STAT_HELIUM_COL = "helium_flux_error_statistical m^-2sr^-1s^-1GV^-1"
TIME_HELIUM_COL = "helium_flux_error_timedependent m^-2sr^-1s^-1GV^-1"
STAT_TAB_COL = "helium_flux_error_statistical m^-2sr^-1s^-1GV^-1"
TIME_TAB_COL = "helium_flux_error_time_dependent m^-2sr^-1s^-1GV^-1"

flux_col_helium = pick_flux_col(helium)
flux_col_tab = pick_flux_col(tab)
stat_helium_col = STAT_HELIUM_COL if STAT_HELIUM_COL in helium.columns else None
time_helium_col = TIME_HELIUM_COL if TIME_HELIUM_COL in helium.columns else None
stat_tab_col = STAT_TAB_COL if STAT_TAB_COL in tab.columns else None
time_tab_col = TIME_TAB_COL if TIME_TAB_COL in tab.columns else None

# ---------- Plot
rigidities = sorted(pred["rigidity_GV"].unique())
out_files = []

for r in rigidities:
    pred_r = pred[pred["rigidity_GV"] == r].sort_values("date")
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
    
    # model
    ax.scatter(pred_r["date"], pred_r["predicted_flux"],
            marker='o', s=4, color="tab:blue", linewidth=1.6, alpha=0.8, zorder=2)

    # Bartels rotations (tab)
    if flux_col_tab and len(tab_r):
        yerr_t = None
        if stat_tab_col and time_tab_col:
            yerr_t = np.sqrt(tab_r[stat_tab_col]**2 + tab_r[time_tab_col]**2)
        ax.errorbar(tab_r["center_date"], tab_r[flux_col_tab], yerr=yerr_t,
                    fmt='o', markersize=4, color='tab:green',
                    ecolor='tab:green', elinewidth=1, alpha=0.8, capsize=2, zorder=3)
    
    # ax.set_xlabel("Date")  # 已由 rcParams 控制大小与粗细
    ax.set_ylabel("Flux (m$^{-2}$ sr$^{-1}$ s$^{-1}$ GV$^{-1}$)")
    # ax.set_title(f"Rigidity min = {r:.2f} GV")  # 粗体大小由 rcParams

    # 彩色类别文字（单独指定字号与粗体）
    ax.text(0.98, 0.85, "Prediction", color="tab:blue",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=16, fontweight="bold")
    ax.text(0.98, 0.8, "Daily", color="tab:orange",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=16, fontweight="bold")
    ax.text(0.98, 0.9, "Bartels rotations", color="tab:green",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=16, fontweight="bold")
    
    ax.text(0.02, 0.9, f"Rigidity min = {r:.2f} GV",# color="tab:green",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=16, fontweight="bold")
    
    # ax.grid(True, alpha=0.3)
    
    out_path = f"figs/helium_compare_{r:.2f}GV.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    out_files.append(out_path)
    plt.close(fig)

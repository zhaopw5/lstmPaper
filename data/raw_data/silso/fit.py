import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# SSN-anchored Similar-Cycle Forecast + Target Mapping
# (Full pipeline: evaluation + plots + predictions)
# -----------------------------------------------------
# Notes:
# - Code comments are in English.
# - Outputs go to /home/phil/Files/lstmPaper/data/outputs/cycle_analysis/.
# - Lag sign convention:
#     "lag" > 0 means SSN leads Y by 'lag' months.
#     We fit/map with: Y(t) = f( SSN(t - lag) ).
# =====================================================

# -------------------- Paths --------------------
SSN_CSV = "/home/phil/Files/lstmPaper/data/raw_data/silso/SN_d_tot_V2.0.csv"     # anchor (daily SSN)
TARGET_CSV = "/home/phil/Files/lstmPaper/data/outputs/cycle_analysis/solar_physics_data_1985_2025_osf.csv"         # user target with daily columns
OUT_EVAL_CSV = "/home/phil/Files/lstmPaper/data/outputs/cycle_analysis/anchor_evaluation_summary.csv"
OUT_FORMULA_CSV = "/home/phil/Files/lstmPaper/data/outputs/cycle_analysis/anchor_mapping_formulas.csv"
OUT_PRED_CSV = "/home/phil/Files/lstmPaper/data/outputs/cycle_analysis/target_new.csv"

# -------------------- Settings --------------------
SMOOTH_WIN = 13
MIN_SEP = 108           # months between cycle minima (>=9 years)
MATCH_MONTHS = 60       # similarity window length
TOP_K = 3
CYCLE25_GUESS = pd.Timestamp("2019-01-01")
FORECAST_END = pd.Timestamp("2030-12-31")

# Target variables to evaluate (must exist in target.csv)
VARS = ["HMF", "wind_speed", "HCS_tilt", "polarity", "daily_OSF"]

# -------------------- Helper functions --------------------
def to_monthly_smooth(df, value_col, smooth_win=13, treat_minus1_as_nan=True):
    """Daily -> monthly mean -> centered smoothing (default 13 months)."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    if treat_minus1_as_nan:
        df[value_col] = df[value_col].replace(-1, np.nan)
    monthly = df.groupby(pd.Grouper(key="date", freq="M"))[value_col].mean()
    smoothed = monthly.rolling(smooth_win, center=True, min_periods=6).mean()
    return monthly, smoothed

def find_cycle_minima(sm_series, min_sep=MIN_SEP):
    """Simple local minima with min separation (in monthly index steps)."""
    s = sm_series.dropna()
    v = s.values; t = s.index
    mins = []
    for i in range(1, len(v)-1):
        if np.isfinite(v[i-1]) and np.isfinite(v[i]) and np.isfinite(v[i+1]):
            if v[i] <= v[i-1] and v[i] < v[i+1]:
                if len(mins) == 0 or (i - mins[-1]) >= min_sep:
                    mins.append(i)
                else:
                    if v[i] < v[mins[-1]]:
                        mins[-1] = i
    return [t[i] for i in mins]

def take_segment(series, start_date, months):
    """Return a monthly segment of given length from start_date (inclusive)."""
    end_date = start_date + pd.DateOffset(months=months-1)
    idx = pd.date_range(start=start_date, end=end_date, freq="M")
    return series.reindex(idx)

def best_lag_corr_safe(ssn_s, y_s, max_lag=36):
    """Find lag with maximum |cross-correlation| between smoothed SSN and Y."""
    ssn = pd.Series(ssn_s.values, index=ssn_s.index).dropna()
    y   = pd.Series(y_s.values, index=y_s.index).dropna()
    df = pd.concat([ssn.rename("SSN"), y.rename("Y")], axis=1).dropna()
    if len(df) < 24 or df["SSN"].std() == 0 or df["Y"].std() == 0:
        return 0, np.nan, pd.DataFrame({"lag": [], "corr": []})
    lags = np.arange(-max_lag, max_lag+1)
    cc = []
    for L in lags:
        if L >= 0:
            # SSN leads Y by L months -> correlate SSN(t) with Y(t+L)
            y_shift = df["Y"].shift(-L)
            valid = df["SSN"].notna() & y_shift.notna()
            if valid.sum() > 12 and df["SSN"][valid].std() > 0 and y_shift[valid].std() > 0:
                r = np.corrcoef(df["SSN"][valid], y_shift[valid])[0,1]
            else:
                r = np.nan
        else:
            # Y leads SSN by |L|
            ssn_shift = df["SSN"].shift(L)  # negative shift
            valid = ssn_shift.notna() & df["Y"].notna()
            if valid.sum() > 12 and ssn_shift[valid].std() > 0 and df["Y"][valid].std() > 0:
                r = np.corrcoef(ssn_shift[valid], df["Y"][valid])[0,1]
            else:
                r = np.nan
        cc.append(r)
    cc = np.array(cc)
    if np.all(np.isnan(cc)):
        return 0, np.nan, pd.DataFrame({"lag": lags, "corr": cc})
    idx = int(np.nanargmax(np.abs(cc)))
    return int(lags[idx]), float(cc[idx]), pd.DataFrame({"lag": lags, "corr": cc})

def fit_linear_with_lag(ssn_s, y_s, lag):
    """Fit Y(t) = a + b * SSN(t - lag)."""
    df = pd.concat([ssn_s.rename("SSN"), y_s.rename("Y")], axis=1)
    x = df["SSN"].shift(lag)       # SSN(t - lag)
    aligned = pd.concat([x.rename("SSN_shift"), df["Y"]], axis=1).dropna()
    if len(aligned) < 24:
        return {"a": np.nan, "b": np.nan, "rmse": np.nan, "r2": np.nan, "n": len(aligned)}
    X = np.vstack([np.ones_like(aligned["SSN_shift"].values), aligned["SSN_shift"].values]).T
    y = aligned["Y"].values
    a, b = np.linalg.lstsq(X, y, rcond=None)[0]
    yhat = a + b * aligned["SSN_shift"].values
    resid = y - yhat
    rmse = float(np.sqrt(np.mean(resid**2)))
    r2 = float(1 - np.sum(resid**2)/np.sum((y - y.mean())**2))
    return {"a": float(a), "b": float(b), "rmse": rmse, "r2": r2, "n": len(aligned)}

def fit_quadratic_with_lag(ssn_s, y_s, lag):
    """Fit Y(t) = a + b * SSN(t - lag) + c * SSN(t - lag)^2."""
    df = pd.concat([ssn_s.rename("SSN"), y_s.rename("Y")], axis=1)
    x = df["SSN"].shift(lag)
    aligned = pd.concat([x.rename("SSN_shift"), df["Y"]], axis=1).dropna()
    if len(aligned) < 24:
        return {"a": np.nan, "b": np.nan, "c": np.nan, "rmse": np.nan, "r2": np.nan, "n": len(aligned)}
    X = np.vstack([np.ones_like(aligned["SSN_shift"].values),
                   aligned["SSN_shift"].values,
                   aligned["SSN_shift"].values**2]).T
    y = aligned["Y"].values
    a, b, c = np.linalg.lstsq(X, y, rcond=None)[0]
    yhat = a + b*aligned["SSN_shift"].values + c*(aligned["SSN_shift"].values**2)
    resid = y - yhat
    rmse = float(np.sqrt(np.mean(resid**2)))
    r2 = float(1 - np.sum(resid**2)/np.sum((y - y.mean())**2))
    return {"a": float(a), "b": float(b), "c": float(c), "rmse": rmse, "r2": r2, "n": len(aligned)}

def zscore(s):
    s = s.copy()
    return (s - s.mean()) / s.std()

# -------------------- 1) Load SSN and smooth --------------------
ssn = pd.read_csv(SSN_CSV)
ssn["date"] = pd.to_datetime(ssn["date"], errors="coerce")
ssn["sunspot_num"] = pd.to_numeric(ssn["sunspot_num"], errors="coerce").replace(-1, np.nan)
ssn_monthly = ssn.groupby(pd.Grouper(key="date", freq="M"))["sunspot_num"].mean()
ssn_smoothed = ssn_monthly.rolling(SMOOTH_WIN, center=True, min_periods=6).mean()

# -------------------- 2) Similar-cycle on SSN --------------------
starts = find_cycle_minima(ssn_smoothed)
cycle25_start = min(starts, key=lambda d: abs(d - CYCLE25_GUESS))

# similarity scoring over first MATCH_MONTHS months after start
obs_seg = take_segment(ssn_smoothed, cycle25_start, MATCH_MONTHS).dropna()
L = len(obs_seg)

scores = []
for s0 in starts:
    if s0 < cycle25_start - pd.DateOffset(years=1):
        seg = take_segment(ssn_smoothed, s0, L).dropna()
        if len(seg) == L:
            x = obs_seg.values; y = seg.values
            corr = 0.0 if (np.std(x)==0 or np.std(y)==0) else np.corrcoef(x, y)[0,1]
            mse = float(np.mean((x - y)**2))
            scores.append((s0, corr, mse))

scores_sorted = sorted(scores, key=lambda t: (-t[1], t[2]))
top_starts = [t[0] for t in scores_sorted[:TOP_K]]

# Build SSN forecast to 2030-12
months_total = (FORECAST_END.year - cycle25_start.year)*12 + (FORECAST_END.month - cycle25_start.month) + 1
t_idx = pd.date_range(start=cycle25_start, periods=months_total, freq="M")
mat = []
for s0 in top_starts:
    mat.append(take_segment(ssn_smoothed, s0, months_total).values)
mat = np.array(mat)
ssn_mean = np.nanmean(mat, axis=0)
ssn_std  = np.nanstd(mat, axis=0)
ssn_forecast_df = pd.DataFrame({"date": t_idx, "SSN_pred": ssn_mean, "SSN_std": ssn_std})

# Plots: SSN smoothed & minima
plt.figure(figsize=(12,5))
plt.plot(ssn_smoothed.index, ssn_smoothed.values, label="SSN (13m smoothed)")
for d in starts:
    plt.axvline(d, linestyle="--", alpha=0.25)
plt.title("SSN smoothed with detected minima")
plt.xlabel("Year"); plt.ylabel("SSN"); plt.grid(True); plt.legend()
plt.savefig("/home/phil/Files/lstmPaper/data/outputs/cycle_analysis/ssn_minima.png", bbox_inches="tight")
plt.show()

# Plot: SSN forecast (mean ±1 std) with top-K paths
plt.figure(figsize=(12,5))
for i in range(mat.shape[0]):
    plt.plot(t_idx, mat[i], alpha=0.5, linewidth=1, label=f"similar start: {top_starts[i].date()}")
plt.plot(t_idx, ssn_mean, linewidth=2, label="SSN forecast (mean of similar cycles)")
plt.fill_between(t_idx, ssn_mean - ssn_std, ssn_mean + ssn_std, alpha=0.2, label="±1 std")
plt.title("SSN forecast to 2030 via similar-cycle method")
plt.xlabel("Year"); plt.ylabel("SSN"); plt.grid(True); plt.legend()
plt.savefig("/home/phil/Files/lstmPaper/data/outputs/cycle_analysis/ssn_forecast.png", bbox_inches="tight")
plt.show()

# -------------------- 3) Load target and evaluate each variable --------------------
tar = pd.read_csv(TARGET_CSV)
# normalize date column if needed
if "date" not in tar.columns:
    for alt in ["Date", "DATE", "time", "Time", "timestamp"]:
        if alt in tar.columns:
            tar = tar.rename(columns={alt: "date"})
            break
tar["date"] = pd.to_datetime(tar["date"], errors="coerce")

eval_rows = []
formula_rows = []
pred_frames = []  # to assemble final target_new.csv

for var in VARS:
    if var not in tar.columns:
        eval_rows.append({"variable": var, "pearson": np.nan, "spearman": np.nan,
                          "best_lag": np.nan, "best_lag_corr": np.nan,
                          "model": "N/A", "r2": np.nan, "rmse": np.nan,
                          "conclusion": "column not found"})
        continue

    # Monthly + smoothed
    monthly, smoothed = to_monthly_smooth(tar[["date", var]].copy(), var, smooth_win=SMOOTH_WIN, treat_minus1_as_nan=True)

    # Overlap for metrics
    df_ov = pd.concat([ssn_smoothed.rename("SSN"), smoothed.rename(var)], axis=1).dropna()
    if len(df_ov) < 24 or df_ov["SSN"].std()==0 or df_ov[var].std()==0:
        eval_rows.append({"variable": var, "pearson": np.nan, "spearman": np.nan,
                          "best_lag": np.nan, "best_lag_corr": np.nan,
                          "model": "N/A", "r2": np.nan, "rmse": np.nan,
                          "conclusion": "insufficient or unsuitable"})
        continue

    pearson = float(np.corrcoef(df_ov["SSN"].values, df_ov[var].values)[0,1])
    spearman = float(pd.Series(df_ov["SSN"]).corr(pd.Series(df_ov[var]), method="spearman"))
    lag, lagcorr, xcorr_df = best_lag_corr_safe(ssn_smoothed, smoothed, max_lag=36)

    # Fits at best lag
    lin  = fit_linear_with_lag(ssn_smoothed, smoothed, lag)
    quad = fit_quadratic_with_lag(ssn_smoothed, smoothed, lag)
    use_quad = (quad["r2"] > lin["r2"] + 0.01)
    model = "quadratic" if use_quad else "linear"
    r2 = quad["r2"] if use_quad else lin["r2"]
    rmse = quad["rmse"] if use_quad else lin["rmse"]

    # Suitability rule
    suitable = (abs(lagcorr) >= 0.5) and (r2 >= 0.4)
    conclusion = "use SSN as anchor" if suitable else "not suitable as anchor"

    eval_rows.append({"variable": var, "pearson": pearson, "spearman": spearman,
                      "best_lag": lag, "best_lag_corr": lagcorr,
                      "model": model, "r2": r2, "rmse": rmse,
                      "conclusion": conclusion})

    # Save formula row (coefficients)
    if model == "linear":
        a, b, c = lin["a"], lin["b"], np.nan
    else:
        a, b, c = quad["a"], quad["b"], quad["c"]
    formula_rows.append({"variable": var, "best_lag": lag, "model": model,
                         "a": a, "b": b, "c": c, "r2": r2, "rmse": rmse,
                         "pearson": pearson, "spearman": spearman})

    # --------- Plots for diagnostics (per variable) ---------
    # (1) Z-score overlay (overlap period)
    z_df = pd.DataFrame({
        "SSN_z": zscore(df_ov["SSN"]),
        f"{var}_z": zscore(df_ov[var])
    }, index=df_ov.index)
    plt.figure(figsize=(12,4))
    plt.plot(z_df.index, z_df["SSN_z"], label="SSN (z)")
    plt.plot(z_df.index, z_df[f"{var}_z"], label=f"{var} (z)")
    plt.title(f"Z-score overlay: SSN vs {var}")
    plt.xlabel("Year"); plt.ylabel("Z-score"); plt.grid(True); plt.legend()
    plt.savefig(f"/home/phil/Files/lstmPaper/data/outputs/cycle_analysis/{var}_z.png", bbox_inches="tight")
    plt.show()

    # (2) Scatter + fit curve
    plt.figure(figsize=(6,6))
    plt.scatter(df_ov["SSN"].values, df_ov[var].values, s=10)
    xline = np.linspace(df_ov["SSN"].min(), df_ov["SSN"].max(), 200)
    if model == "linear":
        yline = a + b*xline
    else:
        yline = a + b*xline + c*(xline**2)
    plt.plot(xline, yline, linewidth=2)
    plt.title(f"{var} vs SSN (smoothed)  |  model={model}, lag={lag}")
    plt.xlabel("SSN (13m smoothed)"); plt.ylabel(f"{var} (13m smoothed)"); plt.grid(True)
    plt.savefig(f"/home/phil/Files/lstmPaper/data/outputs/cycle_analysis/{var}_scatter.png", bbox_inches="tight")
    plt.show()

    # (3) Cross-correlation vs lag
    plt.figure(figsize=(10,3.5))
    plt.plot(xcorr_df["lag"], xcorr_df["corr"])
    plt.axhline(0, linestyle="--")
    plt.axvline(lag, linestyle=":")
    plt.title(f"Cross-correlation: SSN vs {var}  (best lag={lag}, r={lagcorr:.3f})")
    plt.xlabel("Lag (months)  [>0: SSN leads]"); plt.ylabel("Correlation"); plt.grid(True)
    plt.savefig(f"/home/phil/Files/lstmPaper/data/outputs/cycle_analysis/{var}_xcorr.png", bbox_inches="tight")
    plt.show()

    # --------- Forecast to 2030 for suitable variables ---------
    if suitable:
        ssn_for_map = pd.Series(ssn_forecast_df["SSN_pred"].values, index=ssn_forecast_df["date"])
        ssn_for_std = pd.Series(ssn_forecast_df["SSN_std"].values,  index=ssn_forecast_df["date"])
        ssn_shifted  = ssn_for_map.shift(lag)       # SSN(t - lag)
        ssn_std_shift = ssn_for_std.shift(lag)

        if model == "linear":
            y_pred = a + b*ssn_shifted
            y_std  = np.sqrt((b**2)*(ssn_std_shift**2) + (rmse**2))
        else:
            y_pred = a + b*ssn_shifted + c*(ssn_shifted**2)
            fprime = b + 2*c*ssn_shifted
            y_std  = np.sqrt((fprime**2)*(ssn_std_shift**2) + (rmse**2))

        # Assemble frame for plotting
        frame = pd.DataFrame({
            "obs": smoothed, 
            f"{var}_pred": y_pred,
            f"{var}_pred_std": y_std
        })
        # restrict plotting horizon for clarity
        plot_start = pd.Timestamp("2015-01-01")
        frame_plot = frame[(frame.index >= plot_start) & (frame.index <= FORECAST_END)].copy()

        # (4) Observed vs forecast ±1 std band
        plt.figure(figsize=(12,5))
        plt.plot(frame_plot.index, frame_plot["obs"], label=f"{var} observed (smoothed)")
        plt.plot(frame_plot.index, frame_plot[f"{var}_pred"], label=f"{var} forecast (mean)", linewidth=2)
        finite = np.isfinite(frame_plot[f"{var}_pred"].values) & np.isfinite(frame_plot[f"{var}_pred_std"].values)
        plt.fill_between(frame_plot.index[finite],
                         (frame_plot[f"{var}_pred"].values[finite] - frame_plot[f"{var}_pred_std"].values[finite]),
                         (frame_plot[f"{var}_pred"].values[finite] + frame_plot[f"{var}_pred_std"].values[finite]),
                         alpha=0.2, label="±1 std")
        plt.title(f"{var}: observed vs forecast (SSN-anchored)")
        plt.xlabel("Year"); plt.ylabel(var); plt.grid(True); plt.legend()
        plt.savefig(f"/home/phil/Files/lstmPaper/data/outputs/cycle_analysis/pred_{var}.png", bbox_inches="tight")
        plt.show()

        # Keep predictions to build target_new.csv
        pred_frames.append(pd.concat([
            y_pred.rename(f"{var}_pred"),
            y_std.rename(f"{var}_pred_std")
        ], axis=1))

# -------------------- 4) Save CSV outputs --------------------
eval_df = pd.DataFrame(eval_rows)
eval_df.to_csv(OUT_EVAL_CSV, index=False)

formulas_df = pd.DataFrame(formula_rows)
formulas_df.to_csv(OUT_FORMULA_CSV, index=False)

# Assemble predictions table on the SSN forecast timeline
out_df = pd.DataFrame({"date": ssn_forecast_df["date"]})
for pf in pred_frames:
    out_df = out_df.merge(pf.reset_index().rename(columns={"index": "date"}), on="date", how="left")
out_df = out_df[out_df["date"] <= FORECAST_END].copy()
out_df.to_csv(OUT_PRED_CSV, index=False)

print("Top similar SSN cycle starts:", [d.date() for d in top_starts])
print("Saved evaluation summary to:", OUT_EVAL_CSV)
print("Saved formulas to:", OUT_FORMULA_CSV)
print("Saved predictions to:", OUT_PRED_CSV)

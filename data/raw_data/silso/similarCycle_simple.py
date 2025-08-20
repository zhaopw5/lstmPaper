import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 0) Basic settings
# =========================
DATA_PATH = "SN_d_tot_V2.0.csv"    # your input file
OUT_CSV   = "sunspot_similar_cycle_forecast_to_2030.csv"

# Similar-cycle method parameters
SMOOTH_WIN = 13        # 13-month centered smoothing
MIN_SEP = 108          # min separation (months) between detected minima (~>= 9 years)
MATCH_MONTHS = 60      # months used to compare similarity
K_SIMILAR = 3          # number of similar cycles to average
CYCLE25_START_GUESS = pd.Timestamp("2019-01-01")
FORECAST_END = pd.Timestamp("2030-12-31")

# =========================
# 1) Load and preprocess
# =========================
# Read CSV (the uploaded file has headers like: date,date_frac,sunspot_num,...)
df = pd.read_csv(DATA_PATH)

# Parse date column
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Treat -1 as missing values (NaN) before any averaging
# This is the key fix: -1 indicates "no data" in SILSO V2.0
df["sunspot_num"] = df["sunspot_num"].replace(-1, np.nan)

# Keep rows that have a valid timestamp (drop totally invalid dates)
df = df.dropna(subset=["date"]).copy()

# Monthly mean (pandas .mean() skips NaN by default)
monthly = df.groupby(pd.Grouper(key="date", freq="M"))["sunspot_num"].mean()

# 13-month centered smoothing (set min_periods=6 to keep edges partially)
smoothed = monthly.rolling(window=SMOOTH_WIN, center=True, min_periods=6).mean()

# Put into one DataFrame for convenience
m = pd.DataFrame({"monthly": monthly, "smoothed": smoothed})

# =========================
# 2) Detect cycle minima (simple rule)
# =========================
# We look for local minima in the smoothed series with at least MIN_SEP months apart.
mins_idx = []
vals = m["smoothed"].values
idx = m.index

for i in range(1, len(vals)-1):
    # local minimum condition on smoothed curve
    if pd.notna(vals[i-1]) and pd.notna(vals[i]) and pd.notna(vals[i+1]):
        if vals[i] <= vals[i-1] and vals[i] < vals[i+1]:
            if len(mins_idx) == 0:
                mins_idx.append(i)
            else:
                # ensure minima are at least MIN_SEP months apart (monthly index, so use index distance)
                if i - mins_idx[-1] >= MIN_SEP:
                    mins_idx.append(i)
                else:
                    # if too close, keep the deeper minimum
                    if vals[i] < vals[mins_idx[-1]]:
                        mins_idx[-1] = i

cycle_starts = [idx[i] for i in mins_idx]

# Choose the detected minimum closest to 2019-01 as Cycle 25 start
current_start = min(cycle_starts, key=lambda d: abs(d - CYCLE25_START_GUESS))

# =========================
# 3) Build the observed segment (first MATCH_MONTHS from Cycle 25 start)
# =========================
def take_segment(series, start_date, months):
    """Return a monthly segment of given length starting from start_date."""
    end_date = start_date + pd.DateOffset(months=months-1)
    full_index = pd.date_range(start=start_date, end=end_date, freq="M")
    seg = series.reindex(full_index)
    return seg

observed_seg = take_segment(m["smoothed"], current_start, MATCH_MONTHS).dropna()
obs_len = len(observed_seg)

# Historical cycle starts before Cycle 25 (leave 1 year margin)
hist_starts = [d for d in cycle_starts if d < current_start - pd.DateOffset(years=1)]

# =========================
# 4) Similarity scoring (corr high, MSE low)
# =========================
scores = []  # each item: (start_date, corr, mse)
for s0 in hist_starts:
    seg = take_segment(m["smoothed"], s0, obs_len).dropna()
    if len(seg) == obs_len:
        x = observed_seg.values
        y = seg.values
        # correlation
        if np.std(x) == 0 or np.std(y) == 0:
            corr = 0.0
        else:
            corr = np.corrcoef(x, y)[0, 1]
        # mean squared error
        mse = np.mean((x - y) ** 2)
        scores.append((s0, corr, mse))

# Sort: higher corr first, then lower MSE
scores_sorted = sorted(scores, key=lambda t: (-t[1], t[2]))
top_similar = [t[0] for t in scores_sorted[:K_SIMILAR]]

# =========================
# 5) Forecast to 2030-12 by averaging similar cycles
# =========================
# Months from current_start to FORECAST_END (inclusive)
months_total = (FORECAST_END.year - current_start.year) * 12 + (FORECAST_END.month - current_start.month) + 1
forecast_index = pd.date_range(start=current_start, periods=months_total, freq="M")

# Observed smoothed series for the full horizon (will contain NaN in the future)
obs_full = take_segment(m["smoothed"], current_start, months_total)

# Collect similar cycles' full paths (aligned by relative months)
sim_matrix = []
for s0 in top_similar:
    sim_full = take_segment(m["smoothed"], s0, months_total)
    sim_matrix.append(sim_full.values)

sim_matrix = np.array(sim_matrix)           # shape: (K, N)
sim_mean = np.nanmean(sim_matrix, axis=0)   # average across similar cycles
sim_std  = np.nanstd(sim_matrix, axis=0)    # std dev across similar cycles

# Build forecast: keep observed where available; use similar-mean for NaN parts
forecast = obs_full.values.copy()
future_mask = np.isnan(forecast)
forecast[future_mask] = sim_mean[future_mask]

# =========================
# 6) Save CSV and plot
# =========================
out = pd.DataFrame({
    "date": forecast_index,
    "observed_smoothed_ssn": obs_full.values,
    "forecast_smoothed_ssn": forecast,
    "similar_cycle_mean": sim_mean,
    "similar_cycle_std": sim_std
})
out = out[out["date"] <= FORECAST_END].copy()
out.to_csv(OUT_CSV, index=False)

# --- Plot A: smoothed series + detected minima ---
fig, ax = plt.subplots(figsize=(12,5))
ax.plot(m.index, m["smoothed"], label="13-month smoothed")
for d in cycle_starts:
    ax.axvline(d, linestyle="--", alpha=0.25)
ax.set_title("Smoothed Sunspot Numbers (minima shown)")
ax.set_xlabel("Year")
ax.set_ylabel("Smoothed SSN")
ax.grid(True)
ax.legend()
fig.savefig("smoothed_sunspot_numbers.png", dpi=300)


# --- Plot B: Cycle 25 vs top similar cycles (first MATCH_MONTHS months) ---
fig, ax = plt.subplots(figsize=(12,5))
ax.plot(observed_seg.index, observed_seg.values, label="Cycle 25 (observed seg)", linewidth=2)
for s0 in top_similar:
    seg = take_segment(m["smoothed"], s0, obs_len)
    ax.plot(seg.index, seg.values, label=f"Similar start: {s0.date()}")
ax.set_title(f"Top-{len(top_similar)} Similar Cycles (first {obs_len} months)")
ax.set_xlabel("Year")
ax.set_ylabel("Smoothed SSN")
ax.grid(True)
ax.legend()
fig.savefig("similar_cycles_comparison.png", dpi=300)

# --- Plot C: Forecast with similar-cycle mean and ±1 std band ---
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(forecast_index, obs_full.values, label="Observed (smoothed)", linewidth=2)
for i, s0 in enumerate(top_similar):
    ax.plot(forecast_index, sim_matrix[i], alpha=0.5, linewidth=1, label=f"Similar start: {s0.date()}")
ax.plot(forecast_index, forecast, label="Forecast (mean of similar cycles)", linewidth=3)
ax.fill_between(forecast_index, sim_mean - sim_std, sim_mean + sim_std, alpha=0.2, label="±1 std")
ax.axvline(forecast_index[0], linestyle="--", alpha=0.5)
ax.axvline(FORECAST_END, linestyle=":")
ax.set_title("Cycle 25 Forecast to 2030 (Similar-Cycle)")
ax.set_xlabel("Year")
ax.set_ylabel("Smoothed SSN")
ax.grid(True)
ax.legend()
fig.savefig("similar_cycle_forecast.png", dpi=300)

print("Top similar cycle starts:", [d.date() for d in top_similar])
print("Saved CSV to:", OUT_CSV)

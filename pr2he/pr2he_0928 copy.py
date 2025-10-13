import pandas as pd
import numpy as np
import ROOT
from array import array
import os

# -----------------------------
# User-configurable parameters
# -----------------------------
# Full time window for analysis (no hard-coded gaps/ranges)
full_start = '2014-07-01'
full_end   = '2015-05-01'

# Rigidity bin edges (min/max GV)
rig_bin = [1.71,1.92,2.15,2.4,2.67,2.97,3.29,3.64,4.02,4.43,4.88,5.37,5.9,
           6.47,7.09,7.76,8.48,9.26,10.1,11,13,16.6,22.8,33.5,48.5,69.7,100]

rig_bin_he = [1.71,1.92,2.15,2.4,2.67,2.97,3.29,3.64,4.02,4.43,4.88,5.37,5.9,
           6.47,7.09,7.76,8.48,9.26,10.1,11,13,16.6,22.8,33.5,48.5,69.7,100]

rig_bin_pr = [1.00, 1.16, 1.33, 1.51, 1.71,1.92,2.15,2.4,2.67,2.97,
              3.29,3.64,4.02,4.43,4.88,5.37,5.9,6.47,7.09,7.76,8.48,
              9.26,10.1,11,13,16.6,22.8,33.5,48.5,69.7,100]

# Input paths
PROTON_CSV = '/home/zpw/Files/lstmPaper/data/raw_data/ams/proton.csv'
HELIUM_CSV = '/home/zpw/Files/lstmPaper/data/raw_data/ams/helium.csv'
ML_CSV     = '/home/zpw/Files/AMS_NeutronMonitors/AMS/paperPlots/data/lightning_logs/version_3/2011-01-01-2024-07-31_pred_ams_updated.csv'

# Output directory
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()
results_dir = os.path.join(script_dir, 'results_root')
os.makedirs(results_dir, exist_ok=True)

# -----------------------------
# ROOT global styles
# -----------------------------
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptFit(1111)
ROOT.gStyle.SetTimeOffset(0)  # Use Unix epoch to avoid year drift

def dt_to_unix(d):
    """datetime-like -> Unix seconds (int)."""
    return int(pd.Timestamp(d).timestamp())


xlim_start = dt_to_unix(full_start)
xlim_end   = dt_to_unix(full_end)

# -----------------------------
# Column names in AMS CSVs
# -----------------------------
p_date     = 'date YYYY-MM-DD'
p_flux_col = 'proton_flux m^-2sr^-1s^-1GV^-1'
p_stat_col = 'proton_flux_error_statistical m^-2sr^-1s^-1GV^-1'
p_td_col   = 'proton_flux_error_timedependent m^-2sr^-1s^-1GV^-1'

he_flux_col = 'helium_flux m^-2sr^-1s^-1GV^-1'
he_stat_col = 'helium_flux_error_statistical m^-2sr^-1s^-1GV^-1'
he_td_col   = 'helium_flux_error_timedependent m^-2sr^-1s^-1GV^-1'

# -----------------------------
# Load data
# -----------------------------
print("Loading data...")
df_p = pd.read_csv(PROTON_CSV)
df_he = pd.read_csv(HELIUM_CSV)
df_ml = pd.read_csv(ML_CSV)

df_p[p_date] = pd.to_datetime(df_p[p_date])
df_he[p_date] = pd.to_datetime(df_he[p_date])
df_ml['date'] = pd.to_datetime(df_ml['date'])

# Clip all frames to the user-specified full window
full_start_dt = pd.to_datetime(full_start)
full_end_dt   = pd.to_datetime(full_end)

df_p = df_p[(df_p[p_date] >= full_start_dt) & (df_p[p_date] <= full_end_dt)].copy()
df_he = df_he[(df_he[p_date] >= full_start_dt) & (df_he[p_date] <= full_end_dt)].copy()
df_ml_range = df_ml[(df_ml['date'] >= full_start_dt) & (df_ml['date'] <= full_end_dt)].copy()

# -----------------------------
# Helper: select a rigidity bin and inner join P/He by date (observed dates only)
# -----------------------------
def select_observed_pair(df_p, df_he, rig_min, rig_max):
    """Return inner-joined observed P and He within the full window for a given rigidity bin."""
    p_sel = df_p[(df_p['rigidity_min GV'] == rig_min) & (df_p['rigidity_max GV'] == rig_max)].copy()
    he_sel = df_he[(df_he['rigidity_min GV'] == rig_min) & (df_he['rigidity_max GV'] == rig_max)].copy()

    if p_sel.empty or he_sel.empty:
        return pd.DataFrame()

    # Quadrature errors
    p_sel['proton_flux_error'] = np.sqrt(p_sel[p_stat_col]**2 + p_sel[p_td_col]**2)
    he_sel['helium_flux_error'] = np.sqrt(he_sel[he_stat_col]**2 + he_sel[he_td_col]**2)

    # Inner join by date -> only keep dates with both P and He observed
    df_obs = pd.merge(
        p_sel[[p_date, p_flux_col, 'proton_flux_error']],
        he_sel[[p_date, he_flux_col, 'helium_flux_error']],
        on=p_date, how='inner'
    ).sort_values(p_date)

    return df_obs

# -----------------------------
# Helper style functions (NEW)
# -----------------------------
def set_pad_margins(pad, left=0.11, bottom=0.10, top=0.02, right=0.05):
    pad.SetLeftMargin(left)
    pad.SetBottomMargin(bottom)
    pad.SetTopMargin(top)
    pad.SetRightMargin(right)

def config_time_axis(axis, xmin, xmax):
    axis.SetTimeDisplay(1)
    axis.SetTimeOffset(0, "gmt")
    axis.SetTimeFormat("#splitline{%b %d}{%Y}")
    axis.SetNdivisions(-505)
    axis.SetLabelOffset(0.02)
    axis.SetLimits(xmin, xmax)

def draw_time_frame(xmin, ymin, xmax, ymax, ytitle):
    frame = ROOT.gPad.DrawFrame(xmin, ymin, xmax, ymax)
    frame.SetTitle('')
    frame.GetYaxis().SetTitle(ytitle)
    config_time_axis(frame.GetXaxis(), xmin, xmax)
    return frame

# -----------------------------
# Main loop over rigidity bins
# -----------------------------
for i in range(len(rig_bin) - 1):
    rig_min_cur = rig_bin[i]
    rig_max_cur = rig_bin[i + 1]
    rig_label = f'{rig_min_cur:.2f}-{rig_max_cur:.2f} GV'
    print(f"\nProcessing rigidity bin: {rig_label} (bin {i+1})")

    # 1) Build observed P-He pairs in full window
    df_observed = select_observed_pair(df_p, df_he, rig_min_cur, rig_max_cur)
    if df_observed.empty or len(df_observed) < 2:
        print(f"  Not enough observed P-He pairs ({len(df_observed)}). Skip.")
        continue
    print(f"  Found {len(df_observed)} observed P-He paired dates in [{full_start} .. {full_end}].")

    # 2) Prepare ML column for this rigidity in full window
    ml_col_idx = str(i + 5)  # ML file uses '1','2',... for rigidity bins,proton比helium多了4个rigidity bins
    if ml_col_idx not in df_ml_range.columns:
        print(f"  ML column '{ml_col_idx}' not in ML data. Skip.")
        continue
    df_ml_range_local = df_ml_range[['date', ml_col_idx]].copy()
    df_ml_range_local = df_ml_range_local.rename(columns={ml_col_idx: 'ml_proton'})

    # 3) Flags: observed dates (based on df_observed), missing dates (in ML but not observed)
    observed_dates = set(df_observed[p_date])
    df_ml_range_local['is_observed'] = df_ml_range_local['date'].isin(observed_dates)
    df_ml_range_local['is_missing_date'] = ~df_ml_range_local['is_observed']

    # 4) Build ROOT canvas with 4 pads
    c_main = ROOT.TCanvas(f'c_main_{i}', f'Proton-Helium Analysis {rig_label}', 1400, 1000)
    c_main.Divide(2, 2, 0.0001, 0.0001)

    # ======== (1) Top-Left ========
    c_main.cd(1)
    set_pad_margins(ROOT.gPad)

    # --- Observed series (dates, values, errors) ---
    dates_obs = df_observed[p_date].to_numpy()
    ts_obs = np.array([dt_to_unix(d) for d in dates_obs], dtype=float)
    proton_obs = df_observed[p_flux_col].to_numpy(dtype=float)
    proton_err = df_observed['proton_flux_error'].to_numpy(dtype=float)

    g_proton = ROOT.TGraphErrors(len(ts_obs), array('d', ts_obs), array('d', proton_obs),
                                array('d', [0]*len(ts_obs)), array('d', proton_err))
    g_proton.SetTitle('')
    g_proton.SetMarkerStyle(20)
    g_proton.SetMarkerSize(0.5)
    g_proton.SetMarkerColor(ROOT.kBlue)
    g_proton.SetLineColor(ROOT.kBlue)

    # --- ML FULL series over the whole window ---
    ml_dates_full = df_ml_range_local['date'].to_numpy()
    ts_ml_full = np.array([dt_to_unix(d) for d in ml_dates_full], dtype=float)
    ml_vals_full = df_ml_range_local['ml_proton'].to_numpy(dtype=float)

    g_proton_ml_full = ROOT.TGraph(len(ts_ml_full), array('d', ts_ml_full), array('d', ml_vals_full))
    g_proton_ml_full.SetMarkerStyle(24)
    g_proton_ml_full.SetMarkerSize(0.5)
    g_proton_ml_full.SetMarkerColor(ROOT.kRed)
    g_proton_ml_full.SetLineColor(ROOT.kRed)
    g_proton_ml_full.SetLineWidth(1)        # thin line to show continuity

    # --- ML aligned ONLY to observed dates (for point-by-point comparison) ---
    df_obs_with_ml = pd.merge(
        df_observed[[p_date]],
        df_ml_range_local[['date', 'ml_proton']],
        left_on=p_date, right_on='date', how='inner'
    )
    ts_ml_aligned  = np.array([dt_to_unix(d) for d in df_obs_with_ml['date']], dtype=float)
    ml_vals_aligned = df_obs_with_ml['ml_proton'].to_numpy(dtype=float)


    # ---- Compute unified y-range: observed ± error & ML FULL series ----
    proton_low  = proton_obs - np.nan_to_num(proton_err, nan=0.0)
    proton_high = proton_obs + np.nan_to_num(proton_err, nan=0.0)

    ymins = [np.nanmin(proton_low)]
    ymaxs = [np.nanmax(proton_high)]
    if len(ml_vals_full) > 0:
        ymins.append(np.nanmin(ml_vals_full))
        ymaxs.append(np.nanmax(ml_vals_full))

    ymin = float(np.nanmin(ymins)); ymax = float(np.nanmax(ymaxs))
    if not np.isfinite(ymin) or not np.isfinite(ymax):
        ymin, ymax = 0.0, 1.0
    pad = 0.08
    yr = ymax - ymin
    if yr <= 0: yr = abs(ymax) if ymax != 0 else 1.0
    ymin_pad = ymin - pad*yr
    ymax_pad = ymax + pad*yr

    # 统一：时间轴使用用户设定窗口，不再用数据最值
    frame = ROOT.gPad.DrawFrame(xlim_start, ymin_pad, xlim_end, ymax_pad)
    frame.SetTitle('')
    frame.GetYaxis().SetTitle('Proton Flux')

    xa = frame.GetXaxis()
    xa.SetTimeDisplay(1); xa.SetTimeOffset(0, "gmt")
    xa.SetTimeFormat("#splitline{%b %d}{%Y}")
    xa.SetNdivisions(-505); xa.SetLabelOffset(0.02)
    xa.SetLimits(xlim_start, xlim_end)  # 统一方式

    # 1) Observed (blue, with errors)
    g_proton.Draw('P SAME')
    # 2) ML FULL series (red thin line + small markers)
    g_proton_ml_full.Draw('P SAME')

    # Legend & label
    leg_p = ROOT.TLegend(0.7, 0.77, 0.93, 0.95)
    leg_p.SetBorderSize(0)
    leg_p.AddEntry(g_proton,            'AMS', 'pe')
    leg_p.AddEntry(g_proton_ml_full,    'Model', 'lp')
    # leg_p.AddEntry(g_proton_ml_aligned, 'ML Proton (observed dates)', 'p')
    leg_p.Draw()

    latex_r1 = ROOT.TLatex()
    latex_r1.SetNDC(); latex_r1.SetTextSize(0.045)
    latex_r1.DrawLatex(0.15, 0.86, rig_label)

    # ---------- (2) Top-Right: SINGLE fit on observed P vs He ----------
    c_main.cd(2)
    # ROOT.gPad.SetGrid()
    ROOT.gPad.SetLeftMargin(0.11); ROOT.gPad.SetBottomMargin(0.10)
    ROOT.gPad.SetTopMargin(0.02);  ROOT.gPad.SetRightMargin(0.05)
    ROOT.gStyle.SetOptFit(0)  # don't show stats box

    x_fit = df_observed[p_flux_col].to_numpy(dtype=float)
    y_fit = df_observed[he_flux_col].to_numpy(dtype=float)
    ex_fit = df_observed['proton_flux_error'].to_numpy(dtype=float)
    ey_fit = df_observed['helium_flux_error'].to_numpy(dtype=float)

    xarr, yarr = array('d', x_fit), array('d', y_fit)
    exarr, eyarr = array('d', ex_fit), array('d', ey_fit)
    g_fit = ROOT.TGraphErrors(len(xarr), xarr, yarr, exarr, eyarr)
    g_fit.SetTitle('')
    g_fit.SetMarkerStyle(20)
    g_fit.SetMarkerColor(ROOT.kBlue)
    g_fit.Draw('AP')
    g_fit.GetXaxis().SetTitle('Proton Flux')
    g_fit.GetYaxis().SetTitle('Helium Flux')

    xmin, xmax = float(x_fit.min()), float(x_fit.max())
    fit_func = ROOT.TF1(f'lin_{i}', '[0]+[1]*x', xmin*0.9, xmax*1.1)
    fit_func.SetLineColor(ROOT.kRed); fit_func.SetLineWidth(2)

    # Perform ONLY fit here
    g_fit.Fit(fit_func, 'Q S')  # quiet + store
    p0 = fit_func.GetParameter(0)
    p1 = fit_func.GetParameter(1)
    p0_err = fit_func.GetParError(0)
    p1_err = fit_func.GetParError(1)
    chi2 = fit_func.GetChisquare()
    ndf  = fit_func.GetNDF()
    print(f"  Fit: p0={p0:.3e}±{p0_err:.3e}, p1={p1:.3f}±{p1_err:.3f}, chi2/ndf={chi2:.2f}/{ndf} -> {chi2/ndf if ndf>0 else 0:.2f}")

    fit_func.Draw('SAME')

    leg_fit = ROOT.TLegend(0.15, 0.80, 0.55, 0.95)
    leg_fit.SetBorderSize(0)
    leg_fit.AddEntry(g_fit,    'Observed Data', 'pe')
    leg_fit.AddEntry(fit_func, f'y = {p0:.2e} + {p1:.3f} x', 'l')
    leg_fit.Draw()

    lat2 = ROOT.TLatex(); lat2.SetNDC(); lat2.SetTextSize(0.045)
    # lat2.DrawLatex(0.70, 0.86, rig_label)

    # ---------- (3) Bottom-Left: Observed Helium time series ----------
    c_main.cd(3)
    # ROOT.gPad.SetGrid()
    ROOT.gPad.SetLeftMargin(0.11); ROOT.gPad.SetBottomMargin(0.10)
    ROOT.gPad.SetTopMargin(0.02);  ROOT.gPad.SetRightMargin(0.05)

    he_obs = df_observed[he_flux_col].to_numpy(dtype=float)
    he_err = df_observed['helium_flux_error'].to_numpy(dtype=float)
    ts_he  = ts_obs  # same dates as observed P-He pairs

    g_he = ROOT.TGraphErrors(len(ts_he), array('d', ts_he), array('d', he_obs),
                             array('d', [0]*len(ts_he)), array('d', he_err))
    g_he.SetTitle('')
    g_he.SetMarkerStyle(20)
    g_he.SetMarkerColor(ROOT.kGreen+2)
    g_he.SetLineColor(ROOT.kGreen+2)
    g_he.Draw('AP')
    xh = g_he.GetXaxis()
    config_time_axis(xh, xlim_start, xlim_end)
    g_he.GetYaxis().SetTitle('Helium Flux')

    lat3 = ROOT.TLatex(); 
    lat3.SetNDC(); 
    lat3.SetTextSize(0.05)
    # lat3.DrawLatex(0.15, 0.86, rig_label)

    # ---------- (4) Bottom-Right: Predicted Helium using ML (fill only missing dates in red) ----------
    # Compute predictions for all dates in full window
    df_ml_range_local['predicted_helium'] = p0 + p1 * df_ml_range_local['ml_proton']

    c_main.cd(4)
    # ROOT.gPad.SetGrid()
    ROOT.gPad.SetLeftMargin(0.11); ROOT.gPad.SetBottomMargin(0.10)
    ROOT.gPad.SetTopMargin(0.02);  ROOT.gPad.SetRightMargin(0.05)

    all_dates = df_ml_range_local['date'].to_numpy()
    all_ts    = np.array([dt_to_unix(d) for d in all_dates], dtype=float)

    # Observed helium at observed dates (blue)
    obs_mask = df_ml_range_local['is_observed'].to_numpy()
    obs_idx  = np.where(obs_mask)[0]
    if len(obs_idx) > 0:
        # Match each observed date to helium value from df_observed
        obs_ts = []
        obs_he = []
        obs_dates_series = df_ml_range_local.iloc[obs_idx]['date']
        obs_map = dict(zip(df_observed[p_date].astype('datetime64[ns]'), df_observed[he_flux_col].to_numpy()))
        for j, d in enumerate(obs_dates_series):
            if d in obs_map:
                obs_ts.append(all_ts[obs_idx[j]])
                obs_he.append(float(obs_map[d]))
        if len(obs_ts) > 0:
            g_obs_he = ROOT.TGraphErrors(len(obs_ts), array('d', obs_ts), array('d', obs_he))
            g_obs_he.SetMarkerStyle(20)
            g_obs_he.SetMarkerColor(ROOT.kBlue)
            g_obs_he.SetMarkerSize(0.8)

    # Predicted helium for missing dates (red)
    miss_mask = df_ml_range_local['is_missing_date'].to_numpy()
    miss_idx  = np.where(miss_mask)[0]
    if len(miss_idx) > 0:
        miss_ts  = all_ts[miss_idx]
        miss_val = df_ml_range_local.iloc[miss_idx]['predicted_helium'].to_numpy(dtype=float)
        g_pred_miss = ROOT.TGraphErrors(len(miss_ts), array('d', miss_ts), array('d', miss_val))
        g_pred_miss.SetMarkerStyle(21)
        g_pred_miss.SetMarkerColor(ROOT.kRed)
        g_pred_miss.SetMarkerSize(0.8)
    # 去掉此前对 g_pred_miss.GetXaxis().SetRangeUser(...) 的不统一操作
    mg = ROOT.TMultiGraph()
    mg.SetTitle('')
    if 'g_obs_he' in locals():     mg.Add(g_obs_he)
    if 'g_pred_miss' in locals():  mg.Add(g_pred_miss)
    mg.Draw('AP')
    xa_all = mg.GetXaxis()
    config_time_axis(xa_all, xlim_start, xlim_end)
    mg.GetYaxis().SetTitle('Helium Flux')

    leg_he = ROOT.TLegend(0.15, 0.80, 0.55, 0.95)
    leg_he.SetBorderSize(0)
    leg_he.SetTextSize(0.045)
    if 'g_obs_he' in locals():     leg_he.AddEntry(g_obs_he, 'Observed', 'p')
    if 'g_pred_miss' in locals():  leg_he.AddEntry(g_pred_miss, 'Predicted (missing dates)', 'p')
    # if 'g_pred_other' in locals(): leg_he.AddEntry(g_pred_other, 'Predicted (other)', 'p')
    leg_he.Draw()

    lat4 = ROOT.TLatex(); lat4.SetNDC(); lat4.SetTextSize(0.045)
    # lat4.DrawLatex(0.70, 0.86, rig_label)

    # ---------- Save outputs ----------
    c_main.Modified(); c_main.Update()
    png_path = os.path.join(results_dir, f'analysis_{rig_min_cur:.2f}_{rig_max_cur:.2f}_GV.png')
    c_main.SaveAs(png_path)
    print(f"  Figure saved: {png_path}")

    # Save CSV: full window dates with flags and predictions
    out_csv = os.path.join(results_dir, f'helium_prediction_{rig_min_cur:.2f}_{rig_max_cur:.2f}_GV.csv')
    df_save = df_ml_range_local[['date', 'ml_proton', 'predicted_helium', 'is_observed', 'is_missing_date']].copy()
    df_save.to_csv(out_csv, index=False)
    print(f"  CSV saved: {out_csv}")

    # Cleanup ROOT objects to avoid memory growth (仅一次)
    del c_main, g_fit, fit_func
    if 'g_obs_he' in locals():    del g_obs_he
    if 'g_pred_miss' in locals(): del g_pred_miss
    # if 'g_pred_other' in locals(): del g_pred_other

# 循环结束后统一提示
print("\nAll rigidity bins processed!")

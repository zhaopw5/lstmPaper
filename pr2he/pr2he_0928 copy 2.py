# -*- coding: utf-8 -*-
# Full program with fixes for:
# (1) Unify date key types using Unix seconds (int) to avoid Timestamp vs datetime64 mismatch;
# (3) Use DrawFrame to enforce a unified time window on all time-series pads;
# (5) Better memory handling and object ownership to mitigate growth in long loops.

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
# Helper style functions
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
    ROOT.SetOwnership(frame, True)
    return frame

def padded_limits(vals, pad_frac=0.08, fallback=(0.0, 1.0)):
    """Compute (ymin,ymax) with padding; ignore NaNs."""
    vals = np.array(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return fallback
    ymin = float(np.min(vals))
    ymax = float(np.max(vals))
    if ymin == ymax:
        if ymin == 0:
            ymin, ymax = -1.0, 1.0
        else:
            ymin *= 0.9
            ymax *= 1.1
    yr = ymax - ymin
    return ymin - pad_frac*yr, ymax + pad_frac*yr

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
    ml_col_idx = str(i + 5)  # ML file uses '1','2',... for rigidity bins; proton has +4 bins below 1.71 GV
    if ml_col_idx not in df_ml_range.columns:
        print(f"  ML column '{ml_col_idx}' not in ML data. Skip.")
        continue
    df_ml_range_local = df_ml_range[['date', ml_col_idx]].copy()
    df_ml_range_local = df_ml_range_local.rename(columns={ml_col_idx: 'ml_proton'})

    # 3) Flags: observed dates (based on df_observed), missing dates (in ML but not observed)
    # NOTE (1): we will also use Unix int seconds as the *key* for joins/mapping later.
    observed_dates = set(df_observed[p_date])
    df_ml_range_local['is_observed'] = df_ml_range_local['date'].isin(observed_dates)
    df_ml_range_local['is_missing_date'] = ~df_ml_range_local['is_observed']

    # 4) Build ROOT canvas with 4 pads
    c_main = ROOT.TCanvas(f'c_main_{i}', f'Proton-Helium Analysis {rig_label}', 1400, 1000)
    ROOT.SetOwnership(c_main, True)
    c_main.Divide(2, 2, 0.0001, 0.0001)

    # ======== (1) Top-Left: Observed Proton vs ML Proton (full window) ========
    c_main.cd(1)
    set_pad_margins(ROOT.gPad)

    # --- Observed series (dates, values, errors) ---
    dates_obs = df_observed[p_date].to_numpy()
    ts_obs = np.array([dt_to_unix(d) for d in dates_obs], dtype=float)
    proton_obs = df_observed[p_flux_col].to_numpy(dtype=float)
    proton_err = df_observed['proton_flux_error'].to_numpy(dtype=float)

    g_proton = ROOT.TGraphErrors(len(ts_obs), array('d', ts_obs), array('d', proton_obs),
                                 array('d', [0]*len(ts_obs)), array('d', proton_err))
    ROOT.SetOwnership(g_proton, True)
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
    ROOT.SetOwnership(g_proton_ml_full, True)
    g_proton_ml_full.SetMarkerStyle(24)
    g_proton_ml_full.SetMarkerSize(0.5)
    g_proton_ml_full.SetMarkerColor(ROOT.kRed)
    g_proton_ml_full.SetLineColor(ROOT.kRed)
    g_proton_ml_full.SetLineWidth(1)  # thin line to show continuity (drawn as points unless 'LP' used)

    # ---- Compute unified y-range: observed ± error & ML FULL series ----
    proton_low  = proton_obs - np.nan_to_num(proton_err, nan=0.0)
    proton_high = proton_obs + np.nan_to_num(proton_err, nan=0.0)
    ymin_pad, ymax_pad = padded_limits(np.concatenate([proton_low, proton_high, ml_vals_full]) if ml_vals_full.size>0
                                       else np.concatenate([proton_low, proton_high]))

    # Draw a frame with unified time window (require y range as well)
    frame1 = draw_time_frame(xlim_start, ymin_pad, xlim_end, ymax_pad, 'Proton Flux')

    # 1) Observed (blue, with errors)
    g_proton.Draw('P SAME')
    # 2) ML FULL series (red thin markers)
    g_proton_ml_full.Draw('P SAME')  # keep as points; use 'LP SAME' if you want lines

    # Legend & label
    leg_p = ROOT.TLegend(0.7, 0.77, 0.93, 0.95)
    ROOT.SetOwnership(leg_p, True)
    leg_p.SetBorderSize(0)
    leg_p.AddEntry(g_proton,            'AMS', 'pe')
    leg_p.AddEntry(g_proton_ml_full,    'Model', 'lp')
    leg_p.Draw()

    latex_r1 = ROOT.TLatex()
    ROOT.SetOwnership(latex_r1, True)
    latex_r1.SetNDC(); latex_r1.SetTextSize(0.045)
    latex_r1.DrawLatex(0.15, 0.86, rig_label)

    # ---------- (2) Top-Right: SINGLE fit on observed P vs He ----------
    c_main.cd(2)
    set_pad_margins(ROOT.gPad)
    ROOT.gStyle.SetOptFit(0)  # don't show default stats box

    x_fit = df_observed[p_flux_col].to_numpy(dtype=float)
    y_fit = df_observed[he_flux_col].to_numpy(dtype=float)
    ex_fit = df_observed['proton_flux_error'].to_numpy(dtype=float)
    ey_fit = df_observed['helium_flux_error'].to_numpy(dtype=float)

    xarr, yarr = array('d', x_fit), array('d', y_fit)
    exarr, eyarr = array('d', ex_fit), array('d', ey_fit)
    g_fit = ROOT.TGraphErrors(len(xarr), xarr, yarr, exarr, eyarr)
    ROOT.SetOwnership(g_fit, True)
    g_fit.SetTitle('')
    g_fit.SetMarkerStyle(20)
    g_fit.SetMarkerColor(ROOT.kBlue)
    g_fit.Draw('AP')
    g_fit.GetXaxis().SetTitle('Proton Flux')
    g_fit.GetYaxis().SetTitle('Helium Flux')

    xmin, xmax = float(np.nanmin(x_fit)), float(np.nanmax(x_fit))
    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmin == xmax:
        xmin, xmax = 0.0, 1.0
    fit_func = ROOT.TF1(f'lin_{i}', '[0]+[1]*x', xmin*0.9, xmax*1.1)
    ROOT.SetOwnership(fit_func, True)
    fit_func.SetLineColor(ROOT.kRed); fit_func.SetLineWidth(2)

    # Perform ONLY fit here
    g_fit.Fit(fit_func, 'Q S')  # quiet + store
    p0 = fit_func.GetParameter(0); p1 = fit_func.GetParameter(1)
    p0_err = fit_func.GetParError(0); p1_err = fit_func.GetParError(1)
    chi2 = fit_func.GetChisquare(); ndf  = fit_func.GetNDF()
    print(f"  Fit: p0={p0:.3e}±{p0_err:.3e}, p1={p1:.3f}±{p1_err:.3f}, chi2/ndf={chi2:.2f}/{ndf} -> {chi2/ndf if ndf>0 else 0:.2f}")

    fit_func.Draw('SAME')

    leg_fit = ROOT.TLegend(0.15, 0.80, 0.55, 0.95)
    ROOT.SetOwnership(leg_fit, True)
    leg_fit.SetBorderSize(0)
    leg_fit.AddEntry(g_fit,    'Observed Data', 'pe')
    leg_fit.AddEntry(fit_func, f'y = {p0:.2e} + {p1:.3f} x', 'l')
    leg_fit.Draw()

    # ---------- (3) Bottom-Left: Observed Helium time series ----------
    c_main.cd(3)
    set_pad_margins(ROOT.gPad)

    he_obs = df_observed[he_flux_col].to_numpy(dtype=float)
    he_err = df_observed['helium_flux_error'].to_numpy(dtype=float)
    ts_he  = ts_obs  # same dates as observed P-He pairs

    g_he = ROOT.TGraphErrors(len(ts_he), array('d', ts_he), array('d', he_obs),
                             array('d', [0]*len(ts_he)), array('d', he_err))
    ROOT.SetOwnership(g_he, True)
    g_he.SetTitle('')
    g_he.SetMarkerStyle(20)
    g_he.SetMarkerColor(ROOT.kGreen+2)
    g_he.SetLineColor(ROOT.kGreen+2)

    # y-range with padding (based on observed helium and its error)
    he_low  = he_obs - np.nan_to_num(he_err, nan=0.0)
    he_high = he_obs + np.nan_to_num(he_err, nan=0.0)
    ymin_he, ymax_he = padded_limits(np.concatenate([he_low, he_high]))

    frame3 = draw_time_frame(xlim_start, ymin_he, xlim_end, ymax_he, 'Helium Flux')
    g_he.Draw('P SAME')

    # ---------- (4) Bottom-Right: Predicted Helium using ML (fill only missing dates in red) ----------
    # Compute predictions for all dates in full window
    df_ml_range_local['predicted_helium'] = p0 + p1 * df_ml_range_local['ml_proton']

    c_main.cd(4)
    set_pad_margins(ROOT.gPad)

    all_dates = df_ml_range_local['date'].to_numpy()
    all_ts    = np.array([dt_to_unix(d) for d in all_dates], dtype=float)

    # Observed helium at observed dates (blue) -- FIX (1): unify keys as Unix seconds (int)
    g_obs_he = None
    obs_mask = df_ml_range_local['is_observed'].to_numpy()
    obs_idx  = np.where(obs_mask)[0]
    if len(obs_idx) > 0:
        obs_ts = []
        obs_he_vals = []
        # Build a mapping from observed date (as int seconds) -> observed helium
        obs_map = dict(zip(
            df_observed[p_date].map(dt_to_unix),
            df_observed[he_flux_col].to_numpy(dtype=float)
        ))
        # Iterate over ML-side observed dates, also keyed as int seconds
        obs_dates_series = df_ml_range_local.iloc[obs_idx]['date']
        for j, d in enumerate(obs_dates_series):
            key = dt_to_unix(d)
            if key in obs_map:
                obs_ts.append(all_ts[obs_idx[j]])           # X from ML grid (same int seconds)
                obs_he_vals.append(float(obs_map[key]))      # Y from observed helium
        if len(obs_ts) > 0:
            g_obs_he = ROOT.TGraphErrors(len(obs_ts), array('d', obs_ts), array('d', obs_he_vals))
            ROOT.SetOwnership(g_obs_he, True)
            g_obs_he.SetMarkerStyle(20)
            g_obs_he.SetMarkerColor(ROOT.kBlue)
            g_obs_he.SetMarkerSize(0.8)

    # Predicted helium for missing dates (red)
    g_pred_miss = None
    miss_mask = df_ml_range_local['is_missing_date'].to_numpy()
    miss_idx  = np.where(miss_mask)[0]
    if len(miss_idx) > 0:
        miss_ts  = all_ts[miss_idx]
        miss_val = df_ml_range_local.iloc[miss_idx]['predicted_helium'].to_numpy(dtype=float)
        g_pred_miss = ROOT.TGraphErrors(len(miss_ts), array('d', miss_ts), array('d', miss_val))
        ROOT.SetOwnership(g_pred_miss, True)
        g_pred_miss.SetMarkerStyle(21)
        g_pred_miss.SetMarkerColor(ROOT.kRed)
        g_pred_miss.SetMarkerSize(0.8)

    # Build multigraph and compute y-limits with padding for what we actually draw
    mg = ROOT.TMultiGraph()
    ROOT.SetOwnership(mg, True)
    mg.SetTitle('')
    y_stack = []
    if g_obs_he is not None:
        mg.Add(g_obs_he)
        # collect y values
        y_stack.extend(obs_he_vals)
    if g_pred_miss is not None:
        mg.Add(g_pred_miss)
        y_stack.extend(miss_val if miss_mask.any() else [])

    # If both empty, set a fallback
    ymin_r4, ymax_r4 = padded_limits(y_stack if len(y_stack) > 0 else [0.0, 1.0])

    # Enforce unified time window via a frame (draw before mg)
    frame4 = draw_time_frame(xlim_start, ymin_r4, xlim_end, ymax_r4, 'Helium Flux')
    mg.Draw('AP')

    leg_he = ROOT.TLegend(0.15, 0.80, 0.55, 0.95)
    ROOT.SetOwnership(leg_he, True)
    leg_he.SetBorderSize(0)
    leg_he.SetTextSize(0.045)
    if g_obs_he is not None:     leg_he.AddEntry(g_obs_he, 'Observed', 'p')
    if g_pred_miss is not None:  leg_he.AddEntry(g_pred_miss, 'Predicted (missing dates)', 'p')
    leg_he.Draw()

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

    # ---------- Cleanup to avoid memory growth ----------
    # Top-left
    del frame1, g_proton, g_proton_ml_full, leg_p, latex_r1
    # Top-right
    del g_fit, fit_func, leg_fit
    # Bottom-left
    del frame3, g_he
    # Bottom-right
    if 'frame4' in locals(): del frame4
    if mg: del mg
    if g_obs_he is not None: del g_obs_he
    if g_pred_miss is not None: del g_pred_miss
    if 'leg_he' in locals(): del leg_he

    # Canvas
    del c_main

# Done
print("\nAll rigidity bins processed!")

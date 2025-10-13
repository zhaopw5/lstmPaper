# -*- coding: utf-8 -*-
# Full program with fixes:
# (A) Right-bottom pad uses date axis identical to other pads by drawing a frame first,
#     then drawing the multigraph with 'P SAME' (no axis override).
# (B) Missing-data completion logic:
#     - If a missing day has ANY AMS data (proton or helium at ANY rigidity): treat as SEP-like -> linear interpolation on observed He time series.
#     - Else (no AMS data at all that day): treat as full-missing -> fill using ML mapping (p0 + p1 * ML_proton).
# (C) Memory handling: SetOwnership + explicit deletions.

import pandas as pd
import numpy as np
import ROOT
from array import array
import os

# -----------------------------
# User-configurable parameters
# -----------------------------
full_start = '2014-07-01'
full_end   = '2015-05-01'

# Rigidity bins (min/max GV)
rig_bin = [1.71,1.92,2.15,2.4,2.67,2.97,3.29,3.64,4.02,4.43,4.88,5.37,5.9,
           6.47,7.09,7.76,8.48,9.26,10.1,11,13,16.6,22.8,33.5,48.5,69.7,100]

rig_bin_he = [1.71,1.92,2.15,2.4,2.67,2.97,3.29,3.64,4.02,4.43,4.88,5.37,5.9,
              6.47,7.09,7.76,8.48,9.26,10.1,11,13,16.6,22.8,33.5,48.5,69.7,100]

rig_bin_pr = [1.00,1.16,1.33,1.51,1.71,1.92,2.15,2.4,2.67,2.97,3.29,3.64,4.02,
              4.43,4.88,5.37,5.9,6.47,7.09,7.76,8.48,9.26,10.1,11,13,16.6,22.8,
              33.5,48.5,69.7,100]

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
ROOT.gStyle.SetTimeOffset(0)  # Unix epoch

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

# Precompute "any-data" date sets for SEP-like detection (ANY rigidity, ANY of P/He)
dates_any_p  = set(df_p[p_date].map(dt_to_unix))
dates_any_he = set(df_he[p_date].map(dt_to_unix))
dates_any_all = dates_any_p.union(dates_any_he)  # if a day has any AMS data at any rigidity, it goes here

# -----------------------------
# Helper: observed P-He pairs for a rigidity bin
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

    # Inner join by date (keep only dates with both P and He)
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

    # 1) Observed P-He pairs for this bin
    df_observed = select_observed_pair(df_p, df_he, rig_min_cur, rig_max_cur)
    if df_observed.empty or len(df_observed) < 2:
        print(f"  Not enough observed P-He pairs ({len(df_observed)}). Skip.")
        continue
    print(f"  Found {len(df_observed)} observed P-He paired dates in [{full_start} .. {full_end}].")

    # 2) ML column mapping for this bin
    ml_col_idx = str(i + 5)  # proton ML bins start 4 bins below 1.71 GV
    if ml_col_idx not in df_ml_range.columns:
        print(f"  ML column '{ml_col_idx}' not in ML data. Skip.")
        continue
    df_ml_range_local = df_ml_range[['date', ml_col_idx]].copy()
    df_ml_range_local = df_ml_range_local.rename(columns={ml_col_idx: 'ml_proton'})

    # Flags by observed dates (inner-joined dates for this bin)
    observed_dates = set(df_observed[p_date])
    df_ml_range_local['is_observed'] = df_ml_range_local['date'].isin(observed_dates)
    df_ml_range_local['is_missing_date'] = ~df_ml_range_local['is_observed']

    # For CSV: pre-fill flags and placeholders
    df_ml_range_local['is_sep_like'] = False
    df_ml_range_local['fill_method'] = 'none'  # 'observed' | 'sep_interp' | 'ml_pred' | 'none'

    # 3) Build canvas
    c_main = ROOT.TCanvas(f'c_main_{i}', f'Proton-Helium Analysis {rig_label}', 1400, 1000)
    ROOT.SetOwnership(c_main, True)
    c_main.Divide(2, 2, 0.0001, 0.0001)

    # ======== (1) Top-Left: Observed Proton vs ML Proton (full window) ========
    c_main.cd(1)
    set_pad_margins(ROOT.gPad)

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

    ml_dates_full = df_ml_range_local['date'].to_numpy()
    ts_ml_full = np.array([dt_to_unix(d) for d in ml_dates_full], dtype=float)
    ml_vals_full = df_ml_range_local['ml_proton'].to_numpy(dtype=float)

    g_proton_ml_full = ROOT.TGraph(len(ts_ml_full), array('d', ts_ml_full), array('d', ml_vals_full))
    ROOT.SetOwnership(g_proton_ml_full, True)
    g_proton_ml_full.SetMarkerStyle(24)
    g_proton_ml_full.SetMarkerSize(0.5)
    g_proton_ml_full.SetMarkerColor(ROOT.kRed)
    g_proton_ml_full.SetLineColor(ROOT.kRed)
    g_proton_ml_full.SetLineWidth(1)

    # y-range
    proton_low  = proton_obs - np.nan_to_num(proton_err, nan=0.0)
    proton_high = proton_obs + np.nan_to_num(proton_err, nan=0.0)
    ymin_pad, ymax_pad = padded_limits(np.concatenate([proton_low, proton_high, ml_vals_full]) if ml_vals_full.size>0
                                       else np.concatenate([proton_low, proton_high]))

    frame1 = draw_time_frame(xlim_start, ymin_pad, xlim_end, ymax_pad, 'Proton Flux')
    g_proton.Draw('P SAME')
    g_proton_ml_full.Draw('P SAME')  # use 'LP SAME' if you prefer a connecting line

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

    # ---------- (2) Top-Right: He vs P fit ----------
    c_main.cd(2)
    set_pad_margins(ROOT.gPad)
    ROOT.gStyle.SetOptFit(0)

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

    g_fit.Fit(fit_func, 'Q S')
    p0 = fit_func.GetParameter(0); p1 = fit_func.GetParameter(1)
    p0_err = fit_func.GetParError(0); p1_err = fit_func.GetParError(1)
    chi2 = fit_func.GetChisquare();  ndf   = fit_func.GetNDF()
    print(f"  Fit: p0={p0:.3e}±{p0_err:.3e}, p1={p1:.3f}±{p1_err:.3f}, chi2/ndf={chi2:.2f}/{ndf} -> {chi2/ndf if ndf>0 else 0:.2f}")

    fit_func.Draw('SAME')

    leg_fit = ROOT.TLegend(0.15, 0.80, 0.55, 0.95)
    ROOT.SetOwnership(leg_fit, True)
    leg_fit.SetBorderSize(0)
    leg_fit.AddEntry(g_fit,    'Observed Data', 'pe')
    leg_fit.AddEntry(fit_func, f'y = {p0:.2e} + {p1:.3f} x', 'l')
    leg_fit.Draw()

    # ---------- (3) Bottom-Left: Observed Helium vs time ----------
    c_main.cd(3)
    set_pad_margins(ROOT.gPad)

    he_obs = df_observed[he_flux_col].to_numpy(dtype=float)
    he_err = df_observed['helium_flux_error'].to_numpy(dtype=float)
    ts_he  = ts_obs  # same as observed P-He pair dates (already sorted)

    g_he = ROOT.TGraphErrors(len(ts_he), array('d', ts_he), array('d', he_obs),
                             array('d', [0]*len(ts_he)), array('d', he_err))
    ROOT.SetOwnership(g_he, True)
    g_he.SetTitle('')
    g_he.SetMarkerStyle(20)
    g_he.SetMarkerColor(ROOT.kGreen+2)
    g_he.SetLineColor(ROOT.kGreen+2)

    he_low  = he_obs - np.nan_to_num(he_err, nan=0.0)
    he_high = he_obs + np.nan_to_num(he_err, nan=0.0)
    ymin_he, ymax_he = padded_limits(np.concatenate([he_low, he_high]))

    frame3 = draw_time_frame(xlim_start, ymin_he, xlim_end, ymax_he, 'Helium Flux')
    g_he.Draw('P SAME')

    # ---------- (4) Bottom-Right: Observed (blue) + SEP-interp (black) + ML-pred (red) ----------
    c_main.cd(4)
    set_pad_margins(ROOT.gPad)

    # Predictions from ML mapping for all dates
    df_ml_range_local['predicted_helium'] = p0 + p1 * df_ml_range_local['ml_proton']

    all_dates = df_ml_range_local['date'].to_numpy()
    all_ts    = np.array([dt_to_unix(d) for d in all_dates], dtype=float)

    # Build observed helium at observed dates (blue) using Unix-sec keyed map
    g_obs_he = None
    obs_mask = df_ml_range_local['is_observed'].to_numpy()
    obs_idx  = np.where(obs_mask)[0]
    if len(obs_idx) > 0:
        obs_ts_draw = []
        obs_he_vals = []
        obs_map = dict(zip(
            df_observed[p_date].map(dt_to_unix),
            df_observed[he_flux_col].to_numpy(dtype=float)
        ))
        obs_dates_series = df_ml_range_local.iloc[obs_idx]['date']
        for j, d in enumerate(obs_dates_series):
            key = dt_to_unix(d)
            if key in obs_map:
                obs_ts_draw.append(all_ts[obs_idx[j]])
                obs_he_vals.append(float(obs_map[key]))
                # mark CSV fill method
                df_ml_range_local.loc[df_ml_range_local.index[obs_idx[j]], 'fill_method'] = 'observed'
        if len(obs_ts_draw) > 0:
            g_obs_he = ROOT.TGraphErrors(len(obs_ts_draw), array('d', obs_ts_draw), array('d', obs_he_vals))
            ROOT.SetOwnership(g_obs_he, True)
            g_obs_he.SetMarkerStyle(20)
            g_obs_he.SetMarkerColor(ROOT.kBlue)
            g_obs_he.SetMarkerSize(0.8)

    # Split missing dates into SEP-like (has any AMS data that day) vs full-missing (no AMS data that day)
    miss_mask = df_ml_range_local['is_missing_date'].to_numpy()
    miss_idx  = np.where(miss_mask)[0]
    idx_sep_like = []     # indices (in df_ml_range_local) for SEP-like missing days
    idx_full_missing = [] # indices for days without any AMS data

    for idx in miss_idx:
        key = dt_to_unix(df_ml_range_local.iloc[idx]['date'])
        if key in dates_any_all:
            idx_sep_like.append(idx)
            df_ml_range_local.loc[df_ml_range_local.index[idx], 'is_sep_like'] = True
        else:
            idx_full_missing.append(idx)

    # (a) SEP-like days -> linear interpolation on observed helium time series
    g_sep_interp = None
    y_sep_interp = []
    ts_sep_plot  = []
    if len(idx_sep_like) > 0 and len(ts_he) >= 2:
        ts_he_sorted = np.array(sorted(ts_he))           # ensure ascending
        he_obs_sorted = he_obs[np.argsort(ts_he)]        # align with sorted times

        ts_sep = all_ts[idx_sep_like]
        # Keep only SEP dates within interpolation range (no extrapolation)
        in_range_mask = (ts_sep >= ts_he_sorted[0]) & (ts_sep <= ts_he_sorted[-1])
        valid_idx = np.where(in_range_mask)[0]
        if valid_idx.size > 0:
            ts_sep_valid = ts_sep[valid_idx]
            y_interp = np.interp(ts_sep_valid, ts_he_sorted, he_obs_sorted)
            ts_sep_plot = ts_sep_valid.tolist()
            y_sep_interp = y_interp.tolist()

            # mark CSV
            for k in valid_idx:
                df_ml_range_local.loc[df_ml_range_local.index[idx_sep_like[k]], 'fill_method'] = 'sep_interp'

            g_sep_interp = ROOT.TGraphErrors(len(ts_sep_plot), array('d', ts_sep_plot), array('d', y_sep_interp))
            ROOT.SetOwnership(g_sep_interp, True)
            g_sep_interp.SetMarkerStyle(22)
            g_sep_interp.SetMarkerColor(ROOT.kMagenta)
            g_sep_interp.SetMarkerSize(0.9)

    # (b) Full-missing days -> ML mapping
    g_pred_ml = None
    y_pred_ml = []
    ts_ml_plot = []
    if len(idx_full_missing) > 0:
        ts_ml_plot = all_ts[idx_full_missing].tolist()
        y_pred_ml = df_ml_range_local.iloc[idx_full_missing]['predicted_helium'].to_numpy(dtype=float).tolist()

        # mark CSV
        df_ml_range_local.loc[df_ml_range_local.index[idx_full_missing], 'fill_method'] = 'ml_pred'

        g_pred_ml = ROOT.TGraphErrors(len(ts_ml_plot), array('d', ts_ml_plot), array('d', y_pred_ml))
        ROOT.SetOwnership(g_pred_ml, True)
        g_pred_ml.SetMarkerStyle(21)
        g_pred_ml.SetMarkerColor(ROOT.kRed)
        g_pred_ml.SetMarkerSize(0.8)

    # Y-range based on what we'll draw
    y_stack = []
    if g_obs_he is not None:
        y_stack.extend(obs_he_vals)
    if g_sep_interp is not None:
        y_stack.extend(y_sep_interp)
    if g_pred_ml is not None:
        y_stack.extend(y_pred_ml)
    ymin_r4, ymax_r4 = padded_limits(y_stack if len(y_stack) > 0 else [0.0, 1.0])

    # Enforce unified time window with a frame (DATE axis preserved)
    frame4 = draw_time_frame(xlim_start, ymin_r4, xlim_end, ymax_r4, 'Helium Flux')

    # Build multigraph and draw WITHOUT 'A' (do not create axes)
    mg = ROOT.TMultiGraph()
    ROOT.SetOwnership(mg, True)
    mg.SetTitle('')
    if g_obs_he is not None:   mg.Add(g_obs_he)
    if g_sep_interp is not None: mg.Add(g_sep_interp)
    if g_pred_ml is not None: mg.Add(g_pred_ml)
    mg.Draw('P SAME')  # keep the frame's date axis

    leg_he = ROOT.TLegend(0.12, 0.76, 0.55, 0.95)
    ROOT.SetOwnership(leg_he, True)
    leg_he.SetBorderSize(0)
    leg_he.SetTextSize(0.045)
    if g_obs_he is not None:     leg_he.AddEntry(g_obs_he,    'Observed', 'p')
    if g_sep_interp is not None: leg_he.AddEntry(g_sep_interp,'Interpolated (SEP)', 'p')
    if g_pred_ml is not None:    leg_he.AddEntry(g_pred_ml,   'Predicted (ML)', 'p')
    leg_he.Draw()

    # ---------- Save outputs ----------
    c_main.Modified(); c_main.Update()
    png_path = os.path.join(results_dir, f'analysis_{rig_min_cur:.2f}_{rig_max_cur:.2f}_GV.png')
    c_main.SaveAs(png_path)
    print(f"  Figure saved: {png_path}")

    # Save CSV: add classification and fill methods
    out_csv = os.path.join(results_dir, f'helium_prediction_{rig_min_cur:.2f}_{rig_max_cur:.2f}_GV.csv')
    df_save = df_ml_range_local[['date', 'ml_proton', 'predicted_helium',
                                 'is_observed', 'is_missing_date',
                                 'is_sep_like', 'fill_method']].copy()
    df_save.to_csv(out_csv, index=False)
    print(f"  CSV saved: {out_csv}")

    # ---------- Cleanup ----------
    # Top-left
    del frame1, g_proton, g_proton_ml_full, leg_p, latex_r1
    # Top-right
    del g_fit, fit_func, leg_fit
    # Bottom-left
    del frame3, g_he
    # Bottom-right
    del frame4, mg
    if g_obs_he is not None:   del g_obs_he
    if g_sep_interp is not None: del g_sep_interp
    if g_pred_ml is not None:  del g_pred_ml
    del leg_he
    del c_main

print("\nAll rigidity bins processed!")

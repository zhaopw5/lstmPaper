# -*- coding: utf-8 -*-
# Aggregated helium completion over multiple time segments and rigidity bins.
# For each segment & bin:
#   - Build observed P-He pairs, fit He = p0 + p1 * P (if >=2 pairs)
#   - Classify missing days into SEP-like (has any AMS data that day) vs NODATA (no AMS data)
#   - Fill SEP-like via linear interpolation on observed He time series (no extrapolation)
#   - Fill NODATA via ML mapping p0 + p1 * ML_proton (only if fit is available)
# Produce a consolidated CSV with columns:
#   date YYYY-MM-DD, rigidity_min GV, rigidity_max GV,
#   helium_flux m^-2sr^-1s^-1GV^-1, SEPorNODATA  (OBSERVED | SEP | NODATA)

import pandas as pd
import numpy as np
import ROOT
from array import array
import os

# -----------------------------
# User-configurable parameters
# -----------------------------
MAKE_PLOTS = True  # set True to draw 2x2 figures per (segment, bin)

# Full analysis window (absolute bounds for clipping)
GLOBAL_START = '2011-05-20'
GLOBAL_END   = '2019-10-29'

# Time segments (inclusive bounds as strings)
TIME_SEGMENTS = [
    ('2011-05-20', '2011-12-31'),
    ('2012-01-01', '2012-12-31'),
    ('2013-01-01', '2013-12-31'),
    ('2014-01-01', '2014-06-30'),
    ('2014-07-01', '2015-05-01'),
    ('2015-05-01', '2015-12-31'),
    ('2016-01-01', '2016-12-31'),
    ('2017-01-01', '2017-12-31'),
    ('2018-01-01', '2018-12-31'),
    ('2019-01-01', '2019-12-29'),
]

# Rigidity bins (min/max GV) for Helium and for Proton (ML file indexing starts below 1.71 GV)
rig_bin = [1.71,1.92,2.15,2.4,2.67,2.97,3.29,3.64,4.02,4.43,4.88,5.37,5.9,
           6.47,7.09,7.76,8.48,9.26,10.1,11,13,16.6,22.8,33.5,48.5,69.7,100]
rig_bin_he = rig_bin[:]  # same
rig_bin_pr = [1.00,1.16,1.33,1.51,1.71,1.92,2.15,2.4,2.67,2.97,3.29,3.64,4.02,
              4.43,4.88,5.37,5.9,6.47,7.09,7.76,8.48,9.26,10.1,11,13,16.6,22.8,
              33.5,48.5,69.7,100]

# Input paths
PROTON_CSV = '/home/zpw/Files/lstmPaper/data/raw_data/ams/proton.csv'
HELIUM_CSV = '/home/zpw/Files/lstmPaper/data/raw_data/ams/helium.csv'
ML_CSV     = '/home/zpw/Files/AMS_NeutronMonitors/AMS/paperPlots/data/lightning_logs/version_3/2011-01-01-2024-07-31_pred_ams_updated.csv'

# Output directory/files
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()
results_dir = os.path.join(script_dir, 'results_root')
os.makedirs(results_dir, exist_ok=True)

FINAL_OUT = os.path.join(results_dir, f'helium_completed_{GLOBAL_START}_{GLOBAL_END}.csv')

# -----------------------------
# ROOT styles (only used if MAKE_PLOTS=True)
# -----------------------------
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptFit(1111)
ROOT.gStyle.SetTimeOffset(0)  # Unix epoch for ROOT time axes

def dt_to_unix(d):
    """datetime-like -> Unix seconds (int)."""
    return int(pd.Timestamp(d).timestamp())

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

def set_pad_margins(pad, left=0.11, bottom=0.10, top=0.02, right=0.05):
    pad.SetLeftMargin(left); pad.SetBottomMargin(bottom)
    pad.SetTopMargin(top);   pad.SetRightMargin(right)

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
# Load global data (clip once to GLOBAL window)
# -----------------------------
print("Loading data...")
df_p_all  = pd.read_csv(PROTON_CSV)
df_he_all = pd.read_csv(HELIUM_CSV)
df_ml_all = pd.read_csv(ML_CSV)

df_p_all[p_date]  = pd.to_datetime(df_p_all[p_date])
df_he_all[p_date] = pd.to_datetime(df_he_all[p_date])
df_ml_all['date'] = pd.to_datetime(df_ml_all['date'])

g_start = pd.to_datetime(GLOBAL_START)
g_end   = pd.to_datetime(GLOBAL_END)

df_p_all  = df_p_all[(df_p_all[p_date]  >= g_start) & (df_p_all[p_date]  <= g_end)].copy()
df_he_all = df_he_all[(df_he_all[p_date] >= g_start) & (df_he_all[p_date] <= g_end)].copy()
df_ml_all = df_ml_all[(df_ml_all['date'] >= g_start) & (df_ml_all['date'] <= g_end)].copy()

# -----------------------------
# Helpers
# -----------------------------
def select_observed_pair(df_p, df_he, rig_min, rig_max):
    """Return inner-joined observed P and He within the df_p/df_he frames' time span for a given rigidity bin."""
    p_sel = df_p[(df_p['rigidity_min GV'] == rig_min) & (df_p['rigidity_max GV'] == rig_max)].copy()
    he_sel = df_he[(df_he['rigidity_min GV'] == rig_min) & (df_he['rigidity_max GV'] == rig_max)].copy()
    if p_sel.empty or he_sel.empty:
        return pd.DataFrame()

    # Quadrature errors
    p_sel['proton_flux_error'] = np.sqrt(p_sel[p_stat_col]**2 + p_sel[p_td_col]**2)
    he_sel['helium_flux_error'] = np.sqrt(he_sel[he_stat_col]**2 + he_sel[he_td_col]**2)

    df_obs = pd.merge(
        p_sel[[p_date, p_flux_col, 'proton_flux_error']],
        he_sel[[p_date, he_flux_col, 'helium_flux_error']],
        on=p_date, how='inner'
    ).sort_values(p_date)
    return df_obs

def ml_col_index_for_bin(i_bin_starting_at_1p71):
    """Map helium bin index i to ML proton column index string in df_ml (1-based strings)."""
    return str(i_bin_starting_at_1p71 + 5)  # proton ML has 4 bins below 1.71 GV

# -----------------------------
# Aggregation containers
# -----------------------------
final_rows = []  # dicts with target columns
emitted = set()  # to avoid duplicates: key = (date_str, rig_min, rig_max)

# -----------------------------
# Main nested loops: segment -> bin
# -----------------------------
for seg_idx, (seg_start_str, seg_end_str) in enumerate(TIME_SEGMENTS, start=1):
    seg_start = pd.to_datetime(seg_start_str)
    seg_end   = pd.to_datetime(seg_end_str)
    # clip to global window
    if seg_end < g_start or seg_start > g_end:
        continue
    seg_start = max(seg_start, g_start)
    seg_end   = min(seg_end, g_end)

    print(f"\n=== Segment {seg_idx}: {seg_start.date()} .. {seg_end.date()} ===")

    # Slice dataframes to this segment
    df_p_seg  = df_p_all[(df_p_all[p_date] >= seg_start) & (df_p_all[p_date] <= seg_end)].copy()
    df_he_seg = df_he_all[(df_he_all[p_date] >= seg_start) & (df_he_all[p_date] <= seg_end)].copy()
    df_ml_seg = df_ml_all[(df_ml_all['date'] >= seg_start) & (df_ml_all['date'] <= seg_end)].copy()

    # For SEP-like detection: any-data-day set within this segment (ANY rigidity, P or He)
    dates_any_p  = set(df_p_seg[p_date].map(dt_to_unix))
    dates_any_he = set(df_he_seg[p_date].map(dt_to_unix))
    dates_any_all = dates_any_p.union(dates_any_he)

    # For plotting, precompute x-limits (unix)
    xlim_start = dt_to_unix(seg_start)
    xlim_end   = dt_to_unix(seg_end)

    for i in range(len(rig_bin) - 1):
        rig_min_cur = rig_bin[i]
        rig_max_cur = rig_bin[i + 1]
        rig_label = f'{rig_min_cur:.2f}-{rig_max_cur:.2f} GV'
        print(f"  Bin {i+1:02d}: {rig_label}")

        # Observed pairs in this segment and bin
        df_obs = select_observed_pair(df_p_seg, df_he_seg, rig_min_cur, rig_max_cur)
        n_obs = len(df_obs)
        if n_obs == 0:
            print("    No observed P-He pairs. Only original He (if any) will be emitted; no fill.")
        else:
            print(f"    Observed P-He pairs: {n_obs}")

        # Build ML slice and map to a single ML proton column for this bin
        ml_col_idx = ml_col_index_for_bin(i)  # '5','6',...
        if ml_col_idx not in df_ml_seg.columns:
            print(f"    ML column '{ml_col_idx}' not found in ML CSV for this segment. Skip ML fill.")
            df_ml_bin = pd.DataFrame(columns=['date','ml_proton'])
        else:
            df_ml_bin = df_ml_seg[['date', ml_col_idx]].rename(columns={ml_col_idx: 'ml_proton'}).copy()

        # ------------------------
        # Emit OBSERVED helium rows first (these are authoritative)
        # ------------------------
        if not df_he_seg.empty:
            he_sel_emit = df_he_seg[(df_he_seg['rigidity_min GV'] == rig_min_cur) &
                                    (df_he_seg['rigidity_max GV'] == rig_max_cur)][[p_date, he_flux_col]].copy()
            he_sel_emit = he_sel_emit.sort_values(p_date)
            for _, row in he_sel_emit.iterrows():
                d = row[p_date]
                flux = float(row[he_flux_col])
                key = (d.strftime('%Y-%m-%d'), rig_min_cur, rig_max_cur)
                if key in emitted:
                    continue
                final_rows.append({
                    'date YYYY-MM-DD': d.strftime('%Y-%m-%d'),
                    'rigidity_min GV': rig_min_cur,
                    'rigidity_max GV': rig_max_cur,
                    'helium_flux m^-2sr^-1s^-1GV^-1': flux,
                    'SEPorNODATA': 'OBSERVED'
                })
                emitted.add(key)

        # If we have less than 2 observed pairs, we cannot fit; then we cannot do ML mapping for NODATA days.
        can_fit = (n_obs >= 2)

        # Prepare observed-He time series (for interpolation on SEP-like days)
        ts_obs = None
        he_obs_vals = None
        he_err_vals = None
        if n_obs >= 1:
            dates_obs_arr = df_obs[p_date].to_numpy()
            ts_obs = np.array([dt_to_unix(d) for d in dates_obs_arr], dtype=float)
            he_obs_vals = df_obs[he_flux_col].to_numpy(dtype=float)
            he_err_vals = df_obs['helium_flux_error'].to_numpy(dtype=float)
            # Ensure strictly increasing x for np.interp
            order = np.argsort(ts_obs)
            ts_obs = ts_obs[order]
            he_obs_vals = he_obs_vals[order]
            he_err_vals = he_err_vals[order]

        # Fit He vs P, if possible (for ML mapping on NODATA days)
        p0 = p1 = None
        if can_fit:
            x_fit = df_obs[p_flux_col].to_numpy(dtype=float)
            y_fit = df_obs[he_flux_col].to_numpy(dtype=float)
            xmin, xmax = float(np.nanmin(x_fit)), float(np.nanmax(x_fit))
            if not np.isfinite(xmin) or not np.isfinite(xmax) or xmin == xmax:
                can_fit = False
            else:
                fit_func = ROOT.TF1(f'lin_s{seg_idx}_b{i}', '[0]+[1]*x', xmin*0.9, xmax*1.1)
                graph = ROOT.TGraph(len(x_fit), array('d', x_fit), array('d', y_fit))
                graph.Fit(fit_func, 'Q S')
                p0 = fit_func.GetParameter(0)
                p1 = fit_func.GetParameter(1)
                del fit_func, graph

        # ------------------------
        # Classify missing dates inside this segment grid
        # ------------------------
        if df_ml_bin.empty:
            date_grid = pd.date_range(seg_start, seg_end, freq='D')
            df_grid = pd.DataFrame({'date': date_grid})
            df_grid['ml_proton'] = np.nan
        else:
            df_grid = df_ml_bin.copy()

        # Determine observed helium existence for this bin
        observed_dates_set = set()
        if not df_he_seg.empty:
            he_dates_bin = df_he_seg[(df_he_seg['rigidity_min GV'] == rig_min_cur) &
                                     (df_he_seg['rigidity_max GV'] == rig_max_cur)][p_date]
            observed_dates_set = set(pd.to_datetime(he_dates_bin).dt.normalize())

        df_grid['is_observed'] = df_grid['date'].dt.normalize().isin(observed_dates_set)
        df_grid['is_missing_date'] = ~df_grid['is_observed']

        # Split missing: SEP-like vs NODATA
        df_grid['is_sep_like'] = False
        if len(dates_any_all) > 0:
            df_grid.loc[df_grid['is_missing_date'] &
                        df_grid['date'].map(lambda d: dt_to_unix(d) in dates_any_all), 'is_sep_like'] = True

        # Build a map for observed He in this bin (for plotting)
        obs_he_map = {}
        if not df_he_seg.empty:
            he_sel_map = df_he_seg[(df_he_seg['rigidity_min GV'] == rig_min_cur) &
                                   (df_he_seg['rigidity_max GV'] == rig_max_cur)][[p_date, he_flux_col]]
            obs_he_map = {pd.Timestamp(r[p_date]).normalize(): float(r[he_flux_col]) for _, r in he_sel_map.iterrows()}

        # ------------------------
        # Emit filled rows (and plot arrays)
        # ------------------------
        # For plotting on bottom-right:
        br_obs_t = []; br_obs_y = []
        br_sep_t = []; br_sep_y = []
        br_ml_t  = []; br_ml_y  = []

        # (Observed already emitted earlier; but also prepare observed arrays using df_obs)
        if n_obs >= 1:
            # use df_obs times and helium for observed scatter (consistent with top-right pair selection)
            br_obs_t = ts_obs.tolist()
            br_obs_y = he_obs_vals.tolist()

        for _, r in df_grid.iterrows():
            d = pd.Timestamp(r['date']).normalize()
            date_str = d.strftime('%Y-%m-%d')
            key = (date_str, rig_min_cur, rig_max_cur)
            # Skip if already emitted as OBSERVED
            if key in emitted:
                continue
            if r['is_observed']:
                continue

            # Missing day:
            if bool(r['is_sep_like']):
                # SEP-like -> linear interpolation (no extrapolation)
                if ts_obs is not None and len(ts_obs) >= 2:
                    t = float(dt_to_unix(d))
                    if t >= ts_obs[0] and t <= ts_obs[-1]:
                        he_val = float(np.interp(t, ts_obs, he_obs_vals))
                        # aggregate row
                        final_rows.append({
                            'date YYYY-MM-DD': date_str,
                            'rigidity_min GV': rig_min_cur,
                            'rigidity_max GV': rig_max_cur,
                            'helium_flux m^-2sr^-1s^-1GV^-1': he_val,
                            'SEPorNODATA': 'SEP'
                        })
                        emitted.add(key)
                        # plotting arrays
                        br_sep_t.append(t); br_sep_y.append(he_val)
                # else: no interpolation possible -> skip
            else:
                # NODATA -> ML map if fit & ml_proton available
                if can_fit and ('ml_proton' in r) and np.isfinite(r['ml_proton']):
                    he_val = float(p0 + p1 * float(r['ml_proton']))
                    final_rows.append({
                        'date YYYY-MM-DD': date_str,
                        'rigidity_min GV': rig_min_cur,
                        'rigidity_max GV': rig_max_cur,
                        'helium_flux m^-2sr^-1s^-1GV^-1': he_val,
                        'SEPorNODATA': 'NODATA'
                    })
                    emitted.add(key)
                    br_ml_t.append(float(dt_to_unix(d))); br_ml_y.append(he_val)
                # else: skip (cannot fill)

        # ------------------------
        # Optional plotting per (segment, bin)
        # ------------------------
        if MAKE_PLOTS:
            # Create canvas
            c_main = ROOT.TCanvas(f'c_seg{seg_idx}_bin{i}', f'S{seg_idx} {rig_label}', 1400, 1000)
            ROOT.SetOwnership(c_main, True)
            c_main.Divide(2, 2, 0.0001, 0.0001)

            # ======== (1) Top-Left: Observed Proton vs ML Proton (full window) ========
            c_main.cd(1)
            set_pad_margins(ROOT.gPad)

            # Observed proton (from df_obs)
            g_proton = None
            ymin1_vals = []
            if n_obs >= 1:
                ts_obs_proton = np.array([dt_to_unix(d) for d in df_obs[p_date].to_numpy()], dtype=float)
                proton_obs_vals = df_obs[p_flux_col].to_numpy(dtype=float)
                proton_err_vals = df_obs['proton_flux_error'].to_numpy(dtype=float)
                ymin1_vals.extend((proton_obs_vals - np.nan_to_num(proton_err_vals)).tolist())
                ymin1_vals.extend((proton_obs_vals + np.nan_to_num(proton_err_vals)).tolist())
                g_proton = ROOT.TGraphErrors(len(ts_obs_proton),
                                             array('d', ts_obs_proton),
                                             array('d', proton_obs_vals),
                                             array('d', [0]*len(ts_obs_proton)),
                                             array('d', proton_err_vals))
                ROOT.SetOwnership(g_proton, True)
                g_proton.SetMarkerStyle(20); g_proton.SetMarkerSize(0.5)
                g_proton.SetMarkerColor(ROOT.kBlue); g_proton.SetLineColor(ROOT.kBlue)

            # ML proton full series within segment
            g_proton_ml_full = None
            if not df_ml_bin.empty:
                ts_ml_full = np.array([dt_to_unix(d) for d in df_ml_bin['date'].to_numpy()], dtype=float)
                ml_vals_full = df_ml_bin['ml_proton'].to_numpy(dtype=float)
                ymin1_vals.extend(ml_vals_full.tolist())
                g_proton_ml_full = ROOT.TGraph(len(ts_ml_full), array('d', ts_ml_full), array('d', ml_vals_full))
                ROOT.SetOwnership(g_proton_ml_full, True)
                g_proton_ml_full.SetMarkerStyle(24); g_proton_ml_full.SetMarkerSize(0.5)
                g_proton_ml_full.SetMarkerColor(ROOT.kRed); g_proton_ml_full.SetLineColor(ROOT.kRed)
                g_proton_ml_full.SetLineWidth(1)

            # y-range and frame
            ymin1, ymax1 = padded_limits(ymin1_vals if len(ymin1_vals)>0 else [0.0, 1.0])
            frame1 = draw_time_frame(xlim_start, ymin1, xlim_end, ymax1, 'Proton Flux')

            if g_proton is not None:        g_proton.Draw('P SAME')
            if g_proton_ml_full is not None:g_proton_ml_full.Draw('P SAME')

            leg_p = ROOT.TLegend(0.70, 0.77, 0.93, 0.95)
            ROOT.SetOwnership(leg_p, True)
            leg_p.SetBorderSize(0)
            if g_proton is not None:         leg_p.AddEntry(g_proton,         'AMS', 'pe')
            if g_proton_ml_full is not None: leg_p.AddEntry(g_proton_ml_full, 'Model', 'lp')
            leg_p.Draw()

            latex_r1 = ROOT.TLatex()
            ROOT.SetOwnership(latex_r1, True)
            latex_r1.SetNDC(); latex_r1.SetTextSize(0.045)
            latex_r1.DrawLatex(0.15, 0.86, f"S{seg_idx}  {rig_label}")

            # ---------- (2) Top-Right: He vs P fit ----------
            c_main.cd(2)
            set_pad_margins(ROOT.gPad)
            ROOT.gStyle.SetOptFit(0)

            g_fit = None; fit_func = None
            if n_obs >= 1:
                x_fit = df_obs[p_flux_col].to_numpy(dtype=float)
                y_fit = df_obs[he_flux_col].to_numpy(dtype=float)
                ex_fit = df_obs['proton_flux_error'].to_numpy(dtype=float)
                ey_fit = df_obs['helium_flux_error'].to_numpy(dtype=float)
                xarr, yarr = array('d', x_fit), array('d', y_fit)
                exarr, eyarr = array('d', ex_fit), array('d', ey_fit)
                g_fit = ROOT.TGraphErrors(len(xarr), xarr, yarr, exarr, eyarr)
                ROOT.SetOwnership(g_fit, True)
                g_fit.SetMarkerStyle(20); g_fit.SetMarkerColor(ROOT.kBlue)
                g_fit.Draw('AP')
                g_fit.GetXaxis().SetTitle('Proton Flux')
                g_fit.GetYaxis().SetTitle('Helium Flux')

                if can_fit:
                    xmin, xmax = float(np.nanmin(x_fit)), float(np.nanmax(x_fit))
                    fit_func = ROOT.TF1(f'lin_draw_s{seg_idx}_b{i}', '[0]+[1]*x', xmin*0.9, xmax*1.1)
                    ROOT.SetOwnership(fit_func, True)
                    fit_func.SetLineColor(ROOT.kRed); fit_func.SetLineWidth(2)
                    fit_func.SetParameter(0, p0 if p0 is not None else 0.0)
                    fit_func.SetParameter(1, p1 if p1 is not None else 0.0)
                    fit_func.Draw('SAME')

                leg_fit = ROOT.TLegend(0.15, 0.80, 0.55, 0.95)
                ROOT.SetOwnership(leg_fit, True)
                leg_fit.SetBorderSize(0)
                leg_fit.AddEntry(g_fit, 'Observed Data', 'pe')
                if can_fit:
                    leg_fit.AddEntry(fit_func, f'y = {p0:.2e} + {p1:.3f} x', 'l')
                leg_fit.Draw()

            # ---------- (3) Bottom-Left: Observed Helium vs time ----------
            c_main.cd(3)
            set_pad_margins(ROOT.gPad)

            g_he = None
            if n_obs >= 1:
                # Use df_obs (paired days) so there is a 1-1 correspondence with proton in top-left
                he_low  = he_obs_vals - np.nan_to_num(he_err_vals, nan=0.0)
                he_high = he_obs_vals + np.nan_to_num(he_err_vals, nan=0.0)
                ymin3, ymax3 = padded_limits(np.concatenate([he_low, he_high]))
                frame3 = draw_time_frame(xlim_start, ymin3, xlim_end, ymax3, 'Helium Flux')
                g_he = ROOT.TGraphErrors(len(ts_obs), array('d', ts_obs), array('d', he_obs_vals),
                                         array('d', [0]*len(ts_obs)), array('d', he_err_vals))
                ROOT.SetOwnership(g_he, True)
                g_he.SetMarkerStyle(20); g_he.SetMarkerColor(ROOT.kGreen+2); g_he.SetLineColor(ROOT.kGreen+2)
                g_he.Draw('P SAME')
            else:
                frame3 = draw_time_frame(xlim_start, 0.0, xlim_end, 1.0, 'Helium Flux')

            # ---------- (4) Bottom-Right: Observed (blue) + SEP-interp (magenta) + NODATA-ML (red) ----------
            c_main.cd(4)
            set_pad_margins(ROOT.gPad)

            y_stack = []
            if len(br_obs_y) > 0: y_stack.extend(br_obs_y)
            if len(br_sep_y) > 0: y_stack.extend(br_sep_y)
            if len(br_ml_y)  > 0: y_stack.extend(br_ml_y)

            ymin4, ymax4 = padded_limits(y_stack if len(y_stack) > 0 else [0.0, 1.0])
            frame4 = draw_time_frame(xlim_start, ymin4, xlim_end, ymax4, 'Helium Flux')

            mg = ROOT.TMultiGraph(); ROOT.SetOwnership(mg, True)
            mg.SetTitle('')

            # Observed (blue)
            if len(br_obs_t) > 0:
                g_obs_he = ROOT.TGraphErrors(len(br_obs_t), array('d', br_obs_t), array('d', br_obs_y))
                ROOT.SetOwnership(g_obs_he, True)
                g_obs_he.SetMarkerStyle(20); g_obs_he.SetMarkerColor(ROOT.kBlue); g_obs_he.SetMarkerSize(0.8)
                mg.Add(g_obs_he)

            # SEP-like interpolated (magenta)
            if len(br_sep_t) > 0:
                g_sep_interp = ROOT.TGraphErrors(len(br_sep_t), array('d', br_sep_t), array('d', br_sep_y))
                ROOT.SetOwnership(g_sep_interp, True)
                g_sep_interp.SetMarkerStyle(22); g_sep_interp.SetMarkerColor(ROOT.kMagenta); g_sep_interp.SetMarkerSize(0.9)
                mg.Add(g_sep_interp)

            # NODATA ML-pred (red)
            if len(br_ml_t) > 0:
                g_pred_ml = ROOT.TGraphErrors(len(br_ml_t), array('d', br_ml_t), array('d', br_ml_y))
                ROOT.SetOwnership(g_pred_ml, True)
                g_pred_ml.SetMarkerStyle(21); g_pred_ml.SetMarkerColor(ROOT.kRed); g_pred_ml.SetMarkerSize(0.8)
                mg.Add(g_pred_ml)

            mg.Draw('P SAME')  # preserve frame axis

            leg_he = ROOT.TLegend(0.12, 0.76, 0.55, 0.95)
            ROOT.SetOwnership(leg_he, True)
            leg_he.SetBorderSize(0); leg_he.SetTextSize(0.045)
            if len(br_obs_t) > 0:  leg_he.AddEntry(g_obs_he,     'Observed', 'p')
            if len(br_sep_t) > 0:  leg_he.AddEntry(g_sep_interp, 'Interpolated (SEP)', 'p')
            if len(br_ml_t)  > 0:  leg_he.AddEntry(g_pred_ml,    'Predicted (ML)', 'p')
            leg_he.Draw()

            # Save figure
            png_path = os.path.join(
                results_dir, f'plot_seg{seg_idx}_bin{i+1}_{rig_min_cur:.2f}_{rig_max_cur:.2f}_GV.png'
            )
            c_main.Modified(); c_main.Update()
            c_main.SaveAs(png_path)
            print(f"    Figure saved: {png_path}")

            # Cleanup plot objects
            del frame1, leg_p, latex_r1
            if 'g_proton' in locals() and g_proton: del g_proton
            if 'g_proton_ml_full' in locals() and g_proton_ml_full: del g_proton_ml_full

            if 'g_fit' in locals() and g_fit: del g_fit
            if 'fit_func' in locals() and fit_func: del fit_func
            del leg_fit

            del frame3
            if 'g_he' in locals() and g_he: del g_he

            del frame4, mg
            if 'g_obs_he' in locals() and g_obs_he: del g_obs_he
            if 'g_sep_interp' in locals() and g_sep_interp: del g_sep_interp
            if 'g_pred_ml' in locals() and g_pred_ml: del g_pred_ml
            del leg_he
            del c_main

# -----------------------------
# Save the consolidated CSV
# -----------------------------
if len(final_rows) == 0:
    print("\nNo rows generated. Check input CSVs and time segments.")
else:
    df_final = pd.DataFrame(final_rows, columns=[
        'date YYYY-MM-DD',
        'rigidity_min GV',
        'rigidity_max GV',
        'helium_flux m^-2sr^-1s^-1GV^-1',
        'SEPorNODATA'
    ])
    # Sort for readability
    df_final = df_final.sort_values(['date YYYY-MM-DD', 'rigidity_min GV', 'rigidity_max GV']).reset_index(drop=True)
    df_final.to_csv(FINAL_OUT, index=False)
    print(f"\nConsolidated CSV saved: {FINAL_OUT}")

# -*- coding: utf-8 -*-
# Plot full-span helium time series per rigidity bin from the completed CSV.
# Colors:
#   OBSERVED -> black
#   SEP      -> magenta
#   NODATA   -> red
#
# Input CSV columns (exact names expected):
#   date YYYY-MM-DD,rigidity_min GV,rigidity_max GV,helium_flux m^-2sr^-1s^-1GV^-1,SEPorNODATA
#
# One PNG per rigidity bin will be written to 'results_root' (next to this script).

import pandas as pd
import numpy as np
import ROOT
from array import array
import os

# -----------------------------
# User-configurable parameters
# -----------------------------
# Path to the consolidated (completed) CSV produced by the previous script
COMPLETED_CSV = '/home/zpw/Files/lstmPaper/pr2he/results_root/helium_completed_2011-05-20_2019-10-29.csv'

# Output directory (will be created if not exists)
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()
out_dir = os.path.join(script_dir, 'results_root')
os.makedirs(out_dir, exist_ok=True)

# Canvas size
CANVAS_W, CANVAS_H = 1400, 600

# Marker sizes
MS_OBS = 0.7
MS_SEP = 0.9
MS_ML  = 0.8

# -----------------------------
# ROOT global styles
# -----------------------------
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetTimeOffset(0)  # Unix epoch for time axis

def dt_to_unix(d):
    """datetime-like -> Unix seconds (int)."""
    return int(pd.Timestamp(d).timestamp())

def set_pad_margins(pad, left=0.10, bottom=0.12, top=0.05, right=0.04):
    pad.SetLeftMargin(left)
    pad.SetBottomMargin(bottom)
    pad.SetTopMargin(top)
    pad.SetRightMargin(right)

def config_time_axis(axis, xmin, xmax):
    axis.SetTimeDisplay(1)
    axis.SetTimeOffset(0, "gmt")
    axis.SetTimeFormat("#splitline{%b %d}{%Y}")  # two-line: Mon DD / YYYY
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
    a = np.array(vals, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return fallback
    ymin = float(np.min(a))
    ymax = float(np.max(a))
    if ymin == ymax:
        if ymin == 0:
            ymin, ymax = -1.0, 1.0
        else:
            ymin *= 0.9
            ymax *= 1.1
    yr = ymax - ymin
    return ymin - pad_frac*yr, ymax + pad_frac*yr

# -----------------------------
# Load data
# -----------------------------
print("Loading completed CSV:", COMPLETED_CSV)
df = pd.read_csv(COMPLETED_CSV)

# Enforce dtypes and parse dates
date_col = 'date YYYY-MM-DD'
rig_min_col = 'rigidity_min GV'
rig_max_col = 'rigidity_max GV'
flux_col = 'helium_flux m^-2sr^-1s^-1GV^-1'
flag_col = 'SEPorNODATA'

df[date_col] = pd.to_datetime(df[date_col])
df = df.sort_values([rig_min_col, rig_max_col, date_col]).reset_index(drop=True)

# Determine global time span
if df.empty:
    raise RuntimeError("Input CSV is empty.")

global_start = df[date_col].min()
global_end   = df[date_col].max()
xlim_start = dt_to_unix(global_start)
xlim_end   = dt_to_unix(global_end)

# Unique rigidity bins (sorted by rig_min, then rig_max)
bins = (df[[rig_min_col, rig_max_col]]
        .drop_duplicates()
        .sort_values([rig_min_col, rig_max_col])
        .to_numpy())

print(f"Found {len(bins)} rigidity bins.")

# -----------------------------
# Loop over bins and plot
# -----------------------------
for (rmin, rmax) in bins:
    rig_label = f'{rmin:.2f}-{rmax:.2f} GV'
    print(f'Plotting bin: {rig_label}')

    df_bin = df[(df[rig_min_col] == rmin) & (df[rig_max_col] == rmax)].copy()
    if df_bin.empty:
        print('  No rows for this bin. Skip.')
        continue

    # Split by flags
    obs = df_bin[df_bin[flag_col] == 'OBSERVED']
    sep = df_bin[df_bin[flag_col] == 'SEP']
    nod = df_bin[df_bin[flag_col] == 'NODATA']

    # To UNIX time
    t_obs = np.array([dt_to_unix(d) for d in obs[date_col]]) if not obs.empty else np.array([], dtype=float)
    y_obs = obs[flux_col].to_numpy(dtype=float) if not obs.empty else np.array([], dtype=float)

    t_sep = np.array([dt_to_unix(d) for d in sep[date_col]]) if not sep.empty else np.array([], dtype=float)
    y_sep = sep[flux_col].to_numpy(dtype=float) if not sep.empty else np.array([], dtype=float)

    t_nod = np.array([dt_to_unix(d) for d in nod[date_col]]) if not nod.empty else np.array([], dtype=float)
    y_nod = nod[flux_col].to_numpy(dtype=float) if not nod.empty else np.array([], dtype=float)

    # Y-range
    yvals = []
    if y_obs.size: yvals.extend(y_obs.tolist())
    if y_sep.size: yvals.extend(y_sep.tolist())
    if y_nod.size: yvals.extend(y_nod.tolist())
    ymin, ymax = padded_limits(yvals if len(yvals)>0 else [0.0, 1.0])

    # Canvas
    c = ROOT.TCanvas(f'c_full_{rmin:.2f}_{rmax:.2f}', f'{rig_label}  ({global_start.date()}–{global_end.date()})', CANVAS_W, CANVAS_H)
    ROOT.SetOwnership(c, True)
    set_pad_margins(ROOT.gPad)

    # Frame with unified date axis
    frame = draw_time_frame(xlim_start, ymin, xlim_end, ymax, 'Helium Flux')

    # Graphs
    # OBSERVED -> black
    g_obs = None
    if t_obs.size > 0:
        g_obs = ROOT.TGraphErrors(len(t_obs), array('d', t_obs), array('d', y_obs))
        ROOT.SetOwnership(g_obs, True)
        g_obs.SetMarkerStyle(20)
        g_obs.SetMarkerSize(MS_OBS)
        g_obs.SetMarkerColor(ROOT.kBlack)
        g_obs.SetLineColor(ROOT.kBlack)
        g_obs.Draw('P SAME')

    # SEP -> magenta
    g_sep = None
    if t_sep.size > 0:
        g_sep = ROOT.TGraphErrors(len(t_sep), array('d', t_sep), array('d', y_sep))
        ROOT.SetOwnership(g_sep, True)
        g_sep.SetMarkerStyle(22)
        g_sep.SetMarkerSize(MS_SEP)
        g_sep.SetMarkerColor(ROOT.kMagenta)
        g_sep.SetLineColor(ROOT.kMagenta)
        g_sep.Draw('P SAME')

    # NODATA -> red
    g_nod = None
    if t_nod.size > 0:
        g_nod = ROOT.TGraphErrors(len(t_nod), array('d', t_nod), array('d', y_nod))
        ROOT.SetOwnership(g_nod, True)
        g_nod.SetMarkerStyle(21)
        g_nod.SetMarkerSize(MS_ML)
        g_nod.SetMarkerColor(ROOT.kRed)
        g_nod.SetLineColor(ROOT.kRed)
        g_nod.Draw('P SAME')

    # Legend
    leg = ROOT.TLegend(0.80, 0.15, 0.95, 0.35)
    ROOT.SetOwnership(leg, True)
    leg.SetBorderSize(0)
    if g_obs: leg.AddEntry(g_obs, 'Observed', 'p')
    if g_sep: leg.AddEntry(g_sep, 'SEP (Interpolated)', 'p')
    if g_nod: leg.AddEntry(g_nod, 'NODATA (ML)', 'p')
    leg.Draw()

    # Title text
    latex = ROOT.TLatex()
    ROOT.SetOwnership(latex, True)
    latex.SetNDC(); latex.SetTextSize(0.045)
    latex.DrawLatex(0.12, 0.80, f'{rig_label}')

    # Save
    png_path = os.path.join(out_dir, f'fullspan_{rmin:.2f}_{rmax:.2f}_GV.png')
    c.Modified(); c.Update()
    c.SaveAs(png_path)
    print(f'  Saved: {png_path}')

    # Cleanup
    del frame
    if g_obs: del g_obs
    if g_sep: del g_sep
    if g_nod: del g_nod
    del leg, latex, c

print("\nAll rigidity-bin figures are generated.")

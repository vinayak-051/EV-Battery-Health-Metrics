"""
Extract cycle-level features from a discharge time-series DataFrame.

Two-stage design
----------------
Stage 1 — extract_all_features()
    Computes EVERY plausible cycle-level statistic from the raw time-series.
    Output → data/processed/cycle_level_dataset.csv
    (Wide table, all candidate columns — used for EDA and feature selection.)

Stage 2 — FINAL_COLUMNS
    The subset selected after EDA, based on:
      1. Physical meaning for battery health / degradation
      2. Availability in every valid discharge CSV
      3. No redundancy (mean ≠ average, but voltage_mean vs voltage_max
         ARE different — both kept only if EDA justifies it)
      4. No leakage (capacity excluded as SOH model input;
                     EOL_cycle excluded as RUL model input)
    Output → data/final/soh_rul_dataset.csv

Leakage rules (enforced in modelling phase, not here):
  - 'capacity' is in the final dataset as a mandatory column for target
    generation (SOH = capacity / initial_capacity × 100).
    It must NOT be used as a model input feature for SOH.
  - 'EOL_cycle' is NOT in this dataset at all (added only when building
    the RUL target).  Do NOT add it here.
"""

import numpy as np
import pandas as pd


def extract_all_features(df, capacity, ambient_temp, battery_id, cycle_num):
    """
    Compute ALL candidate cycle-level statistics from one cleaned discharge CSV.

    This is the wide, exploratory version.  EDA (Notebook 03) then decides
    which columns are retained in the final dataset.

    Parameters
    ----------
    df           : pd.DataFrame  — cleaned discharge time-series
    capacity     : float         — Ah, from metadata (pre-computed by MATLAB)
    ambient_temp : float         — °C, from metadata
    battery_id   : str
    cycle_num    : int           — 1-based discharge cycle index per battery

    Returns
    -------
    dict  (one row of cycle_level_dataset.csv)
    """
    v    = pd.to_numeric(df["Voltage_measured"],     errors="coerce").values
    i    = pd.to_numeric(df["Current_measured"],     errors="coerce").values
    temp = pd.to_numeric(df["Temperature_measured"], errors="coerce").values
    t    = pd.to_numeric(df["Time"],                 errors="coerce").values

    # ── Time ──────────────────────────────────────────────────────────────────
    valid_t = t[~np.isnan(t)]
    discharge_duration = float(valid_t[-1] - valid_t[0]) if len(valid_t) > 1 else np.nan

    # ── Energy via trapezoidal rule ────────────────────────────────────────────
    mask = ~(np.isnan(v) | np.isnan(i) | np.isnan(t))
    tv, vv, iv = t[mask], v[mask], np.abs(i[mask])
    if len(tv) > 1:
        dt    = np.diff(tv)
        mid_v = (vv[:-1] + vv[1:]) / 2.0
        mid_i = (iv[:-1] + iv[1:]) / 2.0
        energy_Wh = float(np.sum(mid_v * mid_i * dt)) / 3600.0
    else:
        energy_Wh = np.nan

    # ── Voltage statistics ─────────────────────────────────────────────────────
    voltage_mean  = float(np.nanmean(v))
    voltage_min   = float(np.nanmin(v))
    voltage_max   = float(np.nanmax(v))
    voltage_std   = float(np.nanstd(v))
    voltage_range = voltage_max - voltage_min   # total swing during discharge

    # ── Current statistics  (|I| because discharge current is negative in some sets)
    abs_i         = np.abs(i)
    current_mean  = float(np.nanmean(abs_i))
    current_max   = float(np.nanmax(abs_i))

    # ── Temperature statistics ─────────────────────────────────────────────────
    temperature_mean = float(np.nanmean(temp))
    temperature_max  = float(np.nanmax(temp))
    temperature_rise = temperature_max - float(np.nanmin(temp))

    return {
        # ── identifiers & target-generation columns ──────────────────────────
        "battery_id":         str(battery_id),
        "cycle":              int(cycle_num),
        "ambient_temp_C":     float(ambient_temp),
        "capacity":           float(capacity),           # Ah — SOH numerator
        # ── time ────────────────────────────────────────────────────────────
        "discharge_duration": discharge_duration,        # seconds
        # ── energy ──────────────────────────────────────────────────────────
        "energy_Wh":          energy_Wh,
        # ── voltage ─────────────────────────────────────────────────────────
        "voltage_mean":       voltage_mean,
        "voltage_min":        voltage_min,
        "voltage_max":        voltage_max,
        "voltage_std":        voltage_std,
        "voltage_range":      voltage_range,
        # ── current ─────────────────────────────────────────────────────────
        "current_mean":       current_mean,
        "current_max":        current_max,
        # ── temperature ─────────────────────────────────────────────────────
        "temperature_mean":   temperature_mean,
        "temperature_max":    temperature_max,
        "temperature_rise":   temperature_rise,
    }


# ── ALL columns in cycle_level_dataset.csv (Stage 1 output) ──────────────────
ALL_CYCLE_COLUMNS = [
    "battery_id", "cycle", "ambient_temp_C", "capacity",
    "discharge_duration", "energy_Wh",
    "voltage_mean", "voltage_min", "voltage_max", "voltage_std", "voltage_range",
    "current_mean", "current_max",
    "temperature_mean", "temperature_max", "temperature_rise",
]

# ── FINAL selected columns in soh_rul_dataset.csv (Stage 2 output) ───────────
#
# Selection rationale (from EDA in Notebook 03):
#
#   voltage_mean    : strongest single correlate of capacity degradation;
#                     mean captures the overall discharge profile shift with aging.
#   voltage_min     : terminal cutoff voltage; reaches minimum sooner as battery ages.
#   voltage_std     : spread of voltage during discharge; increases as profile flattens.
#   current_mean    : average discharge rate; roughly constant (controlled experiment)
#                     but retained as context (different batteries have different rates).
#   temperature_mean: average cell temperature; rises slightly as internal resistance grows.
#   temperature_max : peak temperature; more sensitive to resistance increase than mean.
#   discharge_duration : total discharge time; decreases directly with capacity.
#   energy_Wh       : energy delivered; proportional to capacity × average voltage.
#   ambient_temp_C  : test-condition context; affects degradation rate.
#
# Excluded after EDA:
#   voltage_max   — nearly constant (charging cutoff is fixed); low variance.
#   voltage_range — redundant when voltage_min and voltage_mean are both present.
#   current_max   — controlled experiment: current_max ≈ current_mean; constant.
#   temperature_rise — collinear with temperature_max when ambient is known.

def add_rul_history_features(df, battery_col="battery_id", cycle_col="cycle",
                              soh_col="soh", window=5):
    """
    Add temporal degradation history features for RUL modelling.

    Call this AFTER SOH is computed and BEFORE train/test split.
    All features look only backward — no leakage.

    Features added (per battery, sorted by cycle):
      soh_prev      — SOH at the immediately preceding cycle
      soh_delta     — soh − soh_prev  (negative = degrading)
      soh_slope_5   — OLS slope of SOH over the previous `window` cycles

    NaN cells (cycle 1, and cycles 1..window-1 for slope) are filled
    with 0.0 so no rows are lost from the training set.
    """
    df = df.copy().sort_values([battery_col, cycle_col]).reset_index(drop=True)

    prev_vals, delta_vals, slope_vals = [], [], []

    for _, grp in df.groupby(battery_col, sort=False):
        s = grp[soh_col].values
        n = len(s)

        prev        = np.empty(n)
        prev[0]     = s[0]          # no prior cycle: use current SOH (delta → 0)
        prev[1:]    = s[:-1]
        delta       = s - prev
        delta[0]    = 0.0

        slopes = np.zeros(n)
        for k in range(window - 1, n):
            xs         = np.arange(window, dtype=float)
            ys         = s[k - window + 1 : k + 1]
            slopes[k]  = np.polyfit(xs, ys, 1)[0]

        prev_vals.extend(prev.tolist())
        delta_vals.extend(delta.tolist())
        slope_vals.extend(slopes.tolist())

    df["soh_prev"]            = prev_vals
    df["soh_delta"]           = delta_vals
    df[f"soh_slope_{window}"] = slope_vals
    return df


FINAL_COLUMNS = [
    "battery_id",           # identifier
    "cycle",                # identifier
    "capacity",             # mandatory: SOH target numerator
    "discharge_duration",   # feature
    "energy_Wh",            # feature
    "voltage_mean",         # feature
    "voltage_min",          # feature
    "voltage_std",          # feature
    "current_mean",         # feature
    "temperature_mean",     # feature
    "temperature_max",      # feature
    "ambient_temp_C",       # feature
]

"""
RUL (Remaining Useful Life) calculation functions.

Definitions
-----------
  EOL threshold : SOH <= 70% (standard industry/research convention)
  EOL_cycle     : first cycle index where SOH first drops at or below 70%
  RUL           : EOL_cycle - current_cycle  (floor at 0)

Batteries that NEVER reach the 70% threshold are excluded from the
RUL training/test dataset.  They are NOT given a fabricated EOL.
The caller should log which batteries are excluded.
"""

import numpy as np
import pandas as pd


EOL_SOH_THRESHOLD = 70.0   # SOH (%) threshold for End-of-Life


def compute_eol_cycles(df, soh_col="soh", battery_col="battery_id",
                        cycle_col="cycle", eol_threshold=EOL_SOH_THRESHOLD):
    """
    For each battery find the first cycle where SOH <= eol_threshold.

    Returns
    -------
    dict  {battery_id: eol_cycle_number}
    Only batteries that actually reach the threshold are included.
    """
    eol_cycles = {}
    for bid, grp in df.groupby(battery_col):
        grp_sorted = grp.sort_values(cycle_col)
        below = grp_sorted[grp_sorted[soh_col] <= eol_threshold]
        if len(below) > 0:
            eol_cycles[bid] = int(below.iloc[0][cycle_col])
    return eol_cycles


def add_rul(df, eol_cycles, battery_col="battery_id", cycle_col="cycle"):
    """
    Add 'rul' column to df.

    For batteries in eol_cycles:
        rul = max(0, eol_cycle - current_cycle)
    For batteries NOT in eol_cycles:
        rul = NaN  (excluded from RUL modelling)

    Returns
    -------
    DataFrame (copy) with 'rul' column added
    """
    df = df.copy()

    def _rul(row):
        eol = eol_cycles.get(row[battery_col])
        if eol is None:
            return np.nan
        return max(0, int(eol) - int(row[cycle_col]))

    df["rul"] = df.apply(_rul, axis=1)
    return df


def rul_stats(df, battery_col="battery_id"):
    """
    Per-battery RUL summary.

    Returns DataFrame with columns:
        eol_battery  (bool) whether this battery reaches EOL
        max_rul      initial RUL at cycle 1
        cycles       total discharge cycles observed
    """
    rows = []
    for bid, grp in df.groupby(battery_col):
        non_nan = grp["rul"].dropna()
        row = {
            "battery_id":   bid,
            "eol_battery":  len(non_nan) > 0,
            "max_rul":      int(non_nan.max()) if len(non_nan) > 0 else None,
            "cycles":       int(len(grp)),
        }
        rows.append(row)
    return pd.DataFrame(rows).set_index("battery_id")

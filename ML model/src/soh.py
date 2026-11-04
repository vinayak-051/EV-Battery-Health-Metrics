"""
SOH calculation functions.

SOH (%) = (Current Capacity / Initial Capacity) × 100

Initial capacity = maximum capacity observed for that battery.
Using max is more robust than "first cycle" because some batteries
in this NASA dataset have anomalously low first-cycle capacities
due to mixed discharge protocols (see dataset/extra_infos/README_*.txt).
"""

import numpy as np
import pandas as pd


EOL_THRESHOLD = 70.0   # SOH (%) below which a battery is considered at end-of-life


def compute_initial_capacities(df, battery_col="battery_id", cap_col="capacity"):
    """
    Compute the initial (reference) capacity per battery as max(capacity).

    Returns
    -------
    dict  {battery_id: initial_capacity_Ah}
    """
    return (
        df.groupby(battery_col)[cap_col]
        .max()
        .to_dict()
    )


def add_soh(df, initial_caps=None, battery_col="battery_id", cap_col="capacity"):
    """
    Add 'initial_capacity' and 'soh' columns to df (in-place copy).

    Parameters
    ----------
    df            : DataFrame with battery_id and capacity columns
    initial_caps  : dict {battery_id: initial_capacity}; computed if None

    Returns
    -------
    DataFrame with two new columns:
        initial_capacity  (Ah)
        soh               (%, clipped to [0, 100])
    """
    df = df.copy()
    if initial_caps is None:
        initial_caps = compute_initial_capacities(df, battery_col, cap_col)

    df["initial_capacity"] = df[battery_col].map(initial_caps)
    df["soh"] = (df[cap_col] / df["initial_capacity"] * 100.0).clip(0.0, 100.0)
    return df


def soh_stats(df, battery_col="battery_id"):
    """Return a per-battery SOH summary DataFrame."""
    stats = (
        df.groupby(battery_col)["soh"]
        .agg(
            soh_initial="first",
            soh_final="last",
            soh_min="min",
            soh_max="max",
        )
        .round(2)
    )
    stats["soh_drop"] = (stats["soh_initial"] - stats["soh_final"]).round(2)
    return stats


def flag_non_monotonic(df, battery_col="battery_id", cycle_col="cycle",
                        soh_col="soh", window=5, rise_threshold=5.0):
    """
    Identify batteries where SOH rises by more than rise_threshold % over
    any window-cycle span — indicating non-monotonic / anomalous capacity data.

    Returns
    -------
    set of battery_id strings
    """
    flagged = set()
    for bid, grp in df.groupby(battery_col):
        soh = grp.sort_values(cycle_col)["soh"].values
        # Look for any upward trend larger than the threshold
        if len(soh) > window:
            diffs = soh[window:] - soh[:-window]
            if diffs.max() > rise_threshold:
                flagged.add(bid)
    return flagged

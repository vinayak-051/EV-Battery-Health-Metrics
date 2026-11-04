"""
Data-quality checks and cleaning for discharge time-series CSVs.
"""

import numpy as np
import pandas as pd

# Minimum columns a discharge CSV must contain
DISCHARGE_REQUIRED = {"Voltage_measured", "Current_measured",
                      "Temperature_measured", "Time"}

# Physical plausibility bounds for Li-ion cells
VOLTAGE_MIN, VOLTAGE_MAX = 2.0, 4.5   # V
TEMP_MIN,    TEMP_MAX    = -20.0, 80.0 # °C
MIN_ROWS = 10                           # cycles with fewer points are unusable


def check_discharge_csv(df, filename=""):
    """
    Validate a discharge DataFrame.

    Returns
    -------
    (is_valid : bool, reason : str)
    """
    if df is None:
        return False, "file not readable"

    if len(df) == 0:
        return False, "empty file"

    missing = DISCHARGE_REQUIRED - set(df.columns)
    if missing:
        return False, f"missing columns: {missing}"

    if len(df) < MIN_ROWS:
        return False, f"only {len(df)} rows (< {MIN_ROWS})"

    # All-NaN key columns
    for col in DISCHARGE_REQUIRED:
        if col in df.columns and df[col].isna().all():
            return False, f"all-NaN column: {col}"

    # Infinite values
    num_cols = df.select_dtypes(include="number").columns
    if np.isinf(df[num_cols].values).any():
        return False, "contains Inf values"

    # Voltage sanity (median check - robust to a few bad points)
    v = pd.to_numeric(df["Voltage_measured"], errors="coerce").dropna()
    if len(v) > 0 and not (VOLTAGE_MIN <= v.median() <= VOLTAGE_MAX):
        return False, f"voltage median {v.median():.2f} V outside [{VOLTAGE_MIN}, {VOLTAGE_MAX}]"

    # Temperature sanity
    t = pd.to_numeric(df["Temperature_measured"], errors="coerce").dropna()
    if len(t) > 0 and not (TEMP_MIN <= t.median() <= TEMP_MAX):
        return False, f"temp median {t.median():.1f} °C outside [{TEMP_MIN}, {TEMP_MAX}]"

    return True, "ok"


def clean_discharge_csv(df):
    """
    Remove rows with NaN/Inf in required columns.
    Converts columns to float.  Returns cleaned copy.
    """
    df = df.copy()

    # Convert to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove Inf
    num_cols = df.select_dtypes(include="number").columns
    inf_mask = np.isinf(df[num_cols].values).any(axis=1)
    df = df[~inf_mask]

    # Drop rows missing any required column
    required_present = [c for c in DISCHARGE_REQUIRED if c in df.columns]
    df = df.dropna(subset=required_present)

    return df.reset_index(drop=True)


def cleaning_report(results):
    """
    Print a structured cleaning report.

    Parameters
    ----------
    results : list of dicts with keys
        filename, battery_id, valid, reason
    """
    total   = len(results)
    valid   = sum(r["valid"] for r in results)
    invalid = total - valid

    print("=" * 50)
    print("  CLEANING REPORT")
    print("=" * 50)
    print(f"  Total discharge cycles checked : {total}")
    print(f"  Valid cycles retained          : {valid}")
    print(f"  Invalid / removed              : {invalid}")
    if invalid:
        from collections import Counter
        reasons = Counter(r["reason"] for r in results if not r["valid"])
        print("\n  Removal reasons:")
        for reason, count in reasons.most_common():
            print(f"    {reason:45s}  {count}")
    print("=" * 50)

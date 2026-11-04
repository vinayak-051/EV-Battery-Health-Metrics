"""
Load metadata and individual cycle CSV files from the NASA battery dataset.

Dataset layout
--------------
dataset/metadata.csv          - master index (one row per CSV file)
dataset/data/00001.csv …      - one time-series per battery operation
"""

import os
import re
import numpy as np
import pandas as pd

# Absolute paths derived from this file's location (src/ -> project root)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_META   = os.path.join(_ROOT, "dataset", "metadata.csv")
DEFAULT_RAWDIR = os.path.join(_ROOT, "dataset", "data")


def load_metadata(meta_path=None):
    """Return metadata.csv as a DataFrame (raw, un-parsed)."""
    path = meta_path or DEFAULT_META
    return pd.read_csv(path)


def parse_start_time(st_str):
    """
    Parse MATLAB array-string timestamps into a pandas Timestamp.

    Handles formats like:
        [2010.       7.      21.      15.       0.      35.093]
        [2.0100e+03 7.0000e+00 2.1000e+01 ...]
        [2010    7   21   20   31    5]
    Returns pd.NaT on failure.
    """
    try:
        nums = re.findall(r"[\d]+(?:\.\d*)?(?:[eE][+-]?\d+)?", str(st_str))
        if len(nums) >= 6:
            y, mo, d, h, mi, s = (float(x) for x in nums[:6])
            sec_int = int(s)
            us = int(round((s - sec_int) * 1_000_000))
            return pd.Timestamp(
                year=int(y), month=int(mo), day=int(d),
                hour=int(h), minute=int(mi), second=sec_int,
                microsecond=us,
            )
    except Exception:
        pass
    return pd.NaT


def load_cycle_csv(filename, raw_dir=None):
    """
    Load one cycle CSV.

    Returns the DataFrame on success, None on any read error.
    """
    directory = raw_dir or DEFAULT_RAWDIR
    path = os.path.join(directory, filename)
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def get_discharge_metadata(meta_path=None):
    """
    Return a cleaned discharge-only view of metadata.

    Adds:
        start_dt      - parsed datetime
        cycle         - 1-based discharge cycle number per battery
    """
    df = load_metadata(meta_path)
    dis = df[df["type"] == "discharge"].copy()

    # Parse capacity to float (some rows may be empty string or 0)
    dis["Capacity"] = pd.to_numeric(dis["Capacity"], errors="coerce")
    dis = dis[dis["Capacity"].notna() & (dis["Capacity"] > 0)].copy()

    # Parse timestamp
    dis["start_dt"] = dis["start_time"].apply(parse_start_time)

    # Sort each battery chronologically and number discharge cycles
    dis = dis.sort_values(["battery_id", "uid"]).reset_index(drop=True)
    dis["cycle"] = dis.groupby("battery_id").cumcount() + 1

    return dis

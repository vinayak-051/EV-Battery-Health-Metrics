"""
Reusable plotting functions for EDA and feature analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns


def _battery_palette(battery_ids):
    """Return a dict {battery_id: color} using tab20 colormap."""
    ids = sorted(battery_ids)
    cmap = plt.get_cmap("tab20", len(ids))
    return {bid: cmap(i) for i, bid in enumerate(ids)}


# ── Time-series plots ──────────────────────────────────────────────────────────

def plot_timeseries(df_ts, battery_id, cycle, save_path=None):
    """
    Plot Voltage, Current, Temperature vs Time for one discharge cycle.
    df_ts must contain those four columns.
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    pairs = [
        ("Voltage_measured",     "Voltage (V)"),
        ("Current_measured",     "Current (A)"),
        ("Temperature_measured", "Temperature (°C)"),
    ]
    for ax, (col, ylabel) in zip(axes, pairs):
        ax.plot(df_ts["Time"], df_ts[col], lw=1.2, color="steelblue")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"Battery {battery_id} — Discharge Cycle {cycle}", fontsize=11)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ── Degradation / cycle-level plots ───────────────────────────────────────────

def plot_capacity_degradation(df, save_path=None):
    """Capacity (Ah) vs Cycle for all batteries."""
    palette = _battery_palette(df["battery_id"].unique())
    fig, ax = plt.subplots(figsize=(14, 5))
    for bid, grp in df.groupby("battery_id"):
        g = grp.sort_values("cycle")
        ax.plot(g["cycle"], g["capacity"],
                label=bid, color=palette[bid], lw=1.0, alpha=0.85)
    ax.set_xlabel("Discharge Cycle")
    ax.set_ylabel("Capacity (Ah)")
    ax.set_title("Capacity Degradation — All Batteries")
    ax.legend(ncol=5, fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_feature_vs_cycle(df, feature, ylabel=None, save_path=None):
    """One feature vs cycle number for all batteries."""
    palette = _battery_palette(df["battery_id"].unique())
    fig, ax = plt.subplots(figsize=(14, 4))
    for bid, grp in df.groupby("battery_id"):
        g = grp.sort_values("cycle")
        ax.plot(g["cycle"], g[feature],
                label=bid, color=palette[bid], lw=0.9, alpha=0.8)
    ax.set_xlabel("Discharge Cycle")
    ax.set_ylabel(ylabel or feature)
    ax.set_title(f"{feature} vs Cycle — All Batteries")
    ax.legend(ncol=5, fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_feature_distributions(df, feature_cols, save_path=None):
    """Histogram grid for a list of features."""
    n = len(feature_cols)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.5 * nrows), squeeze=False)
    all_axes = list(axes.flat)
    for ax, col in zip(all_axes, feature_cols):
        ax.hist(df[col].dropna(), bins=40, color="steelblue",
                edgecolor="white", alpha=0.8)
        ax.set_title(col, fontsize=9)
        ax.set_xlabel(col, fontsize=8)
        ax.set_ylabel("Count", fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")
    for ax in all_axes[n:]:
        ax.set_visible(False)
    plt.suptitle("Feature Distributions", fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_correlation_heatmap(df, feature_cols, save_path=None):
    """Pearson correlation heatmap among selected columns."""
    corr = df[feature_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="coolwarm", center=0, ax=ax,
                linewidths=0.5, annot_kws={"size": 8})
    ax.set_title("Feature Correlation Matrix", fontsize=11)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig

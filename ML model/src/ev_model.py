"""
Single deployable model: operating values in -> SOH and RUL out.

EVBatteryModel bundles the tuned XGBoost SOH model and the tuned XGBoost RUL model.
SOH is predicted first; its output (and the SOH history across cycles) feeds the RUL model.
"""

import numpy as np
import pandas as pd

from feature_engineering import add_rul_history_features

OPERATING_COLUMNS = [
    'cycle', 'discharge_duration', 'energy_Wh',
    'voltage_mean', 'voltage_min', 'voltage_std',
    'current_mean', 'temperature_mean', 'temperature_max', 'ambient_temp_C',
]


def _rolling_slope(df, col, window, out_col):
    """OLS slope of `col` over the last `window` cycles (0 until enough history)."""
    s = df[col].values
    sl = np.zeros(len(s))
    xs = np.arange(window, dtype=float)
    for k in range(window - 1, len(s)):
        sl[k] = np.polyfit(xs, s[k - window + 1: k + 1], 1)[0]
    df[out_col] = sl
    return df


class EVBatteryModel:
    def __init__(self, soh_model, rul_model, soh_features, rul_features, rul_log1p=True):
        self.soh_model = soh_model
        self.rul_model = rul_model
        self.soh_features = list(soh_features)
        self.rul_features = list(rul_features)
        self.rul_log1p = rul_log1p

    # -- helpers ---------------------------------------------------------------
    def _history(self, g):
        """RUL features for one battery's cycles (sorted by cycle); `soh` = predicted SOH."""
        g = g.copy()
        g['soh'] = g['soh_pred']
        if len(g) == 1:
            # a single cycle has no history: neutral trend values
            g['soh_prev'] = g['soh']
            for c in ('soh_delta', 'soh_slope_5', 'soh_slope_10',
                      'energy_trend_5', 'voltage_trend_5', 'dur_trend_5'):
                g[c] = 0.0
            g['energy_cumsum'] = g['energy_Wh']
            g['soh_ema_10'] = g['soh']
            return g
        g = add_rul_history_features(g, window=5)
        g = add_rul_history_features(g, window=10)
        g = g.sort_values('cycle').reset_index(drop=True)
        g['energy_cumsum'] = g['energy_Wh'].cumsum()
        g = _rolling_slope(g, 'energy_Wh', 5, 'energy_trend_5')
        g = _rolling_slope(g, 'voltage_mean', 5, 'voltage_trend_5')
        g = _rolling_slope(g, 'discharge_duration', 5, 'dur_trend_5')
        g['soh_ema_10'] = g['soh'].ewm(span=10, adjust=False).mean().values
        return g

    def features_frame(self, df):
        """Rows sorted by (battery_id, cycle) with soh_pred and every RUL feature."""
        d = df.copy()
        if 'battery_id' not in d.columns:
            d['battery_id'] = 'battery'
        d = d.sort_values(['battery_id', 'cycle']).reset_index(drop=True)
        work = d[['battery_id'] + self.soh_features].copy()
        work['soh_pred'] = self.soh_model.predict(work[self.soh_features].values)
        parts = [self._history(g) for _, g in work.groupby('battery_id', sort=False)]
        return pd.concat(parts, ignore_index=True)

    # -- public ----------------------------------------------------------------
    def predict(self, df):
        """
        df: one row per discharge cycle with the OPERATING_COLUMNS (plus optional battery_id).
        For several cycles of one battery pass them all: the RUL step uses the SOH history.
        Returns df (sorted by battery_id, cycle) with soh_pred (%) and rul_pred (cycles) added.
        """
        missing = [c for c in self.soh_features if c not in df.columns]
        if missing:
            raise ValueError(f'missing input columns: {missing}')
        d = df.copy()
        if 'battery_id' not in d.columns:
            d['battery_id'] = 'battery'
        d = d.sort_values(['battery_id', 'cycle']).reset_index(drop=True)
        f = self.features_frame(d)
        raw = self.rul_model.predict(f[self.rul_features].values)
        d['soh_pred'] = f['soh_pred'].values
        d['rul_pred'] = np.clip(np.expm1(raw), 0, None) if self.rul_log1p else np.clip(raw, 0, None)
        return d

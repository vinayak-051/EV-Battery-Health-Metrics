import os
import numpy as np
import joblib

_BASE = os.path.join(os.path.dirname(__file__), "..", "ML model", "models")

_soh_model = None
_rul_model = None

# Exact feature order from training notebooks
_SOH_FEATURES = [
    "cycle", "discharge_duration", "energy_Wh",
    "voltage_mean", "voltage_min", "voltage_std",
    "current_mean", "temperature_mean", "temperature_max", "ambient_temp_C",
]

_RUL_FEATURES = [
    "cycle", "soh", "soh_prev", "soh_delta", "soh_slope_5",
    "discharge_duration", "energy_Wh",
    "voltage_mean", "voltage_min", "voltage_std",
    "current_mean", "temperature_mean", "temperature_max", "ambient_temp_C",
]


def _load():
    global _soh_model, _rul_model
    if _soh_model is not None:
        return
    _soh_model = joblib.load(os.path.join(_BASE, "soh", "best_soh_model.pkl"))
    _rul_model = joblib.load(os.path.join(_BASE, "rul", "best_rul_model.pkl"))


def _get_status(soh: float) -> str:
    if soh >= 90: return "Excellent"
    if soh >= 80: return "Good"
    if soh >= 70: return "Fair"
    if soh >= 60: return "Poor"
    return "Critical"


def predict(
    cycle: float,
    energy_Wh: float,
    voltage_mean: float,
    current_mean: float,
    temperature_mean: float,
) -> dict:
    _load()

    # ── Derived measurements ────────────────────────────────────────────────
    discharge_duration = energy_Wh * 3600 / max(abs(voltage_mean * current_mean), 1e-6)
    voltage_min        = max(voltage_mean - 0.30, 2.5)
    voltage_std        = 0.08
    temperature_max    = temperature_mean + 4.0
    ambient_temp_C     = 24.0

    # ── SOH prediction (XGBRegressor, 10 features) ─────────────────────────
    soh_row = np.array([[
        cycle, discharge_duration, energy_Wh,
        voltage_mean, voltage_min, voltage_std,
        current_mean, temperature_mean, temperature_max, ambient_temp_C,
    ]])
    soh = float(np.clip(_soh_model.predict(soh_row)[0], 0.0, 100.0))

    # ── RUL prediction (Pipeline: StandardScaler + RF, 14 features) ────────
    # History features are unknown for single-cycle input; use neutral values.
    soh_prev    = soh    # assume previous SOH ≈ current SOH (no degradation observed)
    soh_delta   = 0.0   # no measured change
    soh_slope_5 = 0.0   # no trend window available

    rul_row = np.array([[
        cycle, soh, soh_prev, soh_delta, soh_slope_5,
        discharge_duration, energy_Wh,
        voltage_mean, voltage_min, voltage_std,
        current_mean, temperature_mean, temperature_max, ambient_temp_C,
    ]])
    rul = max(0, int(round(float(_rul_model.predict(rul_row)[0]))))

    return {
        "soh":  round(soh, 2),
        "rul":  rul,
        "status": _get_status(soh),
        "discharge_duration_min": round(discharge_duration / 60, 2),
    }

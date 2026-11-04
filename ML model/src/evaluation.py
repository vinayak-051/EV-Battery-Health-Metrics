"""
Model evaluation helpers.

Functions
---------
  compute_metrics(y_true, y_pred)   → dict {MAE, RMSE, R2}
  build_comparison_table(results)   → DataFrame ranked by RMSE
  select_best_model(results)        → name of best model (lowest test RMSE)
  print_metrics_table(table)        → formatted console output
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def _rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def compute_metrics(y_true, y_pred, label=""):
    """
    Compute MAE, RMSE, R² for one model.

    Returns
    -------
    dict  {model, MAE, RMSE, R2}
    """
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = _rmse(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    return {
        "model": label,
        "MAE":   round(mae,  4),
        "RMSE":  round(rmse, 4),
        "R2":    round(r2,   4),
    }


def build_comparison_table(results_list, sort_by="RMSE_test"):
    """
    Build a comparison DataFrame from a list of result dicts.

    Each dict should have keys:
        model, MAE_train, RMSE_train, R2_train,
                MAE_test,  RMSE_test,  R2_test

    Returns
    -------
    DataFrame sorted by sort_by ascending (lowest RMSE first).
    """
    df = pd.DataFrame(results_list)
    df = df.sort_values(sort_by).reset_index(drop=True)
    df.index += 1   # rank starts at 1
    return df


def select_best_model(results_list, sort_by="RMSE_test"):
    """Return the model name with the best (lowest) sort_by metric."""
    table = build_comparison_table(results_list, sort_by)
    return table.iloc[0]["model"]


def print_metrics_table(table, title="Model Comparison"):
    """Print comparison table to console."""
    line = "-" * 72
    print(f"\n{title}")
    print(line)
    header = (f"{'Rank':<5} {'Model':<26} "
              f"{'RMSE_tr':>8} {'MAE_tr':>8} {'R2_tr':>7} "
              f"{'RMSE_te':>8} {'MAE_te':>8} {'R2_te':>7}")
    print(header)
    print(line)
    for rank, row in table.iterrows():
        print(f"{rank:<5} {row['model']:<26} "
              f"{row['RMSE_train']:>8.4f} {row['MAE_train']:>8.4f} {row['R2_train']:>7.4f} "
              f"{row['RMSE_test']:>8.4f}  {row['MAE_test']:>8.4f} {row['R2_test']:>7.4f}")
    print(line)


def cv_summary(cv_scores, model_name=""):
    """Summarise a cross-validation result dict from RandomizedSearchCV."""
    rmse_mean = -cv_scores["mean_test_score"].mean()
    rmse_std  =  cv_scores["mean_test_score"].std()
    return {
        "model":      model_name,
        "CV_RMSE":    round(rmse_mean, 4),
        "CV_RMSE_std": round(rmse_std,  4),
    }

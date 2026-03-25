import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error

def evaluate_binary(y_true, y_prob, name: str):
    mask = pd.notnull(y_true) & pd.notnull(y_prob)
    y_true = np.asarray(y_true[mask])
    y_prob = np.asarray(y_prob[mask])

    if len(np.unique(y_true)) < 2:
        return {f"{name}_auc": np.nan, f"{name}_ap": np.nan}

    return {
        f"{name}_auc": roc_auc_score(y_true, y_prob),
        f"{name}_ap": average_precision_score(y_true, y_prob),
    }

def evaluate_regression(y_true, y_pred_log, name: str):
    mask = pd.notnull(y_true) & pd.notnull(y_pred_log)
    y_true = np.asarray(y_true[mask])
    y_pred_log = np.asarray(y_pred_log[mask])

    y_true_log = np.log1p(np.clip(y_true, 0, None))
    return {
        f"{name}_mae_log1p": mean_absolute_error(y_true_log, y_pred_log)
    }

def topk_summary(df: pd.DataFrame, score_col: str, k: int = 50):
    top = df.sort_values(score_col, ascending=False).head(k)
    return {
        "k": k,
        "avg_expected_pog_prize": top["expected_pog_prize"].mean(),
        "sum_actual_pog_prize": top["pog_total_prize"].sum() if "pog_total_prize" in top.columns else np.nan,
        "n_win_flag": top["win_flag"].sum() if "win_flag" in top.columns else np.nan,
        "n_bt_place_flag": top["bt_place_flag"].sum() if "bt_place_flag" in top.columns else np.nan,
        "n_graded_win_flag": top["graded_win_flag"].sum() if "graded_win_flag" in top.columns else np.nan,
    }

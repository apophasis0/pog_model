"""
Rolling Backtest for POG Model

Trains the full model suite on multiple rolling train/valid/test splits
and compares Top-k performance of different scores across years.

Usage:
    uv run python src/backtest.py
"""
import os
import sys
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from config import Config
from data import load_all_labeled_frame, load_completed_birth_years
from train import (
    split_by_birth_year,
    describe_split,
    train_binary_stage_model,
    train_positive_regressor,
    train_quantile_regressor,
    predict_binary,
    predict_regressor,
    predict_nested_milestones,
    build_blended_scores,
    build_topk_report,
    fit_ceiling_weights,
)

# ==============================
# Rolling fold definition
# ==============================

def generate_folds(
    completed_years: list[int],
    min_train_years: int = 10,
    max_folds: int = 4,
):
    """
    Generate rolling folds from the most recent completed birth years.
    Each fold: train = [start..train_end], valid = valid_year, test = test_year.
    Limited to `max_folds` most recent folds to avoid data sparsity in early years.
    """
    folds = []
    start = completed_years[0]

    # Build from most recent backward
    for i in range(len(completed_years) - 2, 0, -1):
        test_year = completed_years[i + 1] if i + 1 < len(completed_years) else None
        if test_year is None:
            continue
        valid_year = completed_years[i]
        train_end = completed_years[i - 1]

        if (train_end - start + 1) < min_train_years:
            break

        folds.append({
            "train_start": start,
            "train_end": train_end,
            "valid_year": valid_year,
            "test_year": test_year,
        })

        if len(folds) >= max_folds:
            break

    folds.reverse()  # chronological order
    return folds


# ==============================
# Single fold training & eval
# ==============================

def run_single_fold(
    df: pd.DataFrame,
    fold: dict,
    score_cols: list[str],
    ks: list[int],
    verbose: int = 0,
) -> tuple[pd.DataFrame, dict]:
    """Train all models for one fold, return (topk_report_df, metrics_dict)."""

    test_year = fold["test_year"]
    valid_year = fold["valid_year"]
    train_start = fold["train_start"]
    train_end = fold["train_end"]

    print(f"\n{'='*60}")
    print(f"FOLD: train {train_start}-{train_end}, valid {valid_year}, test {test_year}")
    print(f"{'='*60}")

    # Split
    train_df = df[(df["birth_year"] >= train_start) & (df["birth_year"] <= train_end)].copy()
    valid_df = df[df["birth_year"] == valid_year].copy()
    test_df = df[df["birth_year"] == test_year].copy()

    describe_split("train", train_df)
    describe_split("valid", valid_df)
    describe_split("test", test_df)

    if len(train_df) == 0 or len(valid_df) == 0 or len(test_df) == 0:
        print(f"[SKIP] Fold {test_year}: empty split")
        return pd.DataFrame(), {}

    # --- Train all models ---
    common_kw = dict(use_dynamic=False)

    # Milestone chain
    win_model = train_binary_stage_model(
        train_df, valid_df, target="win_flag", condition_col=None,
        depth=6, l2_leaf_reg=5.0, learning_rate=0.03, iterations=1000,
        auto_class_weights="Balanced", **common_kw,
    )
    bt_place_given_win_model = train_binary_stage_model(
        train_df, valid_df, target="bt_place_flag", condition_col="win_flag",
        depth=5, l2_leaf_reg=10.0, learning_rate=0.03, iterations=1000,
        auto_class_weights="Balanced", **common_kw,
    )

    # Deep conditional models may fail on small subsets — graceful fallback
    try:
        bt_win_given_bt_place_model = train_binary_stage_model(
            train_df, valid_df, target="bt_win_flag", condition_col="bt_place_flag",
            depth=5, l2_leaf_reg=12.0, learning_rate=0.03, iterations=1200,
            auto_class_weights="Balanced", **common_kw,
        )
    except ValueError as e:
        print(f"[WARN] bt_win_given_bt_place failed: {e}. Using base-rate fallback.")
        bt_win_given_bt_place_model = None

    try:
        graded_given_bt_win_model = train_binary_stage_model(
            train_df, valid_df, target="graded_win_flag", condition_col="bt_win_flag",
            depth=4, l2_leaf_reg=20.0, learning_rate=0.02, iterations=1500,
            auto_class_weights="Balanced", **common_kw,
        )
    except ValueError as e:
        print(f"[WARN] graded_given_bt_win failed: {e}. Using base-rate fallback.")
        graded_given_bt_win_model = None

    # Prize models
    positive_prize_model = train_binary_stage_model(
        train_df, valid_df, target="positive_prize_flag", condition_col=None,
        depth=6, l2_leaf_reg=5.0, learning_rate=0.03, iterations=1000,
        auto_class_weights=None, **common_kw,
    )
    prize_model = train_positive_regressor(train_df, valid_df, **common_kw)

    # Ceiling models
    prize_ge_10m_model = train_binary_stage_model(
        train_df, valid_df, target="pog_total_prize_ge_10m_flag", condition_col=None,
        depth=5, l2_leaf_reg=10.0, learning_rate=0.03, iterations=1200,
        auto_class_weights="Balanced", **common_kw,
    )
    prize_ge_30m_model = train_binary_stage_model(
        train_df, valid_df, target="pog_total_prize_ge_30m_flag", condition_col=None,
        depth=4, l2_leaf_reg=15.0, learning_rate=0.02, iterations=1500,
        auto_class_weights="Balanced", **common_kw,
    )

    # Quantile regressors
    q80_model = train_quantile_regressor(
        train_df, valid_df, alpha=0.8, depth=4, l2_leaf_reg=15.0,
        learning_rate=0.02, iterations=2000, **common_kw,
    )
    q90_model = train_quantile_regressor(
        train_df, valid_df, alpha=0.9, depth=4, l2_leaf_reg=15.0,
        learning_rate=0.02, iterations=2000, **common_kw,
    )

    # --- Build test predictions ---
    keep_cols = [
        c for c in [
            "ketto_num", "birth_year",
            "pog_total_prize", "win_flag", "bt_place_flag",
            "bt_win_flag", "graded_win_flag", "positive_prize_flag",
            "pog_total_prize_ge_10m_flag", "pog_total_prize_ge_30m_flag",
        ]
        if c in test_df.columns
    ]
    test_pred = test_df[keep_cols].copy()

    # Milestone predictions (with None-model fallback)
    p_win = predict_binary(win_model, test_df, "win_flag")
    p_bt_place_given_win = predict_binary(bt_place_given_win_model, test_df, "bt_place_flag")

    if bt_win_given_bt_place_model is not None:
        p_bt_win_given_bt_place = predict_binary(bt_win_given_bt_place_model, test_df, "bt_win_flag")
    else:
        base_rate_bt_win = train_df.loc[train_df["bt_place_flag"] == 1, "bt_win_flag"].mean()
        p_bt_win_given_bt_place = np.full(len(test_df), base_rate_bt_win)
        print(f"  [fallback] p_bt_win_given_bt_place = {base_rate_bt_win:.4f}")

    if graded_given_bt_win_model is not None:
        p_graded_given_bt_win = predict_binary(graded_given_bt_win_model, test_df, "graded_win_flag")
    else:
        base_rate_graded_win = train_df.loc[train_df["bt_win_flag"] == 1, "graded_win_flag"].mean()
        p_graded_given_bt_win = np.full(len(test_df), base_rate_graded_win)
        print(f"  [fallback] p_graded_given_bt_win = {base_rate_graded_win:.4f}")

    p_bt_place = p_win * p_bt_place_given_win
    p_bt_win = p_bt_place * p_bt_win_given_bt_place
    p_graded_win = p_bt_win * p_graded_given_bt_win

    test_pred["p_win"] = p_win
    test_pred["p_bt_place_given_win"] = p_bt_place_given_win
    test_pred["p_bt_place"] = p_bt_place
    test_pred["p_bt_win_given_bt_place"] = p_bt_win_given_bt_place
    test_pred["p_bt_win"] = p_bt_win
    test_pred["p_graded_given_bt_win"] = p_graded_given_bt_win
    test_pred["p_graded_win"] = p_graded_win

    # Prize predictions
    test_pred["p_positive_prize"] = predict_binary(positive_prize_model, test_df, "positive_prize_flag")
    test_pred["pred_log_prize_pos"] = predict_regressor(prize_model, test_df)
    test_pred["pred_positive_prize_amount"] = np.clip(np.expm1(test_pred["pred_log_prize_pos"]), 0, None)
    test_pred["expected_pog_prize"] = test_pred["p_positive_prize"] * test_pred["pred_positive_prize_amount"]

    # Ceiling predictions
    test_pred["p_prize_ge_10m"] = predict_binary(prize_ge_10m_model, test_df, "pog_total_prize_ge_10m_flag")
    test_pred["p_prize_ge_30m"] = predict_binary(prize_ge_30m_model, test_df, "pog_total_prize_ge_30m_flag")

    # Quantile predictions
    test_pred["q80_log_prize"] = predict_regressor(q80_model, test_df)
    test_pred["q90_log_prize"] = predict_regressor(q90_model, test_df)
    test_pred["q80_prize"] = np.clip(np.expm1(test_pred["q80_log_prize"]), 0, None)
    test_pred["q90_prize"] = np.clip(np.expm1(test_pred["q90_log_prize"]), 0, None)

    # Blended scores
    valid_pred = valid_df.copy()
    valid_pred["p_win"] = predict_binary(win_model, valid_df, "win_flag")
    valid_pred["p_bt_place_given_win"] = predict_binary(bt_place_given_win_model, valid_df, "bt_place_flag")
    if bt_win_given_bt_place_model is not None:
        valid_pred["p_bt_win_given_bt_place"] = predict_binary(bt_win_given_bt_place_model, valid_df, "bt_win_flag")
    else:
        valid_pred["p_bt_win_given_bt_place"] = np.full(len(valid_df), base_rate_bt_win)
    if graded_given_bt_win_model is not None:
        valid_pred["p_graded_given_bt_win"] = predict_binary(graded_given_bt_win_model, valid_df, "graded_win_flag")
    else:
        valid_pred["p_graded_given_bt_win"] = np.full(len(valid_df), base_rate_graded_win)
        
    valid_pred["p_bt_place"] = valid_pred["p_win"] * valid_pred["p_bt_place_given_win"]
    valid_pred["p_bt_win"] = valid_pred["p_bt_place"] * valid_pred["p_bt_win_given_bt_place"]
    valid_pred["p_graded_win"] = valid_pred["p_bt_win"] * valid_pred["p_graded_given_bt_win"]

    valid_pred["p_positive_prize"] = predict_binary(positive_prize_model, valid_df, "positive_prize_flag")
    valid_pred["pred_log_prize_pos"] = predict_regressor(prize_model, valid_df)
    valid_pred["pred_positive_prize_amount"] = np.clip(np.expm1(valid_pred["pred_log_prize_pos"]), 0, None)
    valid_pred["expected_pog_prize"] = valid_pred["p_positive_prize"] * valid_pred["pred_positive_prize_amount"]
    valid_pred["p_prize_ge_10m"] = predict_binary(prize_ge_10m_model, valid_df, "pog_total_prize_ge_10m_flag")
    valid_pred["p_prize_ge_30m"] = predict_binary(prize_ge_30m_model, valid_df, "pog_total_prize_ge_30m_flag")
    valid_pred["q90_log_prize"] = predict_regressor(q90_model, valid_df)
    valid_pred["q90_prize"] = np.clip(np.expm1(valid_pred["q90_log_prize"]), 0, None)

    ceiling_weights = fit_ceiling_weights(valid_pred)
    
    test_pred = build_blended_scores(test_pred, ceiling_weights=ceiling_weights)

    # --- Top-k report ---
    topk = build_topk_report(test_pred, score_cols=score_cols, ks=ks)
    topk["test_year"] = test_year
    topk["train_range"] = f"{train_start}-{train_end}"

    # --- Metrics ---
    from eval import evaluate_binary
    metrics = {"test_year": test_year}
    metrics.update(evaluate_binary(test_pred["win_flag"], test_pred["p_win"], "win"))
    metrics.update(evaluate_binary(test_pred["bt_place_flag"], test_pred["p_bt_place"], "bt_place"))
    metrics.update(evaluate_binary(test_pred["graded_win_flag"], test_pred["p_graded_win"], "graded_win"))
    metrics.update(evaluate_binary(test_pred["pog_total_prize_ge_10m_flag"], test_pred["p_prize_ge_10m"], "prize_ge_10m"))
    metrics.update(evaluate_binary(test_pred["pog_total_prize_ge_30m_flag"], test_pred["p_prize_ge_30m"], "prize_ge_30m"))

    return topk, metrics


# ==============================
# Aggregation
# ==============================

def aggregate_rolling_results(detail_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate rolling backtest results: mean ± std across test years."""
    agg_cols = [
        "actual_prize_sum", "actual_prize_mean",
        "win_n", "bt_place_n", "bt_win_n", "graded_win_n",
        "win_precision", "bt_place_precision", "bt_win_precision", "graded_win_precision",
        "win_lift", "bt_place_lift", "bt_win_lift", "graded_win_lift",
    ]
    existing_agg = [c for c in agg_cols if c in detail_df.columns]

    summary = detail_df.groupby(["score_col", "k"])[existing_agg].agg(["mean", "std"])
    summary.columns = ["_".join(col).strip() for col in summary.columns]
    summary = summary.reset_index()
    summary["n_folds"] = detail_df.groupby(["score_col", "k"])["test_year"].nunique().values

    return summary


# ==============================
# Main
# ==============================

def main():
    cfg = Config()

    print("=== LOADING DATA ===")
    completed_years = load_completed_birth_years(cfg)
    print(f"Completed birth years: {completed_years}")

    df = load_all_labeled_frame(cfg)
    print(f"Total labeled horses: {len(df)}")

    # Generate folds
    folds = generate_folds(completed_years, min_train_years=10)
    print(f"\n=== ROLLING BACKTEST: {len(folds)} folds ===")
    for f in folds:
        print(f"  train {f['train_start']}-{f['train_end']}, valid {f['valid_year']}, test {f['test_year']}")

    if len(folds) == 0:
        print("[ERROR] No folds generated. Need more completed cohorts.")
        sys.exit(1)

    # Score columns to evaluate
    score_cols = [
        "expected_pog_prize",
        "p_win",
        "p_bt_place",
        "p_bt_win",
        "p_graded_win",
        "p_prize_ge_10m",
        "p_prize_ge_30m",
        "q90_prize",
        "score_balanced",
        "score_ceiling",
        "score_ceiling_old",
    ]
    ks = [20, 50, 100]

    # Run folds
    all_topk = []
    all_metrics = []

    for fold in folds:
        topk, metrics = run_single_fold(df, fold, score_cols, ks)
        if len(topk) > 0:
            all_topk.append(topk)
            all_metrics.append(metrics)

    if len(all_topk) == 0:
        print("[ERROR] No fold produced results.")
        sys.exit(1)

    # Combine results
    os.makedirs("outputs", exist_ok=True)

    detail_df = pd.concat(all_topk, ignore_index=True)
    detail_df.to_csv("outputs/rolling_backtest_detail.csv", index=False)

    summary_df = aggregate_rolling_results(detail_df)
    summary_df.to_csv("outputs/rolling_backtest_summary.csv", index=False)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv("outputs/rolling_backtest_metrics.csv", index=False)

    # Print summary
    print("\n" + "=" * 80)
    print("ROLLING BACKTEST SUMMARY")
    print("=" * 80)

    print(f"\nFolds completed: {len(all_metrics)}")
    print(f"Test years: {[m['test_year'] for m in all_metrics]}")

    print("\n--- Metrics across folds ---")
    print(metrics_df.to_string(index=False))

    print("\n--- Top-k Summary (mean across folds) ---")
    # Show key comparisons for score_balanced vs score_ceiling vs score_ceiling_old
    key_scores = ["score_balanced", "score_ceiling", "score_ceiling_old", "p_bt_win"]
    key_summary = summary_df[summary_df["score_col"].isin(key_scores)].copy()
    display_cols = [c for c in [
        "score_col", "k", "n_folds",
        "actual_prize_sum_mean", "actual_prize_sum_std",
        "bt_place_n_mean", "bt_win_n_mean", "graded_win_n_mean",
        "graded_win_lift_mean",
    ] if c in key_summary.columns]
    print(key_summary[display_cols].to_string(index=False))

    print("\n--- Full detail per fold (Top-20 only) ---")
    top20_detail = detail_df[detail_df["k"] == 20].copy()
    top20_display = [c for c in [
        "test_year", "score_col", "actual_prize_sum",
        "win_n", "bt_place_n", "bt_win_n", "graded_win_n",
    ] if c in top20_detail.columns]
    key_top20 = top20_detail[top20_detail["score_col"].isin(key_scores)]
    print(key_top20[top20_display].sort_values(["test_year", "score_col"]).to_string(index=False))

    print("\n=== DONE ===")
    print("Files outputted:")
    print("- outputs/rolling_backtest_detail.csv")
    print("- outputs/rolling_backtest_summary.csv")
    print("- outputs/rolling_backtest_metrics.csv")


if __name__ == "__main__":
    main()

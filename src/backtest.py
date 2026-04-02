"""
Rolling Backtest for POG Model

Trains the full model suite on multiple rolling train/valid/test splits
and compares Top-k performance of different scores across years.

Usage:
    uv run python src/backtest.py
"""
import os
import sys
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from config import Config
from data import load_all_labeled_frame, load_completed_birth_years
from features import FeatureSet
from train import describe_split
from pipeline import train_and_evaluate


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
    feature_set: FeatureSet,
    score_cols: list[str],
    ks: list[int],
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

    # Train and evaluate via pipeline
    topk, metrics, _ = train_and_evaluate(
        train_df, valid_df, test_df,
        feature_set=feature_set,
        score_cols=score_cols,
        ks=ks,
        graceful_conditional=True,
    )

    topk["test_year"] = test_year
    topk["train_range"] = f"{train_start}-{train_end}"
    metrics["test_year"] = test_year

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
    feature_set = FeatureSet()  # all features ON

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
        topk, metrics = run_single_fold(df, fold, feature_set, score_cols, ks)
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

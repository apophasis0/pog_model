"""
rolling_backtest.py - 滚动回测工具函数
"""
import pandas as pd

from .config import Config
from .features import FeatureSet
from .pipeline import train_and_evaluate


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

    from .split import describe_split
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
"""
Feature Ablation Experiment Runner

Runs rolling backtests with different feature subsets to measure the
marginal contribution of each newly added feature group.

Usage:
    uv run python src/ablation.py
"""
import os
import sys
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from pog_model.config import Config
from pog_model.data import load_all_labeled_frame, load_completed_birth_years
from pog_model.features import FeatureSet
from pog_model.rolling_backtest import generate_folds, run_single_fold, aggregate_rolling_results


# ==============================
# Ablation configurations
# ==============================

def _baseline(**overrides) -> FeatureSet:
    """Start from base-only features and selectively enable groups."""
    defaults = dict(
        base_categorical=True,
        new_categorical=False,
        base_numeric=True,
        new_numeric_dam_sire_age=False,
        new_numeric_damsire_stats=False,
        new_numeric_maternal_sib=False,
        new_numeric_full_sib=False,
        new_numeric_nick_stats=False,
        new_numeric_breeder_trainer=False,
    )
    defaults.update(overrides)
    return FeatureSet(**defaults)


ABLATION_CONFIGS: dict[str, FeatureSet] = {
    # ---- Reference points ----
    "baseline": _baseline(),
    "full": FeatureSet(),  # all ON

    # ---- Single group additions (additive ablation) ----
    "+new_cat": _baseline(new_categorical=True),
    "+dam_sire_age": _baseline(new_numeric_dam_sire_age=True),
    "+damsire_stats": _baseline(new_numeric_damsire_stats=True),
    "+maternal_sib": _baseline(new_numeric_maternal_sib=True),
    "+full_sib": _baseline(new_numeric_full_sib=True),
    "+nick_stats": _baseline(new_numeric_nick_stats=True),
    "+breeder_trainer": _baseline(new_numeric_breeder_trainer=True),

    # ---- Single group removals (subtractive ablation from full) ----
    "full-new_cat": FeatureSet(new_categorical=False),
    "full-dam_sire_age": FeatureSet(new_numeric_dam_sire_age=False),
    "full-damsire_stats": FeatureSet(new_numeric_damsire_stats=False),
    "full-maternal_sib": FeatureSet(new_numeric_maternal_sib=False),
    "full-full_sib": FeatureSet(new_numeric_full_sib=False),
    "full-nick_stats": FeatureSet(new_numeric_nick_stats=False),
    "full-breeder_trainer": FeatureSet(new_numeric_breeder_trainer=False),
}


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

    folds = generate_folds(completed_years, min_train_years=10)
    print(f"\n=== ABLATION: {len(folds)} folds × {len(ABLATION_CONFIGS)} configs ===")
    for f in folds:
        print(f"  train {f['train_start']}-{f['train_end']}, valid {f['valid_year']}, test {f['test_year']}")

    if len(folds) == 0:
        print("[ERROR] No folds generated. Need more completed cohorts.")
        sys.exit(1)

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

    all_topk = []
    all_metrics = []

    for exp_name, feature_set in ABLATION_CONFIGS.items():
        print(f"\n{'#'*80}")
        print(f"# EXPERIMENT: {exp_name}")
        print(f"# Features: {feature_set.describe()}")
        print(f"{'#'*80}")

        for fold in folds:
            topk, metrics = run_single_fold(df, fold, feature_set, score_cols, ks)
            if len(topk) > 0:
                topk["experiment"] = exp_name
                all_topk.append(topk)
                metrics["experiment"] = exp_name
                all_metrics.append(metrics)

    if len(all_topk) == 0:
        print("[ERROR] No experiment produced results.")
        sys.exit(1)

    os.makedirs("outputs", exist_ok=True)

    detail_df = pd.concat(all_topk, ignore_index=True)
    detail_df.to_csv("outputs/ablation_detail.csv", index=False)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv("outputs/ablation_metrics.csv", index=False)

    # ---- Summary per experiment ----
    all_summaries = []
    for exp_name in ABLATION_CONFIGS:
        exp_detail = detail_df[detail_df["experiment"] == exp_name]
        if len(exp_detail) == 0:
            continue
        summary = aggregate_rolling_results(exp_detail)
        summary["experiment"] = exp_name
        all_summaries.append(summary)

    if all_summaries:
        summary_df = pd.concat(all_summaries, ignore_index=True)
        summary_df.to_csv("outputs/ablation_summary.csv", index=False)
    else:
        summary_df = pd.DataFrame()

    # ---- Print comparison ----
    print("\n" + "=" * 80)
    print("ABLATION EXPERIMENT SUMMARY")
    print("=" * 80)

    print(f"\nExperiments completed: {metrics_df['experiment'].nunique()}")
    print(f"Total fold-experiment pairs: {len(all_metrics)}")

    # Quick comparison: show Top-20 score_balanced across experiments
    print("\n--- Top-20 score_balanced comparison (mean across folds) ---")
    if len(summary_df) > 0:
        cmp = summary_df[
            (summary_df["score_col"] == "score_balanced") & (summary_df["k"] == 20)
        ].copy()
        display_cols = [c for c in [
            "experiment",
            "actual_prize_sum_mean", "actual_prize_sum_std",
            "bt_place_n_mean", "bt_win_n_mean", "graded_win_n_mean",
            "graded_win_lift_mean",
        ] if c in cmp.columns]
        if display_cols:
            print(cmp[display_cols].to_string(index=False))

    # AUC comparison
    print("\n--- AUC comparison across experiments ---")
    auc_cols = [c for c in metrics_df.columns if c.endswith("_auc")]
    if auc_cols:
        auc_summary = metrics_df.groupby("experiment")[auc_cols].mean()
        print(auc_summary.to_string())

    print("\n=== DONE ===")
    print("Files outputted:")
    print("- outputs/ablation_detail.csv")
    print("- outputs/ablation_metrics.csv")
    print("- outputs/ablation_summary.csv")


if __name__ == "__main__":
    main()

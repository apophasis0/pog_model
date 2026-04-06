"""
Ablation Experiment for New Features (Upper Limit Rates + Granddam Stats)

Tests the marginal contribution of:
1. Upper Limit Rates (ceiling conversion rates): *_graded_per_win features
2. Granddam Statistics: granddam_prior_* features

Usage:
    python -m src.ablation_new_features
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
# Ablation configurations for NEW features
# ==============================

def _old_features(**overrides) -> FeatureSet:
    """All old features enabled, new features disabled by default."""
    defaults = dict(
        base_categorical=True,
        new_categorical=True,
        base_numeric=True,
        new_numeric_dam_sire_age=True,
        new_numeric_damsire_stats=True,
        new_numeric_maternal_sib=True,
        new_numeric_full_sib=True,
        new_numeric_nick_stats=True,
        new_numeric_breeder_trainer=True,
        # NEW features OFF by default
        new_numeric_upper_limit_rates=False,
        new_numeric_granddam_stats=False,
    )
    defaults.update(overrides)
    return FeatureSet(**defaults)


ABLATION_CONFIGS: dict[str, FeatureSet] = {
    # ---- Reference points ----
    "baseline": _old_features(),  # All old features, no new features
    "full": FeatureSet(),  # All features including new ones

    # ---- Additive ablation: add one new group at a time ----
    "+upper_limit_rates": _old_features(new_numeric_upper_limit_rates=True),
    "+granddam_stats": _old_features(new_numeric_granddam_stats=True),
    "+both_new_features": _old_features(
        new_numeric_upper_limit_rates=True,
        new_numeric_granddam_stats=True,
    ),

    # ---- Subtractive ablation: remove one new group from full ----
    "full-upper_limit_rates": FeatureSet(new_numeric_upper_limit_rates=False),
    "full-granddam_stats": FeatureSet(new_numeric_granddam_stats=False),
    "full-both_new_features": FeatureSet(
        new_numeric_upper_limit_rates=False,
        new_numeric_granddam_stats=False,
    ),
}


# ==============================
# Main
# ==============================

def main():
    cfg = Config()

    print("=" * 80)
    print("ABLATION EXPERIMENT: NEW FEATURES")
    print("=" * 80)
    print("Testing contribution of:")
    print("  1. Upper Limit Rates (5 features: *_graded_per_win)")
    print("  2. Granddam Statistics (9 features: granddam_prior_*)")
    print("=" * 80)

    print("\n=== LOADING DATA ===")
    completed_years = load_completed_birth_years(cfg)
    print(f"Completed birth years: {completed_years}")

    df = load_all_labeled_frame(cfg)
    print(f"Total labeled horses: {len(df)}")

    folds = generate_folds(completed_years, min_train_years=10)
    print(f"\n=== ROLLING BACKTEST: {len(folds)} folds × {len(ABLATION_CONFIGS)} configs ===")
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
        print(f"# Upper Limit Rates: {'ON' if feature_set.new_numeric_upper_limit_rates else 'OFF'}")
        print(f"# Granddam Stats:    {'ON' if feature_set.new_numeric_granddam_stats else 'OFF'}")
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
    detail_df.to_csv("outputs/ablation_new_features_detail.csv", index=False)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv("outputs/ablation_new_features_metrics.csv", index=False)

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
        summary_df.to_csv("outputs/ablation_new_features_summary.csv", index=False)
    else:
        summary_df = pd.DataFrame()

    # ---- Print comparison ----
    print("\n" + "=" * 80)
    print("ABLATION EXPERIMENT RESULTS: NEW FEATURES")
    print("=" * 80)

    print(f"\nExperiments completed: {metrics_df['experiment'].nunique()}")
    print(f"Total fold-experiment pairs: {len(all_metrics)}")

    # Top-20 comparison for key metrics
    print("\n" + "-" * 80)
    print("TOP-20 PERFORMANCE COMPARISON (mean ± std across folds)")
    print("-" * 80)
    
    if len(summary_df) > 0:
        for score_col in ["score_balanced", "score_ceiling", "p_graded_win"]:
            print(f"\n--- {score_col.upper()} ---")
            cmp = summary_df[
                (summary_df["score_col"] == score_col) & (summary_df["k"] == 20)
            ].copy()
            
            if len(cmp) == 0:
                continue
                
            display_cols = [c for c in [
                "experiment",
                "graded_win_n_mean", "graded_win_n_std",
                "graded_win_lift_mean", "graded_win_lift_std",
                "actual_prize_sum_mean", "actual_prize_sum_std",
            ] if c in cmp.columns]
            
            if display_cols:
                cmp_sorted = cmp.sort_values("graded_win_lift_mean", ascending=False)
                print(cmp_sorted[display_cols].to_string(index=False))

    # AUC comparison
    print("\n" + "-" * 80)
    print("AUC COMPARISON (mean across folds)")
    print("-" * 80)
    auc_cols = [c for c in metrics_df.columns if c.endswith("_auc")]
    if auc_cols:
        auc_summary = metrics_df.groupby("experiment")[auc_cols].mean()
        print(auc_summary.to_string())

    # Feature contribution analysis
    print("\n" + "=" * 80)
    print("FEATURE CONTRIBUTION ANALYSIS")
    print("=" * 80)
    
    if len(summary_df) > 0:
        # Get baseline and full performance for Top-20 score_balanced
        baseline_df = summary_df[
            (summary_df["experiment"] == "baseline") &
            (summary_df["score_col"] == "score_balanced") &
            (summary_df["k"] == 20)
        ]
        full_df = summary_df[
            (summary_df["experiment"] == "full") &
            (summary_df["score_col"] == "score_balanced") &
            (summary_df["k"] == 20)
        ]
        
        if len(baseline_df) > 0 and len(full_df) > 0:
            baseline_lift = baseline_df["graded_win_lift_mean"].iloc[0]
            full_lift = full_df["graded_win_lift_mean"].iloc[0]
            total_gain = full_lift - baseline_lift
            
            print(f"\nBaseline graded_win_lift (Top-20 score_balanced): {baseline_lift:.3f}")
            print(f"Full graded_win_lift (Top-20 score_balanced):     {full_lift:.3f}")
            print(f"Total gain from new features:                     +{total_gain:.3f} ({100*total_gain/baseline_lift:.1f}%)")
            
            # Individual contribution
            upper_only = summary_df[
                (summary_df["experiment"] == "+upper_limit_rates") &
                (summary_df["score_col"] == "score_balanced") &
                (summary_df["k"] == 20)
            ]
            granddam_only = summary_df[
                (summary_df["experiment"] == "+granddam_stats") &
                (summary_df["score_col"] == "score_balanced") &
                (summary_df["k"] == 20)
            ]
            
            if len(upper_only) > 0:
                upper_gain = upper_only["graded_win_lift_mean"].iloc[0] - baseline_lift
                print(f"\nUpper Limit Rates contribution:                   +{upper_gain:.3f} ({100*upper_gain/baseline_lift:.1f}%)")
            
            if len(granddam_only) > 0:
                granddam_gain = granddam_only["graded_win_lift_mean"].iloc[0] - baseline_lift
                print(f"Granddam Stats contribution:                      +{granddam_gain:.3f} ({100*granddam_gain/baseline_lift:.1f}%)")

    print("\n" + "=" * 80)
    print("=== DONE ===")
    print("=" * 80)
    print("\nFiles outputted:")
    print("- outputs/ablation_new_features_detail.csv")
    print("- outputs/ablation_new_features_metrics.csv")
    print("- outputs/ablation_new_features_summary.csv")
    print("\nTip: Use analyze_ablation.py to visualize these results")


if __name__ == "__main__":
    main()

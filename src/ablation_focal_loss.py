"""
Focal Loss Ablation Experiment

Compares Focal Loss vs Logloss+Balanced for highly imbalanced binary targets.
Tests each target model in isolation with different gamma values, using the
same rolling backtest folds as the main ablation suite.

Targets tested:
  - graded_given_bt_win: conditional on bt_win_flag=1, ~93% positive (minority = negatives)
  - prize_ge_30m: marginal, ~4% positive (minority = positives)
  - prize_ge_10m: marginal, ~10% positive (minority = positives)
  - bt_win_given_bt_place: conditional on bt_place_flag=1, moderate imbalance

Usage:
    uv run python src/ablation_focal_loss.py
"""
from __future__ import annotations

import os
import sys
import traceback
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

from config import Config
from data import load_all_labeled_frame, load_completed_birth_years
from features import FeatureSet
from eval import evaluate_binary
from pipeline import (
    train_binary_stage_model,
    predict_binary,
    subset_by_condition,
)
from backtest import generate_folds


# ==============================
# Experiment definitions
# ==============================

# Each experiment targets a single binary model.
# "baseline" uses Logloss + auto_class_weights="Balanced" (no focal).
# "focal_gamma_X" uses Focal Loss with specified alpha and gamma.

TARGET_CONFIGS = {
    "graded_given_bt_win": {
        "target": "graded_win_flag",
        "condition_col": "bt_win_flag",
        "severity": "extreme",  # ~93% positive in conditional subset
        "base_params": {
            "depth": 4, "l2_leaf_reg": 20.0, "learning_rate": 0.02,
            "iterations": 1500, "ctr_leaf_count_limit": 4,
        },
        # For this target, minority = negatives, so focal_alpha < 0.5
        "focal_alpha": 0.25,
        "gammas": [1.0, 2.0, 3.0],
    },
    "prize_ge_30m": {
        "target": "pog_total_prize_ge_30m_flag",
        "condition_col": None,
        "severity": "high",  # ~4% positive
        "base_params": {
            "depth": 4, "l2_leaf_reg": 15.0, "learning_rate": 0.02,
            "iterations": 1500, "ctr_leaf_count_limit": 4,
        },
        # For this target, minority = positives, so focal_alpha > 0.5
        "focal_alpha": 0.75,
        "gammas": [1.0, 2.0, 3.0],
    },
    "prize_ge_10m": {
        "target": "pog_total_prize_ge_10m_flag",
        "condition_col": None,
        "severity": "moderate",  # ~10% positive
        "base_params": {
            "depth": 5, "l2_leaf_reg": 10.0, "learning_rate": 0.03,
            "iterations": 1200, "ctr_leaf_count_limit": 4,
        },
        "focal_alpha": 0.75,
        "gammas": [1.0, 1.5, 2.0],
    },
    "bt_win_given_bt_place": {
        "target": "bt_win_flag",
        "condition_col": "bt_place_flag",
        "severity": "moderate",  # moderate imbalance in conditional subset
        "base_params": {
            "depth": 5, "l2_leaf_reg": 12.0, "learning_rate": 0.03,
            "iterations": 1200, "ctr_leaf_count_limit": 4,
        },
        "focal_alpha": 0.75,
        "gammas": [1.0, 2.0],
    },
}


def _build_experiment_list() -> list[dict]:
    """Build flat list of (name, config) for all experiments."""
    experiments = []

    for model_key, tcfg in TARGET_CONFIGS.items():
        # Baseline: Logloss + Balanced
        experiments.append({
            "name": f"{model_key}_baseline",
            "model_key": model_key,
            "target": tcfg["target"],
            "condition_col": tcfg["condition_col"],
            "severity": tcfg["severity"],
            "loss_type": "baseline",
            "train_kwargs": {
                **tcfg["base_params"],
                "auto_class_weights": "Balanced",
                "focal_alpha": None,
                "focal_gamma": None,
            },
        })

        # Focal variants
        for gamma in tcfg["gammas"]:
            experiments.append({
                "name": f"{model_key}_focal_gamma_{gamma}",
                "model_key": model_key,
                "target": tcfg["target"],
                "condition_col": tcfg["condition_col"],
                "severity": tcfg["severity"],
                "loss_type": f"focal_gamma_{gamma}",
                "train_kwargs": {
                    **tcfg["base_params"],
                    "auto_class_weights": "Balanced",  # will be ignored by focal
                    "focal_alpha": tcfg["focal_alpha"],
                    "focal_gamma": gamma,
                },
            })

    return experiments


# ==============================
# Single experiment runner
# ==============================

def run_single_experiment(
    exp: dict,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_set: FeatureSet,
) -> dict | None:
    """Train one model, evaluate on test set, return metrics dict."""
    target = exp["target"]
    condition_col = exp["condition_col"]

    # Check subset sizes
    train_sub = subset_by_condition(train_df, condition_col)
    test_sub = subset_by_condition(test_df, condition_col)

    pos_rate = train_sub[target].mean()
    print(f"  Train subset: n={len(train_sub)}, pos_rate={pos_rate:.4f}")

    if len(train_sub) < 30 or train_sub[target].nunique() < 2:
        print(f"  [SKIP] Insufficient training data for {target}")
        return None

    # Train
    model = train_binary_stage_model(
        train_df, valid_df,
        target=target,
        condition_col=condition_col,
        feature_set=feature_set,
        **exp["train_kwargs"],
    )

    # Predict on full test set (not just conditional subset)
    # This matches how the model is used in production (predict_binary on all horses)
    p_test_all = predict_binary(model, test_df, target, feature_set)

    # Evaluate on conditional subset (matches training distribution)
    if condition_col is not None:
        cond_mask = test_df[condition_col] == 1
        y_true_cond = test_df.loc[cond_mask, target]
        p_cond = p_test_all[cond_mask.values]
        cond_metrics = evaluate_binary(y_true_cond, p_cond, f"{exp['model_key']}_cond")
    else:
        cond_metrics = {}

    # Evaluate on full test set (marginal performance)
    full_metrics = evaluate_binary(test_df[target], p_test_all, f"{exp['model_key']}_full")

    return {**cond_metrics, **full_metrics}


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

    # Generate rolling folds
    folds = generate_folds(completed_years, min_train_years=10)
    print(f"\n=== FOCAL LOSS ABLATION: {len(folds)} folds ===")
    for f in folds:
        print(f"  train {f['train_start']}-{f['train_end']}, valid {f['valid_year']}, test {f['test_year']}")

    if len(folds) == 0:
        print("[ERROR] No folds generated. Need more completed cohorts.")
        sys.exit(1)

    experiments = _build_experiment_list()
    print(f"\n=== EXPERIMENTS: {len(experiments)} configs ===")
    for exp in experiments:
        print(f"  {exp['name']}: target={exp['target']}, cond={exp['condition_col']}, "
              f"loss={exp['loss_type']}, severity={exp['severity']}")

    # Run all experiments across all folds
    all_results = []

    for exp in experiments:
        print(f"\n{'#'*80}")
        print(f"# EXPERIMENT: {exp['name']}")
        print(f"# Target: {exp['target']}, Condition: {exp['condition_col']}")
        focal_a = exp['train_kwargs'].get('focal_alpha')
        focal_g = exp['train_kwargs'].get('focal_gamma')
        if focal_a is not None and focal_g is not None:
            print(f"# Loss: Focal, Alpha: {focal_a}, Gamma: {focal_g}")
        else:
            print(f"# Loss: Logloss + Balanced")
        print(f"{'#'*80}")

        for fold in folds:
            test_year = fold["test_year"]
            print(f"\n  Fold: train {fold['train_start']}-{fold['train_end']}, "
                  f"valid {fold['valid_year']}, test {test_year}")

            train_df = df[
                (df["birth_year"] >= fold["train_start"]) &
                (df["birth_year"] <= fold["train_end"])
            ].copy()
            valid_df = df[df["birth_year"] == fold["valid_year"]].copy()
            test_df = df[df["birth_year"] == fold["test_year"]].copy()

            try:
                metrics = run_single_experiment(
                    exp, train_df, valid_df, test_df, feature_set
                )
                if metrics is not None:
                    metrics["experiment"] = exp["name"]
                    metrics["model_key"] = exp["model_key"]
                    metrics["loss_type"] = exp["loss_type"]
                    metrics["severity"] = exp["severity"]
                    metrics["test_year"] = test_year
                    metrics["focal_alpha"] = focal_a
                    metrics["focal_gamma"] = focal_g
                    all_results.append(metrics)
            except Exception as e:
                print(f"  [ERROR] {exp['name']} fold {test_year}: {e}")
                traceback.print_exc()

    if len(all_results) == 0:
        print("[ERROR] No experiment produced results.")
        sys.exit(1)

    # ---- Save results ----
    os.makedirs("outputs", exist_ok=True)
    results_df = pd.DataFrame(all_results)
    results_df.to_csv("outputs/ablation_focal_loss_results.csv", index=False)

    # ---- Print summary ----
    print("\n" + "=" * 80)
    print("FOCAL LOSS ABLATION SUMMARY")
    print("=" * 80)

    # For each model_key, compare AUC/AP across loss types
    for model_key in TARGET_CONFIGS:
        mk_df = results_df[results_df["model_key"] == model_key]
        if len(mk_df) == 0:
            continue

        print(f"\n--- {model_key} ---")

        # Identify AUC and AP columns for this model
        auc_cond = f"{model_key}_cond_auc"
        ap_cond = f"{model_key}_cond_ap"
        auc_full = f"{model_key}_full_auc"
        ap_full = f"{model_key}_full_ap"

        summary_cols = ["loss_type"]
        agg_dict = {}
        for col in [auc_cond, ap_cond, auc_full, ap_full]:
            if col in mk_df.columns:
                summary_cols.append(col)
                agg_dict[col] = ["mean", "std"]

        if agg_dict:
            summary = mk_df.groupby("loss_type").agg(agg_dict)
            summary.columns = ["_".join(c).strip() for c in summary.columns]
            summary["n_folds"] = mk_df.groupby("loss_type")["test_year"].count().values
            print(summary.to_string())

    # ---- Quick comparison table ----
    print("\n--- Quick Comparison: Mean AUC (full) across folds ---")
    pivot_rows = []
    for model_key in TARGET_CONFIGS:
        auc_col = f"{model_key}_full_auc"
        mk_df = results_df[results_df["model_key"] == model_key]
        if auc_col not in mk_df.columns or len(mk_df) == 0:
            continue
        for loss_type, grp in mk_df.groupby("loss_type"):
            mean_auc = grp[auc_col].mean()
            std_auc = grp[auc_col].std()
            pivot_rows.append({
                "model": model_key,
                "loss": loss_type,
                "auc_mean": f"{mean_auc:.4f}",
                "auc_std": f"{std_auc:.4f}",
                "n_folds": len(grp),
            })
    if pivot_rows:
        print(pd.DataFrame(pivot_rows).to_string(index=False))

    print("\n=== DONE ===")
    print("Files outputted:")
    print("- outputs/ablation_focal_loss_results.csv")


if __name__ == "__main__":
    main()

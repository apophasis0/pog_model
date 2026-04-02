"""Analyze ablation experiment results and produce a concise report."""
import pandas as pd
import sys

# --- Load data ---
metrics = pd.read_csv("outputs/ablation_metrics.csv")
summary = pd.read_csv("outputs/ablation_summary.csv")
detail = pd.read_csv("outputs/ablation_detail.csv")

print("=" * 100)
print("ABLATION EXPERIMENT ANALYSIS")
print("=" * 100)

# --- 1. AUC comparison ---
print("\n### 1. AUC Comparison (mean across 4 folds)")
auc_cols = [c for c in metrics.columns if c.endswith("_auc")]
auc_mean = metrics.groupby("experiment")[auc_cols].mean()
# Compute overall mean AUC
auc_mean["avg_auc"] = auc_mean[auc_cols].mean(axis=1)
auc_mean = auc_mean.sort_values("avg_auc", ascending=False)
print(auc_mean.round(4).to_string())

# --- 2. AP comparison ---
print("\n### 2. Average Precision Comparison (mean across 4 folds)")
ap_cols = [c for c in metrics.columns if c.endswith("_ap")]
ap_mean = metrics.groupby("experiment")[ap_cols].mean()
ap_mean["avg_ap"] = ap_mean[ap_cols].mean(axis=1)
ap_mean = ap_mean.sort_values("avg_ap", ascending=False)
print(ap_mean.round(4).to_string())

# --- 3. Top-20 score_balanced comparison ---
print("\n### 3. Top-20 score_balanced (mean across folds)")
mask = (summary["score_col"] == "score_balanced") & (summary["k"] == 20)
sb20 = summary[mask][["experiment", "actual_prize_sum_mean", "actual_prize_sum_std",
                        "bt_place_n_mean", "bt_win_n_mean", "graded_win_n_mean",
                        "graded_win_lift_mean"]].copy()
sb20 = sb20.sort_values("graded_win_lift_mean", ascending=False)
print(sb20.round(2).to_string(index=False))

# --- 4. Top-20 score_ceiling comparison ---
print("\n### 4. Top-20 score_ceiling (mean across folds)")
mask = (summary["score_col"] == "score_ceiling") & (summary["k"] == 20)
sc20 = summary[mask][["experiment", "actual_prize_sum_mean", "actual_prize_sum_std",
                        "bt_place_n_mean", "bt_win_n_mean", "graded_win_n_mean",
                        "graded_win_lift_mean"]].copy()
sc20 = sc20.sort_values("graded_win_lift_mean", ascending=False)
print(sc20.round(2).to_string(index=False))

# --- 5. Top-50 score_balanced comparison ---
print("\n### 5. Top-50 score_balanced (mean across folds)")
mask = (summary["score_col"] == "score_balanced") & (summary["k"] == 50)
sb50 = summary[mask][["experiment", "actual_prize_sum_mean", "actual_prize_sum_std",
                        "bt_place_n_mean", "bt_win_n_mean", "graded_win_n_mean",
                        "graded_win_lift_mean"]].copy()
sb50 = sb50.sort_values("graded_win_lift_mean", ascending=False)
print(sb50.round(2).to_string(index=False))

# --- 6. Top-50 score_ceiling comparison ---
print("\n### 6. Top-50 score_ceiling (mean across folds)")
mask = (summary["score_col"] == "score_ceiling") & (summary["k"] == 50)
sc50 = summary[mask][["experiment", "actual_prize_sum_mean", "actual_prize_sum_std",
                        "bt_place_n_mean", "bt_win_n_mean", "graded_win_n_mean",
                        "graded_win_lift_mean"]].copy()
sc50 = sc50.sort_values("graded_win_lift_mean", ascending=False)
print(sc50.round(2).to_string(index=False))

# --- 7. Marginal impact: additive ablation (delta vs baseline) ---
print("\n### 7. Marginal Impact: Additive Ablation (delta vs baseline Top-20 score_balanced)")
mask_b = (summary["score_col"] == "score_balanced") & (summary["k"] == 20) & (summary["experiment"] == "baseline")
baseline_row = summary[mask_b].iloc[0] if len(summary[mask_b]) > 0 else None
if baseline_row is not None:
    additive_exps = ["+new_cat", "+dam_sire_age", "+damsire_stats", "+maternal_sib",
                     "+full_sib", "+nick_stats", "+breeder_trainer"]
    rows = []
    for exp in additive_exps:
        m = (summary["score_col"] == "score_balanced") & (summary["k"] == 20) & (summary["experiment"] == exp)
        if len(summary[m]) > 0:
            r = summary[m].iloc[0]
            rows.append({
                "experiment": exp,
                "Δ_prize_sum": r["actual_prize_sum_mean"] - baseline_row["actual_prize_sum_mean"],
                "Δ_bt_place_n": r["bt_place_n_mean"] - baseline_row["bt_place_n_mean"],
                "Δ_bt_win_n": r["bt_win_n_mean"] - baseline_row["bt_win_n_mean"],
                "Δ_graded_win_n": r["graded_win_n_mean"] - baseline_row["graded_win_n_mean"],
                "Δ_graded_win_lift": r["graded_win_lift_mean"] - baseline_row["graded_win_lift_mean"],
            })
    delta_df = pd.DataFrame(rows).sort_values("Δ_graded_win_lift", ascending=False)
    print(delta_df.round(3).to_string(index=False))

# --- 8. Marginal impact: subtractive ablation (delta vs full) ---
print("\n### 8. Marginal Impact: Subtractive Ablation (delta vs full Top-20 score_balanced)")
mask_f = (summary["score_col"] == "score_balanced") & (summary["k"] == 20) & (summary["experiment"] == "full")
full_row = summary[mask_f].iloc[0] if len(summary[mask_f]) > 0 else None
if full_row is not None:
    sub_exps = ["full-new_cat", "full-dam_sire_age", "full-damsire_stats",
                "full-maternal_sib", "full-full_sib", "full-nick_stats", "full-breeder_trainer"]
    rows = []
    for exp in sub_exps:
        m = (summary["score_col"] == "score_balanced") & (summary["k"] == 20) & (summary["experiment"] == exp)
        if len(summary[m]) > 0:
            r = summary[m].iloc[0]
            rows.append({
                "experiment": exp,
                "Δ_prize_sum": r["actual_prize_sum_mean"] - full_row["actual_prize_sum_mean"],
                "Δ_bt_place_n": r["bt_place_n_mean"] - full_row["bt_place_n_mean"],
                "Δ_bt_win_n": r["bt_win_n_mean"] - full_row["bt_win_n_mean"],
                "Δ_graded_win_n": r["graded_win_n_mean"] - full_row["graded_win_n_mean"],
                "Δ_graded_win_lift": r["graded_win_lift_mean"] - full_row["graded_win_lift_mean"],
            })
    delta_df = pd.DataFrame(rows).sort_values("Δ_graded_win_lift", ascending=False)
    print(delta_df.round(3).to_string(index=False))

# --- 9. q90_prize Top-20 comparison ---
print("\n### 9. Top-20 q90_prize (mean across folds)")
mask = (summary["score_col"] == "q90_prize") & (summary["k"] == 20)
q20 = summary[mask][["experiment", "actual_prize_sum_mean",
                       "bt_place_n_mean", "bt_win_n_mean", "graded_win_n_mean",
                       "graded_win_lift_mean"]].copy()
q20 = q20.sort_values("graded_win_lift_mean", ascending=False)
print(q20.round(2).to_string(index=False))

# --- 10. p_bt_win Top-20 comparison ---
print("\n### 10. Top-20 p_bt_win (mean across folds)")
mask = (summary["score_col"] == "p_bt_win") & (summary["k"] == 20)
pw20 = summary[mask][["experiment", "actual_prize_sum_mean",
                        "bt_place_n_mean", "bt_win_n_mean", "graded_win_n_mean",
                        "graded_win_lift_mean"]].copy()
pw20 = pw20.sort_values("graded_win_lift_mean", ascending=False)
print(pw20.round(2).to_string(index=False))

print("\n=== ANALYSIS COMPLETE ===")

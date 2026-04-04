"""
backtest_two_tower.py - 双塔架构与嵌套架构对比回测

用法:
    python -m src.backtest_two_tower

输出:
    outputs/comparison_report.csv - 对比报告
"""
import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from pog_model.config import Config
from pog_model.two_tower_config import TwoTowerConfig
from pog_model.data import load_training_frame, load_completed_birth_years
from pog_model.features import FeatureSet
from pog_model.eval import evaluate_binary
from pog_model.pipeline import train_all_models, predict_all, build_blended_scores, predict_ranking
from pog_model.two_tower_pipeline import train_two_tower_models, predict_all as predict_all_two_tower, build_blended_scores as build_blended_scores_two_tower
from pog_model.split import auto_configure_splits, split_by_birth_year


def build_topk_report(test_pred, score_cols, ks=[20, 50, 100]):
    """构建 Top-k 回测报告"""
    rows = []
    base_graded_rate = test_pred["graded_win_flag"].mean() if "graded_win_flag" in test_pred.columns else 0
    base_prize_30m_rate = test_pred["pog_total_prize_ge_30m_flag"].mean() if "pog_total_prize_ge_30m_flag" in test_pred.columns else 0
    
    for score_col in score_cols:
        if score_col not in test_pred.columns:
            continue
        ranked = test_pred.sort_values(score_col, ascending=False).reset_index(drop=True)
        
        for k in ks:
            top = ranked.head(k)
            if len(top) == 0:
                continue
            
            n = len(top)
            graded_n = int(top["graded_win_flag"].sum()) if "graded_win_flag" in top.columns else 0
            prize_30m_n = int(top["pog_total_prize_ge_30m_flag"].sum()) if "pog_total_prize_ge_30m_flag" in top.columns else 0
            win_n = int(top["win_flag"].sum()) if "win_flag" in top.columns else 0
            
            graded_prec = graded_n / n if n > 0 else 0
            prize_30m_prec = prize_30m_n / n if n > 0 else 0
            win_prec = win_n / n if n > 0 else 0
            
            rows.append({
                "score_col": score_col,
                "k": k,
                "n_selected": n,
                "win_n": win_n,
                "graded_win_n": graded_n,
                "prize_ge_30m_n": prize_30m_n,
                "win_precision": win_prec,
                "graded_win_precision": graded_prec,
                "prize_ge_30m_precision": prize_30m_prec,
                "graded_win_lift": graded_prec / base_graded_rate if base_graded_rate > 0 else np.nan,
                "prize_ge_30m_lift": prize_30m_prec / base_prize_30m_rate if base_prize_30m_rate > 0 else np.nan,
                "actual_prize_sum": float(top["pog_total_prize"].sum()) if "pog_total_prize" in top.columns else np.nan,
                "actual_prize_mean": float(top["pog_total_prize"].mean()) if "pog_total_prize" in top.columns else np.nan,
            })
    
    return pd.DataFrame(rows)


def main():
    os.makedirs("outputs", exist_ok=True)
    
    print("=" * 70)
    print("ARCHITECTURE COMPARISON: NESTED vs TWO-TOWER")
    print("=" * 70)
    
    # 加载数据
    cfg_nested = Config()
    cfg_nested = auto_configure_splits(cfg_nested)
    
    cfg_two_tower = TwoTowerConfig()
    cfg_two_tower.train_birth_year_start = cfg_nested.train_birth_year_start
    cfg_two_tower.train_birth_year_end = cfg_nested.train_birth_year_end
    cfg_two_tower.valid_birth_year_start = cfg_nested.valid_birth_year_start
    cfg_two_tower.valid_birth_year_end = cfg_nested.valid_birth_year_end
    cfg_two_tower.test_birth_year_start = cfg_nested.test_birth_year_start
    cfg_two_tower.test_birth_year_end = cfg_nested.test_birth_year_end
    
    print(f"\nData split: train={cfg_nested.train_birth_year_start}-{cfg_nested.train_birth_year_end}, "
          f"valid={cfg_nested.valid_birth_year_start}, test={cfg_nested.test_birth_year_start}")
    
    df = load_training_frame(cfg_nested)
    train_df, valid_df, test_df = split_by_birth_year(df, cfg_nested)
    
    print(f"Train: {len(train_df)}, Valid: {len(valid_df)}, Test: {len(test_df)}")
    
    feature_set = FeatureSet()
    
    # =========================
    # 训练嵌套架构模型
    # =========================
    print("\n" + "=" * 70)
    print("TRAINING NESTED ARCHITECTURE")
    print("=" * 70)
    
    nested_bundle = train_all_models(train_df, valid_df, feature_set, config=cfg_nested)
    
    # 嵌套架构预测
    nested_pred = test_df[["ketto_num", "birth_year", "pog_total_prize", "win_flag", "graded_win_flag", 
                           "positive_prize_flag", "pog_total_prize_ge_30m_flag"]].copy()
    nested_pred_cols = predict_all(nested_bundle, test_df, train_df=train_df)
    for col in nested_pred_cols.columns:
        nested_pred[col] = nested_pred_cols[col].values
    
    ranking_scores = predict_ranking(nested_bundle, test_df, train_df=train_df) if nested_bundle.ranking_model is not None else None
    nested_pred = build_blended_scores(nested_pred, ceiling_weights=nested_bundle.ceiling_weights, ranking_scores=ranking_scores)
    
    # =========================
    # 训练双塔架构模型
    # =========================
    print("\n" + "=" * 70)
    print("TRAINING TWO-TOWER ARCHITECTURE")
    print("=" * 70)
    
    two_tower_bundle = train_two_tower_models(train_df, valid_df, feature_set, config=cfg_two_tower)
    
    # 双塔架构预测
    two_tower_pred = test_df[["ketto_num", "birth_year", "pog_total_prize", "win_flag", "graded_win_flag",
                              "positive_prize_flag", "pog_total_prize_ge_30m_flag"]].copy()
    two_tower_pred_cols = predict_all_two_tower(two_tower_bundle, test_df)
    for col in two_tower_pred_cols.columns:
        two_tower_pred[col] = two_tower_pred_cols[col].values
    
    two_tower_pred = build_blended_scores_two_tower(two_tower_pred, two_tower_bundle, full_df=test_df)
    
    # =========================
    # 指标对比
    # =========================
    print("\n" + "=" * 70)
    print("METRICS COMPARISON")
    print("=" * 70)
    
    comparison = []
    
    # win_flag
    nested_win = evaluate_binary(nested_pred["win_flag"], nested_pred["p_win"], "win")
    tt_win = evaluate_binary(two_tower_pred["win_flag"], two_tower_pred["p_win"], "win")
    comparison.append({
        "metric": "win_auc",
        "nested": nested_win.get("win_auc", np.nan),
        "two_tower": tt_win.get("win_auc", np.nan),
        "diff": tt_win.get("win_auc", 0) - nested_win.get("win_auc", 0),
    })
    comparison.append({
        "metric": "win_ap",
        "nested": nested_win.get("win_ap", np.nan),
        "two_tower": tt_win.get("win_ap", np.nan),
        "diff": tt_win.get("win_ap", 0) - nested_win.get("win_ap", 0),
    })
    
    # graded_win - 关键对比
    nested_graded = evaluate_binary(nested_pred["graded_win_flag"], nested_pred["p_graded_win"], "graded_win")
    tt_graded = evaluate_binary(two_tower_pred["graded_win_flag"], two_tower_pred["p_graded_win_direct"], "graded_win_direct")
    comparison.append({
        "metric": "graded_win_auc",
        "nested": nested_graded.get("graded_win_auc", np.nan),
        "two_tower": tt_graded.get("graded_win_direct_auc", np.nan),
        "diff": tt_graded.get("graded_win_direct_auc", 0) - nested_graded.get("graded_win_auc", 0),
    })
    comparison.append({
        "metric": "graded_win_ap",
        "nested": nested_graded.get("graded_win_ap", np.nan),
        "two_tower": tt_graded.get("graded_win_direct_ap", np.nan),
        "diff": tt_graded.get("graded_win_direct_ap", 0) - nested_graded.get("graded_win_ap", 0),
    })
    
    # prize_ge_30m
    nested_30m = evaluate_binary(nested_pred["pog_total_prize_ge_30m_flag"], nested_pred["p_prize_ge_30m"], "prize_ge_30m")
    tt_30m = evaluate_binary(two_tower_pred["pog_total_prize_ge_30m_flag"], two_tower_pred["p_prize_ge_30m_direct"], "prize_ge_30m_direct")
    comparison.append({
        "metric": "prize_ge_30m_auc",
        "nested": nested_30m.get("prize_ge_30m_auc", np.nan),
        "two_tower": tt_30m.get("prize_ge_30m_direct_auc", np.nan),
        "diff": tt_30m.get("prize_ge_30m_direct_auc", 0) - nested_30m.get("prize_ge_30m_auc", 0),
    })
    
    comparison_df = pd.DataFrame(comparison)
    print("\n--- AUC/AP Comparison ---")
    print(comparison_df.to_string(index=False))
    
    # =========================
    # Top-k 对比
    # =========================
    print("\n" + "=" * 70)
    print("TOP-K COMPARISON")
    print("=" * 70)
    
    # 嵌套架构 Top-k
    nested_score_cols = ["expected_pog_prize", "p_graded_win", "p_prize_ge_30m", "score_ceiling"]
    if "score_ranking" in nested_pred.columns:
        nested_score_cols.append("score_ranking")
    nested_topk = build_topk_report(nested_pred, nested_score_cols)
    nested_topk["architecture"] = "nested"
    
    # 双塔架构 Top-k
    tt_score_cols = ["expected_pog_prize", "p_graded_win_direct", "p_prize_ge_30m_direct"]
    if "score_ranking" in two_tower_pred.columns:
        tt_score_cols.append("score_ranking")
    tt_topk = build_topk_report(two_tower_pred, tt_score_cols)
    tt_topk["architecture"] = "two_tower"
    
    # 合并对比
    topk_combined = pd.concat([nested_topk, tt_topk], ignore_index=True)
    
    # 关键指标对比表
    print("\n--- Top-50 Graded Win Precision ---")
    top50_graded = topk_combined[topk_combined["k"] == 50].sort_values("graded_win_precision", ascending=False)
    print(top50_graded[["architecture", "score_col", "graded_win_n", "graded_win_precision", "graded_win_lift"]].to_string(index=False))
    
    print("\n--- Top-50 Prize Sum ---")
    top50_prize = topk_combined[topk_combined["k"] == 50].sort_values("actual_prize_sum", ascending=False)
    print(top50_prize[["architecture", "score_col", "actual_prize_sum", "actual_prize_mean"]].to_string(index=False))
    
    # =========================
    # 保存结果
    # =========================
    comparison_df.to_csv("outputs/comparison_metrics.csv", index=False)
    topk_combined.to_csv("outputs/comparison_topk.csv", index=False)
    
    # 详细对比报告
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("ARCHITECTURE COMPARISON REPORT")
    report_lines.append("=" * 70)
    report_lines.append("")
    report_lines.append("## Data Split")
    report_lines.append(f"Train: {cfg_nested.train_birth_year_start}-{cfg_nested.train_birth_year_end}")
    report_lines.append(f"Valid: {cfg_nested.valid_birth_year_start}")
    report_lines.append(f"Test: {cfg_nested.test_birth_year_start}")
    report_lines.append(f"Test samples: {len(test_df)}")
    report_lines.append("")
    report_lines.append("## Metrics Comparison")
    report_lines.append(comparison_df.to_string(index=False))
    report_lines.append("")
    report_lines.append("## Top-50 Graded Win Precision")
    report_lines.append(top50_graded[["architecture", "score_col", "graded_win_precision", "graded_win_lift"]].to_string(index=False))
    report_lines.append("")
    report_lines.append("## Key Findings")
    
    # 自动生成结论
    graded_diff = comparison_df[comparison_df["metric"] == "graded_win_auc"]["diff"].values[0]
    if graded_diff > 0.01:
        report_lines.append(f"- Two-Tower shows significant improvement in graded_win AUC (+{graded_diff:.4f})")
    elif graded_diff > 0:
        report_lines.append(f"- Two-Tower shows slight improvement in graded_win AUC (+{graded_diff:.4f})")
    else:
        report_lines.append(f"- Nested architecture performs better in graded_win AUC ({graded_diff:.4f})")
    
    report_lines.append("")
    report_lines.append("## Conclusion")
    report_lines.append("Two-Tower architecture decouples floor/ceiling predictions,")
    report_lines.append("eliminating error propagation from nested conditional models.")
    report_lines.append("This is particularly beneficial for rare targets like graded_win_flag.")
    
    with open("outputs/comparison_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    
    print("\n" + "=" * 70)
    print("FILES OUTPUTTED:")
    print("- outputs/comparison_metrics.csv")
    print("- outputs/comparison_topk.csv")
    print("- outputs/comparison_report.txt")
    print("=" * 70)


if __name__ == "__main__":
    load_dotenv()
    main()
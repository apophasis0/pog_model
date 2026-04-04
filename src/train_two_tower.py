"""
train_two_tower.py - 双塔解耦架构训练入口

用法:
    python -m src.train_two_tower
"""
import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from pog_model.two_tower_config import TwoTowerConfig
from pog_model.data import (
    load_training_frame,
    load_scoring_frame,
    save_predictions,
)
from pog_model.features import FeatureSet
from pog_model.eval import evaluate_binary, evaluate_regression
from pog_model.two_tower_pipeline import (
    train_two_tower_models,
    predict_all,
    build_blended_scores,
    save_bundle,
)
from pog_model.split import (
    auto_configure_two_tower_splits,
    split_by_birth_year,
    describe_split,
)


# =========================
# Main
# =========================

def main():
    cfg = TwoTowerConfig()
    cfg = auto_configure_two_tower_splits(cfg)

    print("=" * 60)
    print("TWO-TOWER ARCHITECTURE TRAINING")
    print("=" * 60)
    print(f"\n=== SPLIT CONFIG ===")
    print(
        f"train: {cfg.train_birth_year_start}-{cfg.train_birth_year_end}, "
        f"valid: {cfg.valid_birth_year_start}-{cfg.valid_birth_year_end}, "
        f"test: {cfg.test_birth_year_start}-{cfg.test_birth_year_end}"
    )

    os.makedirs("models/two_tower", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # -------------------------
    # Load data
    # -------------------------
    df = load_training_frame(cfg)

    required_cols = [
        "ketto_num",
        "birth_year",
        "win_flag",
        "graded_win_flag",
        "positive_prize_flag",
        "pog_total_prize",
        "pog_total_prize_ge_30m_flag",
    ]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"训练数据缺少必要列: {missing_cols}")

    train_df, valid_df, test_df = split_by_birth_year(df, cfg)

    describe_split("train", train_df)
    describe_split("valid", valid_df)
    describe_split("test", test_df)

    if len(train_df) == 0:
        raise ValueError("train_df 为空")
    if len(valid_df) == 0:
        raise ValueError("valid_df 为空")
    if len(test_df) == 0:
        raise ValueError("test_df 为空")

    # -------------------------
    # Train Two-Tower models
    # -------------------------
    feature_set = FeatureSet()
    bundle = train_two_tower_models(train_df, valid_df, feature_set, config=cfg)

    # -------------------------
    # Save models
    # -------------------------
    save_bundle(bundle, "models/two_tower/", meta_extra={
        "model_name": cfg.model_name,
        "model_version": cfg.model_version,
        "architecture": "two_tower_decoupled",
        "train_years": [cfg.train_birth_year_start, cfg.train_birth_year_end],
        "valid_years": [cfg.valid_birth_year_start, cfg.valid_birth_year_end],
        "test_years": [cfg.test_birth_year_start, cfg.test_birth_year_end],
    })

    # -------------------------
    # Test predictions
    # -------------------------
    print("\n=== BUILD TEST PREDICTIONS ===")

    keep_cols = [
        c for c in [
            "ketto_num",
            "birth_year",
            "sire_name",
            "dam_name",
            "damsire_name",
            "chokyosi_ryakusyo",
            "pog_total_prize",
            "win_flag",
            "graded_win_flag",
            "positive_prize_flag",
            "pog_total_prize_ge_30m_flag",
        ]
        if c in test_df.columns
    ]
    test_pred = test_df[keep_cols].copy()

    pred_cols = predict_all(bundle, test_df)
    for col in pred_cols.columns:
        test_pred[col] = pred_cols[col].values

    test_pred = build_blended_scores(test_pred, bundle, full_df=test_df)

    # -------------------------
    # Metrics
    # -------------------------
    metrics = {}

    # 下限塔评估
    metrics.update(evaluate_binary(test_pred["win_flag"], test_pred["p_win"], "win"))
    metrics.update(
        evaluate_binary(
            test_pred["positive_prize_flag"],
            test_pred["p_positive_prize"],
            "positive_prize"
        )
    )

    # 上限塔评估 - 关键指标
    metrics.update(
        evaluate_binary(
            test_pred["graded_win_flag"],
            test_pred["p_graded_win_direct"],
            "graded_win_direct"
        )
    )
    metrics.update(
        evaluate_binary(
            test_pred["pog_total_prize_ge_30m_flag"],
            test_pred["p_prize_ge_30m_direct"],
            "prize_ge_30m_direct"
        )
    )

    # 奖金评估
    pos_mask = test_pred["pog_total_prize"] > 0
    metrics.update(
        evaluate_regression(
            test_pred.loc[pos_mask, "pog_total_prize"],
            test_pred.loc[pos_mask, "pred_log_prize_pos"],
            "prize_pos_only"
        )
    )
    metrics.update(
        evaluate_regression(
            test_pred["pog_total_prize"],
            np.log1p(np.clip(test_pred["expected_pog_prize"], 0, None)),
            "prize_expected_all"
        )
    )

    print("\n=== TEST METRICS (TWO-TOWER) ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    metrics_df = pd.DataFrame(
        [{"metric": k, "value": v} for k, v in metrics.items()]
    )
    metrics_df.to_csv("outputs/test_metrics_two_tower.csv", index=False)

    # -------------------------
    # Top-k report
    # -------------------------
    score_cols = [
        "expected_pog_prize",
        "p_win",
        "p_graded_win_direct",
        "p_prize_ge_30m_direct",
        "q90_prize",
    ]
    if "score_ranking" in test_pred.columns:
        score_cols.append("score_ranking")

    ks = [20, 50, 100]
    topk_rows = []

    base_graded_rate = test_pred["graded_win_flag"].mean()
    base_prize_30m_rate = test_pred["pog_total_prize_ge_30m_flag"].mean()

    for score_col in score_cols:
        if score_col not in test_pred.columns:
            continue

        ranked = test_pred.sort_values(score_col, ascending=False).reset_index(drop=True)

        for k in ks:
            top = ranked.head(k).copy()
            if len(top) == 0:
                continue

            n = len(top)
            graded_n = int(top["graded_win_flag"].sum()) if "graded_win_flag" in top.columns else 0
            prize_30m_n = int(top["pog_total_prize_ge_30m_flag"].sum()) if "pog_total_prize_ge_30m_flag" in top.columns else 0

            graded_prec = graded_n / n if n > 0 else 0
            prize_30m_prec = prize_30m_n / n if n > 0 else 0

            topk_rows.append({
                "score_col": score_col,
                "k": k,
                "n_selected": n,
                "graded_win_n": graded_n,
                "prize_ge_30m_n": prize_30m_n,
                "graded_win_precision": graded_prec,
                "prize_ge_30m_precision": prize_30m_prec,
                "graded_win_lift": graded_prec / base_graded_rate if base_graded_rate > 0 else np.nan,
                "prize_ge_30m_lift": prize_30m_prec / base_prize_30m_rate if base_prize_30m_rate > 0 else np.nan,
                "actual_prize_sum": float(top["pog_total_prize"].sum()) if "pog_total_prize" in top.columns else np.nan,
                "actual_prize_mean": float(top["pog_total_prize"].mean()) if "pog_total_prize" in top.columns else np.nan,
            })

    topk_report = pd.DataFrame(topk_rows)
    topk_report.to_csv("outputs/topk_backtest_two_tower.csv", index=False)

    print("\n=== TOP-K BACKTEST REPORT (TWO-TOWER) ===")
    print(topk_report.sort_values(["score_col", "k"]).to_string(index=False))

    test_pred.to_csv("outputs/test_predictions_two_tower.csv", index=False)

    # -------------------------
    # Score current cohort
    # -------------------------
    print("\n=== SCORE CURRENT COHORT ===")

    score_df = load_scoring_frame(cfg)

    score_keep_cols = [
        c for c in [
            "ketto_num",
            "birth_year",
            "sire_name",
            "dam_name",
            "damsire_name",
            "chokyosi_ryakusyo",
            "breeder_code",
            "banusi_name",
        ]
        if c in score_df.columns
    ]
    current_pred = score_df[score_keep_cols].copy()

    current_pred_cols = predict_all(bundle, score_df)
    for col in current_pred_cols.columns:
        current_pred[col] = current_pred_cols[col].values

    current_pred = build_blended_scores(current_pred, bundle, full_df=score_df)

    current_pred["model_name"] = cfg.model_name
    current_pred["model_version"] = cfg.model_version
    current_pred["asof_date"] = pd.to_datetime(cfg.asof_date)

    # 排序输出
    if "score_ranking" in current_pred.columns:
        current_pred = current_pred.sort_values("score_ranking", ascending=False).reset_index(drop=True)
    elif "p_graded_win_direct" in current_pred.columns:
        current_pred = current_pred.sort_values("p_graded_win_direct", ascending=False).reset_index(drop=True)
    else:
        current_pred = current_pred.sort_values("expected_pog_prize", ascending=False).reset_index(drop=True)

    current_pred.to_csv("outputs/current_cohort_predictions_two_tower.csv", index=False)

    # Shortlist outputs
    shortlist_cols = [
        c for c in [
            "ketto_num",
            "birth_year",
            "sire_name",
            "dam_name",
            "damsire_name",
            "chokyosi_ryakusyo",
            "banusi_name",
            "p_win",
            "p_positive_prize",
            "p_graded_win_direct",
            "p_prize_ge_30m_direct",
            "q90_prize",
            "expected_pog_prize",
            "score_ranking",
        ]
        if c in current_pred.columns
    ]

    if "score_ranking" in current_pred.columns:
        current_pred.sort_values("score_ranking", ascending=False).head(100)[shortlist_cols].to_csv(
            "outputs/current_top100_two_tower_ranking.csv", index=False
        )
    current_pred.sort_values("p_graded_win_direct", ascending=False).head(100)[shortlist_cols].to_csv(
        "outputs/current_top100_two_tower_graded.csv", index=False
    )
    current_pred.sort_values("p_prize_ge_30m_direct", ascending=False).head(100)[shortlist_cols].to_csv(
        "outputs/current_top100_two_tower_prize_30m.csv", index=False
    )

    print("\n=== DONE ===")
    print("Files outputted:")
    print("- models/two_tower/")
    print("- outputs/test_metrics_two_tower.csv")
    print("- outputs/topk_backtest_two_tower.csv")
    print("- outputs/test_predictions_two_tower.csv")
    print("- outputs/current_cohort_predictions_two_tower.csv")
    if "score_ranking" in current_pred.columns:
        print("- outputs/current_top100_two_tower_ranking.csv")
    print("- outputs/current_top100_two_tower_graded.csv")
    print("- outputs/current_top100_two_tower_prize_30m.csv")


if __name__ == "__main__":
    load_dotenv()
    main()
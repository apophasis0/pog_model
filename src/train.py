import os
import json
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from pog_model.config import Config
from pog_model.data import (
    load_training_frame,
    load_scoring_frame,
    save_predictions,
)
from pog_model.features import FeatureSet
from pog_model.eval import evaluate_binary, evaluate_regression
from pog_model.pipeline import (
    train_all_models,
    predict_all,
    predict_ranking,
    build_blended_scores,
    build_topk_report,
    print_metrics,
    save_bundle,
)
from pog_model.split import (
    auto_configure_splits,
    split_by_birth_year,
    describe_split,
)


# =========================
# Main
# =========================

def main():
    cfg = Config()
    cfg = auto_configure_splits(cfg)

    print("=== SPLIT CONFIG ===")
    print(
        f"train: {cfg.train_birth_year_start}-{cfg.train_birth_year_end}, "
        f"valid: {cfg.valid_birth_year_start}-{cfg.valid_birth_year_end}, "
        f"test: {cfg.test_birth_year_start}-{cfg.test_birth_year_end}"
    )

    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # -------------------------
    # Load data
    # -------------------------
    df = load_training_frame(cfg)

    required_cols = [
        "ketto_num",
        "birth_year",
        "win_flag",
        "bt_place_flag",
        "bt_win_flag",
        "graded_win_flag",
        "positive_prize_flag",
        "pog_total_prize",
        "pog_total_prize_ge_10m_flag",
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
    # Train all models
    # -------------------------
    feature_set = FeatureSet()  # all features ON
    bundle = train_all_models(train_df, valid_df, feature_set)

    # -------------------------
    # Save models
    # -------------------------
    save_bundle(bundle, "models/", meta_extra={
        "model_name": cfg.model_name,
        "model_version": cfg.model_version,
        "nested_chain": ["win_flag", "bt_place_flag", "bt_win_flag", "graded_win_flag"],
        "ceiling_targets": ["pog_total_prize_ge_10m_flag", "pog_total_prize_ge_30m_flag"],
        "quantile_targets": ["q80", "q90"],
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
            "bt_place_flag",
            "bt_win_flag",
            "graded_win_flag",
            "positive_prize_flag",
            "pog_total_prize_ge_10m_flag",
            "pog_total_prize_ge_30m_flag",
        ]
        if c in test_df.columns
    ]
    test_pred = test_df[keep_cols].copy()

    pred_cols = predict_all(bundle, test_df, train_df=train_df)
    for col in pred_cols.columns:
        test_pred[col] = pred_cols[col].values

    # Ranking scores
    ranking_scores = predict_ranking(bundle, test_df, train_df=train_df) if bundle.ranking_model is not None else None

    test_pred = build_blended_scores(
        test_pred,
        ceiling_weights=bundle.ceiling_weights,
        ranking_scores=ranking_scores,
    )

    # -------------------------
    # Metrics
    # -------------------------
    metrics = {}

    # 边际概率评估
    metrics.update(evaluate_binary(test_pred["win_flag"], test_pred["p_win"], "win"))
    metrics.update(evaluate_binary(test_pred["bt_place_flag"], test_pred["p_bt_place"], "bt_place"))
    metrics.update(evaluate_binary(test_pred["bt_win_flag"], test_pred["p_bt_win"], "bt_win"))
    metrics.update(evaluate_binary(test_pred["graded_win_flag"], test_pred["p_graded_win"], "graded_win"))
    metrics.update(
        evaluate_binary(
            test_pred["positive_prize_flag"],
            test_pred["p_positive_prize"],
            "positive_prize"
        )
    )

    # ceiling 模型评估
    metrics.update(
        evaluate_binary(
            test_pred["pog_total_prize_ge_10m_flag"],
            test_pred["p_prize_ge_10m"],
            "prize_ge_10m"
        )
    )
    metrics.update(
        evaluate_binary(
            test_pred["pog_total_prize_ge_30m_flag"],
            test_pred["p_prize_ge_30m"],
            "prize_ge_30m"
        )
    )

    # 条件概率评估
    mask_win = test_df["win_flag"] == 1
    metrics.update(
        evaluate_binary(
            test_df.loc[mask_win, "bt_place_flag"],
            test_pred.loc[mask_win, "p_bt_place_given_win"],
            "bt_place_given_win"
        )
    )

    mask_bt_place = test_df["bt_place_flag"] == 1
    metrics.update(
        evaluate_binary(
            test_df.loc[mask_bt_place, "bt_win_flag"],
            test_pred.loc[mask_bt_place, "p_bt_win_given_bt_place"],
            "bt_win_given_bt_place"
        )
    )

    mask_bt_win = test_df["bt_win_flag"] == 1
    metrics.update(
        evaluate_binary(
            test_df.loc[mask_bt_win, "graded_win_flag"],
            test_pred.loc[mask_bt_win, "p_graded_given_bt_win"],
            "graded_given_bt_win"
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

    print_metrics(metrics)

    metrics_df = pd.DataFrame(
        [{"metric": k, "value": v} for k, v in metrics.items()]
    )
    metrics_df.to_csv("outputs/test_metrics_nested.csv", index=False)

    # -------------------------
    # Top-k backtest
    # -------------------------
    score_cols = [
        "expected_pog_prize",
        "p_win",
        "p_bt_place",
        "p_bt_win",
        "p_graded_win",
        "p_prize_ge_10m",
        "p_prize_ge_30m",
        "q90_prize",
        "score_ceiling",
        "score_ranking",
    ]
    ks = [20, 50, 100]

    topk_report = build_topk_report(test_pred, score_cols=score_cols, ks=ks)
    topk_report.to_csv("outputs/topk_backtest_report.csv", index=False)

    print("\n=== TOP-K BACKTEST REPORT ===")
    print(topk_report.sort_values(["score_col", "k"]).to_string(index=False))

    test_pred.to_csv("outputs/test_predictions_nested.csv", index=False)

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

    # Ranking scores for current cohort
    current_ranking_scores = predict_ranking(bundle, score_df) if bundle.ranking_model is not None else None

    current_pred = build_blended_scores(
        current_pred,
        ceiling_weights=bundle.ceiling_weights,
        ranking_scores=current_ranking_scores,
    )

    current_pred["model_name"] = cfg.model_name
    current_pred["model_version"] = cfg.model_version
    current_pred["asof_date"] = pd.to_datetime(cfg.asof_date)

    # 排序输出
    if "score_ranking" in current_pred.columns:
        current_pred = current_pred.sort_values("score_ranking", ascending=False).reset_index(drop=True)
    elif "score_ceiling" in current_pred.columns:
        current_pred = current_pred.sort_values("score_ceiling", ascending=False).reset_index(drop=True)
    else:
        current_pred = current_pred.sort_values("expected_pog_prize", ascending=False).reset_index(drop=True)
    current_pred.to_csv("outputs/current_cohort_predictions_nested.csv", index=False)

    # 存入数据库
    pred_db_cols = [
        "model_name",
        "model_version",
        "asof_date",
        "birth_year",
        "ketto_num",
        "p_win",
        "p_bt_place",
        "p_graded_win",
        "p_positive_prize",
        "expected_pog_prize",
        "pred_log_prize_pos",
    ]
    pred_db_df = current_pred[[c for c in pred_db_cols if c in current_pred.columns]].copy()

    try:
        save_predictions(cfg, pred_db_df)
    except Exception as e:
        print(f"[WARN] 保存到 pog.model_predictions 失败：{e}")
        print("[WARN] 已生成 CSV：outputs/current_cohort_predictions_nested.csv")

    # 额外输出几个 shortlist
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
            "p_bt_place",
            "p_bt_win",
            "p_graded_win",
            "p_prize_ge_10m",
            "p_prize_ge_30m",
            "q90_prize",
            "expected_pog_prize",
            "score_ceiling",
            "score_ranking",
        ]
        if c in current_pred.columns
    ]

    if "score_ranking" in current_pred.columns:
        current_pred.sort_values("score_ranking", ascending=False).head(100)[shortlist_cols].to_csv(
            "outputs/current_top100_ranking.csv", index=False
        )
    if "score_ceiling" in current_pred.columns:
        current_pred.sort_values("score_ceiling", ascending=False).head(100)[shortlist_cols].to_csv(
            "outputs/current_top100_ceiling.csv", index=False
        )
    current_pred.sort_values("p_graded_win", ascending=False).head(100)[shortlist_cols].to_csv(
        "outputs/current_top100_graded_win.csv", index=False
    )

    print("\n=== DONE ===")
    print("Files outputted:")
    print("- outputs/test_metrics_nested.csv")
    print("- outputs/topk_backtest_report.csv")
    print("- outputs/test_predictions_nested.csv")
    print("- outputs/current_cohort_predictions_nested.csv")
    if "score_ranking" in current_pred.columns:
        print("- outputs/current_top100_ranking.csv")
    if "score_ceiling" in current_pred.columns:
        print("- outputs/current_top100_ceiling.csv")
    print("- outputs/current_top100_graded_win.csv")


if __name__ == "__main__":
    load_dotenv()
    main()

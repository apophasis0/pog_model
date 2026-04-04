"""
score_cohorts.py — Load a saved ModelBundle and run inference on specified birth year cohorts.

Usage:
    uv run python src/score_cohorts.py                         # defaults: models_recommended/, years 2023-2024
    uv run python src/score_cohorts.py --model-dir models/     # use full-feature models
    uv run python src/score_cohorts.py --years 2023            # score only 2023
"""
import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from pog_model.config import Config
from pog_model.features import FeatureSet, prepare_matrix
from pog_model.pipeline import (
    ModelBundle,
    predict_all,
    predict_ranking,
    build_blended_scores,
    build_topk_report,
    load_bundle,
)
from pog_model.data import load_cohort_frame


def score_cohort(bundle: ModelBundle, cohort_df: pd.DataFrame) -> pd.DataFrame:
    """Run full inference on a cohort and return predictions with metadata."""
    keep_cols = [
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
        if c in cohort_df.columns
    ]
    result = cohort_df[keep_cols].copy()

    pred_cols = predict_all(bundle, cohort_df)
    for col in pred_cols.columns:
        result[col] = pred_cols[col].values

    # Ranking scores
    ranking_scores = predict_ranking(bundle, cohort_df) if bundle.ranking_model is not None else None

    result = build_blended_scores(
        result,
        ceiling_weights=bundle.ceiling_weights,
        ranking_scores=ranking_scores,
    )

    # Sort by available score priority
    if "score_ranking" in result.columns:
        return result.sort_values("score_ranking", ascending=False).reset_index(drop=True)
    elif "score_ceiling" in result.columns:
        return result.sort_values("score_ceiling", ascending=False).reset_index(drop=True)
    else:
        return result.sort_values("expected_pog_prize", ascending=False).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Score cohorts using a saved ModelBundle")
    parser.add_argument(
        "--model-dir",
        default="models_recommended",
        help="Directory containing saved model bundle (default: models_recommended)",
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=[2023, 2024],
        help="Birth years to score (default: 2023 2024)",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/scored_cohorts",
        help="Output directory for predictions (default: outputs/scored_cohorts)",
    )
    parser.add_argument(
        "--top-k",
        nargs="+",
        type=int,
        default=[20, 50, 100],
        help="Top-k values for shortlist outputs (default: 20 50 100)",
    )
    args = parser.parse_args()

    cfg = Config()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model bundle
    print(f"=== LOADING MODEL BUNDLE from {args.model_dir} ===")
    bundle = load_bundle(args.model_dir)
    print(f"Feature set: {bundle.feature_set}")
    print(f"Ceiling weights: {bundle.ceiling_weights}")
    print(f"Has ranking model: {bundle.ranking_model is not None}")

    all_preds = []

    for year in args.years:
        print(f"\n{'='*60}")
        print(f"=== SCORING COHORT: birth_year={year} ===")
        print(f"{'='*60}")

        cohort_df = load_cohort_frame(cfg, year)
        if len(cohort_df) == 0:
            print(f"[WARN] No data for birth_year={year}, skipping.")
            continue

        pred = score_cohort(bundle, cohort_df)
        pred["model_dir"] = os.path.basename(args.model_dir)
        all_preds.append(pred)

        # Save full predictions
        out_path = os.path.join(args.output_dir, f"predictions_{year}.csv")
        pred.to_csv(out_path, index=False)
        print(f"  → Saved full predictions: {out_path} ({len(pred)} rows)")

        # Save shortlists by scoring function
        shortlist_cols = [
            c for c in [
                "ketto_num", "birth_year",
                "sire_name", "dam_name", "damsire_name",
                "chokyosi_ryakusyo", "banusi_name",
                "p_win", "p_bt_place", "p_bt_win", "p_graded_win",
                "p_prize_ge_10m", "p_prize_ge_30m",
                "q90_prize", "expected_pog_prize",
                "score_ceiling", "score_ranking",
            ]
            if c in pred.columns
        ]

        for score_col in ["score_ceiling", "score_ranking", "p_graded_win"]:
            if score_col not in pred.columns:
                continue
            for k in args.top_k:
                top = pred.sort_values(score_col, ascending=False).head(k)
                tag = score_col.replace("score_", "").replace("p_", "")
                out_path = os.path.join(args.output_dir, f"top{k}_{tag}_{year}.csv")
                top[shortlist_cols].to_csv(out_path, index=False)

        print(f"  → Saved shortlists for k={args.top_k}")

    # Save combined predictions
    if all_preds:
        combined = pd.concat(all_preds, ignore_index=True)
        combined_path = os.path.join(args.output_dir, "predictions_all.csv")
        combined.to_csv(combined_path, index=False)
        print(f"\n=== COMBINED: {combined_path} ({len(combined)} rows) ===")

    print("\n=== DONE ===")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    load_dotenv()
    main()

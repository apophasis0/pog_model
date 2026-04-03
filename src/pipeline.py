"""
pipeline.py — Reusable training, prediction, and evaluation pipeline.

Extracts shared logic from train.py and backtest.py into a single module
that can be called programmatically for ablation experiments, rolling
backtests, or production training.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, CatBoostRanker, Pool
from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression

from features import (
    FeatureSet, prepare_matrix, add_log_target,
    fit_category_frequencies, apply_rare_filter,
)
from config import Config


# =========================
# Model helpers
# =========================

def build_pool(X, y, cat_cols):
    return Pool(X, label=y, cat_features=[c for c in cat_cols if c in X.columns])


def subset_by_condition(df: pd.DataFrame, condition_col: str | None) -> pd.DataFrame:
    if condition_col is None:
        return df.copy()
    return df[df[condition_col] == 1].copy()


def describe_stage_split(name: str, df: pd.DataFrame, target: str):
    n = len(df)
    pos = int(df[target].sum()) if target in df.columns else 0
    rate = pos / n if n > 0 else 0.0
    print(f"[{name}] target={target}, n={n}, pos={pos}, rate={rate:.6f}")


def train_binary_stage_model(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    target: str,
    feature_set: FeatureSet,
    condition_col: str | None = None,
    depth: int = 6,
    l2_leaf_reg: float = 5.0,
    learning_rate: float = 0.03,
    iterations: int = 1000,
    auto_class_weights: str | None = "Balanced",
    ctr_leaf_count_limit: int | None = None,
    calibrate_probability: bool = True,
):
    """Train a binary classifier with optional isotonic probability calibration.

    Probability calibration is important because tree models under Logloss tend
    to produce probabilities that deviate from true distributions. When
    chaining multiple models (e.g., P(bt_place) = P(win) * P(bt_place|win)),
    uncalibrated probabilities can compound errors exponentially.
    """
    train_sub = subset_by_condition(train_df, condition_col)
    valid_sub = subset_by_condition(valid_df, condition_col)

    describe_stage_split("train_stage", train_sub, target)
    describe_stage_split("valid_stage", valid_sub, target)

    if len(train_sub) == 0:
        raise ValueError(f"{target}: train_sub 为空")
    if len(valid_sub) == 0:
        raise ValueError(f"{target}: valid_sub 为空")
    if train_sub[target].nunique() < 2:
        raise ValueError(f"{target}: train_sub 只有单一类别")
    if valid_sub[target].nunique() < 2:
        raise ValueError(f"{target}: valid_sub 只有单一类别")

    X_train, y_train, _, cat_cols = prepare_matrix(train_sub, feature_set=feature_set)
    X_valid, y_valid, _, _ = prepare_matrix(valid_sub, feature_set=feature_set)

    cb_kwargs: dict = dict(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        l2_leaf_reg=l2_leaf_reg,
        auto_class_weights=auto_class_weights,
        random_seed=42,
        verbose=100,
        od_type="Iter",
        od_wait=100,
    )
    if ctr_leaf_count_limit is not None:
        cb_kwargs["ctr_leaf_count_limit"] = ctr_leaf_count_limit

    model = CatBoostClassifier(**cb_kwargs)

    model.fit(
        build_pool(X_train, y_train[target], cat_cols),
        eval_set=build_pool(X_valid, y_valid[target], cat_cols),
        use_best_model=True,
    )

    # --- Isotonic probability calibration ---
    # For sklearn >= 1.6, CalibratedClassifierCV no longer supports cv='prefit'.
    # We wrap the CatBoost model in a thin proxy that applies isotonic regression
    # on top of its predicted probabilities.
    if calibrate_probability and len(valid_sub) >= 20:
        print(f"[{target}] Applying isotonic probability calibration on valid set...")

        # Get raw probabilities from the trained CatBoost model on validation set
        raw_probs = model.predict_proba(X_valid)[:, 1]

        # Fit isotonic regressor to map raw probabilities -> true labels
        iso_reg = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso_reg.fit(raw_probs, y_valid[target])

        # Store calibration state on the model so predict_binary can use it
        model._calibration_iso_reg = iso_reg
        print(f"[{target}] Probability calibration complete.")

    return model


def train_positive_regressor(
    train_df, valid_df, feature_set: FeatureSet,
    depth=4, l2_leaf_reg=20.0, learning_rate=0.02, iterations=2000,
    ctr_leaf_count_limit: int | None = None,
):
    train_pos = train_df[train_df["pog_total_prize"] > 0].copy()
    valid_pos = valid_df[valid_df["pog_total_prize"] > 0].copy()

    if len(train_pos) == 0:
        raise ValueError("train_pos 为空，无法训练正奖金回归模型。")
    if len(valid_pos) == 0:
        raise ValueError("valid_pos 为空，无法做正奖金回归验证。")

    train_pos = add_log_target(train_pos)
    valid_pos = add_log_target(valid_pos)

    X_train, _, _, cat_cols = prepare_matrix(train_pos, feature_set=feature_set)
    X_valid, _, _, _ = prepare_matrix(valid_pos, feature_set=feature_set)

    cb_kwargs: dict = dict(
        loss_function="MAE",
        eval_metric="MAE",
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        l2_leaf_reg=l2_leaf_reg,
        random_seed=42,
        verbose=100,
        od_type="Iter",
        od_wait=100,
    )
    if ctr_leaf_count_limit is not None:
        cb_kwargs["ctr_leaf_count_limit"] = ctr_leaf_count_limit

    model = CatBoostRegressor(**cb_kwargs)

    model.fit(
        build_pool(X_train, train_pos["log_pog_total_prize"], cat_cols),
        eval_set=build_pool(X_valid, valid_pos["log_pog_total_prize"], cat_cols),
        use_best_model=True,
    )
    return model


def train_quantile_regressor(
    train_df, valid_df, feature_set: FeatureSet,
    alpha=0.9,
    depth=4, l2_leaf_reg=15.0, learning_rate=0.02, iterations=2000,
    ctr_leaf_count_limit: int | None = None,
):
    """Train a quantile regression on log1p(prize) for positive-prize samples."""
    train_pos = train_df[train_df["pog_total_prize"] > 0].copy()
    valid_pos = valid_df[valid_df["pog_total_prize"] > 0].copy()

    if len(train_pos) == 0:
        raise ValueError(f"Quantile({alpha}): train_pos 为空")
    if len(valid_pos) == 0:
        raise ValueError(f"Quantile({alpha}): valid_pos 为空")

    train_pos = add_log_target(train_pos)
    valid_pos = add_log_target(valid_pos)

    X_train, _, _, cat_cols = prepare_matrix(train_pos, feature_set=feature_set)
    X_valid, _, _, _ = prepare_matrix(valid_pos, feature_set=feature_set)

    cb_kwargs: dict = dict(
        loss_function=f"Quantile:alpha={alpha}",
        eval_metric=f"Quantile:alpha={alpha}",
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        l2_leaf_reg=l2_leaf_reg,
        random_seed=42,
        verbose=100,
        od_type="Iter",
        od_wait=100,
    )
    if ctr_leaf_count_limit is not None:
        cb_kwargs["ctr_leaf_count_limit"] = ctr_leaf_count_limit

    model = CatBoostRegressor(**cb_kwargs)

    model.fit(
        build_pool(X_train, train_pos["log_pog_total_prize"], cat_cols),
        eval_set=build_pool(X_valid, valid_pos["log_pog_total_prize"], cat_cols),
        use_best_model=True,
    )
    return model


def predict_binary(model, df, target, feature_set: FeatureSet):
    X, _, _, _ = prepare_matrix(df, feature_set=feature_set)
    prob = model.predict_proba(X)[:, 1]
    # Apply isotonic calibration if it was fitted during training
    cal = getattr(model, "_calibration_iso_reg", None)
    if cal is not None:
        prob = cal.predict(prob)
    return prob


def predict_regressor(model, df, feature_set: FeatureSet):
    X, _, _, _ = prepare_matrix(df, feature_set=feature_set)
    pred_log = model.predict(X)
    return pred_log


def train_ranking_stacking_model(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_set: FeatureSet,
    bundle: "ModelBundle | None" = None,
    depth: int = 4,
    l2_leaf_reg: float = 10.0,
    learning_rate: float = 0.02,
    iterations: int = 1000,
    ranking_mode: str = "YetiRank",
) -> CatBoostRanker | None:
    """Train a CatBoostRanker (YetiRank/PairLogit) as a stacking model.

    Uses the first-stage model outputs (p_win, p_bt_place, etc.) as meta-features.
    The target is the rank of pog_total_prize among all horses.

    Returns None if there is not enough data to train.
    """
    # Need first-stage models to produce meta-features
    if bundle is None:
        print("[WARN] No bundle provided for ranking stacking model. Skipping.")
        return None

    # Filter to positive-prize horses for ranking
    train_pos = train_df[train_df["pog_total_prize"] > 0].copy()
    valid_pos = valid_df[valid_df["pog_total_prize"] > 0].copy()

    if len(train_pos) < 50 or len(valid_pos) < 10:
        print(f"[WARN] Not enough positive-prize data for ranking: "
              f"train={len(train_pos)}, valid={len(valid_pos)}. Skipping.")
        return None

    # Produce first-stage predictions as meta-features
    train_meta = predict_all(bundle, train_pos, train_df=train_df)
    valid_meta = predict_all(bundle, valid_pos, train_df=train_df)

    ranking_features = [
        "p_win", "p_bt_place", "p_bt_win", "p_graded_win",
        "expected_pog_prize", "p_prize_ge_10m", "p_prize_ge_30m",
        "q80_prize", "q90_prize",
    ]
    available_features = [f for f in ranking_features if f in train_meta.columns]

    X_train = train_meta[available_features]
    y_train_rank = train_pos["pog_total_prize"].rank(method="average", ascending=False).values.astype(int)

    X_valid = valid_meta[available_features]
    y_valid_rank = valid_pos["pog_total_prize"].rank(method="average", ascending=False).values.astype(int)

    # Create group IDs (all 1 group per dataset for global ranking)
    train_group_id = np.zeros(len(X_train), dtype=int)
    valid_group_id = np.zeros(len(X_valid), dtype=int)

    train_pool = Pool(
        X_train, label=y_train_rank, group_id=train_group_id
    )
    valid_pool = Pool(
        X_valid, label=y_valid_rank, group_id=valid_group_id
    )

    print(f"Training ranking stacking model ({ranking_mode}) with features: {available_features}")

    ranking_model = CatBoostRanker(
        loss_function=ranking_mode,
        eval_metric="NDCG",
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        l2_leaf_reg=l2_leaf_reg,
        random_seed=42,
        verbose=100,
        od_type="Iter",
        od_wait=100,
    )

    ranking_model.fit(
        train_pool,
        eval_set=valid_pool,
        use_best_model=True,
    )

    return ranking_model


def predict_ranking(bundle: "ModelBundle", df: pd.DataFrame,
                     train_df: pd.DataFrame | None = None) -> np.ndarray:
    """Produce ranking scores from the stacking ranking model.

    Returns ranking scores (higher = better rank) or None if model is unavailable.
    """
    if bundle.ranking_model is None:
        return np.zeros(len(df))

    # Produce first-stage predictions as meta-features
    meta = predict_all(bundle, df, train_df=train_df)

    ranking_features = [
        "p_win", "p_bt_place", "p_bt_win", "p_graded_win",
        "expected_pog_prize", "p_prize_ge_10m", "p_prize_ge_30m",
        "q80_prize", "q90_prize",
    ]
    available_features = [f for f in ranking_features if f in meta.columns]
    X = meta[available_features]

    # CatBoostRanker.predict returns scores
    scores = bundle.ranking_model.predict(X)
    return scores


# =========================
# Nested milestone logic
# =========================

def predict_nested_milestones(
    df: pd.DataFrame,
    win_model,
    bt_place_given_win_model,
    bt_win_given_bt_place_model,
    graded_given_bt_win_model,
    feature_set: FeatureSet,
) -> pd.DataFrame:
    p_win = predict_binary(win_model, df, "win_flag", feature_set)

    p_bt_place_given_win = predict_binary(
        bt_place_given_win_model, df, "bt_place_flag", feature_set
    )
    p_bt_win_given_bt_place = predict_binary(
        bt_win_given_bt_place_model, df, "bt_win_flag", feature_set
    )
    p_graded_given_bt_win = predict_binary(
        graded_given_bt_win_model, df, "graded_win_flag", feature_set
    )

    p_bt_place = p_win * p_bt_place_given_win
    p_bt_win = p_bt_place * p_bt_win_given_bt_place
    p_graded_win = p_bt_win * p_graded_given_bt_win

    return pd.DataFrame({
        "p_win": p_win,
        "p_bt_place_given_win": p_bt_place_given_win,
        "p_bt_place": p_bt_place,
        "p_bt_win_given_bt_place": p_bt_win_given_bt_place,
        "p_bt_win": p_bt_win,
        "p_graded_given_bt_win": p_graded_given_bt_win,
        "p_graded_win": p_graded_win,
    })


# =========================
# Ranking helpers
# =========================

def rank_to_unit_interval(s: pd.Series) -> pd.Series:
    return s.rank(method="average", pct=True)


def fit_ceiling_weights(df: pd.DataFrame) -> dict:
    """
    Fits a linear combination of rank features to target rank of pog_total_prize.
    Returns the normalized learned weights for score_ceiling.
    """
    if "pog_total_prize" not in df.columns or len(df) == 0:
        return {}

    required_cols = ["expected_pog_prize", "p_graded_win", "p_prize_ge_10m", "p_prize_ge_30m", "q90_prize"]
    if not all(c in df.columns for c in required_cols):
        return {}

    temp = df.copy()
    temp["r_expected_pog_prize"] = rank_to_unit_interval(temp["expected_pog_prize"])
    temp["r_p_graded_win"] = rank_to_unit_interval(temp["p_graded_win"])
    temp["r_p_prize_ge_10m"] = rank_to_unit_interval(temp["p_prize_ge_10m"])
    temp["r_p_prize_ge_30m"] = rank_to_unit_interval(temp["p_prize_ge_30m"])
    temp["r_q90_prize"] = rank_to_unit_interval(temp["q90_prize"])

    features = [
        "r_p_prize_ge_10m",
        "r_p_prize_ge_30m",
        "r_p_graded_win",
        "r_q90_prize",
        "r_expected_pog_prize",
    ]

    y = rank_to_unit_interval(temp["pog_total_prize"])
    X = temp[features]

    reg = LinearRegression(positive=True, fit_intercept=False)
    reg.fit(X, y)

    w = reg.coef_
    if w.sum() > 0:
        w = w / w.sum()
    else:
        w = np.array([0.30, 0.25, 0.20, 0.15, 0.10])

    return dict(zip(features, w))


def build_blended_scores(df: pd.DataFrame, ceiling_weights: dict | None = None) -> pd.DataFrame:
    out = df.copy()

    out["r_expected_pog_prize"] = rank_to_unit_interval(out["expected_pog_prize"])
    out["r_p_win"] = rank_to_unit_interval(out["p_win"])
    out["r_p_bt_place"] = rank_to_unit_interval(out["p_bt_place"])
    out["r_p_bt_win"] = rank_to_unit_interval(out["p_bt_win"])
    out["r_p_graded_win"] = rank_to_unit_interval(out["p_graded_win"])

    # 偏稳健：兼顾均值与层级里程碑
    out["score_balanced"] = (
        0.45 * out["r_expected_pog_prize"]
        + 0.20 * out["r_p_bt_place"]
        + 0.20 * out["r_p_bt_win"]
        + 0.15 * out["r_p_graded_win"]
    )

    # 旧版 ceiling（保留供对比）
    out["score_ceiling_old"] = (
        0.25 * out["r_expected_pog_prize"]
        + 0.15 * out["r_p_win"]
        + 0.20 * out["r_p_bt_place"]
        + 0.20 * out["r_p_bt_win"]
        + 0.20 * out["r_p_graded_win"]
    )

    # 新版 ceiling：以 ceiling 专用信号为核心
    has_ceiling = all(
        c in out.columns
        for c in ["p_prize_ge_10m", "p_prize_ge_30m", "q90_prize"]
    )
    if has_ceiling:
        out["r_p_prize_ge_10m"] = rank_to_unit_interval(out["p_prize_ge_10m"])
        out["r_p_prize_ge_30m"] = rank_to_unit_interval(out["p_prize_ge_30m"])
        out["r_q90_prize"] = rank_to_unit_interval(out["q90_prize"])

        if ceiling_weights:
            out["score_ceiling"] = 0.0
            for feat, w in ceiling_weights.items():
                out["score_ceiling"] += w * out[feat]
        else:
            out["score_ceiling"] = (
                0.30 * out["r_p_prize_ge_10m"]
                + 0.25 * out["r_p_prize_ge_30m"]
                + 0.20 * out["r_p_graded_win"]
                + 0.15 * out["r_q90_prize"]
                + 0.10 * out["r_expected_pog_prize"]
            )
    else:
        out["score_ceiling"] = out["score_ceiling_old"]

    return out


# =========================
# Top-k report
# =========================

def _safe_div(a, b):
    if b == 0:
        return np.nan
    return a / b


def build_topk_report(
    df: pd.DataFrame,
    score_cols: list[str],
    ks: list[int],
) -> pd.DataFrame:
    rows = []

    base_win_rate = df["win_flag"].mean() if "win_flag" in df.columns else np.nan
    base_bt_place_rate = df["bt_place_flag"].mean() if "bt_place_flag" in df.columns else np.nan
    base_bt_win_rate = df["bt_win_flag"].mean() if "bt_win_flag" in df.columns else np.nan
    base_graded_rate = df["graded_win_flag"].mean() if "graded_win_flag" in df.columns else np.nan

    for score_col in score_cols:
        if score_col not in df.columns:
            continue

        ranked = df.sort_values(score_col, ascending=False).reset_index(drop=True)

        for k in ks:
            top = ranked.head(k).copy()
            if len(top) == 0:
                continue

            n = len(top)
            win_n = int(top["win_flag"].sum()) if "win_flag" in top.columns else 0
            bt_place_n = int(top["bt_place_flag"].sum()) if "bt_place_flag" in top.columns else 0
            bt_win_n = int(top["bt_win_flag"].sum()) if "bt_win_flag" in top.columns else 0
            graded_n = int(top["graded_win_flag"].sum()) if "graded_win_flag" in top.columns else 0

            win_prec = _safe_div(win_n, n)
            bt_place_prec = _safe_div(bt_place_n, n)
            bt_win_prec = _safe_div(bt_win_n, n)
            graded_prec = _safe_div(graded_n, n)

            rows.append({
                "score_col": score_col,
                "k": k,
                "n_selected": n,
                "expected_prize_sum": float(top["expected_pog_prize"].sum()) if "expected_pog_prize" in top.columns else np.nan,
                "expected_prize_mean": float(top["expected_pog_prize"].mean()) if "expected_pog_prize" in top.columns else np.nan,
                "actual_prize_sum": float(top["pog_total_prize"].sum()) if "pog_total_prize" in top.columns else np.nan,
                "actual_prize_mean": float(top["pog_total_prize"].mean()) if "pog_total_prize" in top.columns else np.nan,
                "win_n": win_n,
                "bt_place_n": bt_place_n,
                "bt_win_n": bt_win_n,
                "graded_win_n": graded_n,
                "win_precision": win_prec,
                "bt_place_precision": bt_place_prec,
                "bt_win_precision": bt_win_prec,
                "graded_win_precision": graded_prec,
                "win_lift": _safe_div(win_prec, base_win_rate),
                "bt_place_lift": _safe_div(bt_place_prec, base_bt_place_rate),
                "bt_win_lift": _safe_div(bt_win_prec, base_bt_win_rate),
                "graded_win_lift": _safe_div(graded_prec, base_graded_rate),
            })

    return pd.DataFrame(rows)


def print_metrics(metrics: dict):
    print("=== TEST METRICS ===")
    for k, v in metrics.items():
        print(k, v)


# =========================
# ModelBundle
# =========================

@dataclass
class ModelBundle:
    """Holds all trained models as a single unit."""
    win_model: CatBoostClassifier
    bt_place_given_win_model: CatBoostClassifier
    bt_win_given_bt_place_model: CatBoostClassifier | None
    graded_given_bt_win_model: CatBoostClassifier | None
    positive_prize_model: CatBoostClassifier
    prize_model: CatBoostRegressor
    prize_ge_10m_model: CatBoostClassifier
    prize_ge_30m_model: CatBoostClassifier
    q80_model: CatBoostRegressor
    q90_model: CatBoostRegressor
    ranking_model: CatBoostRanker | None  # Optional: Learning to Rank stacking model
    ceiling_weights: dict
    feature_set: FeatureSet


# =========================
# Pipeline: train all models
# =========================

def train_all_models(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_set: FeatureSet,
    graceful_conditional: bool = False,
    config: Config | None = None,
) -> ModelBundle:
    """Train all models and learn ceiling weights.

    Args:
        train_df: Training data.
        valid_df: Validation data.
        feature_set: Which features to use.
        graceful_conditional: If True, allow conditional models (bt_win|bt_place,
            graded|bt_win) to fail gracefully (set to None). Used in rolling
            backtest where subsets can be small.
        config: Configuration containing model hyperparameters.
    """
    if config is None:
        config = Config()

    mc = config.model_configs
    models = {}

    # Category 1: Standard Binary Stage Models
    binary_model_keys = [
        "win", 
        "bt_place_given_win", 
        "positive_prize", 
        "prize_ge_10m", 
        "prize_ge_30m"
    ]
    
    print("\n=== TRAIN STANDARD BINARY MODELS ===")
    for key in binary_model_keys:
        print(f"Training {key} model...")
        models[key] = train_binary_stage_model(
            train_df, valid_df, feature_set=feature_set, **mc[key]
        )

    # Category 2: Conditional Binary Stage Models
    conditional_keys = [
        "bt_win_given_bt_place", 
        "graded_given_bt_win"
    ]
    
    print("\n=== TRAIN CONDITIONAL BINARY MODELS ===")
    for key in conditional_keys:
        print(f"Training {key} model...")
        if graceful_conditional:
            try:
                models[key] = train_binary_stage_model(
                    train_df, valid_df, feature_set=feature_set, **mc[key]
                )
            except ValueError as e:
                print(f"[WARN] {key} failed: {e}. Using base-rate fallback.")
                models[key] = None
        else:
            models[key] = train_binary_stage_model(
                train_df, valid_df, feature_set=feature_set, **mc[key]
            )

    # Category 3: Regressor Models
    print("\n=== TRAIN REGRESSOR MODELS ===")
    print("Training prize_model...")
    prize_model = train_positive_regressor(
        train_df, valid_df, feature_set=feature_set, **mc["prize_regressor"]
    )
    
    print("Training q80_model...")
    q80_model = train_quantile_regressor(
        train_df, valid_df, feature_set=feature_set, **mc["q80_regressor"]
    )
    
    print("Training q90_model...")
    q90_model = train_quantile_regressor(
        train_df, valid_df, feature_set=feature_set, **mc["q90_regressor"]
    )

    # Category 4: Ranking Stacking Model (optional, graceful failure)
    print("\n=== TRAIN RANKING STACKING MODEL ===")
    ranking_model = None
    if "ranking_stacking" in mc:
        try:
            ranking_cfg = mc["ranking_stacking"]
            ranking_model = train_ranking_stacking_model(
                train_df, valid_df, feature_set=feature_set,
                depth=ranking_cfg.get("depth", 4),
                l2_leaf_reg=ranking_cfg.get("l2_leaf_reg", 10.0),
                learning_rate=ranking_cfg.get("learning_rate", 0.02),
                iterations=ranking_cfg.get("iterations", 1000),
                ranking_mode=ranking_cfg.get("ranking_mode", "YetiRank"),
            )
            if ranking_model is not None:
                print("Ranking stacking model trained successfully.")
            else:
                print("Ranking stacking model skipped (insufficient data).")
        except Exception as e:
            print(f"[WARN] Ranking stacking model failed: {e}. Skipping.")
            ranking_model = None

    # Learn ceiling weights on validation set
    print("\n=== LEARN CEILING WEIGHTS ===")

    # Build a partial bundle for ranking-aware ceiling weight learning
    partial_bundle = ModelBundle(
        win_model=models["win"],
        bt_place_given_win_model=models["bt_place_given_win"],
        bt_win_given_bt_place_model=models["bt_win_given_bt_place"],
        graded_given_bt_win_model=models["graded_given_bt_win"],
        positive_prize_model=models["positive_prize"],
        prize_model=prize_model,
        prize_ge_10m_model=models["prize_ge_10m"],
        prize_ge_30m_model=models["prize_ge_30m"],
        q80_model=q80_model,
        q90_model=q90_model,
        ranking_model=ranking_model,
        ceiling_weights={},
        feature_set=feature_set,
    )

    ceiling_weights = _learn_ceiling_weights(
        valid_df, 
        models["win"], 
        models["bt_place_given_win"],
        models["bt_win_given_bt_place"], 
        models["graded_given_bt_win"],
        models["positive_prize"], 
        prize_model,
        models["prize_ge_10m"], 
        models["prize_ge_30m"], 
        q90_model,
        feature_set, train_df,
        ranking_model=ranking_model,
    )
    print("Learned Ceiling Weights:", ceiling_weights)

    return partial_bundle


def _learn_ceiling_weights(
    valid_df, win_model, bt_place_given_win_model,
    bt_win_given_bt_place_model, graded_given_bt_win_model,
    positive_prize_model, prize_model,
    prize_ge_10m_model, prize_ge_30m_model, q90_model,
    feature_set: FeatureSet, train_df: pd.DataFrame,
    ranking_model: CatBoostRanker | None = None,
) -> dict:
    """Build predictions on valid_df and fit ceiling weights."""
    valid_pred = valid_df.copy()

    # Milestone predictions (with None-model fallback)
    p_win = predict_binary(win_model, valid_df, "win_flag", feature_set)
    p_bt_place_given_win = predict_binary(bt_place_given_win_model, valid_df, "bt_place_flag", feature_set)

    if bt_win_given_bt_place_model is not None:
        p_bt_win_given_bt_place = predict_binary(bt_win_given_bt_place_model, valid_df, "bt_win_flag", feature_set)
    else:
        base_rate = float(train_df.loc[train_df["bt_place_flag"] == 1, "bt_win_flag"].mean())
        p_bt_win_given_bt_place = np.full(len(valid_df), base_rate)

    if graded_given_bt_win_model is not None:
        p_graded_given_bt_win = predict_binary(graded_given_bt_win_model, valid_df, "graded_win_flag", feature_set)
    else:
        base_rate = float(train_df.loc[train_df["bt_win_flag"] == 1, "graded_win_flag"].mean())
        p_graded_given_bt_win = np.full(len(valid_df), base_rate)

    valid_pred["p_win"] = p_win
    valid_pred["p_bt_place"] = p_win * p_bt_place_given_win
    valid_pred["p_bt_win"] = valid_pred["p_bt_place"] * p_bt_win_given_bt_place
    valid_pred["p_graded_win"] = valid_pred["p_bt_win"] * p_graded_given_bt_win

    valid_pred["p_positive_prize"] = predict_binary(positive_prize_model, valid_df, "positive_prize_flag", feature_set)
    valid_pred["pred_log_prize_pos"] = predict_regressor(prize_model, valid_df, feature_set)
    valid_pred["pred_positive_prize_amount"] = np.clip(np.expm1(valid_pred["pred_log_prize_pos"]), 0, None)
    valid_pred["expected_pog_prize"] = valid_pred["p_positive_prize"] * valid_pred["pred_positive_prize_amount"]

    valid_pred["p_prize_ge_10m"] = predict_binary(prize_ge_10m_model, valid_df, "pog_total_prize_ge_10m_flag", feature_set)
    valid_pred["p_prize_ge_30m"] = predict_binary(prize_ge_30m_model, valid_df, "pog_total_prize_ge_30m_flag", feature_set)
    valid_pred["q90_log_prize"] = predict_regressor(q90_model, valid_df, feature_set)
    valid_pred["q90_prize"] = np.clip(np.expm1(valid_pred["q90_log_prize"]), 0, None)

    return fit_ceiling_weights(valid_pred)


# =========================
# Pipeline: predict all
# =========================

def predict_all(
    bundle: ModelBundle,
    df: pd.DataFrame,
    train_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Assemble complete predictions from a ModelBundle.

    Args:
        bundle: Trained model bundle.
        df: Data to predict on.
        train_df: Training data, needed for base-rate fallback when
                  conditional models are None. If None and conditional
                  models are None, will use 0.5 as fallback rate.
    """
    fs = bundle.feature_set
    pred = pd.DataFrame(index=df.index)

    # Milestone predictions (with None-model fallback)
    p_win = predict_binary(bundle.win_model, df, "win_flag", fs)
    p_bt_place_given_win = predict_binary(bundle.bt_place_given_win_model, df, "bt_place_flag", fs)

    if bundle.bt_win_given_bt_place_model is not None:
        p_bt_win_given_bt_place = predict_binary(bundle.bt_win_given_bt_place_model, df, "bt_win_flag", fs)
    else:
        if train_df is not None:
            base_rate = train_df.loc[train_df["bt_place_flag"] == 1, "bt_win_flag"].mean()
        else:
            base_rate = 0.5
        p_bt_win_given_bt_place = np.full(len(df), base_rate)
        print(f"  [fallback] p_bt_win_given_bt_place = {base_rate:.4f}")

    if bundle.graded_given_bt_win_model is not None:
        p_graded_given_bt_win = predict_binary(bundle.graded_given_bt_win_model, df, "graded_win_flag", fs)
    else:
        if train_df is not None:
            base_rate = train_df.loc[train_df["bt_win_flag"] == 1, "graded_win_flag"].mean()
        else:
            base_rate = 0.5
        p_graded_given_bt_win = np.full(len(df), base_rate)
        print(f"  [fallback] p_graded_given_bt_win = {base_rate:.4f}")

    p_bt_place = p_win * p_bt_place_given_win
    p_bt_win = p_bt_place * p_bt_win_given_bt_place
    p_graded_win = p_bt_win * p_graded_given_bt_win

    pred["p_win"] = p_win
    pred["p_bt_place_given_win"] = p_bt_place_given_win
    pred["p_bt_place"] = p_bt_place
    pred["p_bt_win_given_bt_place"] = p_bt_win_given_bt_place
    pred["p_bt_win"] = p_bt_win
    pred["p_graded_given_bt_win"] = p_graded_given_bt_win
    pred["p_graded_win"] = p_graded_win

    # Prize predictions
    pred["p_positive_prize"] = predict_binary(bundle.positive_prize_model, df, "positive_prize_flag", fs)
    pred["pred_log_prize_pos"] = predict_regressor(bundle.prize_model, df, fs)
    pred["pred_positive_prize_amount"] = np.clip(np.expm1(pred["pred_log_prize_pos"]), 0, None)
    pred["expected_pog_prize"] = pred["p_positive_prize"] * pred["pred_positive_prize_amount"]

    # Ceiling predictions
    pred["p_prize_ge_10m"] = predict_binary(bundle.prize_ge_10m_model, df, "pog_total_prize_ge_10m_flag", fs)
    pred["p_prize_ge_30m"] = predict_binary(bundle.prize_ge_30m_model, df, "pog_total_prize_ge_30m_flag", fs)

    # Quantile predictions
    pred["q80_log_prize"] = predict_regressor(bundle.q80_model, df, fs)
    pred["q90_log_prize"] = predict_regressor(bundle.q90_model, df, fs)
    pred["q80_prize"] = np.clip(np.expm1(pred["q80_log_prize"]), 0, None)
    pred["q90_prize"] = np.clip(np.expm1(pred["q90_log_prize"]), 0, None)

    return pred


# =========================
# Pipeline: train_and_evaluate
# =========================

def train_and_evaluate(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_set: FeatureSet,
    score_cols: list[str],
    ks: list[int],
    graceful_conditional: bool = False,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Full pipeline: train → predict → evaluate.

    Returns:
        (topk_report_df, metrics_dict, test_pred_df)
    """
    from eval import evaluate_binary

    bundle = train_all_models(train_df, valid_df, feature_set, graceful_conditional=graceful_conditional)

    # Build test predictions
    keep_cols = [
        c for c in [
            "ketto_num", "birth_year",
            "pog_total_prize", "win_flag", "bt_place_flag",
            "bt_win_flag", "graded_win_flag", "positive_prize_flag",
            "pog_total_prize_ge_10m_flag", "pog_total_prize_ge_30m_flag",
        ]
        if c in test_df.columns
    ]
    test_pred = test_df[keep_cols].copy()

    pred_cols = predict_all(bundle, test_df, train_df=train_df)
    for col in pred_cols.columns:
        test_pred[col] = pred_cols[col].values

    test_pred = build_blended_scores(test_pred, ceiling_weights=bundle.ceiling_weights)

    # Top-k report
    topk = build_topk_report(test_pred, score_cols=score_cols, ks=ks)

    # Metrics
    metrics = {}
    metrics.update(evaluate_binary(test_pred["win_flag"], test_pred["p_win"], "win"))
    metrics.update(evaluate_binary(test_pred["bt_place_flag"], test_pred["p_bt_place"], "bt_place"))
    metrics.update(evaluate_binary(test_pred["graded_win_flag"], test_pred["p_graded_win"], "graded_win"))
    if "pog_total_prize_ge_10m_flag" in test_pred.columns:
        metrics.update(evaluate_binary(test_pred["pog_total_prize_ge_10m_flag"], test_pred["p_prize_ge_10m"], "prize_ge_10m"))
    if "pog_total_prize_ge_30m_flag" in test_pred.columns:
        metrics.update(evaluate_binary(test_pred["pog_total_prize_ge_30m_flag"], test_pred["p_prize_ge_30m"], "prize_ge_30m"))

    return topk, metrics, test_pred


# =========================
# Serialization
# =========================

def save_bundle(bundle: ModelBundle, path: str, meta_extra: dict | None = None):
    """Save all models and metadata to a directory."""
    os.makedirs(path, exist_ok=True)

    joblib.dump(bundle.win_model, os.path.join(path, "win_model.joblib"))
    joblib.dump(bundle.bt_place_given_win_model, os.path.join(path, "bt_place_given_win_model.joblib"))
    if bundle.bt_win_given_bt_place_model is not None:
        joblib.dump(bundle.bt_win_given_bt_place_model, os.path.join(path, "bt_win_given_bt_place_model.joblib"))
    if bundle.graded_given_bt_win_model is not None:
        joblib.dump(bundle.graded_given_bt_win_model, os.path.join(path, "graded_given_bt_win_model.joblib"))
    joblib.dump(bundle.positive_prize_model, os.path.join(path, "positive_prize_model.joblib"))
    joblib.dump(bundle.prize_model, os.path.join(path, "prize_model.joblib"))
    joblib.dump(bundle.prize_ge_10m_model, os.path.join(path, "prize_ge_10m_model.joblib"))
    joblib.dump(bundle.prize_ge_30m_model, os.path.join(path, "prize_ge_30m_model.joblib"))
    joblib.dump(bundle.q80_model, os.path.join(path, "q80_model.joblib"))
    joblib.dump(bundle.q90_model, os.path.join(path, "q90_model.joblib"))
    if bundle.ranking_model is not None:
        joblib.dump(bundle.ranking_model, os.path.join(path, "ranking_model.joblib"))

    meta = {
        "ceiling_weights": bundle.ceiling_weights,
        "feature_set": asdict(bundle.feature_set),
        "has_ranking_model": bundle.ranking_model is not None,
    }
    if meta_extra:
        meta.update(meta_extra)

    with open(os.path.join(path, "model_bundle_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

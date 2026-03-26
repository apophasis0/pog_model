import os
import json
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from dotenv import load_dotenv

from config import Config
from data import (
    load_training_frame,
    load_scoring_frame,
    save_predictions,
    load_completed_birth_years,
)
from features import prepare_matrix, add_log_target
from eval import evaluate_binary, evaluate_regression


# =========================
# Split helpers
# =========================

def auto_configure_splits(cfg: Config) -> Config:
    completed_years = load_completed_birth_years(cfg)

    if len(completed_years) < 5:
        raise ValueError(
            f"已完成标签的 birth_year 太少：{completed_years}。至少需要 5 个完整 cohort。"
        )

    # 用最后 1 年做 test，倒数第 2 年做 valid，其余做 train
    cfg.test_birth_year_start = completed_years[-1]
    cfg.test_birth_year_end = completed_years[-1]

    cfg.valid_birth_year_start = completed_years[-2]
    cfg.valid_birth_year_end = completed_years[-2]

    cfg.train_birth_year_start = completed_years[0]
    cfg.train_birth_year_end = completed_years[-3]

    return cfg


def split_by_birth_year(df: pd.DataFrame, cfg: Config):
    train_df = df[
        (df["birth_year"] >= cfg.train_birth_year_start) &
        (df["birth_year"] <= cfg.train_birth_year_end)
    ].copy()

    valid_df = df[
        (df["birth_year"] >= cfg.valid_birth_year_start) &
        (df["birth_year"] <= cfg.valid_birth_year_end)
    ].copy()

    test_df = df[
        (df["birth_year"] >= cfg.test_birth_year_start) &
        (df["birth_year"] <= cfg.test_birth_year_end)
    ].copy()

    return train_df, valid_df, test_df


def describe_split(name: str, df: pd.DataFrame):
    n = len(df)
    pos_prize = int((df["pog_total_prize"] > 0).sum()) if "pog_total_prize" in df.columns else 0
    win_n = int(df["win_flag"].sum()) if "win_flag" in df.columns else 0
    bt_place_n = int(df["bt_place_flag"].sum()) if "bt_place_flag" in df.columns else 0
    bt_win_n = int(df["bt_win_flag"].sum()) if "bt_win_flag" in df.columns else 0
    graded_n = int(df["graded_win_flag"].sum()) if "graded_win_flag" in df.columns else 0

    print(
        f"[{name}] n={n}, positive_prize={pos_prize}, "
        f"win={win_n}, bt_place={bt_place_n}, bt_win={bt_win_n}, graded_win={graded_n}"
    )

    if n == 0:
        raise ValueError(f"{name} split 为空，请检查年份切分。")


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
    condition_col: str | None = None,
    use_dynamic: bool = False,
    depth: int = 6,
    l2_leaf_reg: float = 5.0,
    learning_rate: float = 0.03,
    iterations: int = 1000,
    auto_class_weights: str | None = "Balanced",
):
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

    X_train, y_train, _, cat_cols = prepare_matrix(train_sub, use_dynamic=use_dynamic)
    X_valid, y_valid, _, _ = prepare_matrix(valid_sub, use_dynamic=use_dynamic)

    model = CatBoostClassifier(
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

    model.fit(
        build_pool(X_train, y_train[target], cat_cols),
        eval_set=build_pool(X_valid, y_valid[target], cat_cols),
        use_best_model=True,
    )
    return model


def train_positive_regressor(train_df, valid_df, use_dynamic=False):
    train_pos = train_df[train_df["pog_total_prize"] > 0].copy()
    valid_pos = valid_df[valid_df["pog_total_prize"] > 0].copy()

    if len(train_pos) == 0:
        raise ValueError("train_pos 为空，无法训练正奖金回归模型。")
    if len(valid_pos) == 0:
        raise ValueError("valid_pos 为空，无法做正奖金回归验证。")

    train_pos = add_log_target(train_pos)
    valid_pos = add_log_target(valid_pos)

    X_train, _, _, cat_cols = prepare_matrix(train_pos, use_dynamic=use_dynamic)
    X_valid, _, _, _ = prepare_matrix(valid_pos, use_dynamic=use_dynamic)

    model = CatBoostRegressor(
        loss_function="MAE",
        eval_metric="MAE",
        iterations=2000,
        learning_rate=0.02,
        depth=4,
        l2_leaf_reg=20.0,
        random_seed=42,
        verbose=100,
        od_type="Iter",
        od_wait=100
    )

    model.fit(
        build_pool(X_train, train_pos["log_pog_total_prize"], cat_cols),
        eval_set=build_pool(X_valid, valid_pos["log_pog_total_prize"], cat_cols),
        use_best_model=True
    )
    return model


def train_quantile_regressor(
    train_df, valid_df, alpha=0.9, use_dynamic=False,
    depth=4, l2_leaf_reg=15.0, learning_rate=0.02, iterations=2000,
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

    X_train, _, _, cat_cols = prepare_matrix(train_pos, use_dynamic=use_dynamic)
    X_valid, _, _, _ = prepare_matrix(valid_pos, use_dynamic=use_dynamic)

    model = CatBoostRegressor(
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

    model.fit(
        build_pool(X_train, train_pos["log_pog_total_prize"], cat_cols),
        eval_set=build_pool(X_valid, valid_pos["log_pog_total_prize"], cat_cols),
        use_best_model=True,
    )
    return model


def predict_binary(model, df, target, use_dynamic=False):
    X, _, _, _ = prepare_matrix(df, use_dynamic=use_dynamic)
    prob = model.predict_proba(X)[:, 1]
    return prob


def predict_regressor(model, df, use_dynamic=False):
    X, _, _, _ = prepare_matrix(df, use_dynamic=use_dynamic)
    pred_log = model.predict(X)
    return pred_log


# =========================
# Nested milestone logic
# =========================

def predict_nested_milestones(
    df: pd.DataFrame,
    win_model,
    bt_place_given_win_model,
    bt_win_given_bt_place_model,
    graded_given_bt_win_model,
    use_dynamic: bool = False,
) -> pd.DataFrame:
    p_win = predict_binary(win_model, df, "win_flag", use_dynamic=use_dynamic)

    p_bt_place_given_win = predict_binary(
        bt_place_given_win_model, df, "bt_place_flag", use_dynamic=use_dynamic
    )
    p_bt_win_given_bt_place = predict_binary(
        bt_win_given_bt_place_model, df, "bt_win_flag", use_dynamic=use_dynamic
    )
    p_graded_given_bt_win = predict_binary(
        graded_given_bt_win_model, df, "graded_win_flag", use_dynamic=use_dynamic
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


def build_blended_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["r_expected_pog_prize"] = rank_to_unit_interval(out["expected_pog_prize"])
    out["r_p_win"] = rank_to_unit_interval(out["p_win"])
    out["r_p_bt_place"] = rank_to_unit_interval(out["p_bt_place"])
    out["r_p_bt_win"] = rank_to_unit_interval(out["p_bt_win"])
    out["r_p_graded_win"] = rank_to_unit_interval(out["p_graded_win"])

    # 偏稳健：兼顾均值与层级里程碑（保持不变）
    out["score_balanced"] = (
        0.45 * out["r_expected_pog_prize"] +
        0.20 * out["r_p_bt_place"] +
        0.20 * out["r_p_bt_win"] +
        0.15 * out["r_p_graded_win"]
    )

    # --- 旧版 ceiling（保留供对比） ---
    out["score_ceiling_old"] = (
        0.25 * out["r_expected_pog_prize"] +
        0.15 * out["r_p_win"] +
        0.20 * out["r_p_bt_place"] +
        0.20 * out["r_p_bt_win"] +
        0.20 * out["r_p_graded_win"]
    )

    # --- 新版 ceiling：以 ceiling 专用信号为核心 ---
    has_ceiling = all(
        c in out.columns
        for c in ["p_prize_ge_10m", "p_prize_ge_30m", "q90_prize"]
    )
    if has_ceiling:
        out["r_p_prize_ge_10m"] = rank_to_unit_interval(out["p_prize_ge_10m"])
        out["r_p_prize_ge_30m"] = rank_to_unit_interval(out["p_prize_ge_30m"])
        out["r_q90_prize"] = rank_to_unit_interval(out["q90_prize"])

        out["score_ceiling"] = (
            0.30 * out["r_p_prize_ge_10m"] +
            0.25 * out["r_p_prize_ge_30m"] +
            0.20 * out["r_p_graded_win"] +
            0.15 * out["r_q90_prize"] +
            0.10 * out["r_expected_pog_prize"]
        )
    else:
        # fallback：如果 ceiling 列不存在就使用旧版
        out["score_ceiling"] = out["score_ceiling_old"]

    return out


# =========================
# Backtest / reporting
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

    if cfg.use_dynamic_features:
        raise NotImplementedError(
            "当前版本 train.py 仅支持严格静态特征训练。"
            "动态版本需要按历史 as-of date 构建 cohort-specific snapshot。"
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
    # Train nested milestone models
    # main chain: win -> bt_place -> bt_win -> graded_win
    # -------------------------
    print("\n=== TRAIN NESTED MILESTONE MODELS ===")

    win_model = train_binary_stage_model(
        train_df=train_df,
        valid_df=valid_df,
        target="win_flag",
        condition_col=None,
        use_dynamic=cfg.use_dynamic_features,
        depth=6,
        l2_leaf_reg=5.0,
        learning_rate=0.03,
        iterations=1000,
        auto_class_weights="Balanced",
    )

    bt_place_given_win_model = train_binary_stage_model(
        train_df=train_df,
        valid_df=valid_df,
        target="bt_place_flag",
        condition_col="win_flag",
        use_dynamic=cfg.use_dynamic_features,
        depth=5,
        l2_leaf_reg=10.0,
        learning_rate=0.03,
        iterations=1000,
        auto_class_weights="Balanced",
    )

    bt_win_given_bt_place_model = train_binary_stage_model(
        train_df=train_df,
        valid_df=valid_df,
        target="bt_win_flag",
        condition_col="bt_place_flag",
        use_dynamic=cfg.use_dynamic_features,
        depth=5,
        l2_leaf_reg=12.0,
        learning_rate=0.03,
        iterations=1200,
        auto_class_weights="Balanced",
    )

    graded_given_bt_win_model = train_binary_stage_model(
        train_df=train_df,
        valid_df=valid_df,
        target="graded_win_flag",
        condition_col="bt_win_flag",
        use_dynamic=cfg.use_dynamic_features,
        depth=4,
        l2_leaf_reg=20.0,
        learning_rate=0.02,
        iterations=1500,
        auto_class_weights="Balanced",
    )

    # -------------------------
    # Prize models
    # -------------------------
    print("\n=== TRAIN PRIZE MODELS ===")

    positive_prize_model = train_binary_stage_model(
        train_df=train_df,
        valid_df=valid_df,
        target="positive_prize_flag",
        condition_col=None,
        use_dynamic=cfg.use_dynamic_features,
        depth=6,
        l2_leaf_reg=5.0,
        learning_rate=0.03,
        iterations=1000,
        auto_class_weights=None,
    )

    prize_model = train_positive_regressor(
        train_df=train_df,
        valid_df=valid_df,
        use_dynamic=cfg.use_dynamic_features,
    )

    # -------------------------
    # Ceiling models
    # -------------------------
    print("\n=== TRAIN CEILING MODELS ===")

    prize_ge_10m_model = train_binary_stage_model(
        train_df=train_df,
        valid_df=valid_df,
        target="pog_total_prize_ge_10m_flag",
        condition_col=None,
        use_dynamic=cfg.use_dynamic_features,
        depth=5,
        l2_leaf_reg=10.0,
        learning_rate=0.03,
        iterations=1200,
        auto_class_weights="Balanced",
    )

    prize_ge_30m_model = train_binary_stage_model(
        train_df=train_df,
        valid_df=valid_df,
        target="pog_total_prize_ge_30m_flag",
        condition_col=None,
        use_dynamic=cfg.use_dynamic_features,
        depth=4,
        l2_leaf_reg=15.0,
        learning_rate=0.02,
        iterations=1500,
        auto_class_weights="Balanced",
    )

    # -------------------------
    # Quantile regressors (upper tail)
    # -------------------------
    print("\n=== TRAIN QUANTILE REGRESSORS ===")

    q80_model = train_quantile_regressor(
        train_df=train_df,
        valid_df=valid_df,
        alpha=0.8,
        use_dynamic=cfg.use_dynamic_features,
        depth=4,
        l2_leaf_reg=15.0,
        learning_rate=0.02,
        iterations=2000,
    )

    q90_model = train_quantile_regressor(
        train_df=train_df,
        valid_df=valid_df,
        alpha=0.9,
        use_dynamic=cfg.use_dynamic_features,
        depth=4,
        l2_leaf_reg=15.0,
        learning_rate=0.02,
        iterations=2000,
    )

    # -------------------------
    # Save models
    # -------------------------
    joblib.dump(win_model, "models/win_model.joblib")
    joblib.dump(bt_place_given_win_model, "models/bt_place_given_win_model.joblib")
    joblib.dump(bt_win_given_bt_place_model, "models/bt_win_given_bt_place_model.joblib")
    joblib.dump(graded_given_bt_win_model, "models/graded_given_bt_win_model.joblib")
    joblib.dump(positive_prize_model, "models/positive_prize_model.joblib")
    joblib.dump(prize_model, "models/prize_model.joblib")
    joblib.dump(prize_ge_10m_model, "models/prize_ge_10m_model.joblib")
    joblib.dump(prize_ge_30m_model, "models/prize_ge_30m_model.joblib")
    joblib.dump(q80_model, "models/q80_model.joblib")
    joblib.dump(q90_model, "models/q90_model.joblib")

    with open("models/model_bundle_meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": cfg.model_name,
                "model_version": cfg.model_version,
                "nested_chain": ["win_flag", "bt_place_flag", "bt_win_flag", "graded_win_flag"],
                "ceiling_targets": ["pog_total_prize_ge_10m_flag", "pog_total_prize_ge_30m_flag"],
                "quantile_targets": ["q80", "q90"],
                "train_years": [cfg.train_birth_year_start, cfg.train_birth_year_end],
                "valid_years": [cfg.valid_birth_year_start, cfg.valid_birth_year_end],
                "test_years": [cfg.test_birth_year_start, cfg.test_birth_year_end],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

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

    nested_pred = predict_nested_milestones(
        df=test_df,
        win_model=win_model,
        bt_place_given_win_model=bt_place_given_win_model,
        bt_win_given_bt_place_model=bt_win_given_bt_place_model,
        graded_given_bt_win_model=graded_given_bt_win_model,
        use_dynamic=cfg.use_dynamic_features,
    )
    for col in nested_pred.columns:
        test_pred[col] = nested_pred[col].values

    test_pred["p_positive_prize"] = predict_binary(
        positive_prize_model,
        test_df,
        "positive_prize_flag",
        use_dynamic=cfg.use_dynamic_features,
    )
    test_pred["pred_log_prize_pos"] = predict_regressor(
        prize_model,
        test_df,
        use_dynamic=cfg.use_dynamic_features,
    )
    test_pred["pred_positive_prize_amount"] = np.clip(
        np.expm1(test_pred["pred_log_prize_pos"]),
        0,
        None,
    )
    test_pred["expected_pog_prize"] = (
        test_pred["p_positive_prize"] * test_pred["pred_positive_prize_amount"]
    )

    # Ceiling predictions
    test_pred["p_prize_ge_10m"] = predict_binary(
        prize_ge_10m_model, test_df,
        "pog_total_prize_ge_10m_flag",
        use_dynamic=cfg.use_dynamic_features,
    )
    test_pred["p_prize_ge_30m"] = predict_binary(
        prize_ge_30m_model, test_df,
        "pog_total_prize_ge_30m_flag",
        use_dynamic=cfg.use_dynamic_features,
    )

    # Quantile predictions
    test_pred["q80_log_prize"] = predict_regressor(q80_model, test_df, use_dynamic=cfg.use_dynamic_features)
    test_pred["q90_log_prize"] = predict_regressor(q90_model, test_df, use_dynamic=cfg.use_dynamic_features)
    test_pred["q80_prize"] = np.clip(np.expm1(test_pred["q80_log_prize"]), 0, None)
    test_pred["q90_prize"] = np.clip(np.expm1(test_pred["q90_log_prize"]), 0, None)

    test_pred = build_blended_scores(test_pred)

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
        "score_balanced",
        "score_ceiling",
        "score_ceiling_old",
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

    current_nested = predict_nested_milestones(
        df=score_df,
        win_model=win_model,
        bt_place_given_win_model=bt_place_given_win_model,
        bt_win_given_bt_place_model=bt_win_given_bt_place_model,
        graded_given_bt_win_model=graded_given_bt_win_model,
        use_dynamic=cfg.use_dynamic_features,
    )
    for col in current_nested.columns:
        current_pred[col] = current_nested[col].values

    current_pred["p_positive_prize"] = predict_binary(
        positive_prize_model,
        score_df,
        "positive_prize_flag",
        use_dynamic=cfg.use_dynamic_features,
    )
    current_pred["pred_log_prize_pos"] = predict_regressor(
        prize_model,
        score_df,
        use_dynamic=cfg.use_dynamic_features,
    )
    current_pred["pred_positive_prize_amount"] = np.clip(
        np.expm1(current_pred["pred_log_prize_pos"]),
        0,
        None,
    )
    current_pred["expected_pog_prize"] = (
        current_pred["p_positive_prize"] * current_pred["pred_positive_prize_amount"]
    )

    # Ceiling predictions for current cohort
    current_pred["p_prize_ge_10m"] = predict_binary(
        prize_ge_10m_model, score_df,
        "pog_total_prize_ge_10m_flag",
        use_dynamic=cfg.use_dynamic_features,
    )
    current_pred["p_prize_ge_30m"] = predict_binary(
        prize_ge_30m_model, score_df,
        "pog_total_prize_ge_30m_flag",
        use_dynamic=cfg.use_dynamic_features,
    )
    current_pred["q80_log_prize"] = predict_regressor(q80_model, score_df, use_dynamic=cfg.use_dynamic_features)
    current_pred["q90_log_prize"] = predict_regressor(q90_model, score_df, use_dynamic=cfg.use_dynamic_features)
    current_pred["q80_prize"] = np.clip(np.expm1(current_pred["q80_log_prize"]), 0, None)
    current_pred["q90_prize"] = np.clip(np.expm1(current_pred["q90_log_prize"]), 0, None)

    current_pred = build_blended_scores(current_pred)

    current_pred["model_name"] = cfg.model_name
    current_pred["model_version"] = cfg.model_version
    current_pred["asof_date"] = pd.to_datetime(cfg.asof_date)

    # 排序输出
    current_pred = current_pred.sort_values("score_balanced", ascending=False).reset_index(drop=True)
    current_pred.to_csv("outputs/current_cohort_predictions_nested.csv", index=False)

    # 存入数据库：默认只写兼容旧 schema 的列
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
            "score_balanced",
            "score_ceiling",
            "score_ceiling_old",
        ]
        if c in current_pred.columns
    ]

    current_pred.sort_values("score_balanced", ascending=False).head(100)[shortlist_cols].to_csv(
        "outputs/current_top100_balanced.csv", index=False
    )
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
    print("- outputs/current_top100_balanced.csv")
    print("- outputs/current_top100_ceiling.csv")
    print("- outputs/current_top100_graded_win.csv")


if __name__ == "__main__":
    load_dotenv()
    main()

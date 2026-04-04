"""
two_tower_pipeline.py - 双塔解耦架构训练和预测管道

架构:
- 下限塔 (Floor Tower): win_flag, positive_prize_flag - 全样本训练
- 上限塔 (Ceiling Tower): graded_win_flag, prize_ge_30m_flag - 全样本 + Focal Loss
- 融合层: LTR Ranker (YetiRank)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, CatBoostRanker, Pool
from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression

from .features import (
    FeatureSet, prepare_matrix, add_log_target,
)
from .two_tower_config import TwoTowerConfig


# =========================
# Pool 构建
# =========================

def build_pool(X, y, cat_cols):
    return Pool(X, label=y, cat_features=[c for c in cat_cols if c in X.columns])


def describe_split(name: str, df: pd.DataFrame, target: str):
    n = len(df)
    pos = int(df[target].sum()) if target in df.columns else 0
    rate = pos / n if n > 0 else 0.0
    print(f"[{name}] target={target}, n={n}, pos={pos}, rate={rate:.4f}")


# =========================
# 二分类模型训练
# =========================

def train_binary_model(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    target: str,
    feature_set: FeatureSet,
    depth: int = 6,
    l2_leaf_reg: float = 5.0,
    learning_rate: float = 0.03,
    iterations: int = 1000,
    auto_class_weights: str | None = "Balanced",
    ctr_leaf_count_limit: int | None = None,
    calibrate_probability: bool = True,
    focal_alpha: float | None = None,
    focal_gamma: float | None = None,
) -> CatBoostClassifier:
    """训练二分类模型，支持 Focal Loss 和概率校准。"""
    
    describe_split("train", train_df, target)
    describe_split("valid", valid_df, target)

    if len(train_df) == 0:
        raise ValueError(f"{target}: train_df 为空")
    if len(valid_df) == 0:
        raise ValueError(f"{target}: valid_df 为空")
    if train_df[target].nunique() < 2:
        raise ValueError(f"{target}: train_df 只有单一类别")
    if valid_df[target].nunique() < 2:
        raise ValueError(f"{target}: valid_df 只有单一类别")

    X_train, y_train, _, cat_cols = prepare_matrix(train_df, feature_set=feature_set)
    X_valid, y_valid, _, _ = prepare_matrix(valid_df, feature_set=feature_set)

    # Focal Loss 或 Logloss
    use_focal = focal_alpha is not None and focal_gamma is not None
    if use_focal:
        loss_fn = f"Focal:focal_alpha={focal_alpha};focal_gamma={focal_gamma}"
        effective_class_weights = None
        print(f"[{target}] Using Focal Loss: alpha={focal_alpha}, gamma={focal_gamma}")
    else:
        loss_fn = "Logloss"
        effective_class_weights = auto_class_weights

    cb_kwargs: dict = dict(
        loss_function=loss_fn,
        eval_metric="AUC",
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        l2_leaf_reg=l2_leaf_reg,
        random_seed=42,
        verbose=100,
        od_type="Iter",
        od_wait=100,
    )
    if effective_class_weights is not None:
        cb_kwargs["auto_class_weights"] = effective_class_weights
    if ctr_leaf_count_limit is not None:
        cb_kwargs["ctr_leaf_count_limit"] = ctr_leaf_count_limit

    model = CatBoostClassifier(**cb_kwargs)

    model.fit(
        build_pool(X_train, y_train[target], cat_cols),
        eval_set=build_pool(X_valid, y_valid[target], cat_cols),
        use_best_model=True,
    )

    # 概率校准
    if calibrate_probability and len(valid_df) >= 20:
        print(f"[{target}] Applying isotonic probability calibration...")
        raw_probs = model.predict_proba(X_valid)[:, 1]
        iso_reg = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso_reg.fit(raw_probs, y_valid[target])
        model._calibration_iso_reg = iso_reg
        print(f"[{target}] Calibration complete.")

    return model


def predict_binary(model: CatBoostClassifier, df: pd.DataFrame, feature_set: FeatureSet) -> np.ndarray:
    """预测二分类概率，应用校准。"""
    X, _, _, _ = prepare_matrix(df, feature_set=feature_set)
    prob = model.predict_proba(X)[:, 1]
    cal = getattr(model, "_calibration_iso_reg", None)
    if cal is not None:
        prob = cal.predict(prob)
    return prob


# =========================
# 回归模型训练
# =========================

def train_regressor(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_set: FeatureSet,
    depth: int = 4,
    l2_leaf_reg: float = 20.0,
    learning_rate: float = 0.02,
    iterations: int = 2000,
    ctr_leaf_count_limit: int | None = None,
) -> CatBoostRegressor:
    """训练正奖金回归模型。"""
    train_pos = train_df[train_df["pog_total_prize"] > 0].copy()
    valid_pos = valid_df[valid_df["pog_total_prize"] > 0].copy()

    if len(train_pos) == 0:
        raise ValueError("train_pos 为空")
    if len(valid_pos) == 0:
        raise ValueError("valid_pos 为空")

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
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_set: FeatureSet,
    alpha: float = 0.9,
    depth: int = 4,
    l2_leaf_reg: float = 15.0,
    learning_rate: float = 0.02,
    iterations: int = 2000,
    ctr_leaf_count_limit: int | None = None,
) -> CatBoostRegressor:
    """训练分位数回归模型。"""
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


def predict_regressor(model: CatBoostRegressor, df: pd.DataFrame, feature_set: FeatureSet) -> np.ndarray:
    X, _, _, _ = prepare_matrix(df, feature_set=feature_set)
    return model.predict(X)


# =========================
# Ranking Model
# =========================

def _subsample_groups(
    X: pd.DataFrame,
    y: np.ndarray,
    group_ids: np.ndarray,
    max_group_size: int,
    rng: np.random.Generator | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """对每个组进行下采样。"""
    if max_group_size <= 0:
        return X, y, group_ids

    if rng is None:
        rng = np.random.default_rng(42)

    keep_idx: list[int] = []
    for gid in np.unique(group_ids):
        g_mask = np.where(group_ids == gid)[0]
        if len(g_mask) <= max_group_size:
            keep_idx.extend(g_mask.tolist())
        else:
            keep_idx.extend(rng.choice(g_mask, size=max_group_size, replace=False).tolist())

    keep_idx.sort()
    X_out = X.iloc[keep_idx].reset_index(drop=True)
    group_out = group_ids[keep_idx]

    from scipy.stats import rankdata
    y_out = np.empty(len(keep_idx), dtype=float)
    for gid in np.unique(group_out):
        g_mask = np.where(group_out == gid)[0]
        original_prizes = y[np.array(keep_idx)[g_mask]]
        y_out[g_mask] = rankdata(-original_prizes, method="average")

    return X_out, y_out, group_out


def train_ranking_model(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_set: FeatureSet,
    bundle: "TwoTowerBundle",
    depth: int = 4,
    l2_leaf_reg: float = 10.0,
    learning_rate: float = 0.02,
    iterations: int = 500,
    ranking_mode: str = "YetiRank",
    max_group_size: int = 300,
    fallback_ranking_mode: str = "PairLogit",
) -> CatBoostRanker | None:
    """训练 LTR 排序模型作为融合层。"""
    import gc

    train_pos = train_df[train_df["pog_total_prize"] > 0].copy()
    valid_pos = valid_df[valid_df["pog_total_prize"] > 0].copy()

    if len(train_pos) < 50 or len(valid_pos) < 10:
        print(f"[WARN] Not enough positive-prize data for ranking: "
              f"train={len(train_pos)}, valid={len(valid_pos)}. Skipping.")
        return None

    # 生成两塔预测作为 meta-features
    train_meta = predict_all(bundle, train_pos)
    valid_meta = predict_all(bundle, valid_pos)

    ranking_features = [
        # 下限塔
        "p_win", "p_positive_prize",
        # 上限塔
        "p_graded_win_direct", "p_prize_ge_30m_direct",
        # 奖金预测
        "expected_pog_prize", "q90_prize",
    ]
    available_features = [f for f in ranking_features if f in train_meta.columns]

    X_train_raw = train_meta[available_features]
    X_valid_raw = valid_meta[available_features]

    train_group_id = train_pos["birth_year"].values.astype(int)
    valid_group_id = valid_pos["birth_year"].values.astype(int)

    train_prizes = train_pos["pog_total_prize"].values.astype(float)
    valid_prizes = valid_pos["pog_total_prize"].values.astype(float)

    if max_group_size and max_group_size > 0:
        X_train, y_train_rank, train_group_id = _subsample_groups(
            X_train_raw, train_prizes, train_group_id, max_group_size
        )
        X_valid, y_valid_rank, valid_group_id = _subsample_groups(
            X_valid_raw, valid_prizes, valid_group_id, max_group_size
        )
    else:
        X_train = X_train_raw
        X_valid = X_valid_raw
        from scipy.stats import rankdata
        y_train_rank = np.empty(len(X_train), dtype=float)
        for gid in np.unique(train_group_id):
            mask = train_group_id == gid
            y_train_rank[mask] = rankdata(-train_prizes[mask], method="average")
        y_valid_rank = np.empty(len(X_valid), dtype=float)
        for gid in np.unique(valid_group_id):
            mask = valid_group_id == gid
            y_valid_rank[mask] = rankdata(-valid_prizes[mask], method="average")

    del train_meta, valid_meta, X_train_raw, X_valid_raw
    gc.collect()

    n_train_groups = len(np.unique(train_group_id))
    n_valid_groups = len(np.unique(valid_group_id))
    print(f"Ranking data: train={len(X_train)} samples in {n_train_groups} groups, "
          f"valid={len(X_valid)} samples in {n_valid_groups} groups")

    train_order = np.argsort(train_group_id, kind="stable")
    X_train = X_train.iloc[train_order].reset_index(drop=True)
    y_train_rank = y_train_rank[train_order]
    train_group_id = train_group_id[train_order]

    valid_order = np.argsort(valid_group_id, kind="stable")
    X_valid = X_valid.iloc[valid_order].reset_index(drop=True)
    y_valid_rank = y_valid_rank[valid_order]
    valid_group_id = valid_group_id[valid_order]

    train_pool = Pool(X_train, label=y_train_rank, group_id=train_group_id)
    valid_pool = Pool(X_valid, label=y_valid_rank, group_id=valid_group_id)

    print(f"Training ranking model ({ranking_mode}) with features: {available_features}")

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

    try:
        ranking_model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
    except (MemoryError, Exception) as e:
        err_msg = str(e).lower()
        is_oom = isinstance(e, MemoryError) or "memory" in err_msg or "alloc" in err_msg
        if is_oom and ranking_mode != fallback_ranking_mode:
            print(f"[WARN] {ranking_mode} OOM: {e}")
            print(f"[WARN] Retrying with fallback mode: {fallback_ranking_mode}")
            gc.collect()

            ranking_model = CatBoostRanker(
                loss_function=fallback_ranking_mode,
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
            ranking_model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
        else:
            raise

    return ranking_model


def predict_ranking(bundle: "TwoTowerBundle", df: pd.DataFrame) -> np.ndarray | None:
    """使用 ranking model 生成排序分数。
    
    Args:
        bundle: 包含 ranking_model 和 feature_set 的模型包
        df: 必须包含特征列的完整 DataFrame（不能只包含预测结果列）
    """
    if bundle.ranking_model is None:
        return None

    # 需要从原始特征生成预测
    meta = predict_all(bundle, df)

    ranking_features = [
        "p_win", "p_positive_prize",
        "p_graded_win_direct", "p_prize_ge_30m_direct",
        "expected_pog_prize", "q90_prize",
    ]
    available_features = [f for f in ranking_features if f in meta.columns]
    
    if len(available_features) == 0:
        print("[WARN] No ranking features available, skipping ranking prediction")
        return None
    
    X = meta[available_features]

    return bundle.ranking_model.predict(X)


# =========================
# Model Bundle
# =========================

@dataclass
class TwoTowerBundle:
    """双塔架构模型包"""
    # 下限塔
    win_model: CatBoostClassifier
    positive_prize_model: CatBoostClassifier
    
    # 上限塔
    graded_win_direct_model: CatBoostClassifier
    prize_ge_30m_direct_model: CatBoostClassifier
    
    # 回归器
    prize_model: CatBoostRegressor
    q90_model: CatBoostRegressor
    
    # 融合层
    ranking_model: CatBoostRanker | None
    
    # 特征集
    feature_set: FeatureSet


# =========================
# 训练管道
# =========================

def train_two_tower_models(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_set: FeatureSet,
    config: TwoTowerConfig | None = None,
) -> TwoTowerBundle:
    """训练双塔架构所有模型。"""
    if config is None:
        config = TwoTowerConfig()

    # ========== 下限塔 ==========
    print("\n" + "=" * 60)
    print("FLOOR TOWER (下限塔)")
    print("=" * 60)
    
    floor_models = {}
    for key, cfg in config.floor_tower_configs.items():
        print(f"\n--- Training {key} ---")
        floor_models[key] = train_binary_model(
            train_df, valid_df, feature_set=feature_set, **cfg
        )

    # ========== 上限塔 ==========
    print("\n" + "=" * 60)
    print("CEILING TOWER (上限塔)")
    print("=" * 60)
    
    ceiling_models = {}
    for key, cfg in config.ceiling_tower_configs.items():
        print(f"\n--- Training {key} ---")
        ceiling_models[key] = train_binary_model(
            train_df, valid_df, feature_set=feature_set, **cfg
        )

    # ========== 回归器 ==========
    print("\n" + "=" * 60)
    print("REGRESSORS")
    print("=" * 60)
    
    print("\n--- Training prize_regressor ---")
    prize_model = train_regressor(
        train_df, valid_df, feature_set=feature_set,
        **config.regressor_configs["prize_regressor"]
    )
    
    print("\n--- Training q90_regressor ---")
    q90_model = train_quantile_regressor(
        train_df, valid_df, feature_set=feature_set,
        **config.regressor_configs["q90_regressor"]
    )

    # 构建临时 bundle 用于 ranking model
    temp_bundle = TwoTowerBundle(
        win_model=floor_models["win"],
        positive_prize_model=floor_models["positive_prize"],
        graded_win_direct_model=ceiling_models["graded_win_direct"],
        prize_ge_30m_direct_model=ceiling_models["prize_ge_30m_direct"],
        prize_model=prize_model,
        q90_model=q90_model,
        ranking_model=None,
        feature_set=feature_set,
    )

    # ========== 融合层 (LTR Ranker) ==========
    print("\n" + "=" * 60)
    print("FUSION LAYER (LTR Ranker)")
    print("=" * 60)
    
    ranking_model = None
    try:
        ranking_model = train_ranking_model(
            train_df, valid_df, feature_set=feature_set,
            bundle=temp_bundle,
            **config.ranking_config
        )
        if ranking_model is not None:
            print("Ranking model trained successfully.")
        else:
            print("Ranking model skipped.")
    except Exception as e:
        print(f"[WARN] Ranking model failed: {e}. Skipping.")
        ranking_model = None

    return TwoTowerBundle(
        win_model=floor_models["win"],
        positive_prize_model=floor_models["positive_prize"],
        graded_win_direct_model=ceiling_models["graded_win_direct"],
        prize_ge_30m_direct_model=ceiling_models["prize_ge_30m_direct"],
        prize_model=prize_model,
        q90_model=q90_model,
        ranking_model=ranking_model,
        feature_set=feature_set,
    )


# =========================
# 预测管道
# =========================

def predict_all(bundle: TwoTowerBundle, df: pd.DataFrame) -> pd.DataFrame:
    """生成双塔架构的所有预测。"""
    fs = bundle.feature_set
    pred = pd.DataFrame(index=df.index)

    # 下限塔
    pred["p_win"] = predict_binary(bundle.win_model, df, fs)
    pred["p_positive_prize"] = predict_binary(bundle.positive_prize_model, df, fs)

    # 上限塔
    pred["p_graded_win_direct"] = predict_binary(bundle.graded_win_direct_model, df, fs)
    pred["p_prize_ge_30m_direct"] = predict_binary(bundle.prize_ge_30m_direct_model, df, fs)

    # 回归器
    pred["pred_log_prize_pos"] = predict_regressor(bundle.prize_model, df, fs)
    pred["pred_positive_prize_amount"] = np.clip(np.expm1(pred["pred_log_prize_pos"]), 0, None)
    pred["expected_pog_prize"] = pred["p_positive_prize"] * pred["pred_positive_prize_amount"]

    pred["q90_log_prize"] = predict_regressor(bundle.q90_model, df, fs)
    pred["q90_prize"] = np.clip(np.expm1(pred["q90_log_prize"]), 0, None)

    return pred


def build_blended_scores(
    df: pd.DataFrame,
    bundle: TwoTowerBundle,
    full_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """构建融合分数。
    
    Args:
        df: 包含预测结果的 DataFrame（可能是精简版，只有基础列）
        bundle: 模型包
        full_df: 完整的 DataFrame，包含特征列。用于 ranking 预测。
                 如果为 None，则使用 df 进行预测（假设 df 包含特征列）。
    """
    out = df.copy()
    
    # Ranking scores - 需要完整特征数据
    ranking_df = full_df if full_df is not None else df
    ranking_scores = predict_ranking(bundle, ranking_df)
    if ranking_scores is not None and len(ranking_scores) == len(out):
        out["score_ranking"] = ranking_scores

    return out


# =========================
# 序列化
# =========================

def save_bundle(bundle: TwoTowerBundle, path: str, meta_extra: dict | None = None):
    """保存模型包。"""
    os.makedirs(path, exist_ok=True)

    # 下限塔
    joblib.dump(bundle.win_model, os.path.join(path, "win_model.joblib"))
    joblib.dump(bundle.positive_prize_model, os.path.join(path, "positive_prize_model.joblib"))
    
    # 上限塔
    joblib.dump(bundle.graded_win_direct_model, os.path.join(path, "graded_win_direct_model.joblib"))
    joblib.dump(bundle.prize_ge_30m_direct_model, os.path.join(path, "prize_ge_30m_direct_model.joblib"))
    
    # 回归器
    joblib.dump(bundle.prize_model, os.path.join(path, "prize_model.joblib"))
    joblib.dump(bundle.q90_model, os.path.join(path, "q90_model.joblib"))
    
    # 融合层
    if bundle.ranking_model is not None:
        joblib.dump(bundle.ranking_model, os.path.join(path, "ranking_model.joblib"))

    from dataclasses import asdict
    meta = {
        "architecture": "two_tower",
        "feature_set": asdict(bundle.feature_set),
        "has_ranking_model": bundle.ranking_model is not None,
    }
    if meta_extra:
        meta.update(meta_extra)

    with open(os.path.join(path, "two_tower_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def load_bundle(path: str, feature_set: FeatureSet | None = None) -> TwoTowerBundle:
    """加载模型包。"""
    # 下限塔
    win_model = joblib.load(os.path.join(path, "win_model.joblib"))
    positive_prize_model = joblib.load(os.path.join(path, "positive_prize_model.joblib"))
    
    # 上限塔
    graded_win_direct_model = joblib.load(os.path.join(path, "graded_win_direct_model.joblib"))
    prize_ge_30m_direct_model = joblib.load(os.path.join(path, "prize_ge_30m_direct_model.joblib"))
    
    # 回归器
    prize_model = joblib.load(os.path.join(path, "prize_model.joblib"))
    q90_model = joblib.load(os.path.join(path, "q90_model.joblib"))
    
    # 融合层
    ranking_model_path = os.path.join(path, "ranking_model.joblib")
    ranking_model = joblib.load(ranking_model_path) if os.path.exists(ranking_model_path) else None

    # 元数据
    meta_path = os.path.join(path, "two_tower_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if feature_set is None and "feature_set" in meta:
            feature_set = FeatureSet(**meta["feature_set"])

    if feature_set is None:
        feature_set = FeatureSet()

    return TwoTowerBundle(
        win_model=win_model,
        positive_prize_model=positive_prize_model,
        graded_win_direct_model=graded_win_direct_model,
        prize_ge_30m_direct_model=prize_ge_30m_direct_model,
        prize_model=prize_model,
        q90_model=q90_model,
        ranking_model=ranking_model,
        feature_set=feature_set,
    )
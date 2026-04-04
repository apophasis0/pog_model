"""
pog_model - POG 预测模型核心库

包含配置、数据处理、特征工程、模型管道等共享模块。
"""

from .config import Config
from .two_tower_config import TwoTowerConfig
from .features import FeatureSet
from .data import (
    load_training_frame,
    load_scoring_frame,
    save_predictions,
    load_completed_birth_years,
    load_all_labeled_frame,
    load_cohort_frame,
)
from .eval import evaluate_binary, evaluate_regression
from .pipeline import (
    ModelBundle,
    train_all_models,
    predict_all,
    predict_ranking,
    build_blended_scores,
    build_topk_report,
    print_metrics,
    save_bundle,
    load_bundle,
)
from .two_tower_pipeline import (
    TwoTowerBundle,
    train_two_tower_models,
    predict_all as predict_all_two_tower,
    build_blended_scores as build_blended_scores_two_tower,
    save_bundle as save_bundle_two_tower,
)
from .split import (
    auto_configure_splits,
    auto_configure_two_tower_splits,
    split_by_birth_year,
    describe_split,
)
from .rolling_backtest import (
    generate_folds,
    run_single_fold,
    aggregate_rolling_results,
)

__all__ = [
    "Config",
    "TwoTowerConfig",
    "FeatureSet",
    "load_training_frame",
    "load_scoring_frame",
    "save_predictions",
    "load_completed_birth_years",
    "load_all_labeled_frame",
    "evaluate_binary",
    "evaluate_regression",
    "ModelBundle",
    "train_all_models",
    "predict_all",
    "predict_ranking",
    "build_blended_scores",
    "build_topk_report",
    "print_metrics",
    "save_bundle",
    "TwoTowerBundle",
    "train_two_tower_models",
    "predict_all_two_tower",
    "build_blended_scores_two_tower",
    "save_bundle_two_tower",
]
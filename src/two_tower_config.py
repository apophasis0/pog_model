"""
two_tower_config.py - 双塔解耦架构配置

架构说明:
- 下限塔 (Floor Tower): 预测"会不会赔本"，包含 win_flag 和 positive_prize_flag
- 上限塔 (Ceiling Tower): 直接预测"成为顶级马概率"，在全样本上训练
- 融合层: 使用 LTR Ranker (YetiRank) 组合两塔输出
"""
from dataclasses import dataclass, field
from datetime import date
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass
class TwoTowerConfig:
    """双塔架构专用配置"""
    db_url: str = os.getenv("DATABASE_URL", "postgresql+psycopg2://postgres@localhost:5432/jvlink")
    model_name: str = "pog_two_tower"
    model_version: str = "v1.0.0"

    # 当前要预测的出生年
    target_birth_year: int = date.today().year - 2
    asof_date: date = date.today()

    # 训练/验证/测试 cohort
    train_birth_year_start: int = 2008
    train_birth_year_end: int = 2020
    valid_birth_year_start: int = 2021
    valid_birth_year_end: int = 2021
    test_birth_year_start: int = 2022
    test_birth_year_end: int = 2022

    use_dynamic_features: bool = False

    # =========================
    # 下限塔 (Floor Tower) 配置
    # =========================
    floor_tower_configs: dict = field(default_factory=lambda: {
        "win": {
            "target": "win_flag",
            # 全样本训练，无 condition_col
            "depth": 6, "l2_leaf_reg": 5.0, "learning_rate": 0.03, "iterations": 1000,
            "auto_class_weights": "Balanced",
            "ctr_leaf_count_limit": 4,
        },
        "positive_prize": {
            "target": "positive_prize_flag",
            # 全样本训练，无 condition_col
            "depth": 6, "l2_leaf_reg": 5.0, "learning_rate": 0.03, "iterations": 1000,
            "auto_class_weights": None,
            "ctr_leaf_count_limit": 4,
        },
    })

    # =========================
    # 上限塔 (Ceiling Tower) 配置
    # =========================
    # 关键：全样本训练 + Focal Loss 处理极端不平衡
    ceiling_tower_configs: dict = field(default_factory=lambda: {
        "graded_win_direct": {
            "target": "graded_win_flag",
            # 不使用 condition_col，直接在全样本上预测
            # 正样本约 2-3%，使用 Focal Loss
            "depth": 5, "l2_leaf_reg": 10.0, "learning_rate": 0.02, "iterations": 1500,
            "auto_class_weights": None,  # Focal Loss 内置类别平衡
            "ctr_leaf_count_limit": 4,
            # Focal Loss: alpha < 0.5 给少数类更多关注
            "focal_alpha": 0.25,
            "focal_gamma": 2.0,
        },
        "prize_ge_30m_direct": {
            "target": "pog_total_prize_ge_30m_flag",
            # 全样本训练
            "depth": 4, "l2_leaf_reg": 15.0, "learning_rate": 0.02, "iterations": 1500,
            "auto_class_weights": None,
            "ctr_leaf_count_limit": 4,
            "focal_alpha": 0.25,
            "focal_gamma": 2.0,
        },
    })

    # =========================
    # 奖金回归器配置
    # =========================
    regressor_configs: dict = field(default_factory=lambda: {
        "prize_regressor": {
            "depth": 4, "l2_leaf_reg": 20.0, "learning_rate": 0.02, "iterations": 2000,
            "ctr_leaf_count_limit": 4,
        },
        "q90_regressor": {
            "alpha": 0.9, "depth": 4, "l2_leaf_reg": 15.0, "learning_rate": 0.02, "iterations": 2000,
            "ctr_leaf_count_limit": 4,
        },
    })

    # =========================
    # 融合层 (LTR Ranker) 配置
    # =========================
    ranking_config: dict = field(default_factory=lambda: {
        "depth": 4, "l2_leaf_reg": 10.0, "learning_rate": 0.02, "iterations": 500,
        "ranking_mode": "YetiRank",
        "max_group_size": 300,
        "fallback_ranking_mode": "PairLogit",
    })

    # High-cardinality category frequency threshold
    high_cardinality_min_count: int = 3
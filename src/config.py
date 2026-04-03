from dataclasses import dataclass, field
from datetime import date
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Config:
    db_url: str = os.getenv("DATABASE_URL", "postgresql+psycopg2://postgres@localhost:5432/jvlink")
    model_name: str = "pog_catboost_hurdle"
    model_version: str = "v0.1.0"

    # 当前要预测的出生年
    target_birth_year: int = date.today().year - 2

    # 如果做动态更新，给一个快照日期
    asof_date: date = date.today()

    # 训练/验证/测试 cohort
    train_birth_year_start: int = 2008
    train_birth_year_end: int = 2020
    valid_birth_year_start: int = 2021
    valid_birth_year_end: int = 2021
    test_birth_year_start: int = 2022
    test_birth_year_end: int = 2022

    use_dynamic_features: bool = False

    # 模型训练的超参数字典
    model_configs: dict = field(default_factory=lambda: {
        "win": {
            "target": "win_flag",
            "condition_col": None,
            "depth": 6, "l2_leaf_reg": 5.0, "learning_rate": 0.03, "iterations": 1000,
            "auto_class_weights": "Balanced",
            # High-cardinality regularization: limit CTR leaf count to reduce overfitting
            # on rare categories (banusi_code, breeder_code, etc.)
            "ctr_leaf_count_limit": 4,
        },
        "bt_place_given_win": {
            "target": "bt_place_flag",
            "condition_col": "win_flag",
            "depth": 5, "l2_leaf_reg": 10.0, "learning_rate": 0.03, "iterations": 1000,
            "auto_class_weights": "Balanced",
            "ctr_leaf_count_limit": 4,
        },
        "bt_win_given_bt_place": {
            "target": "bt_win_flag",
            "condition_col": "bt_place_flag",
            "depth": 5, "l2_leaf_reg": 12.0, "learning_rate": 0.03, "iterations": 1200,
            "auto_class_weights": "Balanced",
            "ctr_leaf_count_limit": 4,
        },
        "graded_given_bt_win": {
            "target": "graded_win_flag",
            "condition_col": "bt_win_flag",
            "depth": 4, "l2_leaf_reg": 20.0, "learning_rate": 0.02, "iterations": 1500,
            "auto_class_weights": "Balanced",
            "ctr_leaf_count_limit": 4,
        },
        "positive_prize": {
            "target": "positive_prize_flag",
            "condition_col": None,
            "depth": 6, "l2_leaf_reg": 5.0, "learning_rate": 0.03, "iterations": 1000,
            "auto_class_weights": None,
            "ctr_leaf_count_limit": 4,
        },
        "prize_ge_10m": {
            "target": "pog_total_prize_ge_10m_flag",
            "condition_col": None,
            "depth": 5, "l2_leaf_reg": 10.0, "learning_rate": 0.03, "iterations": 1200,
            "auto_class_weights": "Balanced",
            "ctr_leaf_count_limit": 4,
        },
        "prize_ge_30m": {
            "target": "pog_total_prize_ge_30m_flag",
            "condition_col": None,
            "depth": 4, "l2_leaf_reg": 15.0, "learning_rate": 0.02, "iterations": 1500,
            "auto_class_weights": "Balanced",
            "ctr_leaf_count_limit": 4,
        },
        "prize_regressor": {
            "depth": 4, "l2_leaf_reg": 20.0, "learning_rate": 0.02, "iterations": 2000,
            "ctr_leaf_count_limit": 4,
        },
        "q80_regressor": {
            "alpha": 0.8, "depth": 4, "l2_leaf_reg": 15.0, "learning_rate": 0.02, "iterations": 2000,
            "ctr_leaf_count_limit": 4,
        },
        "q90_regressor": {
            "alpha": 0.9, "depth": 4, "l2_leaf_reg": 15.0, "learning_rate": 0.02, "iterations": 2000,
            "ctr_leaf_count_limit": 4,
        },
        # Ranking model (Learning to Rank) configuration
        "ranking_stacking": {
            "depth": 4, "l2_leaf_reg": 10.0, "learning_rate": 0.02, "iterations": 1000,
            "ranking_mode": "YetiRank",
        }
    })

    # High-cardinality category frequency threshold
    high_cardinality_min_count: int = 3

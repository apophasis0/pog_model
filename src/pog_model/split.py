"""
split.py - 数据切分辅助函数
"""
import pandas as pd

from .config import Config
from .two_tower_config import TwoTowerConfig
from .data import load_completed_birth_years


def auto_configure_splits(cfg: Config) -> Config:
    """自动配置训练/验证/测试集的年份范围。"""
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


def auto_configure_two_tower_splits(cfg: TwoTowerConfig) -> TwoTowerConfig:
    """自动配置双塔架构的训练/验证/测试集的年份范围。"""
    completed_years = load_completed_birth_years(cfg)

    if len(completed_years) < 5:
        raise ValueError(
            f"已完成标签的 birth_year 太少：{completed_years}。至少需要 5 个完整 cohort。"
        )

    cfg.test_birth_year_start = completed_years[-1]
    cfg.test_birth_year_end = completed_years[-1]

    cfg.valid_birth_year_start = completed_years[-2]
    cfg.valid_birth_year_end = completed_years[-2]

    cfg.train_birth_year_start = completed_years[0]
    cfg.train_birth_year_end = completed_years[-3]

    return cfg


def split_by_birth_year(
    df: pd.DataFrame,
    cfg: Config | TwoTowerConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """按出生年份切分训练/验证/测试集。"""
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
    """打印数据集切分统计信息。"""
    n = len(df)
    pos_prize = int((df["pog_total_prize"] > 0).sum()) if "pog_total_prize" in df.columns else 0
    win_n = int(df["win_flag"].sum()) if "win_flag" in df.columns else 0
    bt_place_n = int(df["bt_place_flag"].sum()) if "bt_place_flag" in df.columns else 0
    bt_win_n = int(df["bt_win_flag"].sum()) if "bt_win_flag" in df.columns else 0
    graded_n = int(df["graded_win_flag"].sum()) if "graded_win_flag" in df.columns else 0
    prize_30m_n = int(df["pog_total_prize_ge_30m_flag"].sum()) if "pog_total_prize_ge_30m_flag" in df.columns else 0

    parts = [f"n={n}", f"positive_prize={pos_prize}", f"win={win_n}"]
    if bt_place_n > 0:
        parts.append(f"bt_place={bt_place_n}")
    if bt_win_n > 0:
        parts.append(f"bt_win={bt_win_n}")
    if graded_n > 0:
        parts.append(f"graded_win={graded_n}")
    if prize_30m_n > 0:
        parts.append(f"prize_ge_30m={prize_30m_n}")

    print(f"[{name}] " + ", ".join(parts))

    if n == 0:
        raise ValueError(f"{name} split 为空，请检查年份切分。")
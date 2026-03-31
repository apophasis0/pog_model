from dataclasses import dataclass
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

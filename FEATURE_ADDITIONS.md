# 新增特征说明 (New Features Documentation)

## 概述 (Overview)

本次更新为双塔模型（Two-Tower Architecture）的**上限塔（Upper Limit Tower）**引入了两组全新特征，旨在更准确地捕捉马匹的极限潜力和血统深度。

## 一、极值转化率特征 (Ceiling Conversion Rate Features)

### 业务逻辑
"极值转化率" = **重赏马数量 / 获胜马数量**，即 `P(重赏 | 获胜)`

这个特征衡量的是："一旦马匹能赢，它成为重赏马的概率有多大？"

### 价值分析
- **高下限 vs 高上限区分**：某些种公马获胜率很高但多产低级别赢家（高下限、低上限），而顶级种公马虽然赢马率可能不那么夸张，但一旦赢马，成为G1级别的概率极高（高上限）
- **POG 核心目标对齐**：POG 的本质是挑选未来的"大物"，而非仅仅挑选能赢的马

### 新增字段
在以下视图/表中新增 `*_prior_graded_per_win` 字段：

1. **`mv_sire_hist_stats`**
   - `sire_prior_graded_per_win`: 父系极值转化率

2. **`mv_dam_hist_stats`**
   - `dam_prior_graded_per_win`: 母马极值转化率

3. **`mv_damsire_hist_stats`**
   - `damsire_prior_graded_per_win`: 母父极值转化率

4. **`mv_prior_maternal_sibling_stats`**
   - `prior_maternal_sib_graded_per_win`: 同母半兄姊极值转化率

5. **`mv_sire_damsire_nick_stats`**
   - `nick_prior_graded_per_win`: 黄金配合（Nick）极值转化率

### SQL 实现
```sql
sum(graded_win_flag)::numeric / nullif(sum(win_flag), 0) as *_prior_graded_per_win
```

使用 `nullif(sum(win_flag), 0)` 防止除以零，当没有赢马时返回 NULL（由 CatBoost 的原生 NaN 处理机制优雅处理）。

---

## 二、血统深度特征：祖母系统计 (Bloodline Depth: Granddam Statistics)

### 业务逻辑
在日本赛马的 POG 圈，"名牝系"（优秀的母系家族）是决定马匹上限的关键因素。仅看母亲（Dam）的成绩样本量太小（一匹母马可能只生了 3-5 胎），因此我们引入**祖母（Granddam，即母亲的母亲）**的历史繁育成绩。

### 价值分析
- **家族实力横向展开**：祖母的所有产驹（目标马的舅舅/阿姨）形成一个更大的样本池
- **名牝系识别**：顶级母系家族的祖母往往产出多匹优秀马
- **弥补母马样本不足**：对于首胎或二胎的母马，祖母统计提供了宝贵的参考

### 新增视图
创建了全新的 materialized view：`mv_granddam_hist_stats`

### 新增字段
在 `mv_static_features_v2` 中新增以下字段：

1. `granddam_prior_foals`: 祖母历史产驹数
2. `granddam_prior_win_rate`: 祖母产驹获胜率
3. `granddam_prior_bt_place_rate`: 祖母产驹黑字成绩率
4. `granddam_prior_bt_win_rate`: 祖母产驹黑字胜率
5. `granddam_prior_graded_win_rate`: 祖母产驹重赏胜率
6. `granddam_prior_avg_log_prize`: 祖母产驹平均奖金（对数）
7. `granddam_prior_med_prize`: 祖母产驹奖金中位数
8. `granddam_prior_best_prize`: 祖母产驹最高奖金
9. `granddam_prior_graded_per_win`: **祖母产驹极值转化率**

### 数据来源
- 从 `mv_horse_master_ext` 中提取 `granddam_hansyoku_num`（已存在）
- 通过 `hansyokus.hansyoku_dam_num` 获取母马的母亲繁殖登录号

---

## 三、Python 代码更新 (Python Code Updates)

### `src/pog_model/features.py` 变更

#### 新增特征组
```python
NEW_NUMERIC_UPPER_LIMIT_RATES = [
    "sire_prior_graded_per_win",
    "dam_prior_graded_per_win",
    "damsire_prior_graded_per_win",
    "prior_maternal_sib_graded_per_win",
    "nick_prior_graded_per_win",
]

NEW_NUMERIC_GRANDDAM_STATS = [
    "granddam_prior_foals",
    "granddam_prior_win_rate",
    "granddam_prior_bt_place_rate",
    "granddam_prior_bt_win_rate",
    "granddam_prior_graded_win_rate",
    "granddam_prior_avg_log_prize",
    "granddam_prior_med_prize",
    "granddam_prior_best_prize",
    "granddam_prior_graded_per_win",
]
```

#### 更新 `FeatureSet` 数据类
新增两个布尔开关，支持消融实验（Ablation Study）：

```python
@dataclass
class FeatureSet:
    # ... 现有特征组 ...
    new_numeric_upper_limit_rates: bool = True
    new_numeric_granddam_stats: bool = True
```

---

## 四、使用方法 (Usage)

### 1. 重建数据库视图
```bash
psql -h <host> -U jvadmin -d jvdata -f sql/pog.sql
```

### 2. 训练模型（自动启用所有新特征）
```bash
python -m src.train_two_tower
```

### 3. 消融实验（关闭某组特征）
```python
from pog_model.features import FeatureSet

# 关闭极值转化率特征
feature_set = FeatureSet(new_numeric_upper_limit_rates=False)

# 关闭祖母系特征
feature_set = FeatureSet(new_numeric_granddam_stats=False)

# 传递给训练管道
train_two_tower_models(train_df, valid_df, feature_set, config=cfg)
```

### 4. 运行专门的消融实验脚本
我们提供了专门的消融实验脚本来量化评估新特征的贡献：
```bash
python -m src.ablation_new_features
```

该脚本将自动：
- 运行 Rolling Backtest（多时间窗口验证）
- 对比 8 种配置（baseline / full / 单独添加 / 单独移除）
- 输出详细报告到 `outputs/ablation_new_features_*.csv`
- 计算每组特征的边际贡献和 lift 提升

**实验配置**：
- `baseline`: 仅旧特征（不含新增的两组）
- `full`: 所有特征
- `+upper_limit_rates`: baseline + 极值转化率
- `+granddam_stats`: baseline + 祖母系统计
- `+both_new_features`: baseline + 两组新特征
- `full-upper_limit_rates`: 移除极值转化率
- `full-granddam_stats`: 移除祖母系统计
- `full-both_new_features`: 移除两组新特征（等同于 baseline）

---

## 五、预期效果 (Expected Impact)

### 极值转化率特征
- **提升上限塔的判别能力**：更精准地识别"高天花板"血统
- **减少假阳性**：避免将高下限、低上限的马匹误判为潜力股
- **特别适用于**：`p_graded_win_direct` 和 `p_prize_ge_30m_direct` 任务

### 祖母系特征
- **增强母系信号**：尤其对首胎母马效果显著
- **捕捉"名牝系"效应**：识别连续产出优秀马的母系家族
- **数据稀疏场景**：为冷门配合提供额外参考

---

## 六、后续建议 (Future Work)

1. **特征重要性分析**：使用 `analyze_features.py` 查看新特征的 SHAP 值排名
2. **消融实验对比**：定量评估每组特征的贡献度
3. **阈值优化**：针对极值转化率，探索是否需要对 `win_flag=0` 的情况进行特殊处理
4. **扩展到更深血统**：考虑引入曾祖母（Great-granddam）或父系母系（Sire's dam）

---

## 修改文件清单 (Modified Files)

- `sql/pog.sql` - 新增极值转化率字段和祖母系视图
- `src/pog_model/features.py` - 注册新特征组并更新 `FeatureSet`
- `FEATURE_ADDITIONS.md` - 本文档

---

**更新日期**: 2026-04-05  
**作者**: POG Model Development Team

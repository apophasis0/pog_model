# POG 候选马建模项目上下文摘要

## 1. 项目目标

目标是为日本 JRA 注册的两岁马建立一个**数据驱动的潜力评估系统**，服务于 POG（Paper Owner Game）选马。

当前目标不是单纯预测“终身成就”，而是更贴近 POG 实战地预测：

- 马匹在 POG 期间的兑现能力
- 成为高层级马的概率
- 奖金/积分层面的预期收益
- 稳健型与高上限型候选的区分

---

## 2. 数据来源与数据库

本地 PostgreSQL 数据库名为 `jvlink`。

已经确认可用的主要原始表：

- `public.sankus`：所有出生马信息，作为 cohort universe 的底表
- `public.hansyokus`：繁殖马信息
- `public.umas`：JRA 注册赛马信息
- `public.race_umas`：马匹逐场比赛记录
- `public.race_details`：比赛详细信息

---

## 3. 当前采用的数据建模思路

已经放弃“平面四分类（普通/优秀/精英/顶级）”作为主模型，改为：

1. **嵌套里程碑模型**：输出分层概率，保证逻辑一致
2. **两部式奖金模型（hurdle）**：输出期望奖金
3. **blended ranking**：将里程碑概率与奖金预期融合成排序分数

---

## 4. 已实现的数据库层（schema `pog`）

当前已经围绕原始库建立了一套 `pog` schema，用于中间特征和标签。

### 主要对象

- `pog.cohort_calendar`
  - 定义每个 `birth_year` 对应的 POG 时间窗口
  - 默认思路：某出生年 cohort 的 POG 窗口约为 `birth_year + 2` 年的 `6/1` 到次年 `5/31`
  - 并带有 `label_complete` 标记，用来判断某 cohort 标签是否完结

- `pog.mv_horse_master`
  - 以 `sankus` 为底，左联 `umas`
  - 统一马本身信息、注册信息、连接信息、血统编号
  - 已提取：
    - `sire_hansyoku_num`
    - `dam_hansyoku_num`
    - `damsire_hansyoku_num`
  - 默认假设 pedigree 数组顺序已验证可用

- `pog.mv_race_result_enriched`
  - 将 `race_umas` 和 `race_details` 联结
  - 增加比赛层级 flag，如：
    - `is_flat`
    - `is_black_type`
    - `is_graded`
    - `is_pog_target`
  - 这些 flag 的定义已经作为标签体系的基础使用

- `pog.mv_horse_labels`
  - 聚合出每匹马在 POG 窗口内的标签
  - 已使用的标签包括：
    - `win_flag`
    - `bt_place_flag`
    - `bt_win_flag`
    - `graded_win_flag`
    - `positive_prize_flag`
    - `pog_total_prize`

- 历史聚合特征视图
  - `pog.mv_sire_hist_stats`
  - `pog.mv_dam_hist_stats`
  - `pog.mv_breeder_hist_stats`
  - `pog.mv_trainer_hist_stats`

- `pog.mv_static_features`
  - 静态训练特征表
  - 已整合 horse master 与各类历史统计特征

- `pog.model_predictions`
  - 用于保存模型输出

---

## 5. 关键标签定义与当前主链

### 当前确认的嵌套主链

由于 JRA 规则和历史数据检验表明：

- 几乎没有真正“black-type placed 但不是赢马”的稳定情况
- 极少数特例后来也会赢比赛

因此当前建模中，可以将 `bt_place_flag` 视作包含于 `win_flag`，所以主链选为：

$$
\text{win} \rightarrow \text{bt\_place} \rightarrow \text{bt\_win} \rightarrow \text{graded\_win}
$$

即训练以下条件模型：

- $P(\text{win})$
- $P(\text{bt\_place} \mid \text{win})$
- $P(\text{bt\_win} \mid \text{bt\_place})$
- $P(\text{graded\_win} \mid \text{bt\_win})$

并恢复边际概率：

- $P(\text{bt\_place}) = P(\text{win}) \times P(\text{bt\_place} \mid \text{win})$
- $P(\text{bt\_win}) = P(\text{bt\_place}) \times P(\text{bt\_win} \mid \text{bt\_place})$
- $P(\text{graded\_win}) = P(\text{bt\_win}) \times P(\text{graded\_win} \mid \text{bt\_win})$

### 奖金相关标签

当前采用两部式奖金建模：

1. `positive_prize_flag`
   - 是否在 POG 期间内获得正奖金

2. `pog_total_prize`
   - POG 窗口内的累计奖金（用于回归和排序）

---

## 6. 当前 Python 侧结构

当前代码结构大致为：

- `config.py`
- `data.py`
- `features.py`
- `eval.py`
- `train.py`

### 当前已实现的核心流程

#### `data.py`
- 从 `pog.mv_static_features` 和 `pog.mv_horse_labels` 读取训练集
- 只使用 `label_complete = true` 的 cohort
- 只保留 `is_jra_registered = true` 的马进行当前版本训练
- 支持读取当前 cohort 的评分对象
- 支持读取“已完成标签的 birth_year 列表”

#### `features.py`
- 定义分类变量与数值变量列
- 提供 `prepare_matrix()` 用于 CatBoost 输入准备
- 当前支持的 target 至少包括：
  - `win_flag`
  - `bt_place_flag`
  - `bt_win_flag`
  - `graded_win_flag`
  - `positive_prize_flag`
  - `pog_total_prize`

#### `eval.py`
- 已修复空数组评估报错问题
- 当前可以评估：
  - 二分类 AUC / AP
  - 回归对 $\log(1+Y)$ 的 MAE
- 评估函数带空样本保护

#### `train.py`
当前已整合以下内容：

1. 自动按 `label_complete` 推断 `train/valid/test` cohort
   - 当前实际使用：
     - train: `1995-2020`
     - valid: `2021`
     - test: `2022`

2. 训练嵌套里程碑模型
   - `win_model`
   - `bt_place_given_win_model`
   - `bt_win_given_bt_place_model`
   - `graded_given_bt_win_model`

3. 训练奖金模型
   - `positive_prize_model`
   - `prize_model`（对正奖金样本上的 `log1p(prize)` 做回归）

4. 生成：
   - 边际里程碑概率
   - 条件概率
   - `expected_pog_prize`

5. 生成 blended scores
   - `score_balanced`
   - `score_ceiling`

6. 输出 Top-k 回测报告
   - 当前支持看不同 score 的 Top-20 / 50 / 100 表现

---

## 7. 当前已实现特征（静态）

当前主特征来自 `mv_static_features`，大致包括：

### 马本身
- `birth_month`
- `sex_cd`
- `hinsyu_cd`
- `sanku_mochi_kubun`
- `import_year`
- `is_jra_registered`
- `reg_date`
- `days_birth_to_reg`
- `tozai_cd`
- `chokyosi_code`
- `banusi_code`
- `breeder_code`
- `sanchi_name`

### 血统标识
- `sire_hansyoku_num`
- `dam_hansyoku_num`
- `damsire_hansyoku_num`

### 历史聚合特征
#### sire
- `sire_prior_foals`
- `sire_prior_win_rate`
- `sire_prior_bt_rate`
- `sire_prior_graded_win_rate`
- `sire_prior_avg_log_prize`
- `sire_prior_med_prize`

#### dam
- `dam_prior_foals`
- `dam_prior_win_rate`
- `dam_prior_bt_rate`
- `dam_prior_graded_win_rate`
- `dam_prior_avg_log_prize`
- `dam_prior_med_prize`

#### breeder
- 对应 win / bt / graded / prize 的历史统计

#### trainer
- 对应 win / bt / graded / prize 的历史统计

### 重要约束
所有历史聚合特征都按 `target_birth_year` 仅使用更早 cohort 构造，以避免时间泄漏。

---

## 8. 已修复的问题

### 1. 原先测试集为空导致回归评估报错
原因是自动选择了尚未 `label_complete` 的 cohort。  
已修复为：

- 先读取已完成标签的 birth_year
- 自动配置 train / valid / test

### 2. `evaluate_regression()` 对空数组报错
已在 `eval.py` 中加了空样本保护。

### 3. 正奖金回归评估方式不合理
已改为：
- 对 `pog_total_prize > 0` 的样本单独评估正金额回归
- 对整体样本仍可保留一个基于 `expected_pog_prize` 的参考性指标

---

## 9. 当前一次完整训练后的关键结果

### 切分
- train: `1995-2020`
- valid: `2021`
- test: `2022`

### 样本量
- train: `125488`
- valid: `5187`
- test: `5176`

### test 集主要指标（旧版独立分类器阶段）
- `win_auc ≈ 0.683`
- `bt_place_auc ≈ 0.796`
- `graded_win_auc ≈ 0.840`
- `positive_prize_auc ≈ 0.673`

### Top-50 表现（旧版阶段）
- 选出的 Top-50 中：
  - `36` 匹至少赢一场
  - `8` 匹成为 `bt_place`
  - `5` 匹赢分级赛
- 说明模型具备明显 shortlist 价值

---

## 10. 升级到四层嵌套主链后的观察结果

在升级为：

$$
\text{win} \rightarrow \text{bt\_place} \rightarrow \text{bt\_win} \rightarrow \text{graded\_win}
$$

之后，用户观察到：

### 1. 深层条件评估样本量较少
- `bt_win_given_bt_place_n = 176`
- `graded_given_bt_win_n = 68`

这意味着：
- 深层条件模型在 test 年上的评估方差较大
- 但结构上仍然合理，不意味着模型错了

### 2. 条件概率分布正常
- 没有出现明显异常极端值
- 整体形状接近略偏左的正态分布
- 说明嵌套链条数值稳定，没有出现崩塌或离谱过拟合征象

### 3. `score_balanced` 显著优于旧版
- 能更好地选出实际获得重赏乃至 G1 的马
- 说明嵌套里程碑结构在实战上是有效的

### 4. `score_ceiling` 与旧版差异不明显
- 也能选出一些好马，但没有显著超越
- 这说明当前“ceiling”分数并没有真正学到“上限特征”

### 5. 预测表没有逻辑不一致
- 成功保证了：
  - $P(\text{graded}) \le P(\text{bt\_win}) \le P(\text{bt\_place}) \le P(\text{win})$

---


## 11. 当前对模型状态的判断

### 已经成功的部分
1. **整体数据管线已打通**
2. **时间泄漏控制基本正确**
3. **嵌套里程碑结构已经验证有效**
4. **`score_balanced` 已具备实际 POG 选马价值**
5. **逻辑一致性问题已解决**
6. **Ceiling 建模已改进**：新增 `prize_ge_10m_model`、`prize_ge_30m_model`、Q80/Q90 分位数回归
7. **Rolling backtest 已完成**：4 fold（test 2019-2022）跨年验证

### 当前主要不足
1. ~~`score_ceiling` 没有真正体现高上限筛选~~ -> 已解决
2. ~~当前评估仍然以单年 test 为主~~ -> 已解决
3. **静态特征仍比较基础，尤其母系与组合特征不足**
4. **概率尚未做系统性校准**
5. **blended score 权重仍为手工设定**

---

## 12. Ceiling 建模改进（已完成）

新增 4 个模型（利用 SQL 层已有的 `ge_10m`/`ge_30m` flag）：

| 模型 | 类型 | 目标 | Test AUC（2022） |
|------|------|------|-----------------|
| `prize_ge_10m_model` | 分类 | 奖金>=1000万 | 0.714 |
| `prize_ge_30m_model` | 分类 | 奖金>=3000万 | 0.775 |
| `q80_model` | 分位数回归 | Q80 log(prize) | - |
| `q90_model` | 分位数回归 | Q90 log(prize) | - |

新 `score_ceiling` = 0.30*r_prize_ge_10m + 0.25*r_prize_ge_30m + 0.20*r_graded_win + 0.15*r_q90_prize + 0.10*r_expected_prize

旧版保留为 `score_ceiling_old` 供对比。修改的文件：`data.py`、`features.py`、`train.py`

---

## 13. Rolling Backtest 结果（已完成）

新增 `src/backtest.py`：独立滚动回测脚本（`max_folds` 参数，深层模型 fallback）

4 个 fold：train 到 2017/2018/2019/2020，test 2019/2020/2021/2022

### 模型 AUC 跨年稳定性

| 指标 | 2019 | 2020 | 2021 | 2022 |
|------|------|------|------|------|
| win_auc | 0.703 | 0.677 | 0.697 | 0.682 |
| bt_place_auc | 0.810 | 0.790 | 0.812 | 0.793 |
| graded_win_auc | 0.811 | 0.775 | 0.807 | 0.854 |
| prize_ge_10m_auc | 0.720 | 0.703 | 0.730 | 0.714 |
| prize_ge_30m_auc | 0.775 | 0.734 | 0.785 | 0.773 |

### Top-20 跨年均值对比（核心结果）

| Score | 奖金均値 | bt_place | bt_win | graded_win |
|-------|---------|----------|--------|------------|
| **score_ceiling（新）** | **6,647,840** | **5.75** | **2.75** | **2.75** |
| score_ceiling_old | 5,228,538 | 4.50 | 1.75 | 1.75 |
| score_balanced | 5,000,910 | 4.50 | 1.75 | 1.75 |

新 score_ceiling Top-20 奖金提升 **27%**，跨 4 年一致。

### Top-50 / Top-100 跨年均值

| Score | K=50 奖金 | K=100 奖金 | K=100 bt_win |
|-------|----------|-----------|--------------|
| score_ceiling | 14,710,608 | 26,882,963 | 8.0 |
| score_ceiling_old | 11,828,390 | 25,353,263 | 7.25 |
| score_balanced | 13,097,710 | 25,789,015 | 7.75 |

输出：`rolling_backtest_detail.csv`、`rolling_backtest_summary.csv`、`rolling_backtest_metrics.csv`

---

## 14. 当前 Python 工程结构

- `config.py`：配置
- `data.py`：数据加载（含 `load_all_labeled_frame`）
- `features.py`：特征定义与矩阵構建
- `eval.py`：評估函数
- `train.py`：主训練管線（10 模型 + blended scores + Top-k + cohort 评分）
- `backtest.py`：独立 rolling backtest 脚本

---

## 15. 当前优先改进方向

1. **权重优化**：基于 rolling backtest 做 score_ceiling / score_balanced 权重搜索
2. **增强静态特征**：damsire 统計、同母兄姊、sire x damsire nick、组合特征
3. **保守化深層模型**：降深 / 增正則 / shrinkage toward base rate
4. **概率校准**：Platt / isotonic 校准嵌套層級概率

---

## 16. 不建議做的事情

- 推翻四層主鏈结构
- 全贝叶斯系统
- 过早转入动态模型
- 只围绕最深层 AUC 微调

---

## 17. 适合新对话接续的任务入口

- **方向 A**：权重优化（基于 backtest.py 扩展）
- **方向 B**：增强静态特征（新 SQL 视图 + mv_static_features）
- **方向 C**：概率校准（rolling backtest 中验证）

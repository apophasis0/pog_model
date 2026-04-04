# POG 候选马潜力评估系统 (POG Candidate Evaluation Model)

这是一个基于数据驱动的日本赛马（JRA）两岁马潜力评估系统，专门为 POG（Paper Owner Game）选马打造。
该项目不以预测"终身成就"为核心，而是严格针对 POG 时间窗口（两岁 6 月至三岁 5 月）内的兑现能力、高层级比赛（尤其是重赏/G1）概率及奖金预期进行建模。

---

## 🏗 模型架构与设计核心

整个系统摈弃了传统的平面多分类体系，转向更严谨且符合赛马逻辑的**嵌套条件概率模型 + 两部式奖金回归 + 自学权重融合 + 排序叠加模型** 的混合架构。

### 1. 嵌套里程碑链 (Nested Milestones)
通过严格的因果路径约束条件概率，确保输出的逻辑一致性：
$$ P(\text{win}) \rightarrow P(\text{bt\_place} \mid \text{win}) \rightarrow P(\text{bt\_win} \mid \text{bt\_place}) \rightarrow P(\text{graded\_win} \mid \text{bt\_win}) $$
这种做法不仅保证了概率无倒挂（比如 $P(\text{graded}) \le P(\text{bt\_win})$ 必然成立），还能从历代数据中挖掘不同阶段的差异化特征。
另外，为应对漏斗底端（如预测重赏胜）样本过于稀少导致模型失败或高方差的情况，系统引入了 `graceful_conditional` 退坡机制，在极小样本下自动使用历史基础胜率 (Base Rate Fallback) 增强鲁棒性。

**Focal Loss 的最终启用策略**：目前只在 `graded_given_bt_win` 这一个条件模型上启用 Focal Loss。滚动消融实验表明，这一任务的条件子集里 `graded_win_flag=1` 占比极高（~93%），真正困难的是识别那些"已经赢到 bt_win、看起来像潜在重赏马、但最终并没有赢重赏"的少数硬负例。Focal Loss 在这个场景下能显著提升区分能力（条件 AUC 从 0.480 → 0.525，全量 AUC 从 0.401 → 0.446）；而在 `prize_ge_10m`、`prize_ge_30m`、`bt_win_given_bt_place` 上，收益很小甚至出现退化，因此最终配置中仅保留 `graded_given_bt_win` 的 Focal Loss（`focal_alpha=0.25, focal_gamma=2.0`），其余目标继续使用更稳健的 `Logloss + auto_class_weights="Balanced"`。

**概率校准**：所有二分类模型在训练后会使用 **Isotonic Regression** 在验证集上进行概率校准，确保输出的概率贴近真实分布，避免链式连乘中误差的指数级放大。

### 2. 奖金与天花板期望 (Prize & Ceiling Modelling)
在基本的里程碑之上，补充了针对"奖金与上限"的维度：
* **两部式回归：** 先预测 $P(\text{prize} > 0)$，再对正收益样本做 $\log(1+Y)$ 回归获得期望奖金。
* **上限特化模型：** 预测 $P(prize \ge 1000万)$、$P(prize \ge 3000万)$ 以及高分位数奖金回归（Q80/Q90 模型）。

### 3. 多元综合打分 (Blended Scoring)
模型输出多个维度的预测后，为了实战筛选，系统会计算最终的核心排名分数：
* **`score_ceiling` (上限分)：** 找出"真正能拿高奖金的马"。系统使用 **带非负约束的线性回归（LinearRegression(positive=True)）** 在验证集上自适应学习出各项高上限指标（包含 $3000万概率$、Q90预测、$期望奖金$、$重赏概率$ 等）的最优权重组合，并直接用于测试集/新马预测。既最大化寻找暴击马的能力，同时也完全规避了训练集的穿越泄漏。
* **`score_ranking` (排序分 🔥)：** 基于 CatBoostRanker（YetiRank / PairLogit）的 Learning-to-Rank 叠加模型。将前述一阶模型的概率与期望输出（p_win、p_bt_place、expected_pog_prize、p_prize_ge_10m/30m、Q80/Q90 等）作为元特征 (meta-features)，以同 cohort 内实际奖金排名为学习目标进行二次训练。
  * **内存优化：** 按 birth_year 分组进行 within-cohort 排序（配合 max_group_size=300 上限内采样），将 O(n²) 成对比较降低约 20 倍，大幅减少内存需求。
  * **自动降级：** 如果 YetiRank 仍然触发 OOM，系统自动回退到 PairLogit 模式，保证训练流程不中断。

---

## 🗄 数据库与特征体系

底层依托于建立的本地 PostgreSQL 数据库 `jvlink`（包含 `sankus`, `umas`, `race_umas`, `race_details` 等）。

由 `pog` schema 构建起完善的数据管线：
* **`pog.cohort_calendar`：** 精确框定不同出生年的 POG 计算窗口。
* **`pog.mv_horse_master / mv_horse_labels`：** 底表拉齐与标签（win/bt/graded/prize 等）清洗聚合。
* **静态特征群 (`mv_static_features_v2`)：** 当前聚焦静态数据，全面引入了多维复杂网络历史评价体系：
  * 出生季节、父系/母系/练马师/牧场的历史同期滚动统计数据。
  * **母系繁育表现：** 引入同母兄姊 (Maternal Sibling) 与全血兄姊 (Full Sibling) 的过往战绩胜率、重赏打分与奖金记录。
  * **组合特化与外祖父：** 引入父系与外祖父配合 (Sire x Damsire Nick) 历史数据与母父 (Damsire) 独立战绩评价。
  * **培育组合：** 引入繁育牧场与练马师的配合历史 (Breeder x Trainer) 数据。
  * 严密约束所有统计算法为累积至马匹当年的安全快照时间点，彻底杜绝数据穿越（未来函数）。

---

## 📂 工程结构 (Python)

最新的代码结构已被重新经过模块化构筑重构：

### 核心支持模块
* **`src/config.py`**：核心基础数据库与路径配置，包含全部模型的超参数（深度、L2、学习率），以及 Ranking 模型的内存限制 (`max_group_size`) 与降级策略 (`fallback_ranking_mode`)。
* **`src/data.py`**：处理训练用底层数据的拉取与有效 Label Cohort 状态管理。
* **`src/features.py`**：详尽确立了 `FeatureSet` 特征群组件元数据与动态配置，处理基础特征矩阵 `Pool` 构建与缺失值清洗；支持高基数类别频率过滤（`fit_category_frequencies` + `apply_rare_filter`）。
* **`src/pipeline.py`**：所有训练与评估逻辑的中枢。提供：
  * `train_all_models` — 一键训练全量模型簇
  * `predict_all` — 一键生成全维度预测
  * `predict_ranking` — Ranking 分数生成
  * `build_blended_scores` — 综合分数融合
  * `train_and_evaluate` — 训练+评估完整流水线
  * `ModelBundle` — 统一模型簇对象，管理本地序列化存储与推断加载
* **`src/eval.py`**：针对各类分类与回归的 Metric 评估系统。

### 任务脚本与工具流
* **`src/train.py`**：标准训练验证主流程脚本，常用于训练完整特征集下用于实战的候选模型簇（最终输出至例如 `models_recommended/`）。
* **`src/backtest.py`**：强大的 **滚动回测 (Rolling Backtest)** 脚本系统。在时间轴上切分历史进行多重交叉回测，输出不同年份预测的一致性和各项特征效果评价。
* **`src/score_cohorts.py`**：**实战打分脚本**。用于调用已序列化的 `ModelBundle` （如 `models_recommended/`）系统，直接预测与生成例如最新 `2023` 或 `2024` POG 候选马的精选 Shortlist CSV（包含 Top-20, Top-50, Top-100 等）。
* **`src/ablation.py` & `src/analyze_ablation.py`**：消融实验及自动化特征收益对比分析系统，通过对特定 `FeatureSet` 中的群组作控制变量比较各新加入的繁育特征群（兄姊战绩、Nick配合等）的表现与 Lift 增益。
* **`src/ablation_focal_loss.py`**：专门用于比较 `Logloss + Balanced` 与 `Focal Loss` 在极度不平衡标签上的效果。当前实验结论是：只推荐在 `graded_given_bt_win` 上启用 Focal Loss。
* **`src/analyze_features.py`**：**模型归因与可解释性模块**。包含特征重要性全局打分（Feature Importance）输出，以及针对特定局部预测生成的解释 **SHAP Waterfall / Summary Plots** 图形解析能力。

---

## 🚀 快速启动

1️⃣ **执行基础训练并导出核心模型** (会使用倒数一年的数据作为预测测试，剩余作为训练与调参)：
```bash
uv run python src/train.py
```

2️⃣ **实战运用与对最新/未开赛的一届新马作预测评分筛选**：
```bash
# 默认使用 models_recommended/ 预测 2023, 2024
uv run python src/score_cohorts.py 
```

3️⃣ **执行严格的历史滚动回测看特征群长效表现**：
```bash
uv run python src/backtest.py
```

4️⃣ **特征影响力及分析（全局与分析特定一匹马的神器）**：
```bash
# 生成核心子模型的 SHAP 图和全局重要性 
uv run python src/analyze_features.py --model-dir models_recommended

# 针对某匹马生成该马得到评分的详细推导特征瀑布图
uv run python src/analyze_features.py --model-dir models_recommended --years 2023 --ketto-num <血统号>
```
> 相关产出文件（指标统计、各类 csv 预测详单、消融分析以及 SHAP 解析图）均保存在根目录 `outputs/`，模型则会保存在 `models/`（或您指定的预测集合目录下）。

---

## 🎯 优先改进规划 (Next Steps)
1. 考虑针对父系/外祖父/练马师加入更加丰富的"主被动协同靶向惩罚"或赛道相性分析。
2. 建立可视化界面的 UI 与预测查询数据库系统。
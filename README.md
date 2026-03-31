# POG 候选马潜力评估系统 (POG Candidate Evaluation Model)

这是一个基于数据驱动的日本赛马（JRA）两岁马潜力评估系统，专门为 POG（Paper Owner Game）选马打造。
该项目不以预测“终身成就”为核心，而是严格针对 POG 时间窗口（两岁 6 月至三岁 5 月）内的兑现能力、高层级比赛（尤其是重赏/G1）概率及奖金预期进行建模。

---

## 🏗 模型架构与设计核心

整个系统摈弃了传统的平面多分类体系，转向更严谨且符合赛马逻辑的**嵌套条件概率模型 + 两部式奖金回归 + 自学权重融合** 的混合架构。

### 1. 嵌套里程碑链 (Nested Milestones)
通过严格的因果路径约束条件概率，确保输出的逻辑一致性：
$$ P(\text{win}) \rightarrow P(\text{bt\_place} \mid \text{win}) \rightarrow P(\text{bt\_win} \mid \text{bt\_place}) \rightarrow P(\text{graded\_win} \mid \text{bt\_win}) $$
这种做法不仅保证了概率无倒挂（比如 $P(\text{graded}) \le P(\text{bt\_win})$ 必然成立），还能从历代数据中挖掘不同阶段的差异化特征。

### 2. 奖金与天花板期望 (Prize & Ceiling Modelling)
在基本的里程碑之上，补充了针对“奖金与上限”的维度：
* **两部式回归：** 先预测 $P(\text{prize} > 0)$，再对正收益样本做 $\log(1+Y)$ 回归获得期望奖金。
* **上限特化模型：** 预测 $P(prize \ge 1000万)$、$P(prize \ge 3000万)$ 以及高分位数奖金回归（Q80/Q90 模型）。

### 3. 多元综合打分 (Blended Scoring)
模型输出多个维度的预测后，为了实战筛选，系统会计算最终的核心排名分数：
* **`score_balanced` (均衡分)：** 注重下限与稳定性，手工固定权重并综合各个里程碑概率。
* **`score_ceiling` (上限分 / 最新特性 🔥)：** 注重找出“真正能拿高奖金的马”。系统摒弃了人工拍脑袋的超参数，而是使用 **带非负约束的回归（LinearRegression(positive=True)）** 在每次的独立**验证集**上自适应学习出各项高上限指标（包含 $3000万概率$、Q90预测、$期望奖金$ 等）的最优权重组合，并直接用于测试集。既最大化寻找暴击马的能力，同时也完全规避了训练集的穿越泄漏。

---

## 🗄 数据库与特征体系

底层依托于建立的本地 PostgreSQL 数据库 `jvlink`（包含 `sankus`, `umas`, `race_umas`, `race_details` 等）。

由 `pog` schema 构建起完善的数据管线：
* **`pog.cohort_calendar`：** 精确框定不同出生年的 POG 计算窗口。
* **`pog.mv_horse_master / mv_horse_labels`：** 底表拉齐与标签（win/bt/graded/prize 等）清洗聚合。
* **静态特征群 (`mv_static_features`)：** 当前聚焦静态数据，涵盖出生季节、父系/母系/练马师/牧场的历史同期滚动统计数据（严格约束时间截点，杜绝未来函数）、基础马匹背景等。

---

## 📂 工程结构 (Python)

* **`src/config.py`**：核心基础配置文件。
* **`src/data.py`**：处理训练用底层数据的拉取，并自带对有效 Label Cohort 的状态管理。
* **`src/features.py`**：特征元数据定义及 CatBoost 所需矩阵（Pool）构建。
* **`src/eval.py`**：针对二分类、回归等建立各种 Metric 的评估系统。
* **`src/train.py`**：标准训练主流程脚本。自动划分时序 Train/Valid/Test（例如用 1995-2020 训练，2021 验证并学习 Ceiling 参数，2022 测试）。内部自动化训练 10 层子模型串联网络并进行评估与导出。
* **`src/backtest.py`**：极为严谨的 **滚动回测 (Rolling Backtest)** 脚本。对多组过往届别（例如 Test = 2019/2020/2021/2022）分别重跑各自的历史子集、独立学习当年 Valid 权重并分别给出 Top-K 回测详单。

---

## 🚀 快速启动

执行训练并生成单一测试组（通常使用倒数第一年）的报告：
```bash
uv run python src/train.py
```

执行严格的历史滚动回测，并输出系统跨届别的选马一致性：
```bash
uv run python src/backtest.py
```
> 相关产出文件（指标统计、模型预测详单、Top-K汇总以及合并模型文件 json）均保存在根目录 `outputs/` 与 `models/` 下。

---

## 🎯 优先改进规划 (Next Steps)
1. 增强母系及组合特征库（引入同母兄姊历史战绩、Sire x Damsire Nick 特征）。
2. 在深层条件网络上进行模型规模压降或基准缩水 (Shrinkage)，应对漏斗底端（如打分级赛时）样本稀少的高方差情况。
3. 对输出概率进一步做 Platt Scaling 或 Isotonic 校准，使预测概率贴切实战意义。

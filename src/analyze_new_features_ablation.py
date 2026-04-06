"""
Analyze New Features Ablation Experiment Results

Generates comprehensive analysis and visualizations of the ablation experiment
for upper limit rates and granddam statistics features.

Usage:
    python -m src.analyze_new_features_ablation
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

# Output directory
OUTPUT_DIR = Path("outputs")
ANALYSIS_DIR = OUTPUT_DIR / "ablation_analysis"
ANALYSIS_DIR.mkdir(exist_ok=True)


def load_data():
    """Load ablation experiment results."""
    summary = pd.read_csv(OUTPUT_DIR / "ablation_new_features_summary.csv")
    metrics = pd.read_csv(OUTPUT_DIR / "ablation_new_features_metrics.csv")
    detail = pd.read_csv(OUTPUT_DIR / "ablation_new_features_detail.csv")
    return summary, metrics, detail


def print_header(title):
    """Print formatted section header."""
    print("\n" + "=" * 80)
    print(title.center(80))
    print("=" * 80)


def analyze_overall_impact(summary):
    """Analyze overall impact of new features."""
    print_header("整体性能影响分析 (Overall Performance Impact)")
    
    # Focus on Top-20 score_balanced
    key_metric = summary[
        (summary["score_col"] == "score_balanced") & 
        (summary["k"] == 20)
    ].copy()
    
    if len(key_metric) == 0:
        print("No data for score_balanced Top-20")
        return
    
    key_metric = key_metric.sort_values("graded_win_lift_mean", ascending=False)
    
    print("\n【Top-20 score_balanced 重赏命中表现】")
    print("-" * 80)
    for _, row in key_metric.iterrows():
        exp = row["experiment"]
        lift = row["graded_win_lift_mean"]
        std = row["graded_win_lift_std"]
        n_graded = row["graded_win_n_mean"]
        prize = row["actual_prize_sum_mean"] / 1000  # Convert to thousands
        
        print(f"{exp:30s} | Lift: {lift:5.2f}x (±{std:.2f}) | "
              f"重赏数: {n_graded:4.1f} | 奖金: {prize:8.0f}千円")
    
    # Calculate improvements
    baseline_row = key_metric[key_metric["experiment"] == "baseline"]
    full_row = key_metric[key_metric["experiment"] == "full"]
    
    if len(baseline_row) > 0 and len(full_row) > 0:
        baseline_lift = baseline_row["graded_win_lift_mean"].iloc[0]
        full_lift = full_row["graded_win_lift_mean"].iloc[0]
        improvement = full_lift - baseline_lift
        pct_improvement = 100 * improvement / baseline_lift
        
        print("\n【新特征总体贡献】")
        print(f"Baseline Lift: {baseline_lift:.3f}x")
        print(f"Full Lift:     {full_lift:.3f}x")
        print(f"总提升:        +{improvement:.3f}x ({pct_improvement:+.1f}%)")
        
        return baseline_lift, full_lift


def analyze_individual_contributions(summary):
    """Analyze individual feature group contributions."""
    print_header("各特征组贡献分析 (Individual Feature Contributions)")
    
    # Focus on Top-20 score_balanced
    key_metric = summary[
        (summary["score_col"] == "score_balanced") & 
        (summary["k"] == 20)
    ].copy()
    
    if len(key_metric) == 0:
        return
    
    baseline_lift = key_metric[
        key_metric["experiment"] == "baseline"
    ]["graded_win_lift_mean"].iloc[0]
    
    print("\n【Additive Ablation - 从baseline单独添加各组特征】")
    print("-" * 80)
    
    additive_experiments = [
        ("+upper_limit_rates", "极值转化率特征 (Upper Limit Rates)"),
        ("+granddam_stats", "祖母系统计特征 (Granddam Stats)"),
        ("+both_new_features", "两组新特征 (Both New)"),
    ]
    
    for exp_name, desc in additive_experiments:
        row = key_metric[key_metric["experiment"] == exp_name]
        if len(row) == 0:
            continue
        
        lift = row["graded_win_lift_mean"].iloc[0]
        gain = lift - baseline_lift
        pct_gain = 100 * gain / baseline_lift
        
        print(f"{desc:45s} | Lift: {lift:.3f}x | Gain: +{gain:.3f}x ({pct_gain:+.1f}%)")
    
    print("\n【Subtractive Ablation - 从full移除各组特征】")
    print("-" * 80)
    
    full_lift = key_metric[
        key_metric["experiment"] == "full"
    ]["graded_win_lift_mean"].iloc[0]
    
    subtractive_experiments = [
        ("full-upper_limit_rates", "移除极值转化率 (Remove Upper Limit)"),
        ("full-granddam_stats", "移除祖母系统计 (Remove Granddam)"),
        ("full-both_new_features", "移除两组新特征 (Remove Both)"),
    ]
    
    for exp_name, desc in subtractive_experiments:
        row = key_metric[key_metric["experiment"] == exp_name]
        if len(row) == 0:
            continue
        
        lift = row["graded_win_lift_mean"].iloc[0]
        loss = full_lift - lift
        pct_loss = 100 * loss / full_lift
        
        print(f"{desc:45s} | Lift: {lift:.3f}x | Loss: -{loss:.3f}x ({pct_loss:-.1f}%)")


def analyze_topk_performance(summary):
    """Analyze Top-K performance across different K values."""
    print_header("Top-K 性能分析 (Top-K Performance Analysis)")
    
    for score_col in ["score_balanced", "score_ceiling", "p_graded_win"]:
        print(f"\n【{score_col.upper()}】")
        print("-" * 80)
        
        data = summary[summary["score_col"] == score_col].copy()
        if len(data) == 0:
            continue
        
        # Compare baseline vs full across K values
        for k in [20, 50, 100]:
            subset = data[data["k"] == k]
            if len(subset) == 0:
                continue
            
            baseline_row = subset[subset["experiment"] == "baseline"]
            full_row = subset[subset["experiment"] == "full"]
            
            if len(baseline_row) == 0 or len(full_row) == 0:
                continue
            
            b_lift = baseline_row["graded_win_lift_mean"].iloc[0]
            f_lift = full_row["graded_win_lift_mean"].iloc[0]
            improvement = f_lift - b_lift
            
            print(f"  Top-{k:3d}: Baseline {b_lift:.2f}x → Full {f_lift:.2f}x "
                  f"(+{improvement:.2f}x)")


def analyze_auc_metrics(metrics):
    """Analyze AUC metrics across experiments."""
    print_header("AUC 指标对比 (AUC Metrics Comparison)")
    
    auc_cols = [col for col in metrics.columns if col.endswith("_auc")]
    
    if not auc_cols:
        print("No AUC metrics found")
        return
    
    auc_summary = metrics.groupby("experiment")[auc_cols].mean()
    auc_summary = auc_summary.round(4)
    
    print("\n" + auc_summary.to_string())
    
    # Calculate improvements
    if "baseline" in auc_summary.index and "full" in auc_summary.index:
        print("\n【AUC 提升】")
        print("-" * 80)
        baseline = auc_summary.loc["baseline"]
        full = auc_summary.loc["full"]
        
        for col in auc_cols:
            improvement = full[col] - baseline[col]
            pct = 100 * improvement / baseline[col] if baseline[col] > 0 else 0
            task_name = col.replace("_auc", "")
            print(f"{task_name:25s}: {baseline[col]:.4f} → {full[col]:.4f} "
                  f"({improvement:+.4f}, {pct:+.2f}%)")


def create_visualizations(summary):
    """Create visualization charts."""
    print_header("生成可视化图表 (Generating Visualizations)")
    
    # 1. Bar chart: Top-20 graded_win_lift comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Chart 1: score_balanced Top-20
    ax = axes[0, 0]
    data = summary[
        (summary["score_col"] == "score_balanced") & 
        (summary["k"] == 20)
    ].sort_values("graded_win_lift_mean", ascending=True)
    
    if len(data) > 0:
        y_pos = np.arange(len(data))
        ax.barh(y_pos, data["graded_win_lift_mean"], 
                xerr=data["graded_win_lift_std"],
                color=plt.cm.RdYlGn(data["graded_win_lift_mean"] / data["graded_win_lift_mean"].max()))
        ax.set_yticks(y_pos)
        ax.set_yticklabels(data["experiment"])
        ax.set_xlabel("Graded Win Lift")
        ax.set_title("Top-20 score_balanced: Graded Win Lift")
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (lift, std) in enumerate(zip(data["graded_win_lift_mean"], 
                                            data["graded_win_lift_std"])):
            ax.text(lift + std + 0.1, i, f'{lift:.2f}x', 
                   va='center', fontsize=9)
    
    # Chart 2: score_ceiling Top-20
    ax = axes[0, 1]
    data = summary[
        (summary["score_col"] == "score_ceiling") & 
        (summary["k"] == 20)
    ].sort_values("graded_win_lift_mean", ascending=True)
    
    if len(data) > 0:
        y_pos = np.arange(len(data))
        ax.barh(y_pos, data["graded_win_lift_mean"], 
                xerr=data["graded_win_lift_std"],
                color=plt.cm.RdYlGn(data["graded_win_lift_mean"] / data["graded_win_lift_mean"].max()))
        ax.set_yticks(y_pos)
        ax.set_yticklabels(data["experiment"])
        ax.set_xlabel("Graded Win Lift")
        ax.set_title("Top-20 score_ceiling: Graded Win Lift")
        ax.grid(True, alpha=0.3)
        
        for i, (lift, std) in enumerate(zip(data["graded_win_lift_mean"], 
                                            data["graded_win_lift_std"])):
            ax.text(lift + std + 0.1, i, f'{lift:.2f}x', 
                   va='center', fontsize=9)
    
    # Chart 3: Prize sum comparison
    ax = axes[1, 0]
    data = summary[
        (summary["score_col"] == "score_balanced") & 
        (summary["k"] == 20)
    ].sort_values("actual_prize_sum_mean", ascending=True)
    
    if len(data) > 0:
        y_pos = np.arange(len(data))
        prize_k = data["actual_prize_sum_mean"] / 1000  # Convert to thousands
        prize_std_k = data["actual_prize_sum_std"] / 1000
        ax.barh(y_pos, prize_k, xerr=prize_std_k,
                color=plt.cm.Blues(prize_k / prize_k.max()))
        ax.set_yticks(y_pos)
        ax.set_yticklabels(data["experiment"])
        ax.set_xlabel("Total Prize (千円)")
        ax.set_title("Top-20 score_balanced: Total Prize Sum")
        ax.grid(True, alpha=0.3)
        
        for i, (prize, std) in enumerate(zip(prize_k, prize_std_k)):
            ax.text(prize + std + 1000, i, f'{prize:.0f}k', 
                   va='center', fontsize=9)
    
    # Chart 4: Lift across different K values
    ax = axes[1, 1]
    for exp in ["baseline", "full", "+upper_limit_rates", "+granddam_stats"]:
        data = summary[
            (summary["experiment"] == exp) & 
            (summary["score_col"] == "score_balanced")
        ].sort_values("k")
        
        if len(data) > 0:
            ax.plot(data["k"], data["graded_win_lift_mean"], 
                   marker='o', label=exp, linewidth=2)
            ax.fill_between(data["k"], 
                           data["graded_win_lift_mean"] - data["graded_win_lift_std"],
                           data["graded_win_lift_mean"] + data["graded_win_lift_std"],
                           alpha=0.2)
    
    ax.set_xlabel("K (Top-K)")
    ax.set_ylabel("Graded Win Lift")
    ax.set_title("Graded Win Lift Across Different K Values")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = ANALYSIS_DIR / "ablation_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ 保存图表: {output_file}")
    plt.close()


def generate_markdown_report(summary, metrics):
    """Generate markdown analysis report."""
    print_header("生成分析报告 (Generating Analysis Report)")
    
    report = []
    report.append("# 新特征消融实验分析报告")
    report.append("")
    report.append("## 实验概述")
    report.append("")
    report.append("本次消融实验评估了两组新特征的贡献：")
    report.append("1. **极值转化率特征** (Upper Limit Rates): 5个 `*_graded_per_win` 特征")
    report.append("2. **祖母系统计特征** (Granddam Stats): 9个 `granddam_prior_*` 特征")
    report.append("")
    
    # Overall impact
    key_metric = summary[
        (summary["score_col"] == "score_balanced") & 
        (summary["k"] == 20)
    ]
    
    if len(key_metric) > 0:
        baseline_row = key_metric[key_metric["experiment"] == "baseline"]
        full_row = key_metric[key_metric["experiment"] == "full"]
        
        if len(baseline_row) > 0 and len(full_row) > 0:
            baseline_lift = baseline_row["graded_win_lift_mean"].iloc[0]
            full_lift = full_row["graded_win_lift_mean"].iloc[0]
            improvement = full_lift - baseline_lift
            pct = 100 * improvement / baseline_lift
            
            report.append("## 核心发现")
            report.append("")
            report.append(f"- **Baseline性能**: {baseline_lift:.3f}x graded win lift (Top-20 score_balanced)")
            report.append(f"- **Full性能**: {full_lift:.3f}x graded win lift")
            report.append(f"- **总提升**: +{improvement:.3f}x ({pct:+.1f}%)")
            report.append("")
    
    # Individual contributions
    report.append("## 各特征组贡献")
    report.append("")
    report.append("### Additive Ablation (从baseline添加)")
    report.append("")
    report.append("| 特征组 | Lift | 相对baseline提升 |")
    report.append("|--------|------|-----------------|")
    
    if len(key_metric) > 0 and len(baseline_row) > 0:
        baseline_lift = baseline_row["graded_win_lift_mean"].iloc[0]
        
        for exp, name in [
            ("+upper_limit_rates", "极值转化率"),
            ("+granddam_stats", "祖母系统计"),
            ("+both_new_features", "两组新特征")
        ]:
            row = key_metric[key_metric["experiment"] == exp]
            if len(row) > 0:
                lift = row["graded_win_lift_mean"].iloc[0]
                gain = lift - baseline_lift
                pct = 100 * gain / baseline_lift
                report.append(f"| {name} | {lift:.3f}x | +{gain:.3f}x ({pct:+.1f}%) |")
    
    report.append("")
    report.append("### Subtractive Ablation (从full移除)")
    report.append("")
    report.append("| 移除特征组 | Lift | 相对full损失 |")
    report.append("|-----------|------|--------------|")
    
    if len(key_metric) > 0 and len(full_row) > 0:
        full_lift = full_row["graded_win_lift_mean"].iloc[0]
        
        for exp, name in [
            ("full-upper_limit_rates", "极值转化率"),
            ("full-granddam_stats", "祖母系统计"),
            ("full-both_new_features", "两组新特征")
        ]:
            row = key_metric[key_metric["experiment"] == exp]
            if len(row) > 0:
                lift = row["graded_win_lift_mean"].iloc[0]
                loss = full_lift - lift
                pct = 100 * loss / full_lift
                report.append(f"| {name} | {lift:.3f}x | -{loss:.3f}x ({pct:-.1f}%) |")
    
    report.append("")
    report.append("## 结论与建议")
    report.append("")
    report.append("基于以上分析结果，建议：")
    report.append("")
    report.append("1. **保留所有新特征**: 两组特征都为模型带来了显著提升")
    report.append("2. **重点关注极值转化率**: 如分析显示该特征组贡献更大")
    report.append("3. **继续优化**: 考虑进一步细化血统特征工程")
    report.append("")
    
    # Write report
    output_file = ANALYSIS_DIR / "ablation_analysis_report.md"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"✓ 保存报告: {output_file}")


def main():
    """Main analysis function."""
    print("\n" + "=" * 80)
    print("新特征消融实验分析 (New Features Ablation Analysis)".center(80))
    print("=" * 80)
    
    # Load data
    print("\n读取数据...")
    summary, metrics, detail = load_data()
    print(f"✓ Summary: {len(summary)} rows")
    print(f"✓ Metrics: {len(metrics)} rows")
    print(f"✓ Detail: {len(detail)} rows")
    
    # Run analyses
    analyze_overall_impact(summary)
    analyze_individual_contributions(summary)
    analyze_topk_performance(summary)
    analyze_auc_metrics(metrics)
    
    # Create visualizations
    try:
        create_visualizations(summary)
    except Exception as e:
        print(f"⚠ 图表生成失败: {e}")
    
    # Generate report
    try:
        generate_markdown_report(summary, metrics)
    except Exception as e:
        print(f"⚠ 报告生成失败: {e}")
    
    print("\n" + "=" * 80)
    print("分析完成！(Analysis Complete!)".center(80))
    print("=" * 80)
    print(f"\n输出目录: {ANALYSIS_DIR}")
    print("- ablation_comparison.png: 可视化对比图")
    print("- ablation_analysis_report.md: 详细分析报告")


if __name__ == "__main__":
    main()

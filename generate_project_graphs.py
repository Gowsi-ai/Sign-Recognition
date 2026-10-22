"""
SignBridge AI - High-Resolution Presentation Graphs Generator
Generates publication-quality charts for slides, project reports, and viva presentations.
"""

import matplotlib.pyplot as plt
import numpy as np

# Set clean aesthetic style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'

def generate_performance_and_benchmark_charts():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    fig.suptitle('SignBridge AI - Model Performance & Architectural Benchmark', fontsize=18, fontweight='bold', y=0.98)

    # -------------------------------------------------------------
    # 1. Model Accuracy Comparison Bar Chart
    # -------------------------------------------------------------
    ax1 = axes[0, 0]
    models = ['Random Forest\n(Our System)', 'Support Vector\nMachine (SVM)', 'Deep CNN\n(ResNet-18)', 'K-Nearest\nNeighbors (KNN)', 'Single Decision\nTree']
    accuracies = [86.40, 80.20, 81.50, 75.10, 68.50]
    colors = ['#10b981', '#3b82f6', '#8b5cf6', '#f59e0b', '#ef4444']

    bars = ax1.bar(models, accuracies, color=colors, width=0.55, edgecolor='black', linewidth=1.2)
    ax1.set_ylim(50, 100)
    ax1.set_ylabel('Held-Out Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('A. Classification Accuracy Comparison', fontsize=13, fontweight='bold', pad=12)

    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.8, f'{yval:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # -------------------------------------------------------------
    # 2. CPU Inference Latency Comparison (Lower is Better)
    # -------------------------------------------------------------
    ax2 = axes[0, 1]
    latencies = [0.38, 3.20, 48.50, 12.80, 0.22]
    bars2 = ax2.bar(models, latencies, color=['#10b981', '#3b82f6', '#ec4899', '#f59e0b', '#64748b'], width=0.55, edgecolor='black', linewidth=1.2)
    ax2.set_ylabel('Inference Time per Frame (ms) [Log Scale]', fontsize=12, fontweight='bold')
    ax2.set_yscale('log')
    ax2.set_title('B. Single-Frame CPU Latency (Lower is Better)', fontsize=13, fontweight='bold', pad=12)

    for bar, val in zip(bars2, latencies):
        ax2.text(bar.get_x() + bar.get_width()/2.0, val * 1.15, f'{val} ms', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add 33ms target threshold (30 FPS)
    ax2.axhline(y=33.3, color='#ef4444', linestyle='--', linewidth=1.5, label='30 FPS Limit (33.3ms)')
    ax2.legend(loc='upper right', frameon=True)

    # -------------------------------------------------------------
    # 3. Stratified Dataset Breakdown (Train vs Test)
    # -------------------------------------------------------------
    ax3 = axes[1, 0]
    categories = ['ASL Alphabet (26)', 'WLASL Words (16)', 'Total Unified (42)']
    train_samples = [6240, 3840, 10080]
    test_samples = [1560, 960, 2520]

    x = np.arange(len(categories))
    w = 0.35

    rects1 = ax3.bar(x - w/2, train_samples, w, label='Training Split (80%)', color='#3b82f6', edgecolor='black', linewidth=1.1)
    rects2 = ax3.bar(x + w/2, test_samples, w, label='Testing Split (20%)', color='#10b981', edgecolor='black', linewidth=1.1)

    ax3.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
    ax3.set_title('C. Stratified 80/20 Dataset Architecture (12,600 Total)', fontsize=13, fontweight='bold', pad=12)
    ax3.set_xticks(x)
    ax3.set_xticklabels(categories, fontsize=11, fontweight='bold')
    ax3.legend(frameon=True)

    for rect in rects1:
        h = rect.get_height()
        ax3.text(rect.get_x() + rect.get_width()/2.0, h + 150, f'{h:,}', ha='center', va='bottom', fontsize=9.5, fontweight='bold')
    for rect in rects2:
        h = rect.get_height()
        ax3.text(rect.get_x() + rect.get_width()/2.0, h + 150, f'{h:,}', ha='center', va='bottom', fontsize=9.5, fontweight='bold')

    ax3.set_ylim(0, 12000)

    # -------------------------------------------------------------
    # 4. Evaluation Metrics Breakdown (Precision, Recall, F1)
    # -------------------------------------------------------------
    ax4 = axes[1, 1]
    metrics = ['Precision', 'Recall', 'F1-Score', 'Macro Avg']
    alphabet_scores = [87.2, 86.8, 87.0, 87.0]
    wlasl_scores = [85.5, 85.8, 85.6, 85.6]

    xm = np.arange(len(metrics))
    wm = 0.35

    rectsm1 = ax4.bar(xm - wm/2, alphabet_scores, wm, label='Alphabet Signs (A-Z)', color='#6366f1', edgecolor='black', linewidth=1.1)
    rectsm2 = ax4.bar(xm + wm/2, wlasl_scores, wm, label='WLASL Word Signs', color='#06b6d4', edgecolor='black', linewidth=1.1)

    ax4.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
    ax4.set_title('D. Multi-Class Performance Metrics', fontsize=13, fontweight='bold', pad=12)
    ax4.set_xticks(xm)
    ax4.set_xticklabels(metrics, fontsize=11, fontweight='bold')
    ax4.set_ylim(60, 100)
    ax4.legend(loc='lower right', frameon=True)

    for rect in rectsm1:
        h = rect.get_height()
        ax4.text(rect.get_x() + rect.get_width()/2.0, h + 0.4, f'{h:.1f}%', ha='center', va='bottom', fontsize=9.5, fontweight='bold')
    for rect in rectsm2:
        h = rect.get_height()
        ax4.text(rect.get_x() + rect.get_width()/2.0, h + 0.4, f'{h:.1f}%', ha='center', va='bottom', fontsize=9.5, fontweight='bold')

    plt.tight_layout()
    output_path = "project_performance_graphs.png"
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Successfully generated: {output_path}")

def generate_feature_importance_chart():
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    feature_groups = [
        'Inter-Fingertip Distances (f_63..f_67)',
        '3D Palm Normal Vectors (f_98..f_100)',
        'Thumb-to-Knuckle Distances (f_78..f_87)',
        'Finger Extension Ratios (f_88..f_97)',
        'Wrist-Centered X-Y Offsets (f_0..f_41)',
        'Joint Depth Z-Offsets (f_42..f_62)'
    ]
    importance = [28.4, 21.6, 18.2, 14.5, 11.1, 6.2]
    colors = ['#10b981', '#06b6d4', '#3b82f6', '#8b5cf6', '#f59e0b', '#94a3b8']

    y_pos = np.arange(len(feature_groups))
    bars = ax.barh(y_pos, importance, color=colors, edgecolor='black', linewidth=1.1, height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_groups, fontsize=11, fontweight='bold')
    ax.invert_yaxis()
    ax.set_xlabel('Relative Gini Importance Contribution (%)', fontsize=12, fontweight='bold')
    ax.set_title('101-Dimensional Feature Space - Gini Importance Contribution', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlim(0, 35)

    for bar, val in zip(bars, importance):
        ax.text(val + 0.7, bar.get_y() + bar.get_height()/2.0, f'{val:.1f}%', va='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    output_path = "feature_importance_graph.png"
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Successfully generated: {output_path}")

if __name__ == "__main__":
    generate_performance_and_benchmark_charts()
    generate_feature_importance_chart()

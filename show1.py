import matplotlib.pyplot as plt
import numpy as np
import torch

# 设置全局字体与美化风格，确保论文排版清晰
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False


def visualize_explanation(node_idx, delta, save_path=None):
    """
    【局部个案深挖】可视化单个特定节点的反事实扰动向量 delta
    """
    delta_np = delta.detach().cpu().numpy() if hasattr(delta, 'detach') else np.array(delta)

    # 提取扰动强度最大的前 15 个特征维度
    top_dims = np.argsort(np.abs(delta_np))[-15:]
    impact = delta_np[top_dims]

    plt.figure(figsize=(10, 6))

    # 🎨 升级配色策略：使用更具学术质感的哑光色（珊瑚红 vs 迷雾绿）
    # 负值（红色）：主模型判定这些特征“推高了异常得分”，反事实解释器建议“降低或删除”它们
    # 正值（绿色）：反事实解释器建议“增加”这些特征以使其向正常原型靠拢
    colors = ['#E06666' if x < 0 else '#6AA84F' for x in impact]

    bars = plt.barh(range(15), impact, color=colors, edgecolor='#434343', linewidth=0.8, alpha=0.85)
    plt.yticks(range(15), [f"Feat Dim #{d:03d}" for d in top_dims], fontsize=10, fontweight='bold')

    # 智能添加数值标签
    for bar in bars:
        width = bar.get_width()
        ha_val = 'right' if width < 0 else 'left'
        # 微调偏移量，防止文字与柱状图重叠
        offset = -0.002 if width < 0 else 0.002
        plt.text(width + offset, bar.get_y() + bar.get_height() / 2,
                 f'{width:.4f}', va='center', ha=ha_val, fontsize=9, color='#333333')

    plt.axvline(0, color='#333333', linewidth=1.2, linestyle='-')
    plt.title(f"Instance-Level Counterfactual Perturbation (Node {node_idx})\n[Global Explainer Inference]",
              fontsize=12, pad=15, fontweight='bold')
    plt.xlabel("Perturbation Intensity ($\delta$)", fontsize=11)
    plt.grid(axis='x', linestyle=':', alpha=0.6, color='#999999')

    # 去除上方和右方的边框，使图表更现代、干净
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # plt.tight_layout()
    plt.subplots_adjust(left=0.25, right=0.95, top=0.9, bottom=0.15)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📁 个案因果分析图已成功保存至: {save_path}")

    plt.show()


def visualize_global_feature_importance(mean_anom_delta, dataset_name, save_path=None):
    """
    【全新引入：全局共性透视】可视化全图所有异常节点共同的特征敏感度
    对应 run_xai13.py 中的步骤 8，可直接作为论文中的核心实验图
    """
    if hasattr(mean_anom_delta, 'detach'):
        mean_anom_delta = mean_anom_delta.detach().cpu().numpy()

    # 提取全局平均绝对扰动最大的前 10 个核心特征
    top_k = 10
    top_dims = np.argsort(mean_anom_delta)[-top_k:]
    global_impact = mean_anom_delta[top_dims]

    plt.figure(figsize=(9, 5.5))

    # 全局共性图建议使用统一的偏冷色调（如学术蓝），展现统计严谨性
    bars = plt.barh(range(top_k), global_impact, color='#4A90E2', edgecolor='#2C3E50', linewidth=0.8, alpha=0.85)
    plt.yticks(range(top_k), [f"Feat Dim #{d:03d}" for d in top_dims], fontsize=10, fontweight='bold')

    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.002, bar.get_y() + bar.get_height() / 2,
                 f'{width:.4f}', va='center', ha='left', fontsize=9, fontweight='bold', color='#2C3E50')

    plt.title(f"Global Counterfactual Feature Susceptibility Spectrum\n[Dataset: {dataset_name}]", fontsize=12, pad=15,
              fontweight='bold')
    plt.xlabel("Mean Absolute Perturbation Intensity ($|\delta|$)", fontsize=11)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.grid(axis='x', linestyle=':', alpha=0.5)

    # plt.tight_layout()
    plt.subplots_adjust(left=0.25, right=0.95, top=0.9, bottom=0.15)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📁 宏观全局特征共性谱线图已保存至: {save_path}")

    plt.show()
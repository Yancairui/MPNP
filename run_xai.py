import torch
import numpy as np
import os
from model import NodePrototypeAD
from explainer import CounterfactualExplainer
from utils import load_args
from data_loader import load_data
from show_cf import visualize_explanation


def run_explanation():
    # 1. 环境与参数初始化
    args = load_args()
    data, (train_idx, test_idx) = load_data(args)
    data = data.to(args.device)

    # 2. 加载适配新结构的模型
    model = NodePrototypeAD(args, data.num_node_features).to(args.device)
    model_path = 'best_model.pth'

    if not os.path.exists(model_path):
        print(f"❌ 错误：找不到 {model_path}。请确保已运行 main.py 并生成了新模型。")
        return

    try:
        # map_location 确保在不同设备间加载顺利
        model.load_state_dict(torch.load(model_path, map_location=args.device))
        print(f"✅ 成功加载权重：{model_path}")
    except Exception as e:
        print(f"❌ 加载失败，模型结构可能已变动，请重跑 main.py：\n{e}")
        return

    model.eval()

    # 3. 锁定分析目标（自动寻找测试集中风险最高的点）
    results_path = 'test_results.npy'
    if os.path.exists(results_path):
        results = np.load(results_path, allow_pickle=True).item()
        test_indices = results['test_idx']
        test_scores = results['final_score'][test_indices]
        top_idx_in_test = np.argmax(test_scores)
        top_anomaly_idx = test_indices[top_idx_in_test]
    else:
        # 如果没有结果记录，默认分析测试集第一个节点
        top_anomaly_idx = test_idx[0].item()

    print(f"\n[ 诊断任务启动 ]")
    print(f"🔍 正在深入分析最高风险节点: {top_anomaly_idx}")

    # 4. 执行邻域感知型反事实解释
    explainer = CounterfactualExplainer(model, args)

    # 获取特征扰动 delta，以及结构扰动权重 edge_m 和相关的边信息 edges
    delta, edge_m, related_edges = explainer.explain(
        data,
        top_anomaly_idx,
        iterations=500,
        lr=0.08
    )

    # 5. 输出结构诊断结论（邻域分析）
    print("\n" + "=" * 30)
    print("📢 拓扑结构因果诊断报告")
    print("=" * 30)

    # 筛选出被 Explainer 认为“需要削弱”的边（权重下降明显的边）
    found_causal_edge = False
    for i in range(len(edge_m)):
        weight = edge_m[i].item()
        if weight < 0.5:  # 阈值通常设为 0.5
            neighbor = related_edges[1, i].item()
            print(
                f"🚨 异常源定位：节点 {top_anomaly_idx} 与 邻居 {neighbor} 的连接显著推高了异常得分 (权重降至: {weight:.4f})")
            found_causal_edge = True

    if not found_causal_edge:
        print("✅ 结构检查：未发现明显的异常关联，该异常主要由节点自身属性引起。")
    print("=" * 30 + "\n")

    # 6. 生成特征维度的可视化图
    save_path = f'causal_feat_node_{top_anomaly_idx}.png'
    print(f"📊 正在生成特征层次分析图: {save_path}")
    visualize_explanation(top_anomaly_idx, delta, save_path=save_path)
    print("✨ 诊断完成！")


if __name__ == "__main__":
    run_explanation()
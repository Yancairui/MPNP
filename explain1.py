import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class CounterfactualExplainer:
    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.model.eval()

    def explain(self, data, node_idx, iterations=500, lr=0.05):
        # 1. 特征扰动初始化 (Delta)
        x_orig = data.x[node_idx:node_idx + 1].clone().detach()
        delta = torch.zeros_like(x_orig, requires_grad=True, device=self.args.device)

        # 2. 结构扰动初始化 (Edge Mask)
        # 找到所有与目标节点相关的边 (邻居节点)
        row, col = data.edge_index
        edge_mask_indices = (row == node_idx).nonzero(as_tuple=True)[0]
        # 为这些边创建一个可学习的权重，初始为 1.0 (存在)
        edge_logits = torch.ones(len(edge_mask_indices), requires_grad=True, device=self.args.device)

        optimizer = optim.Adam([delta, edge_logits], lr=lr)

        print(f"--- 正在执行邻域感知型诊断 (Node {node_idx}) ---")

        for i in range(iterations):
            optimizer.zero_grad()

            # --- 特征空间扰动 ---
            x_cf = x_orig + delta
            x_full_cf = data.x.clone()
            x_full_cf[node_idx] = x_cf

            # --- 结构空间扰动 (使用 Sigmoid 模拟边的删除) ---
            m = torch.sigmoid(edge_logits)
            # 构造带权重的邻接关系 (需要模型支持 edge_weight 参数)
            # 如果模型不支持 edge_weight，这里可以根据 m < 0.5 暴力截断，但梯度不连续
            # 我们假设 NodePrototypeAD 的 forward 可以接收 edge_weight

            # 模型推理 (5个返回值)
            z, sim, mask, x_rec, h = self.model(x_full_cf, data.edge_index, mode='test', edge_weight=m,
                                                edge_mask_idx=edge_mask_indices)

            # 3. 复合损失函数
            score_res = torch.mean((x_rec[node_idx:node_idx + 1] - x_cf) ** 2)
            score_proto = 1 - torch.max(sim[node_idx:node_idx + 1])
            loss_target = 0.6 * score_res + 0.4 * score_proto

            # 惩罚项：鼓励删除尽可能少的特征和尽可能少的边
            loss_feat_sparsity = torch.norm(delta, p=1)
            loss_edge_sparsity = torch.norm(1 - m, p=1)  # 鼓励 m 接近 1

            total_loss = loss_target + 0.02 * loss_feat_sparsity + 0.05 * loss_edge_sparsity

            total_loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f"Iter {i:03d} | Score: {loss_target.item():.4f} | Edge_Drop_Sum: {(1 - m).sum().item():.2f}")

        return delta.detach().squeeze(), m.detach(), data.edge_index[:, edge_mask_indices]

class GlobalParametricExplainer(nn.Module):
    """
    全局参数化反事实解释器网络
    输入：节点的初始特征与主模型提取出的高阶嵌入 z
    输出：全图节点的特征扰动量 delta 矩阵，以及全图每条边的保留权重 edge_weights
    """

    def __init__(self, input_dim, out_dim, hidden_dim=64):
        super(GlobalParametricExplainer, self).__init__()

        # 特征扰动生成分支 (输入节点嵌入 z -> 输出与原始特征维度一致的扰动)
        self.feat_explainer = nn.Sequential(
            nn.Linear(out_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh()  # 使用 Tanh 将扰动基础幅度限制在 [-1, 1] 之间
        )

        # 结构边权重生成分支 (输入源节点与目标节点嵌入的拼接 -> 输出边的保留概率)
        self.edge_explainer = nn.Sequential(
            nn.Linear(out_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # 概率输出 [0, 1]，1表示完全保留，0表示完全删除
        )

    def forward(self, x, z, edge_index):
        # 1. 全局预测每个节点的特征扰动量，乘上缩放系数 0.1 保证反事实扰动的“最小化”原则
        all_delta = self.feat_explainer(z) * 0.1

        # 2. 全局预测图上每一条已有边的重要性/保留权重
        row, col = edge_index
        z_src = z[row]
        z_dst = z[col]
        edge_weights = self.edge_explainer(torch.cat([z_src, z_dst], dim=-1)).squeeze(-1)

        return all_delta, edge_weights


class GlobalExplainerTrainer:
    """
    全局解释器的联合训练器
    """

    def __init__(self, model, args):
        self.model = model  # 已经训练好并冻结的主模型
        self.args = args
        self.model.eval()

    def train_global_explainer(self, data, anomaly_indices, epochs=200, lr=0.005):
        # 初始化全局解释器网络
        explainer = GlobalParametricExplainer(
            input_dim=data.x.size(1),
            out_dim=self.args.out_dim
        ).to(self.args.device)

        optimizer = optim.Adam(explainer.parameters(), lr=lr)

        print("\n>>> [XAI 阶段启动] 正在统一训练全局参数化反事实解释器... <<<")

        for epoch in range(epochs):
            explainer.train()
            optimizer.zero_grad()

            # 1. 冻结主模型，提取当前图结构下的基准节点嵌入 z
            with torch.no_grad():
                z_orig, _, _, _, _ = self.model(data.x, data.edge_index, mode='test')

            # 2. 前向传播：通过解释器网络直接生成全图的扰动矩阵与边权重
            all_delta, edge_weights = explainer(data.x, z_orig, data.edge_index)

            # 3. 施加反事实扰动：生成全图反事实特征
            x_cf = data.x + all_delta

            # 4. 将反事实特征和预测的边权重统一送入主模型
            # 此时 edge_mask_idx 传入的是全图所有边的索引，即令全图的边都受 edge_weights 驱动
            full_edge_idx = torch.arange(data.edge_index.size(1)).to(self.args.device)
            z_cf, sim_cf, _, x_rec_cf, _ = self.model(
                x_cf, data.edge_index, mode='test',
                edge_weight=edge_weights,
                edge_mask_idx=full_edge_idx
            )

            # 5. 计算面向异常节点的复合反事实损失函数
            # 目标：促使这些异常节点在受扰动后，重构误差变小且靠近正常原型空间
            score_res = torch.mean((x_rec_cf[anomaly_indices] - x_cf[anomaly_indices]) ** 2, dim=1)
            score_proto = 1 - torch.max(sim_cf[anomaly_indices], dim=1)[0]
            loss_target = (0.6 * score_res + 0.4 * score_proto).mean()

            # 6. 稀疏性与协同惩罚项（鼓励尽可能少地修改特征、尽可能保留原始拓扑边）
            loss_feat_sparsity = torch.norm(all_delta[anomaly_indices], p=1) / len(anomaly_indices)
            loss_edge_sparsity = torch.norm(1.0 - edge_weights, p=1) / edge_weights.size(0)

            # 总损失
            total_loss = loss_target + 0.05 * loss_feat_sparsity + 0.1 * loss_edge_sparsity

            total_loss.backward()
            optimizer.step()

            if epoch % 20 == 0:
                print(
                    f"Global Explainer Epoch {epoch:03d} | Target Loss: {loss_target.item():.4f} | Feat_Sparsity: {loss_feat_sparsity.item():.4f} | Edge_Drop_Sum: {(1.0 - edge_weights).sum().item():.2f}")

        return explainer
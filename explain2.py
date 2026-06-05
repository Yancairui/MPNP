import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class GlobalParametricExplainer(nn.Module):
    def __init__(self, input_dim, out_dim, hidden_dim=64):
        super(GlobalParametricExplainer, self).__init__()

        self.feat_explainer = nn.Sequential(
            nn.Linear(out_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh()
        )

        self.edge_explainer = nn.Sequential(
            nn.Linear(out_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x, z, edge_index, anomaly_indices=None):
        """
        显存精细化控制：
        如果传入了 anomaly_indices，特征扰动器只针对这些高危节点生成变长 Tensor，拒绝全图大矩阵
        """
        if anomaly_indices is not None:
            # 仅为异常节点集生成扰动 [len(anomaly_indices), input_dim]
            z_anomaly = z[anomaly_indices]
            anomaly_delta = self.feat_explainer(z_anomaly) * 0.1

            # 构建一个基础为 0 的全图扰动矩阵，但只在异常行保留梯度关联
            all_delta = torch.zeros_like(x, device=x.device)
            all_delta[anomaly_indices] = anomaly_delta
        else:
            all_delta = self.feat_explainer(z) * 0.1

        # 结构扰动：基于已有的稀疏拓扑边做映射，不需要全矩阵
        row, col = edge_index
        z_src = z[row]
        z_dst = z[col]
        edge_weights = self.edge_explainer(torch.cat([z_src, z_dst], dim=-1)).squeeze(-1)

        return all_delta, edge_weights


class GlobalExplainerTrainer:
    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.model.eval()

    def train_global_explainer(self, data, anomaly_indices, epochs=200, lr=0.005):
        explainer = GlobalParametricExplainer(
            input_dim=data.x.size(1),
            out_dim=self.args.out_dim
        ).to(self.args.device)

        optimizer = optim.Adam(explainer.parameters(), lr=lr)

        print(f"\n>>> [XAI 阶段启动] 正在统一训练面向 {len(anomaly_indices)} 个异常节点的局部参数化解释器... <<<")

        for epoch in range(epochs):
            explainer.train()
            optimizer.zero_grad()

            with torch.no_grad():
                z_orig = self.model.get_embedding(data.x, data.edge_index)

            # 🌟 核心显存技巧：显式传入异常节点索引，强制实行特征扰动局部化生成
            all_delta, edge_weights = explainer(data.x, z_orig, data.edge_index, anomaly_indices=anomaly_indices)

            x_cf = data.x + all_delta

            full_edge_idx = torch.arange(data.edge_index.size(1)).to(self.args.device)
            # 通过适配好的 model 前向传播计算反事实空间状态
            z_cf, sim_cf, _, _ = self.model(
                x_cf, data.edge_index, mode='test',
                edge_weight=edge_weights,
                edge_mask_idx=full_edge_idx
            )

            # 只针对目标异常节点阵列计算反事实约束
            # 显存优化：重构误差分步计算
            x_cf_anom = x_cf[anomaly_indices]
            x_rec_cf_anom = self.model.attr_decoder(z_cf[anomaly_indices])

            score_res = torch.mean((x_rec_cf_anom - x_cf_anom) ** 2, dim=1)
            score_proto = 1 - torch.max(sim_cf[anomaly_indices], dim=1)[0]
            loss_target = (0.6 * score_res + 0.4 * score_proto).mean()

            loss_feat_sparsity = torch.norm(all_delta[anomaly_indices], p=1) / len(anomaly_indices)
            loss_edge_sparsity = torch.norm(1.0 - edge_weights, p=1) / edge_weights.size(0)

            total_loss = loss_target + 0.05 * loss_feat_sparsity + 0.1 * loss_edge_sparsity

            total_loss.backward()
            optimizer.step()

            if epoch % 20 == 0:
                print(
                    f"Global Explainer Epoch {epoch:03d} | Target Loss: {loss_target.item():.4f} | Feat_Sparsity: {loss_feat_sparsity.item():.4f} | Edge_Drop_Sum: {(1.0 - edge_weights).sum().item():.2f}")

        return explainer
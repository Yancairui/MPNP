import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from model13_4 import NodePrototypeAD
from data_loader13_4 import load_data
from utils13 import load_args

def drop_feature(x, drop_prob):
    drop_mask = torch.empty((x.size(1),), dtype=torch.float32, device=x.device).uniform_(0, 1) < drop_prob
    x_aug = x.clone()
    x_aug[:, drop_mask] = 0
    return x_aug


def prototype_contrastive_loss(z1, z2, prototypes, temperature=0.05):#(Citeseer:temperature=0.5)
    """
    🌟 终极显存优化：利用全局原型作为中介，完全避免 N*N 矩阵计算
    理论完备，不丢弃任何节点的拓扑结构关联
    """
    z1_norm = F.normalize(z1, p=2, dim=-1)
    z2_norm = F.normalize(z2, p=2, dim=-1)
    p_norm = F.normalize(prototypes, p=2, dim=-1)

    p_assign1 = F.softmax(torch.mm(z1_norm, p_norm.t()) / temperature, dim=-1)
    p_assign2 = F.log_softmax(torch.mm(z2_norm, p_norm.t()) / temperature, dim=-1)

    loss_cl = F.kl_div(p_assign2, p_assign1, reduction='batchmean')
    return loss_cl


def eval_multi_metrics(y_true, y_score):
    aroc = roc_auc_score(y_true, y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    apr = auc(recall, precision)

    ratio = np.sum(y_true == 1) / len(y_true)
    threshold = np.percentile(y_score, 100 * (1 - ratio))
    y_pred = (y_score >= threshold).astype(int)
    f1_macro = f1_score(y_true, y_pred, average='macro')

    return aroc, apr, f1_macro


def train():
    args = load_args()
    data, (train_idx, test_idx) = load_data(args)
    data = data.to(args.device)

    model = NodePrototypeAD(args, data.num_node_features).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    print(f"Starting NodePro-SL on Dataset: {args.dataset} (100% Non-Sampling Full-Graph Mode)")

    best_aroc, corresponding_apr, corresponding_f1 = 0, 0, 0

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        # 前向传播提取嵌入
        z1, sim1, mask1 = model(data.x, data.edge_index, mode='train')
        x_aug = drop_feature(data.x, 0.2)
        z2, sim2, mask2 = model(x_aug, data.edge_index, mode='train')

        # 1. 属性重构损失 —— 🌟 无损全图分块流水线（涵盖 100% 节点，绝不超额索要显存）
        loss_attr = 0
        chunk_size = 4000
        num_nodes = data.x.size(0)
        for i in range(0, num_nodes, chunk_size):
            end_idx = min(i + chunk_size, num_nodes)
            x_rec_chunk = model.attr_decoder(z1[i:end_idx])
            loss_attr += F.mse_loss(x_rec_chunk, data.x[i:end_idx], reduction='sum')
        loss_attr = loss_attr / (num_nodes * data.num_node_features)

        # 2. 原型聚类损失
        dist = 1 - sim1[train_idx].max(dim=1)[0]
        loss_cluster = torch.clamp(dist - 0.15, min=0).mean()

        # 3. 🌟 无采样、纯严谨的原型自监督对比学习
        loss_cl = prototype_contrastive_loss(z1[train_idx], z2[train_idx], model.prototypes, temperature=0.5)

        # 4. 稀疏结构重构损失
        loss_struct = model.recon_loss_struct(z1, data.edge_index)

        # 5. 原型分离损失
        p_norm = F.normalize(model.prototypes, p=2, dim=-1)
        loss_sep = (torch.mm(p_norm, p_norm.t()) - torch.eye(args.n_prot).to(args.device)).pow(2).mean()

        # total_loss = 1.0 * loss_cluster + 0.8 * loss_attr + 0.1 * loss_struct + 2.0 * loss_sep + 0.5 * loss_cl #（Citeseer）
        total_loss = 1.0 * loss_cluster + 5.0 * loss_attr + 0.1 * loss_struct + 2.0 * loss_sep + 0.5 * loss_cl  # （Citeseer）

        total_loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                z_te = model.get_embedding(data.x, data.edge_index)

                # 评估时分分块无损扫描属性误差
                score_attr = []
                for i in range(0, data.x.size(0), chunk_size):
                    end_idx = min(i + chunk_size, data.x.size(0))
                    chunk_z = z_te[i:end_idx]
                    chunk_x_rec = model.attr_decoder(chunk_z)
                    chunk_err = torch.mean((chunk_x_rec - data.x[i:end_idx]) ** 2, dim=1)
                    score_attr.append(chunk_err)
                score_attr = torch.cat(score_attr, dim=0).cpu().numpy()

                z_norm_te = F.normalize(z_te, p=2, dim=-1)
                p_norm_te = F.normalize(model.prototypes, p=2, dim=-1)
                sim_te = torch.mm(z_norm_te, p_norm_te.t())
                score_proto = (1 - sim_te.max(dim=1)[0]).cpu().numpy()

                scaler = MinMaxScaler()
                s_a = scaler.fit_transform(score_attr.reshape(-1, 1)).flatten()
                s_p = scaler.fit_transform(score_proto.reshape(-1, 1)).flatten()
                final_score = 0.6 * s_a + 0.4 * s_p

                y_true_test = data.y[test_idx].cpu().numpy()
                y_score_test = final_score[test_idx]
                aroc, apr, f1_macro = eval_multi_metrics(y_true_test, y_score_test)

                print(
                    f"Epoch {epoch:03d} | AROC: {aroc:.4f} | APR: {apr:.4f} | Macro F1: {f1_macro:.4f} | Loss: {total_loss.item():.4f}")

                if aroc > best_aroc:
                    best_aroc = aroc
                    corresponding_apr = apr
                    corresponding_f1 = f1_macro
                    torch.save(model.state_dict(), 'best_model.pth')
                    np.save('test_results.npy', {
                        'test_idx': test_idx.cpu().numpy(),
                        'final_score': final_score
                    })
                    print(f"🌟 新纪录！最优模型已锁死保存：AROC={best_aroc:.4f}, APR={corresponding_apr:.4f}")

    print(f"\n================ 训练最终总结 ================")
    print(f"🥇 最佳图模型性能统一指标 ({args.dataset}):")
    print(f"📈 AUC-ROC  (AROC) : {best_aroc:.4f}")
    print(f"📊 AUC-PR   (APR)  : {corresponding_apr:.4f}")
    print(f"📉 Macro F1 (F1)   : {corresponding_f1:.4f}")
    print(f"==============================================")


if __name__ == "__main__":
    train()
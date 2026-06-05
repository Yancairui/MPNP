import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, JumpingKnowledge


class NodePrototypeAD(nn.Module):
    def __init__(self, args, input_dim):
        super(NodePrototypeAD, self).__init__()
        self.args = args
        self.input_dim = input_dim

        # 🌟 针对大图超高维特征（如 Flickr 12047维）的显存防御盾牌：
        # 先用线性层进行块映射，防止超高维矩阵直接参与 GNN 的邻居消息传递
        if input_dim > 2000:
            self.input_projector = nn.Linear(input_dim, args.hidden_dim)
            gnn_in_dim = args.hidden_dim
        else:
            self.input_projector = None
            gnn_in_dim = input_dim

        # 1. 编码器：多尺度 JK-GIN
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(args.conv_layers):
            in_d = gnn_in_dim if i == 0 else args.hidden_dim
            mlp = nn.Sequential(
                nn.Linear(in_d, args.hidden_dim),
                nn.BatchNorm1d(args.hidden_dim),
                nn.ReLU(),
                nn.Linear(args.hidden_dim, args.hidden_dim)
            )
            self.convs.append(GINConv(mlp))
            self.bns.append(nn.BatchNorm1d(args.hidden_dim))

        self.jk = JumpingKnowledge(mode='cat')
        self.project = nn.Linear(args.hidden_dim * args.conv_layers, args.out_dim)

        # 2. 属性重构解码器
        self.attr_decoder = nn.Sequential(
            nn.Linear(args.out_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, input_dim)
        )

        # 3. 全局原型
        self.prototypes = nn.Parameter(torch.Tensor(args.n_prot, args.out_dim))
        nn.init.orthogonal_(self.prototypes)

        # 4. 掩码判定器
        self.explainer_layer = nn.Sequential(
            nn.Linear(args.out_dim * 2, args.out_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(args.out_dim, args.out_dim),
            nn.Sigmoid()
        )

    def get_embedding(self, x, edge_index, edge_weight=None):
        # 如果有大图投影盾牌，先对特征降维，锁死消息传递时的显存
        if self.input_projector is not None:
            # 分块投影，防止全矩阵 Linear 爆显存
            x_low = []
            for i in range(0, x.size(0), 4000):
                x_low.append(self.input_projector(x[i:i + 4000]))
            x = torch.cat(x_low, dim=0)

        if edge_weight is not None:
            from torch_scatter import scatter_add
            node_weight = scatter_add(edge_weight, edge_index[0], dim=0, dim_size=x.size(0))
            node_weight = node_weight.view(-1, 1)
            node_weight = node_weight / (node_weight.max() + 1e-15)
            x = x * node_weight

        layer_outputs = []
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            layer_outputs.append(x)

        h_combined = self.jk(layer_outputs)
        z = self.project(h_combined)
        return z

    def forward(self, x, edge_index, mode='train', edge_weight=None, edge_mask_idx=None):
        full_edge_weight = None
        if edge_weight is not None and edge_mask_idx is not None:
            full_edge_weight = torch.ones(edge_index.size(1), device=x.device)
            full_edge_weight[edge_mask_idx] = edge_weight

        z = self.get_embedding(x, edge_index, edge_weight=full_edge_weight)

        # 彻底移除向前传导中的任何全图属性重构，重构全部移到外部以无损 Chunk 方式进行
        z_norm = F.normalize(z, p=2, dim=-1)
        p_norm = F.normalize(self.prototypes, p=2, dim=-1)
        sim = torch.mm(z_norm, p_norm.t())

        max_sim_idx = torch.argmax(sim, dim=1)
        closest_p = self.prototypes[max_sim_idx]
        node_mask = self.explainer_layer(torch.cat([z, closest_p], dim=-1))

        return z, sim, node_mask

    def recon_loss_struct(self, z, edge_index):
        pos_score = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        pos_loss = -torch.log(torch.sigmoid(pos_score) + 1e-15).mean()

        num_nodes = z.size(0)
        neg_edge_index = torch.stack([
            torch.randint(0, num_nodes, (edge_index.size(1) // 10,), device=z.device),
            torch.randint(0, num_nodes, (edge_index.size(1) // 10,), device=z.device)
        ], dim=0)
        neg_score = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=-1)
        neg_loss = -torch.log(1 - torch.sigmoid(neg_score) + 1e-15).mean()
        return pos_loss + neg_loss
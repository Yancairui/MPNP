import argparse
import torch

def load_args():
    parser = argparse.ArgumentParser()
    # 基础设置
    parser.add_argument('--dataset', type=str, default='YelpChi', choices=['citeseer', 'ACM', 'Amazon', 'Flickr', 'YelpChi'],
                        help='要运行的数据集名称 (可选: citeseer, ACM, Amazon, Flickr, YelpChi)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=int, default=0)

    # 训练超参数
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=1500)
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='L2 regularization')

    # 模型维度设置
    parser.add_argument('--hidden_dim', type=int, default=128, help='Dimension of hidden layers')
    parser.add_argument('--out_dim', type=int, default=32)
    parser.add_argument('--conv_layers', type=int, default=3)

    # 原型与解耦核心参数
    parser.add_argument('--n_prot', type=int, default=6, help='Number of normal prototypes (K)')

    # 损失函数权重 (针对解耦模型优化的比例)
    parser.add_argument('--r', type=float, default=0.3, help='Target sparsity for explainer')
    parser.add_argument('--gae', type=float, default=2.0, help='Weight for Reconstruction loss')
    parser.add_argument('--alpha', type=float, default=0.5, help='Weight for Prototypical loss')
    parser.add_argument('--beta', type=float, default=0.8, help='Weight for Disentanglement loss')

    # 使用 parse_args() 或 parse_known_args()
    args = parser.parse_known_args()[0]

    # 设备配置
    args.device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    return args
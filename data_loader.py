import numpy as np
import torch
import scipy.io as sio
import pickle
import os
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split


def load_data(args, random_state=0):
    dataset_name = args.dataset

    # =========================================================================
    # 1. 🌟 核心载入与分流
    # =========================================================================
    if dataset_name.lower() == 'yelpchi' or os.path.exists(f'./data/{dataset_name}.dat'):
        file_path = f'./data/{dataset_name if dataset_name.endswith(".dat") else dataset_name + ".dat"}'
        print(f"--> 检测到 .dat 格式，正在通过 pickle 序列化载入: {file_path}")
        with open(file_path, 'rb') as f:
            raw_data = pickle.load(f)
    else:
        file_path = f'./data/{dataset_name}.mat'
        print(f"--> 正在通过 scipy 载入标准矩阵文件: {file_path}")
        raw_data = sio.loadmat(file_path)

    # =========================================================================
    # 2. 🌟 智能防御：如果载入的文件已经是直接打包好的 PyG 或自定义类对象
    # =========================================================================
    if hasattr(raw_data, 'x') or hasattr(raw_data, 'edge_index'):
        print("--> 探测成功：当前 .dat 文件为已打包的图对象结构，启动无缝属性映射。")

        # 兼容可能有不同命名的类属性
        x_raw = getattr(raw_data, 'x', getattr(raw_data, 'features', None))
        y_raw = getattr(raw_data, 'y', getattr(raw_data, 'label', getattr(raw_data, 'labels', None)))

        if hasattr(raw_data, 'edge_index'):
            edge_index = raw_data.edge_index.long()
        else:
            # 某些类可能把邻接矩阵存为 A 或 adj
            A_matrix = getattr(raw_data, 'A', getattr(raw_data, 'adj', None))
            if hasattr(A_matrix, 'toarray'): A_matrix = A_matrix.toarray()
            edge_index = torch.from_numpy(np.array(A_matrix.nonzero())).long()

        x = x_raw.float() if isinstance(x_raw, torch.Tensor) else torch.from_numpy(x_raw).float()

        if isinstance(y_raw, torch.Tensor):
            y_np = y_raw.cpu().numpy().flatten()
        else:
            y_np = np.array(y_raw).flatten()
        y = torch.from_numpy(y_np).long()

        data = Data(x=x, edge_index=edge_index, y=y)

    # =========================================================================
    # 3. 🌟 传统字典（Dict）匹配路径（兼容 Amazon, Flickr, ACM, CiteSeer）
    # =========================================================================
    else:
        data_mat = raw_data
        X = None
        for feat_key in ['X', 'Attributes', 'attributes', 'x', 'features', 'feat']:
            if feat_key in data_mat:
                X = data_mat[feat_key]
                break

        A = None
        for adj_key in ['A', 'Network', 'network', 'adj', 'homo', 'g']:
            if adj_key in data_mat:
                A = data_mat[adj_key]
                break

        y = None
        for label_key in ['label', 'gnd', 'Label', 'y', 'Y', 'labels']:
            if label_key in data_mat:
                y = data_mat[label_key]
                break

        assert X is not None, f"无法在数据文件中找到特征矩阵。当前包含的 Keys: {list(data_mat.keys()) if isinstance(data_mat, dict) else '非标准字典'}"
        assert A is not None, f"无法在数据文件中找到邻接矩阵。当前包含的 Keys: {list(data_mat.keys()) if isinstance(data_mat, dict) else '非标准字典'}"
        assert y is not None, f"无法在数据文件中找到标签矩阵。当前包含的 Keys: {list(data_mat.keys()) if isinstance(data_mat, dict) else '非标准字典'}"

        if hasattr(X, 'toarray'):
            X = X.toarray()
        elif hasattr(X, 'todense'):
            X = np.asarray(X.todense())

        if hasattr(A, 'toarray'):
            A = A.toarray()
        elif hasattr(A, 'todense'):
            A = np.asarray(A.todense())

        x = torch.from_numpy(X).float()
        if isinstance(y, torch.Tensor):
            y_np = y.cpu().numpy().flatten()
        else:
            y_np = np.array(y).flatten()
        y = torch.from_numpy(y_np).long()

        edge_index = torch.from_numpy(np.array(A.nonzero())).long()
        data = Data(x=x, edge_index=edge_index, y=y)

    # =========================================================================
    # 4. 🌟 训练集与测试集标准化划分
    # =========================================================================
    indices = np.arange(data.num_nodes)
    normal_idx = indices[y_np == 0]
    anomaly_idx = indices[y_np == 1]

    if len(normal_idx) == 0 or len(anomaly_idx) == 0:
        raise ValueError(f"数据集标签解析异常。解析出正常节点数: {len(normal_idx)}, 异常节点数: {len(anomaly_idx)}")

    train_normal, test_normal = train_test_split(
        normal_idx, test_size=0.2, random_state=args.seed + random_state
    )

    train_idx = torch.from_numpy(train_normal).long()
    test_idx = torch.from_numpy(np.concatenate([test_normal, anomaly_idx])).long()

    return data, [train_idx, test_idx]

# import numpy as np
# import torch
# import scipy.io as sio
# from torch_geometric.data import Data
# from sklearn.model_selection import train_test_split
#
#
# def load_data(args, random_state=0):
#     # 根据路径加载 .mat 文件
#     data_mat = sio.loadmat(f'./data/{args.dataset}.mat')
#
#     # =========================================================================
#     # 1. 特征矩阵兼容 (你的文件里是 'X')
#     # =========================================================================
#     X = None
#     for feat_key in ['X', 'Attributes', 'attributes', 'x', 'features', 'feat']:
#         if feat_key in data_mat:
#             X = data_mat[feat_key]
#             break
#
#     # =========================================================================
#     # 2. 邻接矩阵兼容 (你的文件里是 'A')
#     # =========================================================================
#     A = None
#     for adj_key in ['A', 'Network', 'network', 'adj', 'homo', 'g']:
#         if adj_key in data_mat:
#             A = data_mat[adj_key]
#             break
#
#     # =========================================================================
#     # 3. 标签矩阵兼容 (你的文件里是 'gnd')
#     # =========================================================================
#     y = None
#     for label_key in ['gnd', 'Label', 'label', 'y', 'Y', 'labels']:
#         if label_key in data_mat:
#             y = data_mat[label_key]
#             break
#
#     # 严谨的断言与排查机制
#     assert X is not None, f"无法在 .mat 文件中找到特征矩阵。当前文件的 Keys 包含: {list(data_mat.keys())}"
#     assert A is not None, f"无法在 .mat 文件中找到邻接矩阵。当前文件的 Keys 包含: {list(data_mat.keys())}"
#     assert y is not None, f"无法在 .mat 文件中找到标签矩阵。当前文件的 Keys 包含: {list(data_mat.keys())}"
#
#     if hasattr(X, 'toarray'): X = X.toarray()
#     if hasattr(A, 'toarray'): A = A.toarray()
#
#     x = torch.from_numpy(X).float()
#     y = torch.from_numpy(y).long().flatten()
#     # 确保标签中 0 为正常，1 为异常
#     edge_index = torch.from_numpy(np.array(A.nonzero())).long()
#
#     data = Data(x=x, edge_index=edge_index, y=y)
#
#     # 划分逻辑：训练集仅包含正常节点
#     indices = np.arange(data.num_nodes)
#     normal_idx = indices[y == 0]
#     anomaly_idx = indices[y == 1]
#
#     train_normal, test_normal = train_test_split(
#         normal_idx, test_size=0.2, random_state=args.seed + random_state
#     )
#
#     train_idx = torch.from_numpy(train_normal).long()
#     test_idx = torch.from_numpy(np.concatenate([test_normal, anomaly_idx])).long()
#
#     return data, [train_idx, test_idx]
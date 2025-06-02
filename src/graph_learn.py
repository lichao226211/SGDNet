import torch
import numpy as np
import torch.nn.functional as F


def learn_graph(args, graph_learner, node_features, graph_skip_conn=None, init_adj=None):
    # 计算相似性矩阵新的adj，且经过了掩码操作
    raw_adj = graph_learner(node_features)
    # raw_adj1 = pearson_correlation(node_features).to(args.cuda)
    if args.metric_type in ('kernel', 'weighted_cosine'):
        # assert raw_adj.min().item() >= 0
        adj = raw_adj / torch.clamp(torch.sum(raw_adj, dim=-1, keepdim=True), min=args.min_num)
    elif args.metric_type == 'cosine':
        adj = (raw_adj > 0).double()
        adj = normalize_adj(adj)

    else:
        adj = torch.softmax(raw_adj, dim=-1)
    init_adj = normalize_adj(init_adj)
    adj = init_adj + graph_skip_conn * adj
    # adj[adj > 0] = 1
    # adj = normalize_adj(adj)
    return raw_adj, adj


def normalize_adj(mx):
    """Row-normalize matrix: symmetric normalized Laplacian"""
    rowsum = mx.sum(1)
    r_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
    r_inv_sqrt[torch.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = torch.diag(r_inv_sqrt)
    return torch.mm(torch.mm(mx, r_mat_inv_sqrt).transpose(-1, -2), r_mat_inv_sqrt)


import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# from ..utils.generic_utils import to_cuda
# from ..utils.constants import VERY_SMALL_NUMBER, INF


def compute_normalized_laplacian(adj):
    rowsum = torch.sum(adj, -1)
    d_inv_sqrt = torch.pow(rowsum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diagflat(d_inv_sqrt)
    L_norm = torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
    return L_norm


class GraphLearner(nn.Module):
    def __init__(self, input_size, hidden_size, topk=None, epsilon=None, num_pers=16, metric_type='weighted_cosine', device=None):
        super(GraphLearner, self).__init__()
        self.device = device
        self.topk = topk
        self.epsilon = epsilon
        self.metric_type = metric_type


        if metric_type == 'weighted_cosine':
            self.weight_tensor = torch.Tensor(num_pers, input_size)
            self.weight_tensor = nn.Parameter(nn.init.xavier_uniform_(self.weight_tensor))
            # print('[ Multi-perspective {} GraphLearner: {} ]'.format(metric_type, num_pers))

        elif metric_type == 'kernel':
            self.precision_inv_dis = nn.Parameter(torch.Tensor(1, 1))
            self.precision_inv_dis.data.uniform_(0, 1.0)
            self.weight = nn.Parameter(nn.init.xavier_uniform_(torch.Tensor(input_size, hidden_size)))

        else:
            raise ValueError('Unknown metric_type: {}'.format(metric_type))

        # print('[ Graph Learner metric type: {} ]'.format(metric_type))

    def forward(self, context, ctx_mask=None):
        """
        Parameters
        :context, (batch_size, ctx_size, dim)
        :ctx_mask, (batch_size, ctx_size)

        Returns
        :attention, (batch_size, ctx_size, ctx_size)
        """
        if self.metric_type == 'weighted_cosine':
            # expand_weight_tensor = self.weight_tensor.unsqueeze(1)
            # if len(context.shape) == 3:
            #     expand_weight_tensor = expand_weight_tensor.unsqueeze(1)
            context_fc = context.unsqueeze(0)
            # context_fc = context.unsqueeze(0) * expand_weight_tensor
            context_norm = F.normalize(context_fc, p=2, dim=-1)
            attention = torch.matmul(context_norm, context_norm.transpose(-1, -2)).mean(0)
            markoff_value = 0


        elif self.metric_type == 'kernel':
            dist_weight = torch.mm(self.weight, self.weight.transpose(-1, -2))
            attention = self.compute_distance_mat(context, dist_weight)
            attention = torch.exp(-0.5 * attention * (self.precision_inv_dis**2))

            markoff_value = 0



        if ctx_mask is not None:
            attention = attention.masked_fill_(1 - ctx_mask.byte().unsqueeze(1), markoff_value)
            attention = attention.masked_fill_(1 - ctx_mask.byte().unsqueeze(-1), markoff_value)

        # 对其进行掩码操作，将小于epsilon的值设为0
        if self.epsilon is not None:
            attention = self.build_epsilon_neighbourhood(attention, self.epsilon, markoff_value)

        if self.topk is not None:
            attention = self.build_knn_neighbourhood(attention, self.topk, markoff_value)
        return attention

    def build_knn_neighbourhood(self, attention, topk, markoff_value):
        topk = min(topk, attention.size(-1))
        knn_val, knn_ind = torch.topk(attention, topk, dim=-1)
        weighted_adjacency_matrix = to_cuda((markoff_value * torch.ones_like(attention)).scatter_(-1, knn_ind, knn_val), self.device)
        return weighted_adjacency_matrix

    def build_epsilon_neighbourhood(self, attention, epsilon, markoff_value):
        mask = (attention > epsilon).detach().float()
        weighted_adjacency_matrix = attention * mask + markoff_value * (1 - mask)
        return weighted_adjacency_matrix

    def compute_distance_mat(self, X, weight=None):
        if weight is not None:
            trans_X = torch.mm(X, weight)
        else:
            trans_X = X
        norm = torch.sum(trans_X * X, dim=-1)
        dists = -2 * torch.matmul(trans_X, X.transpose(-1, -2)) + norm.unsqueeze(0) + norm.unsqueeze(1)
        return dists


def get_binarized_kneighbors_graph(features, topk, mask=None, device=None):
    assert features.requires_grad is False
    # Compute cosine similarity matrix
    features_norm = features.div(torch.norm(features, p=2, dim=-1, keepdim=True))
    attention = torch.matmul(features_norm, features_norm.transpose(-1, -2))

    if mask is not None:
        attention = attention.masked_fill_(1 - mask.byte().unsqueeze(1), 0)
        attention = attention.masked_fill_(1 - mask.byte().unsqueeze(-1), 0)

    # Extract and Binarize kNN-graph
    topk = min(topk, attention.size(-1))
    _, knn_ind = torch.topk(attention, topk, dim=-1)
    adj = to_cuda(torch.zeros_like(attention).scatter_(-1, knn_ind, 1), device)
    return adj

def to_cuda(x, device):
    if device:
        x = x.to(device)
    return x
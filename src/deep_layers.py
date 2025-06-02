import torch
import torch.nn as nn
import torch.nn.functional as F
# import dgl.function as fn
#
# from ogb.graphproppred.mol_encoder import BondEncoder
# from dgl.nn.functional import edge_softmax
from modules import MLP, MessageNorm


class GENConv(nn.Module):
    r"""
    
    Description
    -----------
    Generalized Message Aggregator was introduced in `DeeperGCN: All You Need to Train Deeper GCNs <https://arxiv.org/abs/2006.07739>`_

    Parameters
    ----------
    dataset: str
        Name of ogb dataset.
    in_dim: int
        Size of input dimension.
    out_dim: int
        Size of output dimension.
    aggregator: str
        Type of aggregator scheme ('softmax', 'power'), default is 'softmax'.
    beta: float
        A continuous variable called an inverse temperature. Default is 1.0.
    learn_beta: bool
        Whether beta is a learnable variable or not. Default is False.
    p: float
        Initial power for power mean aggregation. Default is 1.0.
    learn_p: bool
        Whether p is a learnable variable or not. Default is False.
    msg_norm: bool
        Whether message normalization is used. Default is False.
    learn_msg_scale: bool
        Whether s is a learnable scaling factor or not in message normalization. Default is False.
    norm: str
        Type of ('batch', 'layer', 'instance') norm layer in MLP layers. Default is 'batch'.
    mlp_layers: int
        The number of MLP layers. Default is 1.
    eps: float
        A small positive constant in message construction function. Default is 1e-7.
    """
    def __init__(self,
                 in_dim,
                 out_dim,
                 aggregator='softmax',
                 beta=1.0,
                 learn_beta=False,
                 p=1.0,
                 learn_p=False,
                 msg_norm=False,
                 learn_msg_scale=False,
                 norm='batch',
                 mlp_layers=1,
                 eps=1e-7):
        super(GENConv, self).__init__()
        
        self.aggr = aggregator
        self.eps = eps

        channels = [in_dim]
        for i in range(mlp_layers - 1):
            channels.append(in_dim * 2)

        channels.append(out_dim)


        self.mlp = MLP(channels, norm=norm)
        # 创建一个消息归一化层，如果msg_norm为True，则使用learn_msg_scale进行缩放，否则为None。
        self.msg_norm = MessageNorm(learn_msg_scale) if msg_norm else None
        # 创建超参数beta，并设置其是否可学习的属性，如果learn_beta为True并且聚合器类型为softmax，则beta为可学习的，否则beta为不可学习的。
        self.beta = nn.Parameter(torch.Tensor([beta]), requires_grad=True) if learn_beta and self.aggr == 'softmax' else beta
        #　创建超参数p，并设置其是否可学习的属性，如果learn_p为True，则p为可学习的，否则p为不可学习的。
        self.p = nn.Parameter(torch.Tensor([p]), requires_grad=True) if learn_p else p

        # if dataset == 'ogbg-molhiv':
        #     self.edge_encoder = BondEncoder(in_dim)
        # elif dataset == 'ogbg-ppa':
        # 定义边的变化形式
        # self.edge_encoder = nn.Linear(in_dim, in_dim)
        # else:
        #     raise ValueError(f'Dataset {dataset} is not supported.')

    def forward(self, edges, node_feats):
        return 0




    # def forward(self, g, node_feats, edge_feats):
    #     # 使用DGL库中的local_scope函数，进入图g的局部范围，确保每次操作只在当前图内生效
    #     with g.local_scope():
    #         # Node and edge feature dimension need to match.
    #         g.ndata['h'] = node_feats
    #         g.edata['h'] = self.edge_encoder(edge_feats)
    #         # 对每一条边，将其源节点和目标节点的特征值进行相加，并保存到边的'm'属性中。
    #         g.apply_edges(fn.u_add_e('h', 'h', 'm'))
    #         # 这一步就是完成了消息构造
    #         if self.aggr == 'softmax':
    #             # 对边的'm'属性进行ReLU激活，并加上一个很小的值eps，防止分母为0。
    #
    #             g.edata['m'] = F.relu(g.edata['m']) + self.eps
    #             # 计算边的注意力系数，并保存到边的'a'属性中，其中注意力系数使用edge_softmax函数进行计算，beta为超参数。
    #             g.edata['a'] = edge_softmax(g, g.edata['m'] * self.beta)
    #             g.update_all(lambda edge: {'x': edge.data['m'] * edge.data['a']},
    #                          fn.sum('x', 'm'))
    #
    #         elif self.aggr == 'power':
    #             minv, maxv = 1e-7, 1e1
    #             torch.clamp_(g.edata['m'], minv, maxv)
    #             g.update_all(lambda edge: {'x': torch.pow(edge.data['m'], self.p)},
    #                          fn.mean('x', 'm'))
    #             torch.clamp_(g.ndata['m'], minv, maxv)
    #             g.ndata['m'] = torch.pow(g.ndata['m'], self.p)
    #         else:
    #             raise NotImplementedError(f'Aggregator {self.aggr} is not supported.')
    #         if self.msg_norm is not None:
    #             g.ndata['m'] = self.msg_norm(node_feats, g.ndata['m'])
    #
    #         feats = node_feats + g.ndata['m']
    #         return self.mlp(feats)

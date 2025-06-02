import math

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt

from modules import norm_layer
from src import graphlearn
from src.utils import sptial_neighbor_matrix, compute_dist
from graph_learn import learn_graph
from graphlearn import GraphLearner
from GCNlayer import GCNLayers, GCN,GCNLayers1
from para_model import Para_model
import multiprocessing as mp


class DeeperGCN(torch.nn.Module):
    def __init__(self,
                 args,
                 node_feat_dim,
                 hid_dim,
                 out_dim,
                 num_layers,
                 norm='batch',
                 beta=1.0,
                 class_num=16,
                 cluster = 0,
                 degree_max=0):
        super(DeeperGCN, self).__init__()
        # para_init
        self.args = args
        self.num_layers = num_layers
        self.dropout = self.args.dropout
        self.norm = norm
        self.hid_dim = hid_dim
        self.node_feat_dim = node_feat_dim
        # model_init

        self.graph_skip_conn = Para_model()
        # model_para
        self.gcns = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.d_norms = nn.ModuleList()
        self.embedding=nn.ModuleList()
        self.linear = nn.ModuleList()
        self.graph_learner = nn.ModuleList()
        self.para_model = []
        # 聚类系数映射
        self.idx_graph = nn.ModuleList()
        self.class_num = class_num
        self.gcns.append(GCNLayers(node_feat_dim, node_feat_dim))
        self.norms.append(norm_layer(norm,  hid_dim))
        self.d_norms.append(norm_layer(norm,  hid_dim))
        self.linear.append(nn.Linear(node_feat_dim, hid_dim))

        self.num_nodes = 0
        self.num_anchors = 0

        self.whole_loader = None
        self.position_flag = 1
        self.anchor_node_list = []
        for cluster in cluster:
            self.embedding.append(nn.Embedding(len(degree_max[cluster]), hid_dim).to(self.args.cuda))
            self.idx_graph.append(nn.Embedding(len(degree_max[cluster]), hid_dim))
            self.graph_learner.append(GraphLearner(node_feat_dim, hid_dim, device=self.args.cuda, epsilon= self.args.epsilon))

            # self.para_model.append(Para_model().to(self.args.cuda))
        # self.embedding.append(nn.Embedding(, hid_dim))


    def forward(self,node_feats, train_nodes, cluster, mode, degeree, index_graph, layer, init_adj, target ,index_all, data):
        init_feat = node_feats
        # arpha = self.arpha(index_graph)
        layer = int(layer)
        # dist = self.compute_spatial(index_all, init_adj)
        # 获取标签类别的数量
        # unique_elements = torch.unique(target)
        num_classes = self.class_num
        # num_classes = len(unique_elements)
        # self.class_num = len(unique_elements)
        # 将标签转换为 one-hot 编码矩阵
        target = self.to_cuda(target, self.args.cuda)
        # one_hot_matrix = torch.eye(num_classes, device=self.args.cuda)[target.flatten()]
        # 打印转换后的one-hot编码矩阵
        dist = 0
        # self.num_nodes = len(init_adj)
        # self.num_anchors = len(train_nodes)
        # self.anchor_node_list = train_nodes
        # self.whole_loader = data
        # self.shortest_path_dists_anchor = np.zeros((self.num_nodes, self.num_nodes))
        # self.shortest_path_dists = np.zeros((self.num_nodes, self.num_nodes))
        # self.graph_learner[cluster] = GraphLearner(self.node_feat_dim, self.hid_dim,self.num_nodes,num_classes,len(train_nodes), epsilon=self.args.epsilon)
        #
        # self.group_pagerank_before = self.cal_group_pagerank(init_adj, self.whole_loader, 0.85)
        # self.group_pagerank_after = self.group_pagerank_before
        # self.group_pagerank_args = torch.from_numpy(
        #     self.cal_group_pagerank_args(self.group_pagerank_before, self.group_pagerank_after)).to(self.args.cuda)
        # self.shortest_path_dists = self.cal_shortest_path_distance(init_adj, 0)
        # self.shortest_path_dists_anchor = torch.from_numpy(self.cal_spd(init_adj, 0)).to(self.args.cuda).to(torch.double)

        if mode == 'train':
            loss, layer,output = self.train_epoch(node_feats, train_nodes, init_adj, layer, target, init_feat, cluster,dist)
            return loss, layer, output
        else:
            output = self.test_epoch(node_feats, init_adj, layer, cluster)
            return output
    def learn_adj(self, init_adj, feature, cluster, training = True):
        # cur_raw_adj, cur_adj =  learn_graph(self.args, self.graph_learner[cluster], feature, self.shortest_path_dists_anchor, self.group_pagerank_args,
        #                                     self.position_flag, graph_skip_conn=self.args.graph_skip_conn, init_adj=init_adj)
        cur_raw_adj, cur_adj = learn_graph(self.args, self.graph_learner[cluster], feature,graph_skip_conn=self.args.graph_skip_conn,init_adj=init_adj)
        cur_raw_adj = F.dropout(cur_raw_adj, self.args.feat_adj_dropout, training=training)
        cur_adj = F.dropout(cur_adj, self.args.feat_adj_dropout, training=training)
        return cur_raw_adj, cur_adj

    def normalize_adj(self, mx):
        """Row-normalize matrix: symmetric normalized Laplacian"""
        # mx[np.nonzero(mx)] = 1
        rowsum = mx.sum(1)
        r_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
        r_inv_sqrt[torch.isinf(r_inv_sqrt)] = 0.
        r_mat_inv_sqrt = torch.diag(r_inv_sqrt)
        return torch.mm(torch.mm(mx, r_mat_inv_sqrt).transpose(-1, -2), r_mat_inv_sqrt)
    def add_graph_loss(self, out_adj, features, dist):
        graph_loss = 0
        L = (torch.diagflat(torch.sum(out_adj, -1)) - out_adj).to(self.args.cuda)
        out_adj = self.to_cuda(out_adj, self.args.cuda)
        graph_loss += self.args.smoothness_ratio * torch.trace(torch.mm(features.transpose(-1, -2).double(), torch.mm(L.double(), features.double()))) / int(np.prod(out_adj.shape))
        graph_loss += self.args.sparsity_ratio * torch.sum(torch.pow(out_adj.double(), 2)) / int(np.prod(out_adj.shape))
        return graph_loss

    def get_degree(self, adj):
        d = adj.sum(0).long()
        return d

    def train_epoch(self, node_feats, train_nodes, init_adj, layer, target, init_feat, cluster, dist):
        # init_adj = self.normalize_adj(init_adj)
        pre_feat = node_feats
        # cur_raw_adj经过掩码处理过的， cur_adj加上了原始的图
        cur_feat = self.norms[layer](pre_feat)
        cur_raw_adj, cur_adj = self.learn_adj(init_adj, cur_feat, cluster)
        d = self.get_degree(cur_adj)
        d = self.embedding[cluster](d)
        cur_feat = F.relu(cur_feat)
        cur_feat = F.dropout(cur_feat, p=self.dropout, training=self.training)
        cur_feat = self.gcns[layer](cur_feat, cur_adj)
        pre_feat = self.args.balanced_degree * pre_feat + (1 - self.args.balanced_degree) * d

        cur_feat = cur_feat + pre_feat
        # 增加网络
        self.add_network(layer, cur_feat)
        # 计算距离
        best_dist = self.compute_dist(cur_feat)
        output = torch.nn.functional.log_softmax(cur_feat, dim=1)
        loss1 = torch.nn.functional.nll_loss(output[train_nodes], target[train_nodes])
        loss1 = loss1 + self.add_graph_loss(cur_raw_adj, init_feat, dist)
        # 循环准备
        first_adj = cur_adj
        totol_loss = loss1
        while (1):
            layer += 1
            pre_feat = cur_feat
            cur_feat = self.norms[layer](pre_feat)
            cur_raw_adj, cur_adj = self.learn_adj(init_adj, cur_feat, cluster)
            d = self.get_degree(cur_adj)
            d = self.embedding[cluster](d)
            cur_feat = F.relu(cur_feat)
            cur_feat = F.dropout(cur_feat, p=self.dropout, training=self.training)
            cur_feat = self.gcns[layer](cur_feat, cur_adj)
            pre_feat = self.args.balanced_degree * pre_feat + (1 - self.args.balanced_degree) * d
            cur_feat = cur_feat + pre_feat
            cur_dist = self.compute_dist(cur_feat)
            if cur_dist < best_dist:
                best_dist = cur_dist
                output = torch.nn.functional.log_softmax(cur_feat, dim=1)
                loss2 = torch.nn.functional.nll_loss(output[train_nodes], target[train_nodes])
                loss2 = loss2 + self.add_graph_loss(cur_raw_adj, init_feat, dist)
                totol_loss +=loss2
                self.add_network(layer, cur_feat)
            else:
                layer -= 1
                output = torch.nn.functional.log_softmax(pre_feat, dim=1)
                loss = torch.nn.functional.nll_loss(output[train_nodes], target[train_nodes])
                loss2 = loss + self.add_graph_loss(cur_raw_adj, init_feat, dist)
                totol_loss = (totol_loss + loss2)/ (layer+1)
                break
        return totol_loss, layer,output

    def test_epoch(self, node_feats, init_adj, layer, cluster):
        pre_feat = node_feats
        # cur_raw_adj经过掩码处理过的， cur_adj加上了原始的图
        cur_feat = self.norms[0](pre_feat)
        cur_raw_adj, cur_adj = self.learn_adj(init_adj, cur_feat, cluster, training=False)

        d = self.get_degree(cur_adj)
        d = self.embedding[cluster](d)
        cur_feat = F.relu(cur_feat)
        cur_feat = F.dropout(cur_feat, p=self.dropout, training=False)
        cur_feat = self.gcns[0](cur_feat, cur_adj)
        pre_feat = self.args.balanced_degree * pre_feat + (1 - self.args.balanced_degree) * d
        cur_feat = cur_feat + pre_feat

        # 循环准备
        first_adj = cur_adj
        for layers in range(layer):
            pre_feat = cur_feat
            cur_feat = self.norms[layers + 1](pre_feat)
            cur_raw_adj, cur_adj = self.learn_adj(init_adj, cur_feat, cluster, training=False)

            d = self.get_degree(cur_adj)
            d = self.embedding[cluster](d)

            cur_feat = F.relu(cur_feat)
            cur_feat = F.dropout(cur_feat, p=self.dropout, training=False)
            cur_feat = self.gcns[layers + 1](cur_feat, cur_adj)


            pre_feat = self.args.balanced_degree * pre_feat + (1 - self.args.balanced_degree) * d

            cur_feat = cur_feat + pre_feat
        output = torch.nn.functional.log_softmax(cur_feat, dim=1)
        return output
    def add_network(self, layer, cur_feat):
        if len(self.gcns) <= layer + 1:
            # self.gcns.append(GCN(self.node_feat_dim, 32, self.node_feat_dim, self.dropout)).to(self.args.cuda)
            self.gcns.append(GCNLayers(self.node_feat_dim, self.node_feat_dim)).to(self.args.cuda)
            self.norms.append(norm_layer(self.norm, self.hid_dim)).to(self.args.cuda)
            self.linear.append(nn.Linear(cur_feat.shape[1], cur_feat.shape[1])).to(self.args.cuda)
            self.d_norms.append(norm_layer(self.norm, self.hid_dim)).to(self.args.cuda)
    def compute_dist(self, cur_feat):
        dist1 = compute_dist(self.args, cur_feat, cur_feat)
        max_dist = torch.max(dist1, dim=0)[0].unsqueeze(1)
        dist1 = torch.exp(dist1 - max_dist.repeat(1, dist1.size(1)))
        intra_dist = torch.sum(dist1).mean()
        return intra_dist

    def compute_spatial(self, index_all, out_adj):
        spatial_corrdinates = sptial_neighbor_matrix(index_all, 3, out_adj)
        dist = compute_dist(self.args, spatial_corrdinates, spatial_corrdinates)
        dist = dist / torch.tile(torch.sqrt(torch.sum(dist ** 2, 1)), (dist.shape[0], 1))
        return dist
    def arpha(self, index_graph):
        index_graph_mean =list(index_graph)
        index_graph_mean = sum(index_graph_mean) /len(index_graph_mean)
        x = np.zeros(len(index_graph))
        for i in range(len(index_graph)):
            if(list(index_graph)[i]-index_graph_mean>0):
                x[i] = 1
            else:
                x[i] = 0
        index_graph = torch.DoubleTensor(x).reshape(-1, 1).to(self.args.cuda)
        return index_graph
    def to_cuda(self, x, device):
        if device:
            x = x.to(device)
        return x

    def cal_shortest_path_distance_anchor(self, adj, anchor_sets, approximate):
        num_nodes = self.num_nodes
        num_classes = self.num_classes
        avg_spd = np.zeros((num_nodes, num_classes))
        shortest_path_distance_mat = self.cal_shortest_path_distance(adj, approximate)
        for iter1 in range(num_nodes):
            for iter2 in range(num_classes):
                avg_spd[iter1][iter2] = self.cal_node_2_anchor_avg_distance(iter1, iter2, anchor_sets, shortest_path_distance_mat)

        max_spd = np.max(avg_spd)
        avg_spd = avg_spd / max_spd

        return avg_spd


    def cal_spd(self, adj, approximate):
        num_anchors = self.num_anchors
        num_nodes = self.num_nodes
        spd_mat = np.zeros((num_nodes, num_anchors))
        shortest_path_distance_mat = self.shortest_path_dists
        for iter1 in range(num_nodes):
            for iter2 in range(num_anchors):
                spd_mat[iter1][iter2] = shortest_path_distance_mat[iter1][self.anchor_node_list[iter2]]

        max_spd = np.max(spd_mat)
        spd_mat = spd_mat / max_spd

        return spd_mat


    def rank_group_pagerank(self, pagerank_before, pagerank_after):
        pagerank_dist = torch.mm(pagerank_before, pagerank_after.transpose(-1, -2)).detach().cpu()
        num_nodes = self.num_nodes
        node_pair_group_pagerank_mat = np.zeros((num_nodes, num_nodes))
        node_pair_group_pagerank_mat_list = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                node_pair_group_pagerank_mat_list.append(pagerank_dist[i, j])
        node_pair_group_pagerank_mat_list = np.array(node_pair_group_pagerank_mat_list)
        index = np.argsort(-node_pair_group_pagerank_mat_list)
        rank = np.argsort(index)
        rank = rank + 1
        iter = 0
        for i in range(num_nodes):
            for j in range(num_nodes):
                node_pair_group_pagerank_mat[i][j] = rank[iter]
                iter = iter + 1

        return node_pair_group_pagerank_mat

    def cal_shortest_path_distance(self, adj, approximate):
        n_nodes = self.num_nodes
        Adj = adj.detach().cpu().numpy()
        G = nx.from_numpy_array(Adj)
        G.edges(data=True)
        dists_array = np.zeros((n_nodes, n_nodes))
        dists_dict = self.all_pairs_shortest_path_length_parallel(G, cutoff=approximate if approximate > 0 else None)

        cnt_disconnected = 0

        for i, node_i in enumerate(G.nodes()):
            shortest_dist = dists_dict[node_i]
            for j, node_j in enumerate(G.nodes()):
                dist = shortest_dist.get(node_j, -1)
                if dist == -1:
                    cnt_disconnected += 1
                if dist != -1:
                    dists_array[node_i, node_j] = dist
        return dists_array

    def all_pairs_shortest_path_length_parallel(self, graph, cutoff=None, num_workers=4):
        nodes = list(graph.nodes)
        if len(nodes) < 50:
            num_workers = int(num_workers / 4)
        elif len(nodes) < 400:
            num_workers = int(num_workers / 2)

        pool = mp.Pool(processes=num_workers)
        results = [pool.apply_async(self.single_source_shortest_path_length_range,
                                    args=(graph, nodes[int(len(nodes) / num_workers * i):int(
                                        len(nodes) / num_workers * (i + 1))], cutoff)) for i in range(num_workers)]
        output = [p.get() for p in results]
        dists_dict = self.merge_dicts(output)
        pool.close()
        pool.join()
        return dists_dict

    def single_source_shortest_path_length_range(self, graph, node_range, cutoff):
        dists_dict = {}
        for node in node_range:
            dists_dict[node] = nx.single_source_shortest_path_length(graph, node, cutoff)  # unweighted
        return dists_dict

    def merge_dicts(self, dicts):
        result = {}
        for dictionary in dicts:
            result.update(dictionary)
        return result

    def cal_node_2_anchor_avg_distance(self, node_index, class_index, anchor_sets, shortest_path_distance_mat):
        spd_sum = 0
        count = len(anchor_sets[class_index])
        for iter in range(count):
            spd_sum += shortest_path_distance_mat[node_index][anchor_sets[class_index][iter]]
        return spd_sum / count

    def cal_group_pagerank(self, adj, data_loader, pagerank_prob):
        num_nodes = self.num_nodes
        num_classes = self.class_num

        labeled_list = [0 for _ in range(num_classes)]
        labeled_node = [[] for _ in range(num_classes)]
        labeled_node_list = []

        idx_train = data_loader['idx_train']
        labels = data_loader['labels']

        for iter1 in idx_train:
            iter_label = labels[iter1]
            labeled_node[iter_label].append(iter1)
            labeled_list[iter_label] += 1
            labeled_node_list.append(iter1)

        A = adj
        A_hat = A.to(self.args.cuda) + torch.eye(A.size(0)).to(self.args.cuda)
        D = torch.diag(torch.sum(A_hat, 1))
        D = D.inverse().sqrt()
        A_hat = torch.mm(torch.mm(D, A_hat), D)
        P = (1 - pagerank_prob) * ((torch.eye(A.size(0)).to(self.args.cuda) - pagerank_prob * A_hat).inverse())

        I_star = torch.zeros(num_nodes)

        for class_index in range(num_classes):
            Lc = labeled_list[class_index]
            if Lc == 0:
                continue  # 当Lc为零时跳过循环体

            Ic = torch.zeros(num_nodes)
            Ic[torch.tensor(labeled_node[class_index])] = 1.0 / Lc

            if class_index == 0:
                I_star = Ic
            if class_index != 0:
                I_star = torch.vstack((I_star, Ic))

        # for class_index in range(num_classes):
        #     Lc = labeled_list[class_index]
        #     Ic = torch.zeros(num_nodes)
        #     Ic[torch.tensor(labeled_node[class_index])] = 1.0 / Lc
        #     if class_index == 0:
        #         I_star = Ic
        #     if class_index != 0:
        #         I_star = torch.vstack((I_star,Ic))

        I_star = I_star.transpose(-1, -2).to(self.args.cuda)

        Z = torch.mm(P.double(), I_star.double())
        return Z

    def cal_group_pagerank_args(self, pagerank_before, pagerank_after):
        node_pair_group_pagerank_mat = self.rank_group_pagerank(pagerank_before, pagerank_after)  # rank
        num_nodes = self.num_nodes
        PI = 3.1415926
        for i in range(num_nodes):
            for j in range(num_nodes):
                node_pair_group_pagerank_mat[i][j] = 2 - (
                            math.cos((node_pair_group_pagerank_mat[i][j] / (num_nodes * num_nodes)) * PI) + 1)

        return node_pair_group_pagerank_mat


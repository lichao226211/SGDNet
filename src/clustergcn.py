import torch
import random
import numpy as np
from tqdm import trange, tqdm
from layers import StackedGCN
import copy
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from src.deep_models3 import DeeperGCN
from utils import *
from clustering import train_test_spilts
class ClusterGCNTrainer(object):
    """
    Training a ClusterGCN.
    """
    def __init__(self, args, features, clustering_machine):
        """
        :param ags: Arguments object.
        :param clustering_machine:
        """
        self.args = args
        self.layer = np.zeros(self.args.cluster_number)
        self.best_layer = np.zeros(self.args.cluster_number)
        self.clustering_machine = clustering_machine
        self.features = features
        # self.device = torch.device("cpu")
        self.device = torch.device(self.args.cuda if torch.cuda.is_available() else "cpu")
        self.create_model()

    def create_model(self):
        """
        Creating a StackedGCN and transferring to CPU/GPU.
        """

        self.model = DeeperGCN(
                      args=self.args,
                      node_feat_dim=self.features.shape[1],
                      hid_dim=self.args.hid_dim,
                      out_dim=self.clustering_machine.class_count,
                      num_layers=self.args.num_layers,
                      cluster=self.clustering_machine.clusters,
                      degree_max=self.clustering_machine.sg_nodes
        )
        # self.model = StackedGCN(self.args, self.clustering_machine.feature_count, self.clustering_machine.class_count)
        self.model = self.model.to(self.device)
        self.train_node, self.test_nodes, self.totol_sample_num, self.totol_test_num, self.mapper_list_total = train_test_spilts(self.args,self.clustering_machine.class_count,self.clustering_machine.clusters,self.clustering_machine.sg_nodes,self.clustering_machine.sg_targets)
        self.model_layer = []
        self.prediction_train = {}
        print(self.totol_sample_num, self.totol_sample_num.sum())
        print(self.totol_test_num, self.totol_test_num.sum())

    def do_forward_pass(self, cluster, target_all):
        """
        Making a forward pass with data from a given partition.
        :param cluster: Cluster index.
        :return average_loss: Average loss on the cluster.
        :return node_count: Number of nodes.
        """
        # edge_all = self.clustering_machine.edge
        # edges = self.clustering_machine.sg_edges[cluster].to(self.device)
        # macro_nodes = self.clustering_machine.sg_nodes[cluster].to(self.device)
        train_nodes = self.train_node[cluster].to(self.device)
        features = self.clustering_machine.sg_features[cluster].to(self.device)
        target = self.clustering_machine.sg_targets[cluster].to(self.device).squeeze()
        sg_node = self.clustering_machine.sg_nodes[cluster].to(self.device)
        degree = self.clustering_machine.degree[cluster].to(self.device)
        index_graph = self.clustering_machine.index_graph[cluster]
        adj = self.clustering_machine.sg_adj[cluster].to(self.device)
        index_all = self.clustering_machine.sg_nodes[cluster].to(self.device)
        # graph = self.clustering_machine.sg_graph[cluster]
        # dgl_graph =dgl.from_networkx(graph)
        # 将子图的边和特征传入，得到预测值，并计算损失
        layer_dyn = np.zeros(self.args.cluster_number)

        data = self.get_para(cluster)

        average_loss, layer, output = self.model(features, train_nodes, cluster, 'train', degree, index_graph, layer_dyn[cluster], adj, target, index_all, data)
        self.layer[cluster] = layer
        self.best_layer[cluster] = self.layer[cluster] + self.best_layer[cluster]
        node_count = train_nodes.shape[0]

        # prediction = self.model(edges, features)
        # average_loss = torch.nn.functional.nll_loss(prediction[train_nodes], target[train_nodes])
        self.prediction_train[cluster] = output
        return average_loss, node_count, self.layer[cluster]

    def update_average_loss(self, batch_average_loss, node_count):
        """
        Updating the average loss in the epoch.
        :param batch_average_loss: Loss of the cluster.
        :param node_count: Number of nodes in currently processed cluster.
        :return average_loss: Average loss in the epoch.
        """
        self.accumulated_training_loss = self.accumulated_training_loss + batch_average_loss.item()*node_count
        self.node_count_seen = self.node_count_seen + node_count
        average_loss = self.accumulated_training_loss/self.node_count_seen
        return average_loss

    def do_prediction(self, cluster, layer):
        """
        Scoring a cluster.
        :param cluster: Cluster index.
        :return prediction: Prediction matrix with probabilities.
        :return target: Target vector.
        """
        # edge_all = self.clustering_machine.edge
        # edges = self.clustering_machine.sg_edges[cluster].to(self.device)
        # macro_nodes = self.clustering_machine.sg_nodes[cluster].to(self.device)
        test_nodes = self.test_nodes[cluster].to(self.device)
        features = self.clustering_machine.sg_features[cluster].to(self.device)
        target = self.clustering_machine.sg_targets[cluster].to(self.device).squeeze()
        target = target[test_nodes]
        degree = self.clustering_machine.degree[cluster].to(self.device)
        index_graph = self.clustering_machine.index_graph[cluster]
        adj = self.clustering_machine.sg_adj[cluster].to(self.device)
        index_all = self.clustering_machine.sg_nodes[cluster].to(self.device)
        data = self.get_para(cluster)
        # 取出测试节点的预测值，size为（n,3）
        prediction= self.model(features, None, cluster, 'test', degree, index_graph, layer, adj, target,index_all, data)
        # prediction = self.model(edges, features)
        prediction = prediction[test_nodes,:]
        return prediction, target
    def normalize_adj(self, mx):
        """Row-normalize matrix: symmetric normalized Laplacian"""
        rowsum = mx.sum(1)
        r_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
        r_inv_sqrt[torch.isinf(r_inv_sqrt)] = 0.
        r_mat_inv_sqrt = torch.diag(r_inv_sqrt)
        return torch.mm(torch.mm(mx, r_mat_inv_sqrt).transpose(-1, -2), r_mat_inv_sqrt)
    def train(self,target_all):
        """
        Training a model.
        """
        print("Training started.\n")
        # 展示进度条
        epochs = trange(self.args.epochs, desc = "Train Loss")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        # 开启训练模式
        self.model.train()
        loss = []
        for epoch in epochs:
            # 打乱子图顺序
            random.shuffle(self.clustering_machine.clusters)
            self.node_count_seen = 0
            self.accumulated_training_loss = 0
            for cluster in self.clustering_machine.clusters:
                self.optimizer.zero_grad()
                batch_average_loss, node_count, layer = self.do_forward_pass(cluster, target_all)
                batch_average_loss.backward()
                self.optimizer.step()
                average_loss = self.update_average_loss(batch_average_loss, node_count)

            # print("layer",self.layer)
            epochs.set_description("Train Loss: %g" % round(average_loss, 4))
            loss.append(average_loss)
            # if average_loss < 0.1:
            #     break
        self.best_layer =np.ceil(self.best_layer / self.args.epochs)
        return loss[epoch-1]
        # fig, ax = plt.subplots()
        # ax.plot(loss, label='训练损失')
        # # 设置图形属性
        # ax.set_xlabel('Epoch')
        # ax.set_ylabel('Loss')
        # ax.set_title('损失变化曲线')
        # ax.legend()
        # plt.show()

    def test(self):
        """
        Scoring the test and printing the F-1 score.
        """
        # 进入评估模式
        self.model.eval()
        self.predictions = {}
        self.targets = {}
        self.node = {}
        mapper={}
        test_node = []
        train_node = {}
        for cluster in self.clustering_machine.clusters:
            prediction, target = self.do_prediction(cluster, self.best_layer[cluster])
            self.predictions[cluster] = prediction.cpu().detach().numpy()
            self.targets[cluster] = target.cpu().detach().numpy()
            self.node[cluster] = self.clustering_machine.sg_nodes[cluster]
            mapper[cluster] = self.clustering_machine.ori_index[cluster]
            train_node[cluster] = self.train_node[cluster]


        for j in self.clustering_machine.clusters:
            temp = []
            for i in range(len(self.test_nodes[j])):
                for key, value in self.mapper_list_total[j]:
                    if self.test_nodes[j][i] == key:
                        self.test_nodes[j][i] = value;
                        break;
            for i in range(len(train_node[j])):
                for key, value in self.mapper_list_total[j]:
                    if train_node[j][i] == key:
                        train_node[j][i] = value;
                        break;
            for i in self.test_nodes[j]:
                for k, node in enumerate(mapper[j]):
                    if i == k:
                        temp.append(node)
                        break
            test_node.append(temp)


        # self.targets = np.concatenate(self.targets)
        # self.predictions = np.concatenate(self.predictions)
        # test_node = np.concatenate(test_node)
        prediction_list = []
        node_list = []
        target_list = []
        train_node_list = []
        prediction_train_list = []
        for i in range(len(self.predictions)):
            prediction_list.extend(self.predictions[i].argmax(1))
            prediction_train_list.extend(self.prediction_train[i].argmax(1))
            node_list.extend(self.test_nodes[i])
            target_list.extend(self.targets[i])
            train_node_list.extend(train_node[i])
        node_list.extend(train_node_list)
        prediction_list.extend(prediction_train_list)
        prediction_map, gt_map = get_map(prediction_list, node_list, target_list)
        Draw_Classification_Map(prediction_map, "test")
        # self.predictions = np.concatenate(self.predictions).argmax(1)
        OA, Kap, AA, CA= Cal_accuracy(prediction_list, target_list)
        # score = f1_score(self.targets, self.predictions.argmax(1), average="micro")
        print("\nOA: {:.4f}".format(OA),
              "Kap:{:.4f}".format(Kap),
              "AA:{:.4f}".format(AA),
              '\nCA:', {"{:.4f}".format(i) for i in CA}
              )
        return OA , Kap, AA, CA

    def get_para(self, cluster):
        feature = self.clustering_machine.sg_features[cluster].to(self.device)
        train_nodes = self.train_node[cluster].to(self.device)
        test_nodes = self.test_nodes[cluster].to(self.device)
        target = self.clustering_machine.sg_targets[cluster].to(self.device).squeeze()
        data = {'adj': self.clustering_machine.sg_adj[cluster].to(self.device) ,
                'features': feature.to(self.device),
                'labels': target.to(self.device),
                'idx_train': train_nodes.to(self.device) ,
                'idx_test': test_nodes.to(self.device) ,
                'labels_train': target[train_nodes],
                'labels_test': target[test_nodes]}
        return data
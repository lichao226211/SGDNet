import math
import scipy.io as sio
import torch
import numpy as np
import pandas as pd
import networkx as nx
from matplotlib import pyplot as plt, colors
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from texttable import Texttable
from scipy.sparse import coo_matrix
from sklearn.model_selection import train_test_split
import scipy.sparse as sp
import numpy as np
import scipy.io
from src.test import _split_with_min_per_class
import spectral as spy

def encode_onehot(labels):
    classes = len(set(list(labels)))
    classes_dict = {c: np.identity(classes)[i, :] for i, c in
                    enumerate(set(list(labels)))}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot
# 转换格式
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(torch.double)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.DoubleTensor(indices, values, shape)

def aug_random_walk(adj):
    adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv = np.power(row_sum, -1.0).flatten()
    d_mat = sp.diags(d_inv)
    return (d_mat.dot(adj)).tocoo()

def aver(adj, hops, feature_list, alpha=0.15):
    input_feature = []
    for i in range(adj.shape[0]):
        hop = hops[i].int().item()
        if hop == 0:
            fea = feature_list[0][i].unsqueeze(0)
            print("fel",feature_list[0][i])
        else:
            fea = 0
            for j in range(hop):
                #  1-alpha 表示当前层的贡献，而 alpha 表示原始特征向量的贡献。最后将加权平均后的特征向量作为该节点的平滑后特征，并将其添加到 input_feature 列表中
                fea += (1-alpha)*feature_list[j][i].unsqueeze(0) + alpha*feature_list[0][i].unsqueeze(0)
            fea = fea / hop
        input_feature.append(fea)
    input_feature = torch.cat(input_feature, dim=0)
    return input_feature

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    # sorted是对字段以一种逻辑进行排序
    keys = sorted(args.keys())
    # 可视化打印
    t = Texttable()
    t.add_rows([["Parameter", "Value"]] +  [[k.replace("_"," ").capitalize(),args[k]] for k in keys])
    print(t.draw())

def graph_reader(path):
    """
    Function to read the graph from the path.
    :param path: Path to the edge list.
    :return graph: NetworkX object returned.
    """
    edges = pd.read_csv(path)
    # # # nx.from_pandas_edgelist() 函数从 Pandas DataFrame 创建图形。edge_attr='e_fet' 参数将 e_fet 列中的值作为边的权重添加到图中。
    # # # create_using=nx.DiGraph() 参数指定创建一个有向图对象。
    # # # create_using 参数的值被更改为 nx.Graph()，表示将创建一个无向图对象。
    # edge_source = edges["id1"].values.tolist()
    # edge_target = edges["id2"].values.tolist()
    # edge_fea1 = edges["e_fet"].values.tolist()
    # edge_fea = coo_matrix((edge_fea1, (edge_source, edge_target)), shape=(edges.shape[0], 1)).toarray()
    # graph = nx.from_pandas_edgelist(edges, source='id1', target='id2', create_using=nx.Graph())
    # from_edgelist返回一个由列表中元素构成的图形
    graph = nx.from_edgelist(edges.values.tolist())
    return graph

def feature_reader(path):
    """
    Reading the sparse feature matrix stored as csv from the disk.
    :param path: Path to the csv file.
    :return features: Dense matrix of features.
    """
    features = pd.read_csv(path)
    node_index = features["node_id"].values.tolist()
    feature_index = features["feature"].values.tolist()
    feature_values = features["value"].values.tolist()
    node_count = max(node_index)+1
    feature_count = max(feature_index)+1
    features = coo_matrix((feature_values, (node_index, feature_index)), shape=(node_count, feature_count)).toarray()
    if (feature_count > 60):
        pca = PCA(n_components=60)
        # 拟合数据并进行降维
        reduced_data = pca.fit_transform(features)
        return reduced_data
    else:
        return features

def target_reader(path):
    """
    Reading the target vector from disk.
    :param path: Path to the target.
    :return target: Target vector.
    """
    target = np.array(pd.read_csv(path)["label"]).reshape(-1,1)
    return target

def target_all_reader(path):
    # gnd = sio.loadmat(path)['PaviaU_gt']
    # gnd = sio.loadmat(path)['Houston_gt']
    gnd = sio.loadmat(path)['Indian_pines_gt']
    gnd = np.array(gnd)
    target_all = gnd
    return target_all

def ind2sub1(array_shape, ind):
    # array_shape is array,an array of two elements where the first element is the number of rows in the matrix
    # and the second element is the number of columns.
    # ind is vector index with the python rule
    ind = np.array(ind.cpu().numpy())
    array_shape = np.array(array_shape)
    rows = (ind.astype('int') // array_shape[1].astype('int'))
    cols = (ind.astype('int') % array_shape[1])  # or numpy.mod(ind.astype('int'), array_shape[1])
    return rows, cols
def ind2sub(array_shape, ind):
    array_shape = torch.tensor(array_shape)  # 将数组形状转换为 PyTorch 张量
    rows = ind // array_shape[1]  # 计算行索引
    cols = ind % array_shape[1]  # 计算列索引
    return rows, cols  # 返回行索引和列索引
def sptial_neighbor_matrix(index_all, neighbor, gt):
    """extract the spatial neighbor matrix, if x_j belong to x_i neighbors, thus S_ij = 1"""
    # index_all = np.concatenate((index_train_all, index_test))
    # index_all = dy11
    L_cor = torch.zeros([2, 1])
    for kkk in range(len(index_all)):
        [X_cor, Y_cor] = ind2sub([gt.size(0), gt.size(1)], index_all[kkk])
        XY_cor = torch.tensor([X_cor, Y_cor]).view(2, 1)
        L_cor = torch.cat((L_cor, XY_cor), dim=1)
    L_cor = L_cor[:, 1:]  # 删除第一列
    return torch.transpose(L_cor, 0, 1)


def compute_dist(args, array1, array2, type='euclidean'):
    """Compute the euclidean or cosine distance of all pairs.
    Args:
        array1: numpy array or torch tensor on GPU with shape [m1, n]
        array2: numpy array or torch tensor on GPU with shape [m2, n]
        type: one of ['cosine', 'euclidean']
    Returns:
        numpy array or torch tensor on GPU with shape [m1, m2]
    """
    device = args.cuda
    assert type in ['cosine', 'euclidean']
    if isinstance(array1, np.ndarray):
        array1 = torch.from_numpy(array1)
    if isinstance(array2, np.ndarray):
        array2 = torch.from_numpy(array2)
    if torch.cuda.is_available():
        array1 = array1.to(device)
        array2 = array2.to(device)
    if type == 'cosine':
        dist = torch.matmul(array1, array2.T)
        return dist
    else:
        square1 = torch.sum(torch.square(array1), dim=1).unsqueeze(1)
        square2 = torch.sum(torch.square(array2), dim=1).unsqueeze(0)
        t = -2 * torch.matmul(array1, array2.T)
        squared_dist = t + square1 + square2
        squared_dist[squared_dist < 0] = 0
        dist = torch.sqrt(squared_dist)
        return dist


def compute_dist1(array1, array2, type='euclidean'):
    """Compute the euclidean or cosine distance of all pairs.
  Args:
    array1: numpy array with shape [m1, n]
    array2: numpy array with shape [m2, n]
    type: one of ['cosine', 'euclidean']
  Returns:
    numpy array with shape [m1, m2]
  """

    assert type in ['cosine', 'euclidean']
    if type == 'cosine':
        #array1 = normalize(array1, axis=1)
       # array2 = normalize(array2, axis=1)
        dist = np.matmul(array1, array2.T)
        return dist
    else:
        # shape [m1, 1]
        # 平方后按列求和，
        square1 = np.sum(np.square(array1), axis=1)[..., np.newaxis]
        # shape [1, m2]
        square2 = np.sum(np.square(array2), axis=1)[np.newaxis, ...]
        t = - 2 * np.matmul(array1, array2.T)
        squared_dist = t + square1 + square2
        squared_dist[squared_dist < 0] = 0
        dist = np.sqrt(squared_dist)#np.exp(dist - np.tile(np.max(dist, axis=0)[..., np.newaxis], np.size(dist, 1)))
        return dist
def train_test_spilts(args, class_count, clusters, sg_nodes, sg_targets):
    lists = np.zeros(int(class_count))
    sg_train_nodes={}
    sg_test_nodes={}
    all_data_nodes={}
    mapper_list_total = {}
    totol_sample_num = np.zeros(int(class_count) )
    totol_test_num = np.zeros(int(class_count) )
    # mask = np.zeros(int(class_count) )
    # mask_totol = np.zeros(int(class_count) )
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    for cluster in clusters:
        sg_train_nodes[cluster], sg_test_nodes[cluster], class_num, test_num, mapper_list = _split_with_min_per_class(
            X=sg_nodes[cluster], y=sg_targets[cluster], test_size=args.test_ratio, list=lists,
            node_index=sg_nodes[cluster])
        mapper_list_total[cluster] = mapper_list
        # sg_train_nodes[cluster], sg_test_nodes[cluster], class_num, test_num = _split_with_min_per_class(
        #     X=sg_nodes[cluster], y=sg_targets[cluster], test_size=args.test_ratio, list=lists)


        print("子图{}训练集数量：{}".format(cluster, class_num))
        # print("子图{}测试集数量：{}".format(cluster, test_num))
        # print(type(all_data_nodes[cluster]),'\n', len(all_data_nodes[cluster]))
        # print(sg_train_nodes[cluster].shape, '\n', sg_test_nodes[cluster].shape)
        all_data_nodes[cluster] = np.array(sg_train_nodes[cluster]+sg_test_nodes[cluster])

        # 保存子图
        # save_figure(all_data_nodes[cluster], sg_nodes[cluster], sg_targets[cluster], cluster)
        sg_test_nodes[cluster] = sorted(sg_test_nodes[cluster])
        sg_train_nodes[cluster] = sorted(sg_train_nodes[cluster])
        sg_train_nodes[cluster] = torch.LongTensor(sg_train_nodes[cluster])
        sg_test_nodes[cluster] = torch.LongTensor(sg_test_nodes[cluster])
        totol_sample_num += class_num
        totol_test_num += test_num
    return sg_train_nodes, sg_test_nodes, totol_sample_num, totol_test_num, mapper_list_total

# def Cal_accuracy(predict, label):
#     estim_label = predict.argmax(1)
#     # estim_label = estim_label.detach().cpu().numpy()
#     # true_label = label.detach().cpu().numpy()
#     true_label = label
#     n = true_label.shape[0]
#     OA = np.sum(estim_label == true_label) * 1.0 / n
#     correct_sum = np.zeros((max(true_label) + 1))
#     reali = np.zeros((max(true_label) + 1))
#     predicti = np.zeros((max(true_label) + 1))
#     producerA = np.zeros((max(true_label) + 1))
#
#     predictions = []
#
#     # 循环计算每个类别的预测结果
#     for i in range(0, max(true_label) + 1):
#         correct_sum[i] = np.sum(true_label[np.where(estim_label == i)] == i)
#         reali[i] = np.sum(true_label == i)
#         predicti[i] = np.sum(estim_label == i)
#         producerA[i] = correct_sum[i] / reali[i]
#
#         # 计算预测结果并添加到列表中
#         predictions.append(producerA[i])
#
#     # 计算预测结果的均值
#     predictions_mean = np.mean(predictions)
#     # print(producerA)
#     Kappa = (n * np.sum(correct_sum) - np.sum(reali * predicti)) * 1.0 / (n * n - np.sum(reali * predicti))
#     return OA, Kappa, predictions_mean, predictions
def Cal_accuracy(predict, label):
    estim_label = predict
    # estim_label = estim_label.detach().cpu().numpy()
    # true_label = label.detach().cpu().numpy()
    true_label = label
    n = true_label.shape[0]
    OA = np.sum(estim_label == true_label) * 1.0 / n
    correct_sum = np.zeros((max(true_label) + 1))
    reali = np.zeros((max(true_label) + 1))
    predicti = np.zeros((max(true_label) + 1))
    producerA = np.zeros((max(true_label) + 1))

    predictions = []

    # 循环计算每个类别的预测结果
    for i in range(1, max(true_label) + 1):
        correct_sum[i] = np.sum(true_label[np.where(estim_label == i)] == i)
        reali[i] = np.sum(true_label == i)
        predicti[i] = np.sum(estim_label == i)
        producerA[i] = correct_sum[i] / reali[i]

        # 计算预测结果并添加到列表中
        predictions.append(producerA[i])

    # 计算预测结果的均值
    predictions_mean = np.mean(predictions)
    # print(producerA)
    Kappa = (n * np.sum(correct_sum) - np.sum(reali * predicti)) * 1.0 / (n * n - np.sum(reali * predicti))
    return OA, Kappa, predictions_mean, predictions

def save_figure(data, feature, target,cluster):
    train_test_data = data
    n = train_test_data.shape[0]
    l = feature.shape[1]

    x = torch.ceil(torch.sqrt(torch.tensor(n)))
    x = int(x)
    # features = np.zeros((n,l))
    # targets = np.zeros((n, 1))

    # 假设节点的数量为 n，特征维度为 60
    feature_dim = 60

    # 创建形状为 (145, 145, 60) 的矩阵，并初始化为零
    matrix = np.zeros((x, x, feature_dim))
    targets = np.zeros((x, x, 1))
    # 将节点映射到矩阵中
    for k in range(n):
        node_index = train_test_data[k]
        # 计算节点在矩阵中的索引
        i = node_index // x
        j = node_index % x
        # 将节点的索引号赋值给对应的矩阵位置
        matrix[i, j, :] = feature[node_index, :]
        targets[i, j, :] = target[node_index, :]
    # for i in range(n):
    #     features[i,:] = feature[train_test_data[i], :]

    # features = np.expand_dims(feature, axis=1)
    feature_dict = {
        'feature': matrix
    }
    label_dict ={
        'target': targets
    }

    # 保存为.mat文件
    scipy.io.savemat('../cluster_figure/Houston'+str(cluster)+'.mat', feature_dict)
    scipy.io.savemat('../cluster_figure/Houston_gt'+str(cluster)+'.mat', label_dict)

def Draw_Classification_Map(label, name: str, scale: float = 4.0, dpi: int = 400):  # 绘制全分类图
    """
            get classification map , then save to given path
            :param label: classification label, 2D
            :param name: saving path and file's name
            :param scale: scale of image. If equals to 1, then saving-size is just the label-size
            :param dpi: default is OK
            :return: null
            """
    height, width = label.shape
    # gt_mat = sio.loadmat('../input/Houston_gt.mat')
    # gt = gt_mat['Houston_gt']

    # gt_mat = sio.loadmat('../input/PaviaU_gt.mat')
    # gt = gt_mat['PaviaU_gt']
    gt_mat = sio.loadmat('../input/Indian_pines_gt.mat')
    gt = gt_mat['Indian_pines_gt']
    custom_colors = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [1, 1, 0],
        [0.62, 0.13, 0.94],
        [0.33, 0.1, 0.55],
        [0, 0, 0.55],
        [0.18, 0.2, 0.34],
        [0.63, 0.32, 0.18],
        [0.85, 0.44, 0.84],
        [0, 0.55, 0],
        [1, 0, 1],
        [0.69, 0.19, 0.38],
        [1, 0.5, 0.31],
        [0.5, 1, 0.83],
        [0.8, 0.8, 0]
    ]

    class_count  = 16
    # 创建自定义颜色映射
    cmap = ListedColormap(custom_colors[:class_count])

    plt.set_cmap(cmap)
    temp_zeros = np.zeros((height, width))
    fig, ax = plt.subplots()

    # 使用颜色映射来给每个类别着色
    truth = np.where(gt != 0, label, temp_zeros)

    # 使用imshow绘制分类地图
    v = plt.imshow(truth.astype(np.int16), cmap=cmap)  # 修改此处

    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    fig.set_size_inches(label.shape[1] * scale / dpi, label.shape[0] * scale / dpi)
    foo_fig = plt.gcf()  # 'get current figure'

    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    foo_fig.show()

    foo_fig.savefig(name + '.png', format='png', transparent=True, dpi=dpi, pad_inches=0)
    pass

# def get_map(prediction, node, mapper):
#     # file_path = '../data/houston/relation.txt'
#     file_path= '../data/indian/relation-indian.txt'
#     list = {}
#     with open(file_path, 'r') as file:
#         data = file.read()
#     key_value_pairs = data.split(',')
#     temp1 = []
#     j = 0
#     for item in key_value_pairs:
#         key, value = map(int, item.split(":"))
#         list[key] = value
#     for i in node:
#         for k, nodes in enumerate(list):
#             if i == k:
#                 temp1.append(nodes)
#                 break
#     # label = torch.zeros((349, 1905))
#     label = torch.zeros((145,145))
#     for i in range(len(temp1)):
#         # row = temp1[i] // 1905
#         # col = temp1[i] %  1905
#         row = temp1[i] // 145
#         col = temp1[i] % 145
#         label[row][col] = prediction[i]
#     return label

def get_map(prediction_list, node_list, target_list):
    # prediction_list = []
    # node_list = []
    # target_list = []
    # for i in range(len(prediction)):
    #     prediction_list.extend(prediction[i].argmax(1))
    #     node_list.extend(node[i])
    #     target_list.extend(target[i])
    # file_path = '../data/paviau/relation-PaviaU.txt'
    # file_path = '../data/houston/relation.txt'
    file_path = '../data/indian/relation-indian.txt'
    list = {}
    with open(file_path, 'r') as file:
        data = file.read()
    key_value_pairs = data.split(',')
    temp1 = []
    # j = 0
    for item in key_value_pairs:
        key, value = map(int, item.split(":"))
        list[key] = value
    for i in node_list:
        for k, nodes in enumerate(list):
            if i == k:
                temp1.append(nodes)
                break
    # label = torch.zeros((610, 340))
    # gt = torch.zeros((610, 340))
    # label = torch.zeros((349, 1905))
    label = torch.zeros((145, 145))
    gt = torch.zeros((145, 145))
    # gt = torch.zeros((349, 1905))
    for i in range(len(temp1)):
        row = temp1[i] // 145
        col = temp1[i] % 145
        # row = temp1[i] // 340
        # col = temp1[i] %  340
        # row = temp1[i] // 1905
        # col = temp1[i] % 1905
        label[row][col] = prediction_list[i]
    # sio.savemat('../pred.mat', {'pred': label})
    # sio.savemat('../gt.mat', {'gt': gt})
    return label, gt
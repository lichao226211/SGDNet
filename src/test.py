import random
from math import ceil, floor
# 111
from random import shuffle

import numpy as np
from itertools import chain, combinations

from sklearn.utils import indexable, _safe_indexing
from sklearn.utils.validation import _num_samples
import matplotlib.pyplot as plt
from operator import truediv
import scipy.io as sio
import torch
import math

def _split_with_min_per_class(X, y, test_size,list, node_index):
    """
    在按照百分比划分训练集时，最少每个类有一个样本作为训练集的训练对象

    参数：
    X: array-like，特征数据
    y: array-like，标签数据
    test_size: float，测试集占比
    min_per_class: int，每个类别的最小训练样本数
    random_state: int or RandomState，随机数种子，可选

    返回：
    train_idx: array，训练集索引
    test_idx: array，测试集索引
    """

    # 检查特征数据和标签数据的长度是否一致
    if len(X) != len(y):
        raise ValueError("The length of X and y must be the same.")

    # 获取每个类别的标签
    classes = np.unique(y)

    # 初始化训练集和测试集索引
    train_idx = []
    test_idx = []
    class_sample_counts = np.zeros(len(list) + 1)
    class_test_counts = np.zeros(len(list) + 1)
    mapper_list = []
    # 对每个类别分别处理
    for c in classes:

        # 获取当前类别下所有样本的索引
        idx_c = np.where(y == c)[0]

        idx = [int(node_index[i]) for i in idx_c]
        mapper = {key: value for key, value in zip(idx_c, idx)}
        mapper_list.extend(mapper.items())
        # for key, value in mapper_list:
        #     print(key, value)
        # 计算当前类别需要保留的测试样本数
        n_test = int(np.floor(len(idx_c) * test_size))

        # 计算当前类别需要保留的训练样本数
        n_train = len(idx) - n_test

        # 随机打乱当前类别下所有样本的索引
        shuffle(idx)

        # 将随机打乱后的前n_train个样本加入训练集
        train_idx.extend(idx_c[:n_train])

        # 将随机打乱后的n_train到n_train+n_test个样本加入测试集
        test_idx.extend(idx_c[n_train:n_train + n_test])
        class_sample_counts[int(c)] = n_train
        class_test_counts[int(c)] = n_test

    return train_idx, test_idx, class_sample_counts[1:], class_test_counts[1:], mapper_list

# def _split_with_min_per_class(X, y, test_size,list, min_per_class=2, target_train_samples=20, random_state=None):
#     """
#     在按照百分比划分训练集时，最少每个类有一个样本作为训练集的训练对象
#
#     参数：
#     X: array-like，特征数据
#     y: array-like，标签数据
#     test_size: float，测试集占比
#     min_per_class: int，每个类别的最小训练样本数
#     random_state: int or RandomState，随机数种子，可选
#
#     返回：
#     train_idx: array，训练集索引
#     test_idx: array，测试集索引
#     """
#
#     # 检查特征数据和标签数据的长度是否一致
#     if len(X) != len(y):
#         raise ValueError("The length of X and y must be the same.")
#
#     # 获取每个类别的标签
#     classes = np.unique(y)
#
#     # 初始化训练集和测试集索引
#     train_idx = []
#     test_idx = []
#     class_sample_counts = np.zeros(len(list))
#     class_test_counts = np.zeros(len(list))
#     # 对每个类别分别处理
#     for c in classes:
#
#         # 获取当前类别下所有样本的索引
#         idx_c = np.where(y == c)[0]
#
#         # 如果当前类别样本数小于最小训练样本数，则将所有样本加入训练集
#         # if len(idx_c) < min_per_class:
#         #     train_idx.extend(idx_c)
#         #     class_sample_counts[int(c)] = len(idx_c)
#
#         # 如果当前类别样本数大于或等于最小训练样本数，则按照给定比例划分训练集和测试集
#         # else:
#         # 计算当前类别需要保留的最小训练样本数
#         # min_train_samples = min_per_class - 1
#
#         # 计算当前类别需要保留的测试样本数
#         n_test = int(np.floor(len(idx_c) * test_size))
#
#         # 计算当前类别需要保留的训练样本数
#         n_train = len(idx_c) - n_test
#
#         # 随机打乱当前类别下所有样本的索引
#         shuffle(idx_c)
#
#         # 将随机打乱后的前n_train个样本加入训练集
#         train_idx.extend(idx_c[:n_train])
#
#         # 将随机打乱后的n_train到n_train+n_test个样本加入测试集
#         test_idx.extend(idx_c[n_train:n_train + n_test])
#         class_sample_counts[int(c)] = n_train
#         class_test_counts[int(c)] = n_test
#
#     return train_idx, test_idx, class_sample_counts, class_test_counts
def _split_with_min_per_class1(X, y, test_size,list, min_per_class=2, target_train_samples=20, random_state=None):
    """
    在按照百分比划分训练集时，最少每个类有一个样本作为训练集的训练对象

    参数：
    X: array-like，特征数据
    y: array-like，标签数据
    test_size: float，测试集占比
    min_per_class: int，每个类别的最小训练样本数
    random_state: int or RandomState，随机数种子，可选

    返回：
    train_idx: array，训练集索引
    test_idx: array，测试集索引
    """

    # 检查特征数据和标签数据的长度是否一致
    if len(X) != len(y):
        raise ValueError("The length of X and y must be the same.")

    # 获取每个类别的标签
    classes = np.unique(y)

    # 初始化训练集和测试集索引
    train_idx = []
    test_idx = []
    class_sample_counts = np.zeros(len(list))
    class_test_counts = np.zeros(len(list))

    # 对每个类别分别处理
    for c in classes:
        idx = np.where(y == c)[0]
        samplesCount = len(idx)
        rand_list = [i for i in range(samplesCount)]  # 用于随机的列表
        sample_num = 5
        if sample_num < samplesCount:
            # 随机数数量 四舍五入(改为上取整)
            rand_idx = random.sample(rand_list, int(sample_num))
            n_train = sample_num
            n_test = len(idx) - n_train
        else:
            rand_idx = random.sample(rand_list, int(samplesCount))
            n_train = samplesCount
            n_test = len(idx) - n_train
        rand_real_idx_per_class = idx[rand_idx]
        train_idx.append(rand_real_idx_per_class)
        class_sample_counts[int(c)] = n_train
        class_test_counts[int(c)] = n_test

    train_rand_idx = np.array(train_idx, dtype=object)
    train_data_index = []
    for c in range(train_rand_idx.shape[0]):
        a = train_rand_idx[c]
        for j in range(a.shape[0]):
            train_data_index.append(a[j])
    train_data_index = np.array(train_data_index)

    ##将测试集（所有样本，包括训练样本）也转化为特定形式
    train_data_index = set(train_data_index)
    all_data_index = [i for i in range(len(y))]
    all_data_index = set(all_data_index)
    test_data_index = all_data_index - train_data_index

    all_data_index = np.array(all_data_index)
    return train_data_index, test_data_index,all_data_index,class_sample_counts, class_test_counts

def train_test_split(
    features,
    targets,
    *arrays,
    test_size=None,
    train_size=None,
    random_state=None,
    shuffle=True,
    stratify=None,
):
    n_arrays = len(arrays)
    if n_arrays == 0:
        raise ValueError("At least on e array required as input")

    arrays = indexable(*arrays)

    n_samples = _num_samples(arrays[0])
    n_train, n_test = _validate_shuffle_split(
        n_samples, test_size, train_size, default_test_size=0.25
    )
    train, test = _split_with_min_per_class(features, targets,16)
    return list(
        chain.from_iterable(
            (_safe_indexing(a, train), _safe_indexing(a, test)) for a in arrays
        )
    )

def sampling(proportion, ground_truth):#proportion划分比率，gound_truth标签
    train = {}
    test = {}
    labels_loc = {}
    m = max(ground_truth)
    #取出有标签数据
    for i in range(m):
        indexes = [j for j, x in enumerate(ground_truth.ravel().tolist()) if x == i + 1]##ravel()方法将数组维度拉成一维数组
        #打乱某一类标签坐标
        np.random.shuffle(indexes)
        labels_loc[i] = indexes
        if proportion != 1:
            nb_val = max(int((1 - proportion) * len(indexes)), 3)#一个类， 最少取3个，最多取百分0.2，所以造成了类间数量不平衡
        else:
            nb_val = 0
        # print(i, nb_val, indexes[:nb_val])
        # train[i] = indexes[:-nb_val]
        # test[i] = indexes[-nb_val:]
        train[i] = indexes[:nb_val]
        test[i] = indexes[nb_val:]
    #每类取不同数量，形成字典
    train_indexes = []
    test_indexes = []
   #将字典展成列表
    for i in range(m):
        train_indexes += train[i]
        test_indexes += test[i]
    #打乱训练集测试机标签的索引
    np.random.shuffle(train_indexes)
    np.random.shuffle(test_indexes)
    return train_indexes, test_indexes
def _validate_shuffle_split(n_samples, test_size, train_size, default_test_size=None):
    """
    Validation helper to check if the test/test sizes are meaningful wrt to the
    size of the data (n_samples)
    """
    if test_size is None and train_size is None:
        test_size = default_test_size

    test_size_type = np.asarray(test_size).dtype.kind
    train_size_type = np.asarray(train_size).dtype.kind

    if (
        test_size_type == "i"
        and (test_size >= n_samples or test_size <= 0)
        or test_size_type == "f"
        and (test_size <= 0 or test_size >= 1)
    ):
        raise ValueError(
            "test_size={0} should be either positive and smaller"
            " than the number of samples {1} or a float in the "
            "(0, 1) range".format(test_size, n_samples)
        )

    if (
        train_size_type == "i"
        and (train_size >= n_samples or train_size <= 0)
        or train_size_type == "f"
        and (train_size <= 0 or train_size >= 1)
    ):
        raise ValueError(
            "train_size={0} should be either positive and smaller"
            " than the number of samples {1} or a float in the "
            "(0, 1) range".format(train_size, n_samples)
        )

    if train_size is not None and train_size_type not in ("i", "f"):
        raise ValueError("Invalid value for train_size: {}".format(train_size))
    if test_size is not None and test_size_type not in ("i", "f"):
        raise ValueError("Invalid value for test_size: {}".format(test_size))

    if train_size_type == "f" and test_size_type == "f" and train_size + test_size > 1:
        raise ValueError(
            "The sum of test_size and train_size = {}, should be in the (0, 1)"
            " range. Reduce test_size and/or train_size.".format(train_size + test_size)
        )

    if test_size_type == "f":
        n_test = ceil(test_size * n_samples)
    elif test_size_type == "i":
        n_test = float(test_size)

    if train_size_type == "f":
        n_train = floor(train_size * n_samples)
    elif train_size_type == "i":
        n_train = float(train_size)

    if train_size is None:
        n_train = n_samples - n_test
    elif test_size is None:
        n_test = n_samples - n_train

    if n_train + n_test > n_samples:
        raise ValueError(
            "The sum of train_size and test_size = %d, "
            "should be smaller than the number of "
            "samples %d. Reduce test_size and/or "
            "train_size." % (n_train + n_test, n_samples)
        )

    n_train, n_test = int(n_train), int(n_test)

    if n_train == 0:
        raise ValueError(
            "With n_samples={}, test_size={} and train_size={}, the "
            "resulting train set will be empty. Adjust any of the "
            "aforementioned parameters.".format(n_samples, test_size, train_size)
        )

    return n_train, n_test
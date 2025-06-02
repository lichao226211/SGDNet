import numpy as np
import pandas as pd
import torch
from src.parser import parameter_parser
from clustering import ClusteringMachine
from clustergcn import ClusterGCNTrainer
from utils import tab_printer, graph_reader, feature_reader, target_reader, target_all_reader

def train_test(args,features, clustering_machine,target_all):

    epoch = 1
    loss_mean = 0.0
    OA_mean= 0.0
    Kap_mean= 0.0
    AA_mean = 0.0
    CA_mean = torch.zeros((len(np.unique(target_all)) + 1))
    OA1 = []
    kap1 = []
    AA1 = []
    CA1 = {}
    for i in range(epoch):
        # 初始化模型
        gcn_trainer = ClusterGCNTrainer(args, features, clustering_machine)
        loss = gcn_trainer.train(target_all)
        OA, Kap, AA, CA = gcn_trainer.test()
        OA1.append(OA)
        kap1.append(Kap)
        AA1.append(AA)

        loss_mean = float(loss) + loss_mean
        OA_mean = float(OA) + OA_mean
        Kap_mean = float(Kap) + Kap_mean
        AA_mean = float(AA) + AA_mean
        result = []
        CA1[i] = CA
        for num1, num2 in zip(CA, CA_mean):
            result.append(num1 + num2)
            CA_mean = result

    OA_mean /= epoch
    Kap_mean /= epoch
    AA_mean /= epoch
    loss_mean /= epoch

    OA1_dict = np.append(OA1, OA_mean)
    AA1_dict = np.append(AA1, AA_mean)
    kap1_dict = np.append(kap1, Kap_mean)

    result_dict = {
        'OA': OA1_dict,
        'Kap': kap1_dict,
        'AA': AA1_dict,
    }

    for i in range(epoch):
        for j, x in enumerate(CA1[i]):
            if i == 0:
                result_dict[j] = [x]
            else:
                result_dict[j].append(x)

    for i, x in enumerate(CA_mean):
        x /= epoch
        CA_mean[i] = x
        result_dict[i].append(x)
    # 将字典转换为DataFrame
    result_df = pd.DataFrame(result_dict)

    # 保存为CSV文件
    save_path = args.save_path
    save_file_path = save_path +'/'+ args.Dataname +'-'+ str(args.test_ratio)+'.csv'
    result_df.to_csv(save_file_path, index=False)
    print("\nOA: ", OA1,
          "\nKap:", kap1,
          "\nAA:", AA1,
              )
    print("\nOA_mean: {:.4f}".format(OA_mean),
          "Kap_mean:{:.4f}".format(Kap_mean),
          "AA_mean:{:.4f}".format(AA_mean),
          "Train_loss:{:.4f}".format(loss_mean))

def main():
    """
    Parsing command line parameters, reading data, graph decomposition, fitting a ClusterGCN and scoring the model.
    """
    # 配置参数
    args = parameter_parser()
    # 设置随机数种子，保证实验结果的可再现
    torch.manual_seed(args.seed)
    # 可视化参数表制作
    tab_printer(args)
    # 读取边索引
    graph = graph_reader(args.edge_path)
    # 读取特征索引,生成特征矩阵
    features = feature_reader(args.features_path)
    # 这里的target指的是标签真值。
    target = target_reader(args.target_path)
    # target_all = target_all_reader(args.Houston_Target)
    target_all = target_all_reader(args.Indian_Target)
    # target_all = target_all_reader(args.PaviaU_Target)
    # 对聚类器做一个初始化
    clustering_machine = ClusteringMachine(args, graph, features, target)
    # 拆分子图
    clustering_machine.decompose()
    # 训练，测试
    train_test(args, features, clustering_machine,target_all)

if __name__ == "__main__":
    main()

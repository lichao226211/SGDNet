import argparse

def parameter_parser():
    parser = argparse.ArgumentParser(description = "Run .")

    # Indina_Pine
    # parser.add_argument("--edge-path",
    #                     nargs = "?",
    #                     default = "D:/paper/paperwithcode/ClusterGCN-master/input/egde_indian2.csv",
	#                 help = "Edge list csv.")
    # parser.add_argument("--Indian-Target",
    #                     nargs="?",
    #                     default="D:/paper/paperwithcode/ClusterGCN-master/input/Indian_pines_gt.mat",
    #                     help="Edge list csv.")
    # parser.add_argument("--features-path",
    #                     nargs = "?",
    #                     default = "D:/paper/paperwithcode/ClusterGCN-master/input/features_indian2.csv",
	#                 help = "Features json.")
    #
    # parser.add_argument("--target-path",
    #                     nargs = "?",
    #                     default = "D:/paper/paperwithcode/ClusterGCN-master/input/target_indian2.csv",
	#                 help = "Target classes csv.")

    parser.add_argument("--edge-path",
                        nargs="?",
                        default="D:\paper\paperwithcode\ClusterGCN-master3\data\indian\egde_indian.csv",
                        help="Edge list csv.")
    parser.add_argument("--Indian-Target",
                        nargs="?",
                        default="D:/paper/paperwithcode/ClusterGCN-master/input/Indian_pines_gt.mat",
                        help="Edge list csv.")
    parser.add_argument("--features-path",
                        nargs="?",
                        default="D:\paper\paperwithcode\ClusterGCN-master3\data\indian/features_indian.csv",
                        help="Features json.")

    parser.add_argument("--target-path",
                        nargs="?",
                        default="D:/paper/paperwithcode/ClusterGCN-master3/data/indian/target_indian.csv",
                        help="Target classes csv.")
    # #
    parser.add_argument("--save-path",
                        nargs="?",
                        default="D:/paper/paperwithcode/ClusterGCN-master3/output/Indian",
                        help="Target classes csv.")
    # parser.add_argument("--save-path",
    #                     nargs="?",
    #                     default="/data/lc/train/Cluster-GCN/ClusterGCN-master/output",
    #                     help="Target classes csv.")

    # PaviaU
    # parser.add_argument("--edge-path",
    #                     nargs="?",
    #                     default="D:/paper/paperwithcode/ClusterGCN-master3/data\paviau/egde_PaviaU_all.csv",
    #                     help="Edge list csv.")
    # parser.add_argument("--PaviaU-Target",
    #                     nargs="?",
    #                     default="D:/paper/paperwithcode/ClusterGCN-master3/input/PaviaU_gt.mat",
    #                     help="Edge list csv.")
    # parser.add_argument("--features-path",
    #                     nargs="?",
    #                     default="D:\paper\paperwithcode\ClusterGCN-master3\data\paviau/features_PaviaU_all.csv",
    #                     help="Features json.")
    #
    # parser.add_argument("--target-path",
    #                     nargs="?",
    #                     default="D:\paper\paperwithcode\ClusterGCN-master3\data\paviau/target_PaviaU_all.csv",
    #                     help="Target classes csv.")

    # PaviaU
    # parser.add_argument("--edge-path",
    #                     nargs="?",
    #                     default="D:\paper\paperwithcode\ClusterGCN-master3\data\houston/egde_Houston2.csv",
    #                     help="Edge list csv.")
    # parser.add_argument("--Houston-Target",
    #                     nargs="?",
    #                     default="D:/paper/paperwithcode/ClusterGCN-master/input/Houston_gt.mat",
    #                     help="Edge list csv.")
    # parser.add_argument("--features-path",
    #                     nargs="?",
    #                     default="D:\paper\paperwithcode\ClusterGCN-master3\data\houston/features_Houston2.csv",
    #                     help="Features json.")


    parser.add_argument("--Dataname",
                        type = str,
                        default="PU",
                        help="HS,IP,PU")


    parser.add_argument("--clustering-method",
                        nargs = "?",
                        default = "metis",
	                help = "Clustering method for graph decomposition. Default is the metis procedure.")

    parser.add_argument("--epochs",
                        type = int,
                        default =100,
	                help = "Number of training epochs. Default is 200.")

    parser.add_argument("--seed",
                        type = int,
                        default =42,
	                help = "Random seed for train-test split. Default is 42.")
    parser.add_argument("--cuda",
                        type=int,
                        default=0,
                        help="Random seed for train-test split. Default is 42.")

    parser.add_argument("--dropout",
                        type = float,
                        default = 0.01,
	                help = "Dropout parameter. Default is 0.5.")

    parser.add_argument("--learning-rate",
                        type = float,
                        default = 0.03,
	                help = "Learning rate. Default is 0.01.")

    parser.add_argument("--test-ratio",
                        type = float,
                        default = 0.985,
	                help = "Test data ratio. Default is 0.1.")
    parser.add_argument("--train_num",
                        type=float,
                        default= 13,
                        help="Test data ratio. Default is 0.1.")

    parser.add_argument("--cluster-number",
                        type = int,
                        default = 8,
                        help = "Number of clusters extracted. Default is 10.")
    # model
    parser.add_argument('--num-layers', type=int, default=10, help='Number of GNN layers.')
    parser.add_argument('--hid-dim', type=int, default=60, help='Hidden channel size.')
    parser.set_defaults(layers=[16, 16, 16])

    parser.add_argument('--min_num',type=float, default=0.25)
    parser.add_argument('--graph_skip_conn', type=float, default=0.15)
    parser.add_argument('--feat_adj_dropout', type=float, default=0.01)
    parser.add_argument('--balanced_degree', type=float, default=0.15)

    parser.add_argument('--epsilon', type=float, default=0.55, help='Value of epsilon in changing adj.')
    parser.add_argument('--metric_type', nargs = "?", default="weighted_cosine", help='Value of epsilon in changing adj.')
    parser.add_argument('--degree_ratio', type=float, default=0.1)
    parser.add_argument('--sparsity_ratio', type=float, default=0.1)
    parser.add_argument('--smoothness_ratio', type=float, default=0.5)

    return parser.parse_args()

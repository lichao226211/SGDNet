confusion_matrix_fdssc = metrics.confusion_matrix(pred_test_fdssc, gt_test)  # 建立混淆矩阵, pred_test_fdssc为预测节点标签，gt_test为原标签
each_acc_fdssc, average_acc_fdssc = aa_and_each_accuracy(confusion_matrix_fdssc)
def aa_and_each_accuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)#取对称轴的值
    list_raw_sum = np.sum(confusion_matrix, axis=1)#混淆矩阵形状：横轴为真实值数量，列为预测值数量
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum)) # x/y
    average_acc = np.mean(each_acc)
    return each_acc, average_acc

#adj_patch图邻接矩阵。
adj_patch = torch.where(adj_patch > 0, 1, adj_patch)
in_degree = torch.sum(adj_patch, dim=1)  # 32 81
out_degree = torch.sum(adj_patch, dim=2)  # 32 81
in_max = in_degree.max(1).values.unsqueeze(-1)  # 32
in_min = (in_degree.min(1).values + 1).unsqueeze(-1)  # 32 81
in_mean = (in_degree.mean(1) + 1).unsqueeze(-1)  # 32
out_max = out_degree.max(1).values.unsqueeze(-1)
out_min = (out_degree.min(1).values + 1).unsqueeze(-1)
out_mean = (out_degree.mean(1) + 1).unsqueeze(-1)
in_degree_rate = (1 / in_max) + 0.5 * ((in_max - in_min) / in_max) * (
            1 + torch.cos(((in_max - in_degree) / (in_max + in_min)) * (math.pi / 2)))
out_degree_rate = (1 / out_max) + 0.5 * ((out_max - out_min) / out_max) * (
            1 + torch.cos(((out_max - out_degree) / (out_max + out_min)) * (math.pi / 2)))

# 计算大图的度/

graph_in = graph_in_degree
graph_out = graph_out_degree
graph_in = rearrange(graph_in, 'b h w  -> b (h w)')
graph_out = rearrange(graph_out, 'b h w  -> b (h w)')
graph_in_max = graph_in.max(1).values.unsqueeze(-1)  # 32
graph_in_min = (graph_in.min(1).values + 1).unsqueeze(-1)  # 32 81
in_mean = (graph_in.mean(1) + 1).unsqueeze(-1)  # 32
graph_out_max = graph_out.max(1).values.unsqueeze(-1)
graph_out_min = (graph_out.min(1).values + 1).unsqueeze(-1)
out_mean = (graph_out.mean(1) + 1).unsqueeze(-1)
graph_in_degree_rate = (1 / graph_in_max) + 0.5 * ((graph_in_max - graph_in_min) / graph_in_max) * (
        1 + torch.cos(((graph_in_max - graph_in) / (graph_in_max + graph_in_min)) * (math.pi / 2)))
graph_out_degree_rate = (1 / graph_out_max) + 0.5 * ((graph_out_max - graph_out_min) / graph_out_max) * (
        1 + torch.cos(((graph_out_max - graph_out) / (graph_out_max + graph_out_min)) * (math.pi / 2)))
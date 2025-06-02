import torch
import torch.nn as nn

class Para_model(nn.Module):
    def _init__(self):
        super(Para_model, self).__init__()
        self.graph_skip_conn = nn.Parameter(torch.Tensor([0]))
        # Define other model components

    def forward(self, adj, init_adj):
        # Calculate combined adj using graph_skip_conn
        adj_combined = self.graph_skip_conn * adj + (1 - self.graph_skip_conn) * init_adj
        # Perform other operations using adj_combined
        return adj_combined

from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv


class GCN(nn.Module):
    """Two-layer Graph Convolutional Network used as the prediction backbone."""

    def __init__(self,in_channels,hidden_channels,out_channels,dropout=0.0,batchnorm=False,):
        super().__init__()

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

        self.dropout = dropout
        self.use_batchnorm = batchnorm
        self.batchnorm1 = nn.BatchNorm1d(hidden_channels) if batchnorm else None

    def reset_parameters(self):
        """Reset all learnable parameters of the model."""
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

        if self.batchnorm1 is not None:
            self.batchnorm1.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight)

        if self.use_batchnorm:
            x = self.batchnorm1(x)

        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)

        return F.log_softmax(x, dim=1)
    
class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=2, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden_channels * heads, out_channels, heads=heads, concat=False, dropout=dropout)

    def forward(self, x, edge_index):
        x = self.gat1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index)
        return F.log_softmax(x, dim=1)
    



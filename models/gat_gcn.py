import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from utils import *
# GCN-CNN based model

class GAT_GCN(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=35, num_features_xt=25,
                 n_filters=16, embed_dim=128, output_dim=128, dropout=0.3):

        super(GAT_GCN, self).__init__()

        self.n_output = n_output
        self.conv1 = GATConv(num_features_xd, num_features_xd, heads=10)
        self.conv2 = GCNConv(num_features_xd*10, num_features_xd*10)
        self.fc_g1 = torch.nn.Linear(num_features_xd*10*2,1500)
        self.fc_g2 = torch.nn.Linear(1500, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # 1D convolution on protein sequence
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.conv_xt_1 = nn.Conv1d(in_channels=128, out_channels=n_filters, kernel_size=8)
        self.fc1_xt = nn.Linear(16*35, output_dim)

        # combined layers
        self.fc1 = nn.Linear(256, 512)
        self.out = nn.Linear(512, self.n_output)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        target = data.target
        x,w = self.conv1(x, edge_index,return_attention_weights= True)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        # apply global max pooling (gmp) and global mean pooling (gap)
        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.fc_g2(x)

        embedded_xt = self.embedding_xt(target)
        embedded_xt = embedded_xt.permute(0,2,1)
        conv_xt = self.conv_xt_1(embedded_xt)
        # flatten
        xt = conv_xt.view(-1, 16*35)
        xt = self.fc1_xt(xt)

        # concat
        xc = torch.cat((x, xt), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = torch.sigmoid(self.out(xc))
        return out,w

from torch import nn
import torch
import torch.nn.functional as F
import math
import numpy as np

def get_humanml3d_adjacency_matrix():
    num_joints = 22
    adjacency = np.zeros((num_joints, num_joints))

    skeleton_links = [
        (0, 1), (0, 2), (0, 3),  # pelvis->left_hip, right_hip, spine1
        (3, 6), (6, 9), (9, 12),  # spine1->spine2->spine3->neck
        (12, 15),  # neck->head
        (12, 13), (12, 14),  # neck->left_collar, neck->right_collar
        # left_collar->left_shoulder->left_elbow->left_wrist
        (13, 16), (16, 18), (18, 20),
        # right_collar->right_shoulder->right_elbow->right_wrist
        (14, 17), (17, 19), (19, 21),
        (1, 4), (4, 7), (7, 10),  # left_hip->left_knee->left_ankle->left_foot
        (2, 5), (5, 8), (8, 11),  # right_hip->right_knee->right_ankle->right_foot
    ]

    for i, j in skeleton_links:
        adjacency[i, j] = 1
        adjacency[j, i] = 1

    for i in range(num_joints):
        adjacency[i, i] = 1

    return adjacency

def normalize_adjacency_matrix(adjacency):
    D = np.sum(adjacency, axis=1)
    D_sqrt_inv = np.diag(np.power(D, -0.5, where=D != 0))
    normalized_adjacency = D_sqrt_inv.dot(adjacency).dot(D_sqrt_inv)

    return normalized_adjacency

class GraphConv(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, feats_in, feats_hidden, feats_out, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConv(feats_in, feats_hidden)
        self.gc2 = GraphConv(feats_hidden, feats_hidden)
        self.gc3 = GraphConv(feats_hidden, feats_out)

        self.norm1 = nn.LayerNorm(feats_hidden)
        self.norm2 = nn.LayerNorm(feats_hidden)
        self.norm3 = nn.LayerNorm(feats_hidden)

        self.dropout = dropout

    def forward(self, x, adj):
        identity = x
        x1 = self.norm1(F.gelu(self.gc1(x, adj)))
        x1 = F.dropout(x1, self.dropout, training=self.training)
        x2 = self.norm2(F.gelu(self.gc2(x1, adj)))
        x2 = F.dropout(x2, self.dropout, training=self.training)
        x3 = self.norm3(F.gelu(self.gc3(x2, adj)))
        x3 = F.dropout(x3, self.dropout, training=self.training)
        return x3 + identity

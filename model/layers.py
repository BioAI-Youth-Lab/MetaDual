"""Selected original layers; tensor allocation follows the input device."""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Parameter, Module
from model.similarity import consinesimilar

class dotproduct(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features):
        super(dotproduct, self).__init__()
        self.in_features = in_features
        self.weight = Parameter(torch.FloatTensor(1, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input):
        output = torch.mul(input, self.weight)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> )'

class adjcentmatrix(Module):

    def __init__(self, in_features, epsilon, n_samples, nhid, dropout1, r):
        super(adjcentmatrix, self).__init__()
        self.n_head = 4
        self.in_features = in_features
        self.dropout1 = dropout1
        self.nhid = nhid
        self.r = r
        self.out_features = 8
        self.out_features = 8
        self.epsilon = epsilon
        self.n_samples = n_samples
        self.weight1 = Parameter(torch.FloatTensor(self.in_features, self.out_features * self.n_head))
        self.weight2 = Parameter(torch.FloatTensor(self.in_features, self.out_features * self.n_head))
        self.weight3 = Parameter(torch.FloatTensor(self.in_features, self.nhid))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight1.size(1))
        self.weight1.data.uniform_(-stdv, stdv)
        stdv = 1.0 / math.sqrt(self.weight2.size(1))
        self.weight2.data.uniform_(-stdv, stdv)
        stdv = 1.0 / math.sqrt(self.weight3.size(1))
        self.weight3.data.uniform_(-stdv, stdv)

    def forward(self, input):
        attention2 = input.new_zeros((self.n_samples, self.n_samples))
        zero_matrix = attention2
        inter_put1 = torch.mm(input, self.weight1)
        inter_put2 = torch.mm(input, self.weight2)
        inter_put3 = torch.mm(input, self.weight3)
        j = 0
        adjacent = zero_matrix
        for i in range(self.n_head):
            input1 = inter_put1[:, j:j + self.out_features]
            input2 = inter_put2[:, j:j + self.out_features]
            adjacent = adjacent + consinesimilar(input1, input2)
            j = j + self.out_features
        attention1 = torch.where(adjacent > self.epsilon, adjacent, zero_matrix)
        attention2 = zero_matrix
        b = torch.topk(adjacent, k=self.r, dim=1, largest=True)
        attention2 = attention2.scatter_(1, b.indices, b.values)
        attention = attention1 + attention2
        attention = torch.where(attention > 0, adjacent, zero_matrix)
        f = torch.isnan(attention)
        attention[f] = 0
        attention = torch.where(attention > attention.t(), attention, attention.t())
        attention = attention + torch.eye(self.n_samples, device=input.device, dtype=input.dtype)
        attention = normalize_adj(attention)
        f = torch.isnan(attention)
        attention[f] = 0
        output = torch.mm(attention, inter_put3)
        output = F.dropout(output, self.dropout1, training=self.training)
        return (output, attention)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.n_samples) + ')'

class MultiHeadAttention(torch.nn.Module):

    def __init__(self, input_dim, n_heads, ouput_dim=None):
        super(MultiHeadAttention, self).__init__()
        self.d_k = self.d_v = input_dim // n_heads
        self.dropout = 0.5
        self.n_heads = n_heads
        if ouput_dim == None:
            self.ouput_dim = input_dim
        else:
            self.ouput_dim = ouput_dim
        self.W_Q = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
        self.W_K = torch.nn.Linear(input_dim, self.d_k * self.n_heads, bias=False)
        self.W_V = torch.nn.Linear(input_dim, self.d_v * self.n_heads, bias=False)
        self.fc = torch.nn.Linear(self.n_heads * self.d_v, self.ouput_dim, bias=False)

    def forward(self, X):
        Q = self.W_Q(X).view(-1, self.n_heads, self.d_k).transpose(0, 1)
        K = self.W_K(X).view(-1, self.n_heads, self.d_k).transpose(0, 1)
        V = self.W_V(X).view(-1, self.n_heads, self.d_v).transpose(0, 1)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        attn = torch.nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).reshape(-1, self.n_heads * self.d_v)
        return context

class EncoderLayer1(torch.nn.Module):

    def __init__(self, input_dim, n_heads, dropout):
        super(EncoderLayer1, self).__init__()
        self.attn = MultiHeadAttention(input_dim, n_heads)
        self.AN1 = torch.nn.LayerNorm(input_dim)
        self.dropout = dropout
        self.l1 = torch.nn.Linear(input_dim, input_dim)
        self.AN2 = torch.nn.LayerNorm(input_dim)

    def forward(self, X):
        output = F.relu(self.attn(X))
        output = F.dropout(output, self.dropout, training=self.training)
        output = output + X
        output = F.relu(output)
        output = F.dropout(output, self.dropout, training=self.training)
        output = output + X
        return output

def normalize_adj(input):
    """Row-normalize sparse matrix"""
    sum_input = torch.sum(input, dim=1)
    mask = sum_input > 1e-09
    r_inv_sqrt = torch.zeros_like(sum_input)
    r_inv_sqrt[mask] = torch.pow(sum_input[mask], -0.5)
    degree_input = torch.diag(r_inv_sqrt)
    input_norm = torch.mm(degree_input, torch.mm(input, degree_input))
    return input_norm

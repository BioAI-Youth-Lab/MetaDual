"""Prompt scorer uses its original structural layer implementation."""
import torch
from torch import nn
from torch.nn import functional as F
from prompt_search.scoring_layers import adjcentmatrix, dotproduct, EncoderLayer1

class GCNSA_two_graph(nn.Module):

    def __init__(self, nfeatures_basic, nfeatures_text, nhid, nclass, dropout1, epsilon, n_samples, r):
        super(GCNSA_two_graph, self).__init__()
        self.fc0_basic = torch.nn.Linear(nfeatures_basic, nhid)
        self.structurel_basic = adjcentmatrix(nfeatures_basic, epsilon, n_samples, 2 * nhid, dropout1, r)
        self.fc0_text = torch.nn.Linear(nfeatures_text, nhid)
        self.structurel_text = adjcentmatrix(nfeatures_text, epsilon, n_samples, 2 * nhid, dropout1, r)
        self.hp = dotproduct(nhid * 16)
        self.fc1 = torch.nn.Linear(nhid * 16, nclass)
        self.dropout1 = dropout1
        self.layers = torch.nn.ModuleList([EncoderLayer1(nclass, 1, dropout1) for _ in range(1)])
        self.modifiedt = EncoderLayer1(nhid, 4, dropout1)
        self.linear = torch.nn.Linear(nhid * 2, nhid * 2)

    def forward(self, features_basic, features_text, adj1, adj2, K):
        y1_basic = F.relu(self.fc0_basic(features_basic))
        y_basic = F.dropout(y1_basic, self.dropout1, training=self.training)
        x_basic = self.modifiedt(y_basic)
        z_basic = torch.cat([x_basic, y_basic], dim=1)
        naz_basic, attention_basic = self.structurel_basic(features_basic)
        aaz_basic = az_basic = z_basic
        for i in range(K - 1):
            az_basic = torch.spmm(adj1, az_basic)
            aaz_basic = torch.spmm(adj1, az_basic)
        y1_text = F.relu(self.fc0_text(features_text))
        y_text = F.dropout(y1_text, self.dropout1, training=self.training)
        x_text = self.modifiedt(y_text)
        z_text = torch.cat([x_text, y_text], dim=1)
        naz_text, attention_text = self.structurel_text(features_text)
        aaz_text = az_text = z_text
        for i in range(K - 1):
            az_text = torch.spmm(adj2, az_text)
            aaz_text = torch.spmm(adj2, az_text)
        z3 = torch.cat([naz_basic, z_basic, az_basic, aaz_basic, naz_text, z_text, az_text, aaz_text], dim=1)
        z3 = F.dropout(z3, self.dropout1, training=self.training)
        z3 = F.relu(self.hp(z3))
        z3 = F.dropout(z3, self.dropout1, training=self.training)
        z3 = self.fc1(z3)
        for layer in self.layers:
            z3 = layer(z3)
        z3 = F.log_softmax(z3, dim=1)
        return (z3, attention_basic, attention_text)

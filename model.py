import numpy as np
import torch
import torch.nn as nn
import dgl.function as fn
import torch.nn.functional as F
from torch import tensor
import torch_geometric.utils
from torch_geometric.nn.conv import MessagePassing


def nSGCConv(graph, feats, order):
    with graph.local_scope():
        degs = graph.in_degrees().float().clamp(min=1)
        norm = torch.pow(degs, -0.5).to(feats.device).unsqueeze(1)
        graph.ndata['norm'] = norm
        graph.apply_edges(fn.u_mul_v('norm', 'norm', 'weight'))
        # x = F.dropout(feats, p=0.5)
        y = 0 + feats
        for i in range(order):
            graph.ndata['h'] = x
            graph.update_all(fn.u_mul_e('h', 'weight', 'm'), fn.sum('m', 'h'))
            x = graph.ndata.pop('h')
            y = torch.cat((y, x), dim=1)
    return y


def nSGCConv2(graph, feats, order):
    def message_func(edges):
        msg = F.dropout(edges.src['h'], p=0.5) * edges.data['weight']
        # F.dropout(tensor('h'), p=0.5)
        return {'m': msg}

    with graph.local_scope():
        degs = graph.in_degrees().float().clamp(min=1)
        norm = torch.pow(degs, -0.5).to(feats.device).unsqueeze(1)
        graph.ndata['norm'] = norm
        graph.apply_edges(fn.u_mul_v('norm', 'norm', 'weight'))
        x = feats
        # x = F.dropout(feats, p=0.5)
        y = 0 + feats
        for i in range(order):
            graph.ndata['h'] = x
            graph.update_all(message_func, fn.sum('m', 'h'))
            x = graph.ndata.pop('h')
            y = torch.cat((y, x), dim=1)
    return y


def nSGCConv3(graph, feats, order):
    def message_func(edges):
        msg = F.dropout(edges.dst['h'], p=0.5) * edges.data['weight']
        # F.dropout(tensor('h'), p=0.5)
        return {'m': msg}

    with graph.local_scope():
        degs = graph.in_degrees().float().clamp(min=1)
        norm = torch.pow(degs, -0.5).to(feats.device).unsqueeze(1)
        graph.ndata['norm'] = norm
        graph.apply_edges(fn.u_mul_v('norm', 'norm', 'weight'))
        x = feats
        # x = F.dropout(feats, p=0.5)
        y = 0 + feats
        for i in range(order):
            graph.ndata['h'] = x
            graph.update_all(message_func, fn.sum('m', 'h'))
            x = graph.ndata.pop('h')
            y = torch.cat((y, x), dim=1)
    return y


def nSGCConv4(self, graph, feats, drop_rate, order):
    with graph.local_scope():
        edge_index, edge_weight = pretreatment(graph, feats)
        y = self.propagate(edge_index=edge_index, size=None, x=feats, drop_rate=drop_rate)
    return y


def pretreatment(graph, feats):
    with graph.local_scope():
        row = graph.edges()[0]
        col = graph.edges()[1]
        edge_index = torch.vstack((row, col))
        deg = torch_geometric.utils.degree(col, feats.size(0), dtype=feats.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]

    return edge_index, edge_weight


class nSGCN(MessagePassing):
    def __init__(self, add_self_loops: bool = True, normalize: bool = True):
        super(nSGCN, self).__init__()
        # self.pt = ModelPretreatment(add_self_loops, normalize)
        self.edge_weight = None

    def reset_parameters(self):
        pass

    def forward(self, graph, feats, drop_rate, order):
        edge_index, self.edge_weight = pretreatment(graph, feats)
        # print(drop_rate)
        y = self.propagate(edge_index=edge_index, size=None, x=feats, drop_rate=drop_rate)
        return y

    def message(self, x_j, drop_rate: float):
        # normalize
        if self.edge_weight is not None:
            x_j = x_j * self.edge_weight.view(-1, 1)
        if not self.training:
            return x_j
        # drop messages
        x_j = F.dropout(x_j, drop_rate)
        return x_j


class nSGC(nn.Module):
    def __init__(self, G, hid_dim, n_class, K, batchnorm, num_diseases, num_mirnas,
                 d_sim_dim, m_sim_dim, out_dim, dropout, slope, node_dropout=0.5, input_droprate=0.0,
                 hidden_droprate=0.0):
        super(nSGC, self).__init__()
        self.G = G
        self.hid_dim = hid_dim

        self.K = K
        self.n_class = n_class
        self.num_diseases = num_diseases
        self.num_mirnas = num_mirnas
        self.disease_nodes = G.filter_nodes(lambda nodes: nodes.data['type'] == 1)
        self.mirna_nodes = G.filter_nodes(lambda nodes: nodes.data['type'] == 0)

        # self.m_fc = nn.Linear(G.ndata['m_sim'].shape[1], hid_dim, bias=False)
        # self.d_fc = nn.Linear(G.ndata['d_sim'].shape[1], hid_dim, bias=False)

        self.m_fc = nn.Linear(G.ndata['m_sim'].shape[1], hid_dim, bias=False)
        self.d_fc = nn.Linear(G.ndata['d_sim'].shape[1], hid_dim, bias=False)
        self.f_fc = nn.Linear(out_dim * (K + 1), out_dim)
        self.f_fc2 = nn.Linear(out_dim * K, out_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.predict = nn.Linear(out_dim * 2, 1)
        self.predict_onlycross = nn.Linear(out_dim, 1)

        self.predict_addcross = nn.Linear(out_dim * 3, 1)

        # self.InnerProductDecoder = InnerProductDecoder()
        self.backbone = nSGCN(True, True)

    def forward(self, graph, diseases, mirnas, training=True):
        # self.G.apply_nodes(lambda nodes: {'z': self.dropout1(self.d_fc(nodes.data['d_sim']))}, self.disease_nodes)
        # self.G.apply_nodes(lambda nodes: {'z': self.dropout1(self.m_fc(nodes.data['m_sim']))}, self.mirna_nodes)
        self.G.apply_nodes(lambda nodes: {'z': self.d_fc(nodes.data['d_sim'])}, self.disease_nodes)
        self.G.apply_nodes(lambda nodes: {'z': self.m_fc(nodes.data['m_sim'])}, self.mirna_nodes)

        feats = self.G.ndata.pop('z')  # 1709*64
        X = feats
        # X = F.dropout(feats, p=0.5)
        if training:
            feat0 = []
            y = 0 + X
            # y = torch.Tensor([])
            x = self.backbone(graph, X, 0.5, 1)
            # x = F.relu(x)
            y = torch.cat((y, x), dim=1)
            for i in range(self.K - 1):
                x = self.backbone(graph, x, 0.5, 1)
                # x = F.relu(x)
                y = torch.cat((y, x), dim=1)

            h = self.f_fc2(y)
            h_diseases = h[diseases]
            h_mirnas = h[mirnas]
            h_cross = h_diseases * h_mirnas
            h_edge = torch.cat((h_diseases, h_mirnas, h_cross), 1)
            predict_score = torch.sigmoid(self.predict_addcross(h_edge))
            return predict_score

        else:
            feat0 = nSGCConv(graph, X, self.K)
            h = self.f_fc(feat0)
            h_diseases = h[diseases]
            h_mirnas = h[mirnas]
            h_cross = h_diseases * h_mirnas
            h_edge = torch.cat((h_diseases, h_mirnas, h_cross), 1)
            predict_score = torch.sigmoid(self.predict_addcross(h_edge))
            return predict_score

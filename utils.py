import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from scipy import interp
from numpy import interp
from sklearn import metrics
import torch
import torch.nn as nn
import dgl

def load_data5(directory, random_seed):
    D_SSM1 = np.loadtxt(directory + '/diseaseSim/DiseaseSimilarity1.txt')  # 792 * 792
    D_SSM2 = np.loadtxt(directory + '/diseaseSim/DiseaseSimilarity2.txt')
    D_GSM = np.loadtxt(directory + '/diseaseSim/D_GIP2.txt')

    np.fill_diagonal(D_SSM1, 1)
    np.fill_diagonal(D_SSM2, 1)

    M_FSM = np.loadtxt(directory + '/miRNASim/FuncSim.txt')  # 917 * 917
    M_SeSM = np.loadtxt(directory + '/miRNASim/SeqSim2.txt')  # 917 * 917
    M_Fam = np.loadtxt(directory + '/miRNASim/Famsim.txt')
    M_GSM = np.loadtxt(directory + '/miRNASim/M_GIP.txt')

    all_associations = pd.read_csv(directory + '/new_adjacency_matrix.csv',
                                   names=['miRNA', 'disease', 'label'])  # 726264 * 3

    D_SSM = (D_SSM1 + D_SSM2) / 2
    ID = D_SSM
    for i in range(D_SSM.shape[0]):
        for j in range(D_SSM.shape[1]):
            if ID[i][j] == 0:
                ID[i][j] = D_GSM[i][j]
            else:
                ID[i][j] = (D_GSM[i][j] + ID[i][j]) / 2

    for i in range(M_FSM.shape[0]):
        for j in range(M_FSM.shape[1]):
            if M_Fam[i][j] == 1:
                M_FSM[i][j] = (M_FSM[i][j] + M_Fam[i][j]) / 2
            else:
                M_FSM[i][j] = M_FSM[i][j]

    M_SSM = (M_FSM + M_SeSM) / 2
    IM = M_SSM
    for i in range(M_FSM.shape[0]):
        for j in range(M_FSM.shape[1]):
            if IM[i][j] == 0:
                IM[i][j] = M_GSM[i][j]
            else:
                # IM[i][j] = (M_GSM[i][j] + M_FSM[i][j]) / 2
                IM[i][j] = (M_GSM[i][j] + M_SSM[i][j]) / 2

    known_associations = all_associations.loc[all_associations['label'] == 1]  # 14550 * 3
    unknown_associations = all_associations.loc[all_associations['label'] == 0]  # 711714 * 3
    random_negative = unknown_associations.sample(n=known_associations.shape[0], random_state=random_seed,
                                                  axis=0)  # 14550 * 3
    sample_df = known_associations.append(random_negative)
    sample_df.reset_index(drop=True, inplace=True)
    samples = sample_df.values
    return ID, IM, samples




def build_graph(directory, random_seed):
    ID, IM, samples = load_data5(directory, random_seed)
    g = dgl.DGLGraph()
    g.add_nodes(ID.shape[0] + IM.shape[0])
    node_type = torch.zeros(g.number_of_nodes(), dtype=torch.int64)
    node_type[: ID.shape[0]] = 1
    g.ndata['type'] = node_type

    d_sim = torch.zeros(g.number_of_nodes(), ID.shape[1])
    d_sim[: ID.shape[0], :] = torch.from_numpy(ID.astype('float32'))
    g.ndata['d_sim'] = d_sim

    m_sim = torch.zeros(g.number_of_nodes(), IM.shape[1])
    m_sim[ID.shape[0]: ID.shape[0] + IM.shape[0], :] = torch.from_numpy(IM.astype('float32'))
    g.ndata['m_sim'] = m_sim

    disease_ids = list(range(1, ID.shape[0] + 1))
    mirna_ids = list(range(1, IM.shape[0] + 1))

    disease_ids_invmap = {id_: i for i, id_ in enumerate(disease_ids)}
    mirna_ids_invmap = {id_: i for i, id_ in enumerate(mirna_ids)}

    sample_disease_vertices = [disease_ids_invmap[id_] for id_ in samples[:, 1]]
    sample_mirna_vertices = [mirna_ids_invmap[id_] + ID.shape[0] for id_ in samples[:, 0]]

    g.add_edges(sample_disease_vertices, sample_mirna_vertices,
                data={'label': torch.from_numpy(samples[:, 2].astype('float32'))})
    g.add_edges(sample_mirna_vertices, sample_disease_vertices,
                data={'label': torch.from_numpy(samples[:, 2].astype('float32'))})
    g.readonly()

    return g, sample_disease_vertices, sample_mirna_vertices, ID, IM, samples

def weight_reset(m):
    if isinstance(m, nn.Linear):
        m.reset_parameters()






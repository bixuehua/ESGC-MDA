import numpy as np
import csv
# 将csv读取成为npy
from numpy import genfromtxt
data = genfromtxt('adjacency_matrix.csv', delimiter=',', skip_header = 0)

def getGosiR(Asso_RNA_Dis):  # 计算高斯核中的r
    nc = Asso_RNA_Dis.shape[0]
    summ = 0
    for i in range(nc):
        x_norm = np.linalg.norm(Asso_RNA_Dis[i, :])
        x_norm = np.square(x_norm)
        summ = summ + x_norm
    r = summ / nc
    return r

def GIP_kernel(Asso_RNA_Dis):
    nc = Asso_RNA_Dis.shape[0]
    matrix = np.zeros((nc, nc))
    # r部分
    r = getGosiR(Asso_RNA_Dis)
    # 计算结果矩阵
    for i in range(nc):
        for j in range(nc):
            # 计算GIP公式的上部分
            temp_up = np.square(np.linalg.norm(Asso_RNA_Dis[i, :] - Asso_RNA_Dis[j, :]))
            if r == 0:
                matrix[i][j] = 0
            elif i == j:
                matrix[i][j] = 1
            else:
                matrix[i][j] = np.e ** (-temp_up / r)
    return matrix

GIP_sim = GIP_kernel(data)
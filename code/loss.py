import torch as t
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import scipy.sparse as sp
import random
import gc
import os
import scipy.io as scio
import time
import csv
from clac_metric import get_metrics
from utils import constructHNet, constructNet, get_edge_index, load_data, Sizes
from torch import optim
#
# data_path = '../data/dataset/'
# data_set = 'Chembl/'
# drug_sim = np.loadtxt(data_path + data_set + 'drugsimilarity.txt', delimiter='\t')
# pro_sim = np.loadtxt(data_path + data_set + 'proteinsimilarity.txt', delimiter='\t')
# adj_triple = np.loadtxt(data_path + data_set + 'adj.txt')
# drug_pro_matrix = sp.csc_matrix((adj_triple[:, 2], (adj_triple[:, 0] - 1, adj_triple[:, 1] - 1)),
#                                 shape=(len(drug_sim), len(pro_sim))).toarray()
# drug_pro = drug_pro_matrix
# pos_index_matrix = np.array(np.where(drug_pro == 1))
# neg_index_matrix = np.array(np.where(drug_pro == 0))
#
# np.random.shuffle(pos_index_matrix.T)
# np.random.shuffle(neg_index_matrix.T)
#
# pos_row_indices = pos_index_matrix[0, :]
# pos_col_indices = pos_index_matrix[1, :]
#
# neg_row_indices = neg_index_matrix[0, :]
# neg_col_indices = neg_index_matrix[1, :]
#
# positive_samples = drug_pro[pos_row_indices, pos_col_indices]
# negative_samples = drug_pro[neg_row_indices, neg_col_indices]
#
#
# class DrugProDataset(Dataset):
#     def __init__(self, drug_pro, positive_samples, negative_samples):
#         self.drug_pro_matrix = drug_pro
#         self.positive_samples = positive_samples
#         self.negative_samples = negative_samples
#
#     def __len__(self):
#         return len(self.positive_samples)
#
#     def __getitem__(self, idx):
#         anchor_index = self.positive_samples[idx]
#         positive_index = self.positive_samples[idx]
#         negative_index = self.negative_samples[idx]
#         anchor_index = int(anchor_index)
#         positive_index = int(positive_index)
#         negative_index = int(negative_index)
#         # print("Anchor Index:", anchor_index)
#         # print("Anchor Index Type:", type(anchor_index))
#         anchor = t.tensor(self.drug_pro_matrix[anchor_index, :], dtype=t.float32)
#         positive = t.tensor(self.drug_pro_matrix[positive_index, :], dtype=t.float32)
#         negative = t.tensor(self.drug_pro_matrix[negative_index, :], dtype=t.float32)
#
#         return anchor, positive, negative
#
# class SiameseNetwork(nn.Module):
#     def __init__(self, input_size):
#         super(SiameseNetwork, self).__init__()
#         self.fc1 = nn.Linear(in_features=input_size, out_features=64)
#         self.fc2 = nn.Linear(in_features=64, out_features=32)
#         self.fc3 = nn.Linear(in_features=32, out_features=16)
#
#     def forward_one(self, x):
#         x = t.relu(self.fc1(x))
#         x = t.relu(self.fc2(x))
#         x = t.relu(self.fc3(x))
#         return x
#
#     def forward(self, input1, input2):
#         output1 = self.forward_one(input1)
#         output2 = self.forward_one(input2)
#         return output1, output2
#
# class TripletLoss(nn.Module):
#     def __init__(self, margin=1.0):
#         super(TripletLoss, self).__init__()
#         self.margin = margin
#
#     def forward(self, anchor, positive, negative):
#         distance_positive = t.pairwise_distance(anchor, positive, keepdim=True)
#         distance_negative = t.pairwise_distance(anchor, negative[ 0 ], keepdim=True)
#         loss = t.relu(distance_positive - distance_negative + self.margin)
#         return t.mean(loss)
# siamese_net = SiameseNetwork(input_size=drug_pro_matrix.shape[1])
# triplet_loss = TripletLoss()
#
# #optimizer = optim.Adam(siamese_net.parameters(), lr=0.001)
#
# # Create DataLoader for the dataset
# dataset = DrugProDataset(drug_pro_matrix, positive_samples, negative_samples)
# dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
# for batch in dataloader:
#     anchor, positive, negative = batch
#
# output1, output2 = siamese_net(anchor, positive)
#
# loss = triplet_loss(output1, output2, siamese_net(negative, negative))

class Myloss(nn.Module):
    def __init__(self):
        super(Myloss, self).__init__()

    def forward(self, target, prediction, pro_l, drug_l, alpha1, alpha2, sizes):
        target = t.tensor(target, dtype=t.float32)
        # 损失函数结合了Sigmoid激活函数和二元交叉熵损失
        criterion = nn.BCEWithLogitsLoss()
        closs = criterion(prediction, target)
        # print(len(alpha1))
        # print(len(alpha2))
        # print(len(pro_l))
        # print(len(drug_l))
        drug_reg = t.trace(t.mm(t.mm(alpha2.T, drug_l), alpha2))
        # drug_reg = t.trace(t.mm(t.mm(alpha1, drug_l), alpha2))
        drug_reg = t.tensor(drug_reg, dtype=t.float32)
        pro_reg = t.trace(t.mm(t.mm(alpha1.T, pro_l), alpha1))
        pro_reg = t.tensor(pro_reg, dtype=t.float32)
        graph_reg = sizes.lambda1 * drug_reg + sizes.lambda2 * pro_reg
        graph_reg = t.tensor(graph_reg, dtype=t.float32)
        # loss_sum = 0.7 * closs + 0.2 * graph_reg + 0.1 * loss
        loss_sum = 0.7 * closs + 0.3 * graph_reg
        return loss_sum.sum()
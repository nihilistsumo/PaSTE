import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
import random
import json
import argparse
import sys
from sklearn.preprocessing import minmax_scale
from data.unsup_data_process import Mu_etAl_PPA, Raunak_etAl_dimred

class Query_Attn_InteractMatrix_Network(nn.Module):
    def __init__(self, ):
        super(Query_Attn_InteractMatrix_Network, self).__init__()
        # parameters
        self.emb_size = 768
        self.l1out_size = 96
        self.l2out_size = 48
        self.cosine_sim = nn.CosineSimilarity()
        self.LL1 = nn.Linear(self.emb_size*self.emb_size, self.l1out_size)
        self.LL2 = nn.Linear(self.l1out_size, self.l2out_size)

    def forward(self, X):
        self.Xq = X[:, :self.emb_size]
        self.Xp1 = X[:, self.emb_size:2*self.emb_size]
        self.Xp2 = X[:, 2*self.emb_size:]
        self.qp1z = torch.einsum('bi,bj -> bij', (self.Xq, self.Xp1))
        self.qp1z = torch.flatten(self.qp1z, start_dim=1)
        self.qp2z = torch.einsum('bi,bj -> bij', (self.Xq, self.Xp2))
        self.qp2z = torch.flatten(self.qp2z, start_dim=1)
        self.p1l1 = torch.relu(self.LL1(self.qp1z))
        self.p2l1 = torch.relu(self.LL1(self.qp2z))
        self.p1l2 = torch.relu(self.LL2(self.p1l1))
        self.p2l2 = torch.relu(self.LL2(self.p2l1))
        o = self.cosine_sim(self.p1l2, self.p2l2)
        return o

    def num_flat_features(self, X):
        size = X.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def predict(self, X_test):
        #print("Predicted data based on trained weights: ")
        #print("Input (scaled): \n" + str(X_test))
        y_pred = self.forward(X_test)
        #print("Output: " + str(y_pred))
        return y_pred

class Query_Attn_ExpandLL_Network(nn.Module):
    def __init__(self, ):
        super(Query_Attn_ExpandLL_Network, self).__init__()
        # parameters
        self.emb_size = 768
        self.cosine_sim = nn.CosineSimilarity()
        self.LL1 = nn.Linear(self.emb_size, self.emb_size*self.emb_size)

    def forward(self, X):
        self.Xq = X[:, :self.emb_size]
        self.Xp1 = X[:, self.emb_size:2*self.emb_size]
        self.Xp2 = X[:, 2*self.emb_size:]
        self.z = torch.relu(self.LL1(self.Xq))
        self.z = self.z.view(-1, self.emb_size, self.emb_size)
        self.sXp1 = torch.matmul(self.Xp1.view(-1, 1, self.emb_size), self.z).view(-1, self.emb_size)
        self.sXp2 = torch.matmul(self.Xp2.view(-1, 1, self.emb_size), self.z).view(-1, self.emb_size)
        o = self.cosine_sim(self.sXp1, self.sXp2)  # final activation function
        return o

    def num_flat_features(self, X):
        size = X.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def predict(self, X_test):
        #print("Predicted data based on trained weights: ")
        #print("Input (scaled): \n" + str(X_test))
        y_pred = self.forward(X_test)
        #print("Output: " + str(y_pred))
        return y_pred

class Query_Attn_LL_Network(nn.Module):
    def __init__(self, ):
        super(Query_Attn_LL_Network, self).__init__()
        # parameters
        self.emb_size = 768
        self.cosine_sim = nn.CosineSimilarity()
        self.LL1 = nn.Linear(self.emb_size, self.emb_size).cuda()
        self.LL2 = nn.Linear(2*self.emb_size, 1).cuda()
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, X):
        self.Xq = X[:, :self.emb_size]
        self.Xp1 = X[:, self.emb_size:2*self.emb_size]
        self.Xp2 = X[:, 2*self.emb_size:]
        self.z = torch.relu(self.LL1(self.dropout(self.Xq)))
        self.sXp1 = torch.mul(self.Xp1, self.z)
        self.sXp2 = torch.mul(self.Xp2, self.z)
        self.sXp = torch.cat((self.sXp1, self.sXp2), dim=1)
        #o = self.cosine_sim(self.sXp1, self.sXp2)  # final activation function
        o = self.LL2(self.sXp)
        o = o.reshape(-1)
        return o

    def num_flat_features(self, X):
        size = X.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def predict(self, X_test):
        #print("Predicted data based on trained weights: ")
        #print("Input (scaled): \n" + str(X_test))
        y_pred = self.forward(X_test)
        #print("Output: " + str(y_pred))
        return y_pred

class Query_Attn_LL_dimred_Network(nn.Module):
    def __init__(self, qd, pd, h1, o):
        super(Query_Attn_LL_dimred_Network, self).__init__()
        # parameters
        self.emb_size = 768
        self.qry_emb_size = qd
        self.para_emb_size = pd
        self.h1_size = h1
        self.out_emb_size = o
        self.cosine_sim = nn.CosineSimilarity()
        self.LL1 = nn.Linear(self.qry_emb_size * self.para_emb_size, self.h1_size).cuda()
        self.out = nn.Linear(self.h1_size, self.out_emb_size).cuda()
        self.dropout = nn.Dropout()

    def forward(self, X):
        self.Xq = X[:, :self.qry_emb_size]
        self.Xp1 = X[:, self.qry_emb_size:self.qry_emb_size+self.para_emb_size]
        self.Xp2 = X[:, self.qry_emb_size+self.para_emb_size:]
        self.outer1 = torch.einsum('bi, bj -> bij', (self.Xq, self.Xp1))
        self.outer1 = self.dropout(torch.flatten(self.outer1, start_dim=1))
        self.outer2 = torch.einsum('bi, bj -> bij', (self.Xq, self.Xp2))
        self.outer2 = self.dropout(torch.flatten(self.outer2, start_dim=1))
        self.z11 = torch.relu(self.LL1(self.outer1))
        self.z12 = torch.relu(self.LL1(self.outer2))
        self.o1 = torch.relu(self.out(self.z11))
        self.o2 = torch.relu(self.out(self.z12))
        o = self.cosine_sim(self.o1, self.o2)  # final activation function
        return o

    def num_flat_features(self, X):
        size = X.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def predict(self, X_test):
        #print("Predicted data based on trained weights: ")
        #print("Input (scaled): \n" + str(X_test))
        y_pred = self.forward(X_test)
        #print("Output: " + str(y_pred))
        return y_pred

class Siamese_Network(nn.Module):
    def __init__(self, ):
        super(Siamese_Network, self).__init__()
        # parameters
        self.emb_size = 768
        self.cosine_sim = nn.CosineSimilarity()
        # self.LL1 = nn.Linear(self.emb_size, self.emb_size)
        self.LL1 = nn.Linear(self.emb_size, self.emb_size)
        #self.LL1 = nn.Linear(self.emb_size, 128)
        #self.LL2 = nn.Linear(128, 64)
        #self.LL2 = nn.Linear(self.emb_size, self.emb_size)
        self.LL3 = nn.Linear(6*self.emb_size, 1)
        #self.LL3 = nn.Linear(6*64, 1)

    def forward(self, X):
        self.Xq = X[:, :self.emb_size]
        self.Xp1 = X[:, self.emb_size:2 * self.emb_size]
        self.Xp2 = X[:, 2 * self.emb_size:]
        self.z1 = torch.abs(self.Xp1 - self.Xq)
        self.z2 = torch.abs(self.Xp2 - self.Xq)
        self.zdiff = torch.abs(self.Xp1 - self.Xp2)
        self.zp1 = torch.relu(self.LL1(self.Xp1))
        self.zp2 = torch.relu(self.LL1(self.Xp2))
        self.zql = torch.relu(self.LL1(self.Xq))
        self.zd = torch.abs(self.zp1 - self.zp2)
        self.zdqp1 = torch.abs(self.zp1 - self.zql)
        self.zdqp2 = torch.abs(self.zp2 - self.zql)
        self.z = torch.cat((self.zql, self.zp1, self.zp2, self.zd, self.zdqp1, self.zdqp2), dim=1)
        #o = self.cosine_sim(self.z1, self.z2)  # final activation function
        o = torch.relu(self.LL3(self.z))
        o = o.reshape(-1)
        return o

    def num_flat_features(self, X):
        size = X.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def predict(self, X_test):
        #print("Predicted data based on trained weights: ")
        #print("Input (scaled): \n" + str(X_test))
        y_pred = self.forward(X_test)
        #print("Output: " + str(y_pred))
        return y_pred

class Siamese_Ablation_Network(nn.Module):
    def __init__(self, ):
        super(Siamese_Ablation_Network, self).__init__()
        # parameters
        self.emb_size = 768
        self.cosine_sim = nn.CosineSimilarity()
        # self.LL1 = nn.Linear(self.emb_size, self.emb_size)
        self.LL1 = nn.Linear(self.emb_size, self.emb_size)
        #self.LL1 = nn.Linear(self.emb_size, 128)
        #self.LL2 = nn.Linear(128, 64)
        #self.LL2 = nn.Linear(self.emb_size, self.emb_size)
        self.LL3 = nn.Linear(4*self.emb_size, 1)
        #self.LL3 = nn.Linear(6*64, 1)

    def forward(self, X):
        self.Xq = X[:, :self.emb_size]
        self.Xp1 = X[:, self.emb_size:2 * self.emb_size]
        self.Xp2 = X[:, 2 * self.emb_size:]
        self.z1 = torch.abs(self.Xp1 - self.Xq)
        self.z2 = torch.abs(self.Xp2 - self.Xq)
        self.zdiff = torch.abs(self.Xp1 - self.Xp2)
        self.zp1 = torch.relu(self.LL1(self.Xp1))
        self.zp2 = torch.relu(self.LL1(self.Xp2))
        self.zql = torch.relu(self.LL1(self.Xq))
        self.zd = torch.abs(self.zp1 - self.zp2)
        self.zdqp1 = torch.abs(self.zp1 - self.zql)
        self.zdqp2 = torch.abs(self.zp2 - self.zql)
        #self.z = torch.cat((self.zql, self.zp1, self.zp2, self.zd, self.zdqp1, self.zdqp2), dim=1)
        self.z = torch.cat((self.zp1, self.zp2, self.zdqp1, self.zdqp2), dim=1)
        #o = self.cosine_sim(self.z1, self.z2)  # final activation function
        o = torch.relu(self.LL3(self.z))
        o = o.reshape(-1)
        return o

    def num_flat_features(self, X):
        size = X.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def predict(self, X_test):
        #print("Predicted data based on trained weights: ")
        #print("Input (scaled): \n" + str(X_test))
        y_pred = self.forward(X_test)
        #print("Output: " + str(y_pred))
        return y_pred

class Siamese_Network_dimred(nn.Module):
    def __init__(self, red_dim):
        super(Siamese_Network_dimred, self).__init__()
        # parameters
        self.emb_size = 768
        self.red_dim = red_dim
        self.cosine_sim = nn.CosineSimilarity()
        # self.LL1 = nn.Linear(self.emb_size, self.emb_size)
        self.LL1 = nn.Linear(self.red_dim, self.red_dim)
        #self.LL1 = nn.Linear(self.emb_size, 128)
        #self.LL2 = nn.Linear(128, 64)
        #self.LL2 = nn.Linear(self.emb_size, self.emb_size)
        self.LL3 = nn.Linear(6*self.red_dim, 1)
        #self.LL3 = nn.Linear(6*64, 1)

    def forward(self, X):
        self.Xq = X[:, :self.red_dim]
        self.Xp1 = X[:, self.red_dim:2 * self.red_dim]
        self.Xp2 = X[:, 2 * self.red_dim:]
        self.z1 = torch.abs(self.Xp1 - self.Xq)
        self.z2 = torch.abs(self.Xp2 - self.Xq)
        self.zdiff = torch.abs(self.Xp1 - self.Xp2)
        self.zp1 = torch.relu(self.LL1(self.Xp1))
        self.zp2 = torch.relu(self.LL1(self.Xp2))
        self.zql = torch.relu(self.LL1(self.Xq))
        self.zd = torch.abs(self.zp1 - self.zp2)
        self.zdqp1 = torch.abs(self.zp1 - self.zql)
        self.zdqp2 = torch.abs(self.zp2 - self.zql)
        self.z = torch.cat((self.zql, self.zp1, self.zp2, self.zd, self.zdqp1, self.zdqp2), dim=1)
        #o = self.cosine_sim(self.z1, self.z2)  # final activation function
        o = torch.relu(self.LL3(self.z))
        o = o.reshape(-1)
        return o

    def num_flat_features(self, X):
        size = X.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def predict(self, X_test):
        #print("Predicted data based on trained weights: ")
        #print("Input (scaled): \n" + str(X_test))
        y_pred = self.forward(X_test)
        #print("Output: " + str(y_pred))
        return y_pred
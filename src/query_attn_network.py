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

def Mu_etAl_PPA(X):
    X = X.numpy()
    #sample_size = X.shape[0]
    #X = np.vstack((X[:, :768], X[:, 768:]))
    print('Applying PPA by Mu et al.')
    pca = PCA()
    X = X - np.mean(X)
    X_fit = pca.fit_transform(X)
    U1 = pca.components_
    z = []
    # Removing Projections on Top Components
    for i, x in enumerate(X):
        for u in U1[0:7]:
            x = x - np.dot(u.transpose(), x) * u
        z.append(x)
    z = np.array(z)
    #z = np.hstack((z[:sample_size, :], z[sample_size:, :]))
    return torch.tensor(z)

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
        self.LL1 = nn.Linear(self.emb_size, self.emb_size)

    def forward(self, X):
        self.Xq = X[:, :self.emb_size]
        self.Xp1 = X[:, self.emb_size:2*self.emb_size]
        self.Xp2 = X[:, 2*self.emb_size:]
        self.z = torch.relu(self.LL1(self.Xq))
        self.sXp1 = torch.mul(self.Xp1, self.z)
        self.sXp2 = torch.mul(self.Xp2, self.z)
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

class Query_Attn_LL_dimred_Network(nn.Module):
    def __init__(self, ):
        super(Query_Attn_LL_dimred_Network, self).__init__()
        # parameters
        self.emb_size = 768
        self.cosine_sim = nn.CosineSimilarity()
        self.LL1 = nn.Linear(self.emb_size, self.emb_size)

    def forward(self, X):
        self.Xq = Mu_etAl_PPA(X[:, :self.emb_size])
        self.Xp1 = Mu_etAl_PPA(X[:, self.emb_size:2*self.emb_size])
        self.Xp2 = Mu_etAl_PPA(X[:, 2*self.emb_size:])
        self.z = torch.relu(self.LL1(self.Xq))
        self.sXp1 = torch.mul(self.Xp1, self.z)
        self.sXp2 = torch.mul(self.Xp2, self.z)
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

class Siamese_Network(nn.Module):
    def __init__(self, ):
        super(Siamese_Network, self).__init__()
        # parameters
        self.emb_size = 768
        self.cosine_sim = nn.CosineSimilarity()
        self.LL1 = nn.Linear(self.emb_size, self.emb_size)

    def forward(self, X):
        self.Xq = X[:, :self.emb_size]
        self.Xp1 = X[:, self.emb_size:2 * self.emb_size]
        self.Xp2 = X[:, 2 * self.emb_size:]
        self.z1 = torch.relu(self.LL1(self.Xp1))
        self.z2 = torch.relu(self.LL1(self.Xp2))
        o = self.cosine_sim(self.z1, self.z2)  # final activation function
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
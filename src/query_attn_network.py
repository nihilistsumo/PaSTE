import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics import roc_auc_score
import random
import json
import argparse
import sys

class Dummy_Similarity_Network():
    def __init__(self, ):
        # parameters
        self.emb_size = 768
        self.cosine_sim = nn.CosineSimilarity()

    def forward(self, X):
        self.Xq = X[:, :self.emb_size]
        self.Xp1 = X[:, self.emb_size:2*self.emb_size]
        self.Xp2 = X[:, 2*self.emb_size:]
        print("Input (scaled): \n" + str(X))
        o = self.cosine_sim(self.Xp1, self.Xp2)
        print("Output: " + str(o))
        return o

    def num_flat_features(self, X):
        size = X.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class Query_Attn_Outprod_Network(nn.Module):
    def __init__(self, ):
        super(Query_Attn_Outprod_Network, self).__init__()
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
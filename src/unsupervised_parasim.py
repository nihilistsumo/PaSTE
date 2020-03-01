import torch
import torch.nn as nn
from scipy.spatial import distance
import numpy as np

class Paravec_Cosine():
    def __init__(self, emb_dim=768):
        # parameters
        self.emb_size = emb_dim
        self.cosine_sim = nn.CosineSimilarity()

    def forward(self, X):
        self.Xp1 = X[:, :self.emb_size]
        self.Xp2 = X[:, self.emb_size:]
        print("Input (scaled): \n" + str(X))
        o = self.cosine_sim(self.Xp1, self.Xp2)
        o = (o+1)/2 # to scale the cosine sim score between (0,1)
        print("Output: " + str(o))
        return o

    def num_flat_features(self, X):
        size = X.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
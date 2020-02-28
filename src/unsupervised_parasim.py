import torch
import torch.nn as nn

class Paravec_Cosine():
    def __init__(self, ):
        # parameters
        self.emb_size = 768
        self.cosine_sim = nn.CosineSimilarity()

    def forward(self, X):
        self.Xp1 = X[:, :self.emb_size]
        self.Xp2 = X[:, self.emb_size:]
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
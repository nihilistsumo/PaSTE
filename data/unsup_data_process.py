import numpy as np
from sklearn.decomposition import PCA
import torch

def Mu_etAl_PPA(X):
    X = X.numpy()
    sample_size = X.shape[0]
    X = np.vstack((X[:, :768], X[:, 768:]))
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
    z = np.hstack((z[:sample_size, :], z[sample_size:, :]))
    return torch.tensor(z)

def Raunak_etAl_dimred(X, d):
    print('Applying dim reduction algo by Raunak et al.')
    X = Mu_etAl_PPA(X).numpy()
    sample_size = X.shape[0]
    X = np.vstack((X[:, :768], X[:, 768:]))
    pca = PCA(n_components=d)
    X_train = X - np.mean(X)
    X_new_final = pca.fit_transform(X_train)

    # PCA to do Post-Processing Again
    pca = PCA(n_components=d)
    X_new = X_new_final - np.mean(X_new_final)
    X_new_fit = pca.fit_transform(X_new)
    Ufit = pca.components_

    X_new_final = X_new_final - np.mean(X_new_final)

    z = []
    for i, x in enumerate(X_new_final):
        for u in Ufit[0:7]:
            x = x - np.dot(u.transpose(), x) * u
        z.append(x)
    z = np.array(z)
    z = np.hstack((z[:sample_size, :], z[sample_size:, :]))
    return torch.tensor(z)
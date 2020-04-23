import numpy as np
from sklearn.decomposition import PCA
import torch
import argparse

def get_pca_transform_mat(emb_vec_file, out_transform_mat_file):
    X = np.load(emb_vec_file)
    pca = PCA()
    X = X - np.mean(X)
    X_fit = pca.fit_transform(X)
    U1 = pca.components_
    np.save(out_transform_mat_file, U1)
    
def Mu_etAl_PPA(X):
    #X = X.numpy()
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
    #return torch.tensor(z)
    return z

def Mu_etAl_PPA_qry_attn_data(X, emb_size):
    X = X.numpy()
    sample_size = X.shape[0]
    Xq = X[:, :emb_size]
    Xp = np.vstack((X[:, emb_size:2*emb_size], X[:, 2*emb_size:]))
    zq = Mu_etAl_PPA(Xq)
    zp = Mu_etAl_PPA(Xp)
    zp = np.hstack((zp[:sample_size, :], zp[sample_size:, :]))
    z = np.hstack((zq, zp))
    return torch.tensor(z)
    # pca = PCA()
    # Xq = Xq - np.mean(Xq)
    # Xq_fit = pca.fit_transform(Xq)
    # Uq1 = pca.components_
    # zq = []
    # # Removing Projections on Top Components
    # for i, x in enumerate(Xq):
    #     for u in Uq1[0:7]:
    #         x = x - np.dot(u.transpose(), x) * u
    #     zq.append(x)
    # zq = np.array(zq)
    #
    # pca = PCA()
    # Xp = Xp - np.mean(Xp)
    # Xp_fit = pca.fit_transform(Xp)
    # Up1 = pca.components_
    # zp = []
    # # Removing Projections on Top Components
    # for i, x in enumerate(Xp):
    #     for u in Up1[0:7]:
    #         x = x - np.dot(u.transpose(), x) * u
    #     zp.append(x)
    # zp = np.array(zp)
    #
    # zp = np.hstack((zp[:sample_size, :], zp[sample_size:, :]))
    # z = np.hstack((zq, zp))
    # return torch.tensor(z)

def Raunak_etAl_dimred(X, d):
    print('Applying dim reduction algo by Raunak et al.')
    X = Mu_etAl_PPA(X)
    #sample_size = X.shape[0]
    #X = np.vstack((X[:, :768], X[:, 768:]))
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
    #z = np.hstack((z[:sample_size, :], z[sample_size:, :]))
    return z

def Raunak_etAl_dimred_qry_attn_data(X, emb_size, dimred):
    X = X.numpy()
    sample_size = X.shape[0]
    Xq = X[:, :emb_size]
    Xp = np.vstack((X[:, emb_size:2 * emb_size], X[:, 2 * emb_size:]))
    zq = Raunak_etAl_dimred(Xq, dimred)
    zp = Raunak_etAl_dimred(Xp, dimred)
    zp = np.hstack((zp[:sample_size, :], zp[sample_size:, :]))
    z = np.hstack((zq, zp))
    return torch.tensor(z)

def main():
    parser = argparse.ArgumentParser(description='Dimensionality reduction algo for embedding vectors')
    parser.add_argument('-ei', '--input_embid', help='Path to input paraid file')
    parser.add_argument('-ev', '--input_embvec', help='Path to input embvec file')
    parser.add_argument('-op', '--option', type=int, help='Options 1: Mu et al, 2: Raunak dimred')
    parser.add_argument('-d', '--red_dim', type=int, help='Reduced dim size for Raunak')
    parser.add_argument('-oi', '--out_paraid', help='Path to output paraid')
    parser.add_argument('-ov', '--out_vec', help='Path to output embvec file')
    args = vars(parser.parse_args())
    input_id = args['input_embid']
    input_vec = args['input_embvec']
    option = args['option']
    outid = args['out_paraid']
    outvec = args['out_vec']
    pids = np.load(input_id)
    vecs = np.load(input_vec)
    if option == 1:
        newvecs = Mu_etAl_PPA(vecs)
        np.save(outid, pids)
        np.save(outvec, newvecs)
    elif option == 2:
        dimsize = args['red_dim']
        newvecs = Raunak_etAl_dimred(vecs, dimsize)
        np.save(outid, pids)
        np.save(outvec, newvecs)
    else:
        print('Wrong option')

if __name__ == '__main__':
    main()
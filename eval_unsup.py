import argparse
import torch
from data import process_pp_data as dat
from src.unsupervised_parasim import Paravec_Cosine
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
import json
import numpy as np

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

def main():
    # X = torch.tensor(torch.randn(256, 12))
    # target = torch.tensor(torch.tensor(([0.1, 0, 0, 0.4], [0, 0.3, 0.2, 0.1], [0.1, 0.2, 0.2, 0.1], [0, 0.8, 0.1, 0.1])))
    #
    # X_q = torch.matmul(X[:, :4], target)
    # X_p1 = torch.matmul(X[:, 4:8], torch.t(X_q))
    # X_p2 = torch.matmul(X[:, 8:], torch.t(X_q))
    # y = cosine_sim(X_p1, X_p2)
    #
    # X_test = torch.tensor(torch.randn(8, 12))
    # X_testq = torch.matmul(X_test[:, :4], target)
    # X_testp1 = torch.matmul(X_test[:, 4:8], torch.t(X_testq))
    # X_testp2 = torch.matmul(X_test[:, 8:], torch.t(X_testq))
    # y_test = cosine_sim(X_testp1, X_testp2)

    parser = argparse.ArgumentParser(description='Train and evaluate query attentive network for paragraph similarity task')
    parser.add_argument('-e', '--emb_dir', help='Path to para embedding directory')
    parser.add_argument('-n', '--variation', help='Model variation (1/2)')
    parser.add_argument('-m', '--emb_file_prefix', help='Name of the model used to embed the paras/ embedding file prefix')
    parser.add_argument('-p', '--emb_paraids_file', help='Path to embedding paraids file')
    parser.add_argument('-em', '--emb_mode', help='Embedding mode: s=single embedding file, m=multi emb files in shards')
    parser.add_argument('-b', '--emb_batch_size', help='Batch size of each embedding file shard')
    parser.add_argument('-d', '--data_file', help='Path to parapair json data file')
    parser.add_argument('-k', '--reduced_dim', default=-1, help='Reduced dimension size if variation 3 is chosen')
    args = vars(parser.parse_args())
    emb_dir = args['emb_dir']
    variation = int(args['variation'])
    emb_prefix = args['emb_file_prefix']
    emb_pids_file = args['emb_paraids_file']
    emb_mode = args['emb_mode']
    emb_batch = int(args['emb_batch_size'])
    data_filepath = args['data_file']
    if(args['reduced_dim'] != -1):
        d = int(args['reduced_dim'])
    with open(data_filepath, 'r') as dt:
        data = json.load(dt)
    X, y = dat.get_data_unsup(emb_dir, emb_prefix, emb_pids_file, data, emb_mode, emb_batch)
    if variation == 1:
        Unsup = Paravec_Cosine()
        y_pred = Unsup.forward(X)
        auc_score = roc_auc_score(y, y_pred)
        print('\n###\nThis cosine sim AUC score is micro averaged. Hence it will give slightly different result when\ncompared to'
              'macro-averaged AUC score (mean of AUC scores per page) obtained from SummerProjectEvaluation\n###\n')
        print('AUC score: ' + str(auc_score))
    elif variation == 2:
        X = Mu_etAl_PPA(X)
        Unsup = Paravec_Cosine()
        y_pred = Unsup.forward(X)
        auc_score = roc_auc_score(y, y_pred)
        print('AUC score: ' + str(auc_score))
    elif variation == 3:
        X = Raunak_etAl_dimred(X, d)
        Unsup = Paravec_Cosine(d)
        y_pred = Unsup.forward(X)
        auc_score = roc_auc_score(y, y_pred)
        print('AUC score: ' + str(auc_score))
    else:
        print('Trying random prediction')
        y_pred = torch.randn(y.shape[0])
        y_pred = (y_pred - torch.min(y_pred)) / (torch.max(y_pred) - torch.min(y_pred))
        auc_score = roc_auc_score(y, y_pred)
        print('AUC score: ' + str(auc_score))

if __name__ == '__main__':
    main()
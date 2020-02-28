import argparse
from data import process_qry_attn_data as dat
from src.query_attn_network import Dummy_Similarity_Network, Query_Attn_ExpandLL_Network, Query_Attn_LL_Network, \
    Siamese_Network, Query_Attn_InteractMatrix_Network
from sklearn.metrics import roc_auc_score
import sys
import torch
import torch.nn as nn
import torch.optim as optim

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
    parser.add_argument('-d', '--data_file', help='Path to query attn data file')
    args = vars(parser.parse_args())
    emb_dir = args['emb_dir']
    variation = int(args['variation'])
    emb_prefix = args['emb_file_prefix']
    emb_pids_file = args['emb_paraids_file']
    emb_mode = args['emb_mode']
    emb_batch = int(args['emb_batch_size'])
    data_filepath = args['data_file']
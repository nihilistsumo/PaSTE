import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics import roc_auc_score
from sentbert_embed import SentbertParaEmbedding
import random
import json
import argparse
import sys

class Dummy_cosine_sim():
    def __init__(self, ):
        # parameters
        self.emb_size = 768
        self.cosine_sim = nn.CosineSimilarity()

    def forward(self, X):
        self.Xq = X[:, :self.emb_size]
        self.Xp1 = X[:, self.emb_size:2*self.emb_size]
        self.Xp2 = X[:, 2*self.emb_size:]

        o = self.cosine_sim(self.Xp1, self.Xp2)
        return o

    def num_flat_features(self, X):
        size = X.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def predict(self, X_test):
        print("Predicted data based on trained weights: ")
        print("Input (scaled): \n" + str(X_test))
        print("Output: " + str(self.forward(X_test)))

class Neural_Network(nn.Module):
    def __init__(self, ):
        super(Neural_Network, self).__init__()
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
        print("Predicted data based on trained weights: ")
        print("Input (scaled): \n" + str(X_test))
        print("Output: " + str(self.forward(X_test)))

class Neural_Network_scale(nn.Module):
    def __init__(self, ):
        super(Neural_Network_scale, self).__init__()
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
        print("Predicted data based on trained weights: ")
        print("Input (scaled): \n" + str(X_test))
        y_pred = self.forward(X_test)
        print("Output: " + str(y_pred))
        return y_pred

class Neural_Network_siamese(nn.Module):
    def __init__(self, ):
        super(Neural_Network_siamese, self).__init__()
        # parameters
        self.emb_size = 768
        self.cosine_sim = nn.CosineSimilarity()
        self.LL1 = nn.Linear(self.emb_size, self.emb_size)

    def forward(self, X):
        self.Xp1 = X[:, :self.emb_size]
        self.Xp2 = X[:, self.emb_size:]
        self.z1 = torch.relu(self.LL1(self.Xp1))
        self.z2 = torch.relu(self.LL1(self.Xp2))
        o = self.cosine_sim(self.z1, self.z2)
        return o

    def num_flat_features(self, X):
        size = X.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def predict(self, X_test):
        print("Predicted data based on trained weights: ")
        print("Input (scaled): \n" + str(X_test))
        y_pred = self.forward(X_test)
        print("Output: " + str(y_pred))
        return y_pred

def write_query_attn_dataset_bert(bert_data, art_qrels, outfile, num_samples=1000, emb_paraids_available=None):
    art_qrels_rev_dict = dict()
    with open(art_qrels, 'r') as aq:
        for l in aq:
            art_qrels_rev_dict[l.split(' ')[2]] = l.split(' ')[0]
    if emb_paraids_available != None:
        emb_paraids = list(np.load(emb_paraids_available))
    posdata = []
    negdata = []
    fl = True
    with open(bert_data, 'r') as bd:
        for l in bd:
            if fl:
                fl = False
                continue
            label = l.split('\t')[0]
            p1 = l.split('\t')[1]
            p2 = l.split('\t')[2]
            if p1 not in emb_paraids or p2 not in emb_paraids:
                continue
            if label == '0' and len(negdata) < num_samples // 2:
                art = art_qrels_rev_dict[p1]
                if art != art_qrels_rev_dict[p2]:
                    print('p1 art: ' + art + ' p2 art: ' + art_qrels_rev_dict[p2])
                else:
                    negdata.append('0\t' + art.split(':')[1].replace('%20', ' ') + '\t' + p1 + '\t' + p2)
            elif len(posdata) < num_samples // 2:
                art = art_qrels_rev_dict[p1]
                if art != art_qrels_rev_dict[p2]:
                    print('p1 art: ' + art + ' p2 art: ' + art_qrels_rev_dict[p2])
                else:
                    posdata.append('1\t' + art.split(':')[1].replace('%20', ' ') + '\t' + p1 + '\t' + p2)
            if len(posdata) + len(negdata) >= num_samples:
                break
    data = posdata + negdata
    print('Output data has ' + str(len(data)) + ' samples with ' + str(len(posdata)) + ' +ve and ' + str(len(negdata)) + ' -ve samples')
    random.shuffle(data)
    with open(outfile, 'w') as out:
        for d in data:
            out.write(d+'\n')

def write_query_attn_dataset_parapair(parapair_data, outfile):
    with open(parapair_data, 'r') as pp:
        pp_data = json.load(pp)
    data = []
    for page in pp_data.keys():
        page_labels = pp_data[page]['labels']
        if len(page_labels) == 0:
            continue
        page_pairs = pp_data[page]['parapairs']
        for i in range(len(page_labels)):
            data.append(str(page_labels[i]) + '\t' + page.split(':')[1].replace('%20', ' ') + '\t' +
                        page_pairs[i].split('_')[0] + '\t' + page_pairs[i].split('_')[1])
    random.shuffle(data)
    with open(outfile, 'w') as out:
        for d in data:
            out.write(d+'\n')

def get_data(emb_dir, emb_file_prefix, emb_paraids_file, query_attn_data_file, emb_mode, batch_size=10000):
    model = SentenceTransformer(emb_file_prefix)
    print("Using " + emb_file_prefix + " to embed query, should be same as the embedding file")
    paraids = list(np.load(emb_paraids_file))
    X_train = []
    y_train = []
    if emb_mode == 's':
        para_emb = np.load(emb_dir + '/' + emb_file_prefix + '-part1.npy')
        para_emb_dict = dict()
        for i in range(len(paraids)):
            para_emb_dict[paraids[i]] = para_emb[i]
    elif emb_mode == 'm':
        emb = SentbertParaEmbedding(emb_paraids_file, emb_dir, emb_file_prefix, batch_size)
        # Not needed if using get_single_embedding of sentbert
        # paraids_dat = set()
        # with open(query_attn_data_file, 'r') as qd:
        #     for l in qd:
        #         paraids_dat.add(l.split('\t')[2])
        #         paraids_dat.add(l.split('\t')[3].rstrip())
        # para_emb_dict = emb.get_embeddings(list(paraids_dat))
    else:
        print('Embedding mode not supported')
        return 1

    with open(query_attn_data_file, 'r') as qd:
        for l in qd:
            qemb = model.encode([l.split('\t')[1]])[0]
            p1 = l.split('\t')[2]
            p2 = l.split('\t')[3].rstrip()
            if emb_mode == 's':
                p1emb = para_emb_dict[p1]
                p2emb = para_emb_dict[p2]
            elif emb_mode == 'm':
                p1emb = emb.get_single_embedding(p1)
                p2emb = emb.get_single_embedding(p2)

            if p1emb is None or p2emb is None:
                continue
            X_train.append(np.hstack((qemb, p1emb, p2emb)))
            y_train.append(float(l.split('\t')[0]))
    return (torch.tensor(X_train), torch.tensor(y_train))

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
    parser.add_argument('-et', '--emb_dir_test', help='Path to para embedding directory for test split paras')
    parser.add_argument('-n', '--neural_model', help='Neural model variation (0/1/2)')
    parser.add_argument('-lr', '--learning_rate', help='Learning rate')
    parser.add_argument('-i', '--num_iteration', help='No. of iteration')
    parser.add_argument('-m', '--emb_file_prefix', help='Name of the model used to embed the paras/ embedding file prefix')
    parser.add_argument('-p', '--emb_paraids_file', help='Path to train embedding paraids file')
    parser.add_argument('-pt', '--test_emb_paraids_file', help='Path to test embedding paraids file')
    parser.add_argument('-em', '--emb_mode', help='Embedding mode: s=single embedding file, m=multi emb files in shards')
    parser.add_argument('-b', '--emb_batch_size', help='Batch size of each embedding file shard')
    parser.add_argument('-d', '--train_data_file', help='Path to train data file')
    parser.add_argument('-t', '--test_data_file', help='Path to test data file')
    parser.add_argument('-o', '--model_outfile', help='Path to save the trained model')
    args = vars(parser.parse_args())
    emb_dir = args['emb_dir']
    emb_dir_test = args['emb_dir_test']
    variation = int(args['neural_model'])
    lrate = float(args['learning_rate'])
    iter = int(args['num_iteration'])
    emb_prefix = args['emb_file_prefix']
    emb_pids_file = args['emb_paraids_file']
    test_emb_pids_file = args['test_emb_paraids_file']
    emb_mode = args['emb_mode']
    emb_batch = int(args['emb_batch_size'])
    train_filepath = args['train_data_file']
    test_filepath = args['test_data_file']
    model_out = args['model_outfile']
    X, y = get_data(emb_dir, emb_prefix, emb_pids_file, train_filepath, emb_mode, emb_batch)
    if emb_dir_test == '':
        X_test, y_test = get_data(emb_dir, emb_prefix, test_emb_pids_file, test_filepath, 's')
    else:
        X_test, y_test = get_data(emb_dir_test, emb_prefix, test_emb_pids_file, test_filepath, 's')

    cosine_sim = nn.CosineSimilarity()
    if variation == 1:
        NN = Neural_Network()
    elif variation == 2:
        NN = Neural_Network_scale()
    elif variation == 0:
        NN = Dummy_cosine_sim()
        y_pred = NN.predict(X_test).numpy()
        auc_score = roc_auc_score(y_test, y_pred)
        print('AUC score: ' + str(auc_score))
        sys.exit(0)
    else:
        print('Wrong model variation selected!')
        exit(1)
    criterion = nn.MSELoss()
    opt = optim.SGD(NN.parameters(), lr=lrate)
    for i in range(iter):  # trains the NN 1,000 times
        opt.zero_grad()
        output = NN(X)
        loss = criterion(output, y)
        print('Iteration: ' + str(i) + ', loss: ' +str(loss))
        loss.backward()
        opt.step()
    # NN.saveWeights(NN)
    y_pred = NN.predict(X_test).detach().numpy()
    auc_score = roc_auc_score(y_test, y_pred)
    print('AUC score: ' + str(auc_score))
    #print(NN.parameters())
    #print('True output: ' + str(y_test))
    #print('Features: ' + str(NN.num_flat_features(X_test)))
    print('Saving model at ' + model_out)
    torch.save(NN.state_dict(), model_out)

if __name__ == '__main__':
    main()
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics import roc_auc_score
import random
import argparse

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
        self.Xq = F.normalize(X[:, :self.emb_size], p=2, dim=1)
        self.Xp1 = F.normalize(X[:, self.emb_size:2*self.emb_size], p=2, dim=1)
        self.Xp2 = F.normalize(X[:, 2*self.emb_size:], p=2, dim=1)
        self.z = torch.relu(self.LL1(self.Xq))
        self.sXp1 = F.normalize(torch.mul(self.Xp1, self.z), p=2, dim=1)
        self.sXp2 = F.normalize(torch.mul(self.Xp2, self.z), p=2, dim=1)
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

def write_query_attn_dataset(bert_data, art_qrels, outfile, num_samples=1000):
    art_qrels_rev_dict = dict()
    with open(art_qrels, 'r') as aq:
        for l in aq:
            art_qrels_rev_dict[l.split(' ')[2]] = l.split(' ')[0]
    posdata = []
    negdata = []
    fl = True
    with open(bert_data, 'r') as bd:
        for l in bd:
            if fl:
                fl = False
                continue
            label = l.split('\t')[0]
            if label == '0' and len(negdata) < num_samples // 2:
                p1 = l.split('\t')[1]
                p2 = l.split('\t')[2]
                art = art_qrels_rev_dict[p1]
                assert art == art_qrels_rev_dict[p2]
                negdata.append('0\t' + art.split(':')[1].replace('%20', ' ') + '\t' + p1 + '\t' + p2)
            elif len(posdata) < num_samples // 2:
                p1 = l.split('\t')[1]
                p2 = l.split('\t')[2]
                art = art_qrels_rev_dict[p1]
                assert art == art_qrels_rev_dict[p2]
                posdata.append('1\t' + art.split(':')[1].replace('%20', ' ') + '\t' + p1 + '\t' + p2)
            if len(posdata) + len(negdata) >= num_samples:
                break
    data = posdata + negdata
    random.shuffle(data)
    with open(outfile, 'w') as out:
        for d in data:
            out.write(d+'\n')

def get_data(emb_dir, emb_model_name, query_attn_data_file):
    model = SentenceTransformer(emb_model_name)
    print("Using " + emb_model_name + " to embed query, should be same as the embedding file")
    para_emb = np.load(emb_dir + '/' + emb_model_name + '-part1.npy')
    paraids = list(np.load(emb_dir + '/paraids.npy'))
    X_train = []
    y_train = []
    with open(query_attn_data_file, 'r') as qd:
        for l in qd:
            y_train.append(float(l.split('\t')[0]))
            qemb = model.encode([l.split('\t')[1]])[0]
            p1emb = para_emb[paraids.index(l.split('\t')[2])]
            p2emb = para_emb[paraids.index(l.split('\t')[3].rstrip())]
            X_train.append(np.hstack((qemb, p1emb, p2emb)))
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
    parser.add_argument('-n', '--neural_model', help='Neural model variation (1/2)')
    parser.add_argument('-lr', '--learning_rate', help='Learning rate')
    parser.add_argument('-i', '--num_iteration', help='No. of iteration')
    parser.add_argument('-m', '--emb_model_name', help='Name of the model used to embed the paras')
    parser.add_argument('-d', '--train_data_file', help='Path to train data file')
    parser.add_argument('-t', '--test_data_file', help='Path to test data file')
    parser.add_argument('-o', '--model_outfile', help='Path to save the trained model')
    args = vars(parser.parse_args())
    emb_dir = args['emb_dir']
    emb_dir_test = args['emb_dir_test']
    variation = int(args['neural_model'])
    lrate = float(args['learning_rate'])
    iter = int(args['num_iteration'])
    model_name = args['emb_model_name']
    train_filepath = args['train_data_file']
    test_filepath = args['test_data_file']
    model_out = args['model_outfile']
    X, y = get_data(emb_dir, model_name, train_filepath)
    if emb_dir_test == '':
        X_test, y_test = get_data(emb_dir, model_name, test_filepath)
    else:
        X_test, y_test = get_data(emb_dir_test, model_name, test_filepath)

    cosine_sim = nn.CosineSimilarity()
    if variation == 1:
        NN = Neural_Network()
    elif variation == 2:
        NN = Neural_Network_scale()
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
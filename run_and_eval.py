import argparse
from data import process_qry_attn_data as dat
from data import unsup_data_process as undat
from src.query_attn_network import Query_Attn_ExpandLL_Network, Query_Attn_LL_Network, \
    Siamese_Network, Query_Attn_InteractMatrix_Network, Siamese_Ablation_Network, Siamese_Network_dimred
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
    parser.add_argument('-e', '--emb_file_train', help='Path to para embedding file for train split paras')
    parser.add_argument('-et', '--emb_file_test', help='Path to para embedding file for test split paras')
    parser.add_argument('-n', '--neural_model', help='Neural model variation (0/1/2)')
    parser.add_argument('-lr', '--learning_rate', help='Learning rate')
    parser.add_argument('-i', '--num_iteration', type=int, help='No. of iteration')
    parser.add_argument('-bt', '--batch_size', type=int, help='Size of each training batch')
    parser.add_argument('-mp', '--emb_model_name', help='Emb model name or path')
    parser.add_argument('-p', '--train_emb_paraids_file', help='Path to train embedding paraids file')
    parser.add_argument('-pt', '--test_emb_paraids_file', help='Path to test embedding paraids file')
    parser.add_argument('-d', '--train_data_file', help='Path to train data file')
    parser.add_argument('-t', '--test_data_file', help='Path to test data file')
    parser.add_argument('-pca', '--pca_mat', help='Path to PCA transformation matrix')
    parser.add_argument('-mu', '--mudim_sub', type=int, help='No of pca dim to subtract')
    parser.add_argument('-pd', '--para_dim', type=int, help='Dimension of para embedding to be reduced by Raunak et al')
    parser.add_argument('-o', '--model_outfile', help='Path to save the trained model')
    args = vars(parser.parse_args())
    emb_file_train = args['emb_file_train']
    emb_file_test = args['emb_file_test']
    variation = int(args['neural_model'])
    lrate = float(args['learning_rate'])
    iter = args['num_iteration']
    batch = args['batch_size']
    emb_model_name = args['emb_model_name']
    emb_pids_file = args['train_emb_paraids_file']
    test_emb_pids_file = args['test_emb_paraids_file']
    train_filepath = args['train_data_file']
    test_filepath = args['test_data_file']
    model_out = args['model_outfile']
    if torch.cuda.is_available():
        device1 = torch.device('cuda:0')
        device2 = torch.device('cuda:1')
    else:
        device1 = torch.device('cpu')
        device2 = device1
    if args['para_dim'] != None:
        reddim = int(args['para_dim'])
    log_out = model_out + '.train.log'
    if variation == 5:
        pca_mat = args['pca_mat']
        mudim = args['mudim_sub']
        X, y = dat.get_data_mu_etal(emb_model_name, emb_file_train, emb_pids_file, train_filepath, pca_mat, mudim)

        # X_val = X[:100, :].cuda(device1)
        X_val = X[:100, :]
        y_val = y[:100]
        X_train = X[100:, :]
        y_train = y[100:]
        X_test, y_test = dat.get_data_mu_etal(emb_model_name, emb_file_test, test_emb_pids_file, test_filepath, pca_mat, mudim)
    elif variation != 0:
        X, y = dat.get_data(emb_model_name, emb_file_train, emb_pids_file, train_filepath)

        #X_val = X[:100, :].cuda(device1)
        X_val = X[:100, :]
        y_val = y[:100]
        X_train = X[100:, :]
        y_train = y[100:]
        X_test, y_test = dat.get_data(emb_model_name, emb_file_test, test_emb_pids_file, test_filepath)

    if variation == 1:
        NN = Query_Attn_ExpandLL_Network().to(device1)
    elif variation == 2:
        NN = Query_Attn_LL_Network().to(device1)
    elif variation == 3 or variation == 5:
        NN = Siamese_Network().to(device1)
    elif variation == 4:
        NN = Query_Attn_InteractMatrix_Network().to(device1)
    elif variation == 6:
        NN = Siamese_Network_dimred(reddim).to(device1)
        X_train = undat.Raunak_etAl_dimred_qry_attn_data(X_train, NN.emb_size, reddim)
        X_val = undat.Raunak_etAl_dimred_qry_attn_data(X_val, NN.emb_size, reddim)
        X_test = undat.Raunak_etAl_dimred_qry_attn_data(X_test, NN.emb_size, reddim)
    elif variation == 7:
        NN = Siamese_Ablation_Network().to(device1)
    else:
        print('Wrong model variation selected!')
        exit(1)

    # X_train = X_train.cuda(device1)
    num_batch = X_train.shape[0] // batch + 1
    y_train = y_train.cuda(device1)
    X_val = X_val.cuda(device1)
    X_test = X_test.cuda(device1)
    criterion = nn.MSELoss().cuda(device1)
    #criterion = nn.BCELoss().cuda(device1)
    opt = optim.SGD(NN.parameters(), lr=lrate)
    print()
    test_auc = 0.0
    with open(log_out, 'w') as lo:
        for i in range(iter):
            for j in range(num_batch):
                X_train_curr = X_train[j*batch: (j+1)*batch]
                X_train_curr = X_train_curr.cuda(device1)
                opt.zero_grad()
                output = NN(X_train_curr)
                loss = criterion(output, y_train)
                y_val_pred = NN.predict(X_val).detach().cpu().numpy()
                val_auc_score = roc_auc_score(y_val, y_val_pred)
                sys.stdout.write('\r' + 'Iteration: ' + str(i) + ', loss: ' +str(loss) + ', val AUC: ' +
                                 '{:.4f}'.format(val_auc_score) + ', test AUC: ' + '{:.4f}'.format(test_auc))
                if i%10 == 0:
                    lo.write('Iteration: ' + str(i) + ', loss: ' +str(loss) + ', val AUC: ' +str(val_auc_score) + '\n')
                    y_pred = NN.predict(X_test).detach().cpu().numpy()
                    test_auc = roc_auc_score(y_test, y_pred)
                loss.backward()
                opt.step()
    # NN.saveWeights(NN)
    y_pred = NN.predict(X_test).detach().cpu().numpy()
    auc_score = roc_auc_score(y_test, y_pred)
    print()
    print("Predicted data based on trained weights: ")
    print("Input (scaled): \n" + str(X_test))
    print("Output: " + str(y_pred))
    print('AUC score: ' + str(auc_score))
    #print(NN.parameters())
    #print('True output: ' + str(y_test))
    #print('Features: ' + str(NN.num_flat_features(X_test)))
    print('Saving model at ' + model_out)
    torch.save(NN.state_dict(), model_out)

if __name__ == '__main__':
    main()
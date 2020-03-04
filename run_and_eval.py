import argparse
from data import process_qry_attn_data as dat
from data import unsup_data_process as undat
from src.query_attn_network import Query_Attn_ExpandLL_Network, Query_Attn_LL_Network, \
    Siamese_Network, Query_Attn_InteractMatrix_Network, Query_Attn_LL_dimred_Network
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
    parser.add_argument('-e', '--emb_dir', help='Path to para embedding directory for train split paras')
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
    parser.add_argument('-qd', '--query_dim', type=int, help='Dimension of query embedding to be reduced by Raunak et al')
    parser.add_argument('-pd', '--para_dim', type=int, help='Dimension of para embedding to be reduced by Raunak et al')
    parser.add_argument('-h1', '--hidden_layer_size', type=int, help='Hidden layer size of dimred NN')
    parser.add_argument('-oe', '--out_emb', type=int, help='Output size of dimred NN')
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
    if torch.cuda.is_available():
        device1 = torch.device('cuda:0')
        device2 = torch.device('cuda:1')
    else:
        device1 = torch.device('cpu')
        device2 = device1
    if args['query_dim'] != None:
        qdim = int(args['query_dim'])
        pdim = int(args['para_dim'])
        h1 = int(args['hidden_layer_size'])
        oemb = int(args['out_emb'])
    log_out = model_out + '.train.log'
    if variation != 0:
        X, y = dat.get_data(emb_dir, emb_prefix, emb_pids_file, train_filepath, emb_mode, emb_batch)

        X_val = X[:100, :]
        y_val = y[:100]
        X_train = X[100:, :]
        y_train = y[100:].cuda(device1)
    if emb_dir_test == '':
        X_test, y_test = dat.get_data(emb_dir, emb_prefix, test_emb_pids_file, test_filepath, 's')
    else:
        X_test, y_test = dat.get_data(emb_dir_test, emb_prefix, test_emb_pids_file, test_filepath, 's')

    if variation == 1:
        NN = Query_Attn_ExpandLL_Network().to(device1)
    elif variation == 2:
        NN = Query_Attn_LL_Network().to(device1)
    elif variation == 3:
        NN = Siamese_Network().to(device1)
    elif variation == 4:
        NN = Query_Attn_InteractMatrix_Network().to(device1)
    elif variation == 5:
        NN = Query_Attn_LL_Network().to(device1)
        X_train = undat.Mu_etAl_PPA_qry_attn_data(X_train, NN.emb_size)
        X_val = undat.Mu_etAl_PPA_qry_attn_data(X_val, NN.emb_size)
        X_test = undat.Mu_etAl_PPA_qry_attn_data(X_test, NN.emb_size)
    elif variation == 6:
        NN = Query_Attn_LL_dimred_Network(qdim, pdim, h1, oemb).to(device1)
        X_train = undat.Raunak_etAl_dimred_qry_attn_data(X_train, NN.emb_size, qdim, pdim).cuda(device1)
        X_val = undat.Raunak_etAl_dimred_qry_attn_data(X_val, NN.emb_size, qdim, pdim).cuda(device1)
        X_test = undat.Raunak_etAl_dimred_qry_attn_data(X_test, NN.emb_size, qdim, pdim).cuda(device1)
    else:
        print('Wrong model variation selected!')
        exit(1)


    criterion = nn.MSELoss().cuda(device1)
    opt = optim.SGD(NN.parameters(), lr=lrate)
    print()
    with open(log_out, 'w') as lo:
        for i in range(iter):  # trains the NN 1,000 times
            opt.zero_grad()
            output = NN(X_train)
            loss = criterion(output, y_train)
            y_val_pred = NN.predict(X_val).detach().cpu().numpy()
            val_auc_score = roc_auc_score(y_val, y_val_pred)
            sys.stdout.write('\r' + 'Iteration: ' + str(i) + ', loss: ' +str(loss) + ', val AUC: ' + '{:.4f}'.format(val_auc_score))
            if i%10 == 0:
                lo.write('Iteration: ' + str(i) + ', loss: ' +str(loss) + ', val AUC: ' +str(val_auc_score) + '\n')
            loss.backward()
            opt.step()
    # NN.saveWeights(NN)
    y_pred = NN.predict(X_test).detach().numpy()
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
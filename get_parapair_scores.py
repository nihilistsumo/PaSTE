import argparse
from data import process_qry_attn_data as dat
import torch
from src.query_attn_network import Siamese_Network
import json
from json import encoder
import numpy as np
encoder.FLOAT_REPR = lambda o: format(o, '.16f')

def write_parapair_scores(nn_model_path, emb_model_name, emb_vec_file, emb_pids_file, qry_attn_file, outfile):
    X_test, y_test = dat.get_data(emb_model_name, emb_vec_file, emb_pids_file, qry_attn_file)
    #X_test = X_test.cuda(device1)
    model = Siamese_Network()
    model.load_state_dict(torch.load(nn_model_path))
    y_pred = model.predict(X_test).detach().cpu().numpy()
    parapairs = []
    with open(qry_attn_file, 'r') as qd:
        for l in qd:
            parapairs.append(l.split('\t')[2]+'_'+l.split('\t')[3].rstrip())
    parapair_score_dict = {}
    for i in range(len(parapairs)):
        parapair_score_dict[parapairs[i]] = float(y_pred[i])
        if i%10000 == 0:
            print(str(i) + ' samples inferred')
    with open(outfile, 'w') as out:
        json.dump(parapair_score_dict, out)

def main():
    parser = argparse.ArgumentParser(description='Write parapair scores for trained Siamese network')
    parser.add_argument('-et', '--emb_vec_file', help='Path to para embedding vec file')
    parser.add_argument('-mn', '--emb_model_name', help='Emb model name or path')
    parser.add_argument('-pt', '--emb_paraids_file', help='Path to embedding paraids file')
    parser.add_argument('-qt', '--qry_attn_file', help='Path to query attn file')
    parser.add_argument('-np', '--neural_model_path', help='Path to saved and trained model')
    parser.add_argument('-op', '--outscore', help='Path to output parascore file')
    args = vars(parser.parse_args())
    emb_vec_file = args['emb_vec_file']
    emb_model_name = args['emb_model_name']
    emb_pids_file = args['emb_paraids_file']
    qry_attn_filepath = args['qry_attn_file']
    neural_model = args['neural_model_path']
    outpath = args['outscore']
    write_parapair_scores(neural_model, emb_model_name, emb_vec_file, emb_pids_file, qry_attn_filepath, outpath)

if __name__ == '__main__':
    main()
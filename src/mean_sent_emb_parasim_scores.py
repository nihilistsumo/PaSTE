import transformers
from transformers import BertForSequenceClassification, XLNetForSequenceClassification, RobertaForSequenceClassification
from transformers import AlbertForSequenceClassification, XLMRobertaForSequenceClassification, FlaubertForSequenceClassification
from transformers import XLMForSequenceClassification

import numpy as np
import json
import torch
import torch.nn.functional as F
import argparse
from scipy.spatial import distance

def get_pair_ids(pairtext_file):
    pair_ids = []
    splitter = '_'
    with open(pairtext_file, 'r') as tst:
        i = 0
        fl = True
        for l in tst:
            if fl:
                fl = False
                continue
            id1 = l.split('\t')[1]
            id2 = l.split('\t')[2]
            if '_' in id1:
                pair_ids.append(id1 + '#' + id2)
                splitter = '#'
            else:
                pair_ids.append(id1 + '_' + id2)
    return pair_ids, splitter

def get_embed_similarity_scores(pair_ids, paraids_dict, splitter, embed_dir, embed_file_prefix, batch_size, norm=-1):
    pred_dict = dict()
    c = 10000
    part_cache = 1
    emb_list = np.load(embed_dir + '/' + embed_file_prefix + '-part1.npy')
    for i in range(len(pair_ids)):
        p1 = pair_ids[i].split(splitter)[0]
        p2 = pair_ids[i].split(splitter)[1]
        part_p1 = int(paraids_dict[p1][0])
        start_index_p1 = int(paraids_dict[p1][1])
        len_p1 = int(paraids_dict[p1][2])
        part_p2 = int(paraids_dict[p2][0])
        start_index_p2 = int(paraids_dict[p2][1])
        len_p2 = int(paraids_dict[p2][2])
        if batch_size == -1:
            if len_p1 == 0:
                p1vec = np.zeros(emb_list.shape[1])
            else:
                p1vec = np.mean(emb_list[start_index_p1: start_index_p1 + len_p1], axis=0)
            if len_p2 == 0:
                p2vec = np.zeros(emb_list.shape[1])
            else:
                p2vec = np.mean(emb_list[start_index_p2: start_index_p2 + len_p2], axis=0)
        else:
            if part_p1 != part_cache:
                emb_list = np.load(embed_dir + '/' + embed_file_prefix + '-part' + str(part_p1) + '.npy')
                part_cache = part_p1
            if len_p1 == 0:
                p1vec = np.zeros(emb_list.shape[1])
            else:
                p1vec = np.mean(emb_list[start_index_p1: start_index_p1 + len_p1], axis=0)
            if part_p2 != part_cache:
                emb_list = np.load(embed_dir + '/' + embed_file_prefix + '-part' + str(part_p2) + '.npy')
                part_cache = part_p2
            if len_p2 == 0:
                p2vec = np.zeros(emb_list.shape[1])
            else:
                p2vec = np.mean(emb_list[start_index_p2: start_index_p2 + len_p2], axis=0)
        if norm == -1:
            pred_dict[pair_ids[i]] = 1 - distance.cosine(p1vec, p2vec)
        else:
            pred_dict[pair_ids[i]] = 1 - distance.cosine(p1vec/np.linalg.norm(p1vec, norm), p2vec/np.linalg.norm(p2vec, norm))
        if i % c == 0:
            print(str(i) + ' predictions received')
    return pred_dict

def main():
    parser = argparse.ArgumentParser(description='Use pre-trained models to predict on para similarity data')
    parser.add_argument('-p', '--parapair_file', help='Path to parapair file in BERT seq pair format')
    parser.add_argument('-b', '--batch_size', help='Size of each para embedding shards if there\n'
                                                   'are multiple shards or -1 if there is a single embedding file')
    parser.add_argument('-i', '--paraids_emb', help='Path to paraids file corresponding to the para embeddings sentwise')
    parser.add_argument('-e', '--emb_dir', help='Path to the para embedding dir sentwise')
    parser.add_argument('-x', '--emb_file_prefix', help='Common part of the file name of each embedding shards')
    parser.add_argument('-nm', '--normalization', default=-1, help='Normalization for embedding vecs (-1 for no norm)')
    parser.add_argument('-o', '--outfile', help='Path to parapair score output directory')
    args = vars(parser.parse_args())
    pp_file = args['parapair_file']
    batch = int(args['batch_size'])
    paraids_file = args['paraids_emb']
    emb_dir = args['emb_dir']
    emb_prefix = args['emb_file_prefix']
    norm = int(args['normalization'])
    outfile = args['outfile']
    parapairids, splitter = get_pair_ids(pp_file)
    # if model_path == '' and model_type == '':
    paraids = list(np.load(paraids_file))
    paraids_dict = {}
    for p in paraids:
        paraids_dict[p.split('\t')[0]] = (p.split('\t')[1], p.split('\t')[2], p.split('\t')[3])
    pred_dict = get_embed_similarity_scores(parapairids, paraids_dict, splitter, emb_dir, emb_prefix, batch, norm)
    print("Writing parapair score file")
    with open(outfile, 'w') as out:
        json.dump(pred_dict, out)
    # else:
    #     with open(proc_text, 'r') as proc:
    #         tokenized = json.load(proc)
    #     pred_dict = get_similarity_scores(tokenized, parapairids, model_type, model_path, batch)
    #     model_name = model_path.split('/')[len(model_path.split('/')) - 1]
    #     print("Writing parapair score file")
    #     with open(outfile, 'w') as out:
    #         json.dump(pred_dict, out)

if __name__ == '__main__':
    main()
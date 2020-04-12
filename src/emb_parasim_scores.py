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

def get_para_embed_vec(pid, paraids, embed_dir, embed_file_prefix, batch_size):
    pindex = paraids.index(pid)
    part = pindex // batch_size + 1
    part_offset = pindex % batch_size
    embed_arr = np.load(embed_dir + '/' + embed_file_prefix + '-part' + str(part) + '.npy')
    emb_vec = embed_arr[part_offset]
    return emb_vec

def get_embed_similarity_scores(pair_ids, paraids, splitter, embed_vec_file, norm=-1):
    pred_dict = dict()
    c = 10000
    emb_list = np.load(embed_vec_file)
    for i in range(len(pair_ids)):
        p1 = pair_ids[i].split(splitter)[0]
        p2 = pair_ids[i].split(splitter)[1]
        p1vec = emb_list[paraids.index(p1)]
        p2vec = emb_list[paraids.index(p2)]

        if norm == -1:
            pred_dict[pair_ids[i]] = 1 - distance.cosine(p1vec, p2vec)
        else:
            pred_dict[pair_ids[i]] = 1 - distance.cosine(p1vec/np.linalg.norm(p1vec, norm), p2vec/np.linalg.norm(p2vec, norm))
        if i % c == 0:
            print(str(i) + ' predictions received')
    return pred_dict

def main():
    parser = argparse.ArgumentParser(description='Use pre-trained models to predict on para similarity data')
    parser.add_argument('-pp', '--parapair_file', help='Path to parapair file in BERT seq pair format')
    parser.add_argument('-ei', '--paraids_emb', help='Path to paraids file corresponding to the para embeddings')
    parser.add_argument('-ev', '--emb_file', help='Path to emb vec file')
    parser.add_argument('-nm', '--normalization', default=-1, help='Normalization for embedding vecs (-1 for no norm)')
    parser.add_argument('-so', '--outfile', help='Path to parapair score output directory')
    args = vars(parser.parse_args())
    pp_file = args['parapair_file']
    paraids_file = args['paraids_emb']
    emb_vec_file = args['emb_file']
    norm = int(args['normalization'])
    outfile = args['outfile']
    parapairids, splitter = get_pair_ids(pp_file)
    # if model_path == '' and model_type == '':
    paraids = list(np.load(paraids_file))
    pred_dict = get_embed_similarity_scores(parapairids, paraids, splitter, emb_vec_file, norm)
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
import numpy as np
import random
import json
from sentence_transformers import SentenceTransformer
from src.sentbert_embed import SentbertParaEmbedding
import torch
from sklearn.preprocessing import minmax_scale


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
    print('Output data has ' + str(len(data)) + ' samples with ' + str(len(posdata)) + ' +ve and ' + str(len(negdata))
          + ' -ve samples')
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
    X_train = minmax_scale(X_train)
    return (torch.tensor(X_train).float(), torch.tensor(y_train))
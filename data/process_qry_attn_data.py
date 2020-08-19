import numpy as np
import random
import json
from sentence_transformers import SentenceTransformer
from src.sentbert_embed import SentbertParaEmbedding
import torch
import sys
import os
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

def get_data(emb_model, emb_file, emb_paraids_file, query_attn_data_file):
    model = SentenceTransformer(emb_model)
    paraids = list(np.load(emb_paraids_file))
    X= []
    y= []

    para_emb = np.load(emb_file)
    para_emb_dict = dict()
    for i in range(len(paraids)):
        para_emb_dict[paraids[i]] = para_emb[i]

    count = 0
    for line in open(query_attn_data_file).readlines(): count += 1
    print('Reading ' + str(count) + ' samples in data file')

    queries = []
    p1_list = []
    p2_list = []
    targets = []

    with open(query_attn_data_file, 'r') as qd:
        for l in qd:
            queries.append(l.split('\t')[1])
            p1_list.append(l.split('\t')[2])
            p2_list.append(l.split('\t')[3].rstrip())
            targets.append(float(l.split('\t')[0]))
    print('Using ' + emb_file + ' to embed query, should be same as the embedding file')
    query_attn_filename = query_attn_data_file.split('/')[len(query_attn_data_file.split('/'))-1]
    if os.path.isfile('./cache/embedded_cached_'+query_attn_filename):
        qemb_list = np.load('./cache/embedded_cached_'+query_attn_filename)
    else:
        qemb_list = model.encode(queries, show_progress_bar=True)
        np.save('./cache/embedded_cached_'+query_attn_filename, qemb_list)
    print('Queries embedded, now formatting the data into tensors')
    c = 0
    for i in range(len(queries)):
        qemb = qemb_list[i]
        p1emb = para_emb_dict[p1_list[i]]
        p2emb = para_emb_dict[p2_list[i]]

        if p1emb is None or p2emb is None:
            continue

        X.append(np.hstack((qemb, p1emb, p2emb)))
        y.append(targets[i])
        c += 1
        if c % 100 == 0:
            sys.stdout.write('\r' + str(c) + ' samples read')
    return (torch.tensor(X), torch.tensor(y))

def get_data_mu_etal(emb_model, emb_file, emb_paraids_file, query_attn_data_file, pca_components_file, mudim_red):
    U1 = np.load(pca_components_file)
    model = SentenceTransformer(emb_model)
    paraids = list(np.load(emb_paraids_file))
    X= []
    y= []

    para_emb = np.load(emb_file)
    para_emb_dict = dict()
    for i in range(len(paraids)):
        para_emb_dict[paraids[i]] = para_emb[i]

    count = 0
    for line in open(query_attn_data_file).readlines(): count += 1
    print('Reading ' + str(count) + ' samples in data file')

    queries = []
    p1_list = []
    p2_list = []
    targets = []

    with open(query_attn_data_file, 'r') as qd:
        for l in qd:
            queries.append(l.split('\t')[1])
            p1_list.append(l.split('\t')[2])
            p2_list.append(l.split('\t')[3].rstrip())
            targets.append(float(l.split('\t')[0]))
    print('Using ' + emb_file + ' to embed query, should be same as the embedding file')
    qemb_list = model.encode(queries, show_progress_bar=True)
    print('Queries embedded, now formatting the data into tensors')
    c = 0
    for i in range(len(queries)):
        qemb = qemb_list[i]
        p1emb = para_emb_dict[p1_list[i]]
        p2emb = para_emb_dict[p2_list[i]]

        if p1emb is None or p2emb is None:
            continue
        qemb = mu_etal_transform(qemb, U1, mudim_red)
        p1emb = mu_etal_transform(p1emb, U1, mudim_red)
        p2emb = mu_etal_transform(p2emb, U1, mudim_red)

        X.append(np.hstack((qemb, p1emb, p2emb)))
        y.append(targets[i])
        c += 1
        if c % 100 == 0:
            sys.stdout.write('\r' + str(c) + ' samples read')
    return (torch.tensor(X), torch.tensor(y))

def mu_etal_transform(x, U1, mudim_red):
    for u in U1[0:mudim_red]:
        x = x - np.dot(u.transpose(), x) * u
    return x
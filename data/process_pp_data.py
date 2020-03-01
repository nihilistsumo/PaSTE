import torch
from src.sentbert_embed import SentbertParaEmbedding
import numpy as np

def get_data_unsup(emb_dir, emb_file_prefix, emb_paraids_file, parapair_data, emb_mode, batch_size=10000):
    paraids = list(np.load(emb_paraids_file))
    X = []
    y = []
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
    for page in parapair_data.keys():
        parapairs = parapair_data[page]['parapairs']
        labels = parapair_data[page]['labels']
        for i in range(len(parapairs)):
            p1 = parapairs[i].split('_')[0]
            p2 = parapairs[i].split('_')[1]
            if emb_mode == 's':
                p1emb = para_emb_dict[p1]
                if p1emb is None:
                    print(p1 + ' not in emb dict')
                p2emb = para_emb_dict[p2]
                if p2emb is None:
                    print(p2 + ' not in emb dict')
            elif emb_mode == 'm':
                p1emb = emb.get_single_embedding(p1)
                p2emb = emb.get_single_embedding(p2)
            if p1emb is None or p2emb is None:
                continue
            X.append(np.hstack((p1emb, p2emb)))
            y.append(float(labels[i]))
    return (torch.tensor(X), torch.tensor(y))
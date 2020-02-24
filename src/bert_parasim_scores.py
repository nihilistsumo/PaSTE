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
    with open(pairtext_file, 'r') as tst:
        i = 0
        fl = True
        for l in tst:
            if fl:
                fl = False
                continue
            id1 = l.split('\t')[1]
            id2 = l.split('\t')[2]
            pair_ids.append(id1 + '_' + id2)
    return pair_ids

def get_similarity_scores(processed_text, pair_ids, model_type, model_path, batch_size=100):
    if model_type == 'bert':
        model = BertForSequenceClassification.from_pretrained(model_path)
    elif model_type == 'xlnet':
        model = XLNetForSequenceClassification.from_pretrained(model_path)
    elif model_type == 'roberta':
        model = RobertaForSequenceClassification.from_pretrained(model_path)
    elif model_type == 'albert':
        model = AlbertForSequenceClassification.from_pretrained(model_path)
    elif model_type == 'xlmroberta':
        model = XLMRobertaForSequenceClassification.from_pretrained(model_path)
    elif model_type == 'flaubert':
        model = FlaubertForSequenceClassification.from_pretrained(model_path)
    elif model_type == 'xlm':
        model = XLMForSequenceClassification.from_pretrained(model_path)
    else:
        print("Fine tuned model in model path and model type does not match")
        exit(0)

    model.eval()
    print("Going to calculate predictions with batch size: " + str(batch_size))
    cuda_avail = False
    if torch.cuda.device_count() > 0:
        cuda_avail = True
        print("Cuda enabled GPU available, using " + torch.cuda.get_device_name(0))
        model.to('cuda')
    sm = torch.nn.Softmax(dim=1)
    predictions = []
    batch_count = len(processed_text) // batch_size
    for b in range(batch_count):
        batch_text = processed_text[b*batch_size:(b+1)*batch_size]
        tokens_tensor = torch.tensor([t['input_ids'] for t in batch_text])
        type_tensor = torch.tensor([t['token_type_ids'] for t in batch_text])
        attn_tensor = torch.tensor([t['attention_mask'] for t in batch_text])
        if cuda_avail:
            tokens_tensor = tokens_tensor.to('cuda')
            type_tensor = type_tensor.to('cuda')
            attn_tensor = attn_tensor.to('cuda')
        print("Batch " + str(b+1) + " Tensors formed", end='')
        with torch.no_grad():
            outputs = model(tokens_tensor, attention_mask=attn_tensor, token_type_ids=type_tensor)
        print(" ..Predictions received")

        # sm(outputs) is an array of 2 valued array [-ve, +ve] in tensor form
        # we save only the second element which is the chance prediciton of the instance to have a positive label

        predictions += [p[1].item() for p in sm(outputs[0])]
    batch_text = processed_text[batch_count * batch_size:]
    tokens_tensor = torch.tensor([t['input_ids'] for t in batch_text])
    type_tensor = torch.tensor([t['token_type_ids'] for t in batch_text])
    attn_tensor = torch.tensor([t['attention_mask'] for t in batch_text])
    if cuda_avail:
        tokens_tensor = tokens_tensor.to('cuda')
        type_tensor = type_tensor.to('cuda')
        attn_tensor = attn_tensor.to('cuda')
    print("Last batch Tensors formed", end='')
    with torch.no_grad():
        outputs = model(tokens_tensor, attention_mask=attn_tensor, token_type_ids=type_tensor)
    print(" ..Predictions received")
    predictions += [p[1].item() for p in sm(outputs[0])]
    assert len(predictions) == len(pair_ids)
    pred_dict = dict()
    for i in range(len(pair_ids)):
        pred_dict[pair_ids[i]] = predictions[i]
    return pred_dict

def get_para_embed_vec(pid, paraids, embed_dir, embed_file_prefix, batch_size):
    pindex = paraids.index(pid)
    part = pindex // batch_size + 1
    part_offset = pindex % batch_size
    embed_arr = np.load(embed_dir + '/' + embed_file_prefix + '-part' + str(part) + '.npy')
    emb_vec = embed_arr[part_offset]
    return emb_vec

def get_embed_similarity_scores(pair_ids, paraids, embed_dir, embed_file_prefix, batch_size, norm=-1):
    pred_dict = dict()
    c = 10000
    if batch_size == -1:
        emb_list = np.load(embed_dir + '/' + embed_file_prefix + '-part1.npy')
    for i in range(len(pair_ids)):
        p1 = pair_ids[i].split('_')[0]
        p2 = pair_ids[i].split('_')[1]
        if batch_size == -1:
            p1vec = emb_list[paraids.index(p1)]
            p2vec = emb_list[paraids.index(p2)]
        else:
            p1vec = get_para_embed_vec(p1, paraids, embed_dir, embed_file_prefix, batch_size)
            p2vec = get_para_embed_vec(p2, paraids, embed_dir, embed_file_prefix, batch_size)
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
    parser.add_argument('-t', '--processed_textfile', help='Path to processed pairtext file')
    parser.add_argument('-n', '--model_type', default='', help='Type of model (bert/roberta/albert/xlnet/xlmroberta/flaubert)')
    parser.add_argument('-m', '--model_path', default='', help='Path to pre-trained/fine tuned model')
    parser.add_argument('-b', '--batch_size', help='In case of transformer models: Batch size of the tensors submitted to GPU\n'
                                                   'In case of embedding models: Size of each para embedding shards if there\n'
                                                   'are multiple shards or -1 if there is a single embedding file')
    parser.add_argument('-i', '--paraids_emb', help='Path to paraids file corresponding to the para embeddings')
    parser.add_argument('-e', '--emb_dir', help='Path to the para embedding dir')
    parser.add_argument('-x', '--emb_file_prefix', help='Common part of the file name of each embedding shards')
    parser.add_argument('-nm', '--normalization', default=-1, help='Normalization for embedding vecs (-1 for no norm)')
    parser.add_argument('-o', '--outfile', help='Path to parapair score output directory')
    args = vars(parser.parse_args())
    pp_file = args['parapair_file']
    proc_text = args['processed_textfile']
    model_type = args['model_type']
    model_path = args['model_path']
    batch = int(args['batch_size'])
    paraids_file = args['paraids_emb']
    emb_dir = args['emb_dir']
    emb_prefix = args['emb_file_prefix']
    norm = int(args['normalization'])
    outfile = args['outfile']
    parapairids = get_pair_ids(pp_file)
    if model_path == '' and model_type == '':
        paraids = list(np.load(paraids_file))
        pred_dict = get_embed_similarity_scores(parapairids, paraids, emb_dir, emb_prefix, batch, norm)
        print("Writing parapair score file")
        with open(outfile, 'w') as out:
            json.dump(pred_dict, out)
    else:
        with open(proc_text, 'r') as proc:
            tokenized = json.load(proc)
        pred_dict = get_similarity_scores(tokenized, parapairids, model_type, model_path, batch)
        model_name = model_path.split('/')[len(model_path.split('/')) - 1]
        print("Writing parapair score file")
        with open(outfile, 'w') as out:
            json.dump(pred_dict, out)

if __name__ == '__main__':
    main()
import transformers
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
import json
import torch
import argparse

def get_similarity_scores(tokenized_pairs, pair_ids, model_path, batch_size=100):
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.eval()
    print("Going to calculate predictions with batch size: " + str(batch_size))
    cuda_avail = False
    if torch.cuda.device_count() > 0:
        cuda_avail = True
        print("Cuda enabled GPU available, using " + torch.cuda.get_device_name(0))
        model.to('cuda')
    sm = torch.nn.Softmax(dim=1)
    predictions = []
    batch_count = len(tokenized_pairs['input_ids']) // batch_size
    for b in range(batch_count):
        #batch_text = processed_text[b*batch_size:(b+1)*batch_size]
        tokens_tensor = torch.tensor(tokenized_pairs['input_ids'][b*batch_size : (b+1)*batch_size])
        type_tensor = torch.tensor(tokenized_pairs['token_type_ids'][b*batch_size : (b+1)*batch_size])
        attn_tensor = torch.tensor(tokenized_pairs['attention_mask'][b*batch_size : (b+1)*batch_size])
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
    #batch_text = processed_text[(len(processed_text)//batch_size) * batch_size:]
    tokens_tensor = torch.tensor(tokenized_pairs['input_ids'][b * batch_size: (b + 1) * batch_size])
    type_tensor = torch.tensor(tokenized_pairs['token_type_ids'][b * batch_size: (b + 1) * batch_size])
    attn_tensor = torch.tensor(tokenized_pairs['attention_mask'][b * batch_size: (b + 1) * batch_size])
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

def main():
    parser = argparse.ArgumentParser(description='Use pre-trained models to predict on para similarity data')
    parser.add_argument('-t', '--tokenized_dirpath', help='Path to tokenized pairtext dir')
    parser.add_argument('-m', '--model_path', help='Path to pre-trained/fine tuned model')
    parser.add_argument('-o', '--outdir', help='Path to parapair score output directory')
    args = vars(parser.parse_args())
    tok_dir = args['tokenized_dirpath']
    model_path = args['model_path']
    outdir = args['outdir']
    with open(tok_dir+'/paraid.json', 'r') as idin:
        with open(tok_dir+'/processed_text.json', 'r') as textin:
            paraids = json.load(idin)
            tokenized = json.load(textin)
    pred_dict = get_similarity_scores(tokenized, paraids, model_path)
    model_name = model_path.split('/')[len(model_path.split('/')) - 1]
    print("Writing parapair score file")
    with open(outdir + '/' + model_name, 'w') as out:
        json.dump(pred_dict, out)

if __name__ == '__main__':
    main()
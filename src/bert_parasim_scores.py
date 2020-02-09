import transformers
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
import numpy as np
import json
import torch
import argparse

def get_similarity_scores(test_file, tokenizer, maxlen, model_path, batch_size=100):
    processed_text = []
    pair_ids = []
    with open(test_file, 'r') as tst:
        i = 0
        fl = True
        for l in tst:
            if fl:
                fl = False
                continue
            id1 = l.split('\t')[1]
            id2 = l.split('\t')[2]
            text1 = l.split('\t')[3]
            text2 = l.split('\t')[4]
            pair_ids.append(id1 + '_' + id2)
            processed_text.append(tokenizer.encode_plus(text1, text2, add_special_tokens=True, max_length=maxlen,
                                                        pad_to_max_length=True))
            i += 1
            if i % 100 == 0:
                print(str(i)+" lines processed")

    print("Text processing done")
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
    for b in range(len(processed_text)//batch_size):
        batch_text = processed_text[b*batch_size:(b+1)*batch_size]
        tokens_tensor = torch.tensor([t['input_ids'] for t in batch_text])
        type_tensor = torch.tensor([t['token_type_ids'] for t in batch_text])
        attn_tensor = torch.tensor([t['attention_mask'] for t in batch_text])
        if cuda_avail:
            tokens_tensor = tokens_tensor.to('cuda')
            type_tensor = type_tensor.to('cuda')
            attn_tensor = attn_tensor.to('cuda')
        print("Batch " + str(b+1) + "Tensors formed")
        with torch.no_grad():
            outputs = model(tokens_tensor, attention_mask=attn_tensor, token_type_ids=type_tensor)
        print("Predictions received")

        # sm(outputs) is an array of 2 valued array [-ve, +ve] in tensor form
        # we save only the second element which is the chance prediciton of the instance to have a positive label

        predictions += [p[1].item() for p in sm(outputs[0])]
    batch_text = processed_text[(len(processed_text)//batch_size) * batch_size:]
    tokens_tensor = torch.tensor([t['input_ids'] for t in batch_text])
    type_tensor = torch.tensor([t['token_type_ids'] for t in batch_text])
    attn_tensor = torch.tensor([t['attention_mask'] for t in batch_text])
    if cuda_avail:
        tokens_tensor = tokens_tensor.to('cuda')
        type_tensor = type_tensor.to('cuda')
        attn_tensor = attn_tensor.to('cuda')
    print("Last batch Tensors formed")
    with torch.no_grad():
        outputs = model(tokens_tensor, attention_mask=attn_tensor, token_type_ids=type_tensor)
    print("Predictions received")
    predictions += [p[1].item() for p in sm(outputs[0])]
    assert len(predictions) == len(pair_ids)
    pred_dict = dict()
    for i in range(len(pair_ids)):
        pred_dict[pair_ids[i]] = predictions[i]
    return pred_dict

def main():
    parser = argparse.ArgumentParser(description='Use pre-trained models to predict on para similarity data')
    parser.add_argument('-t', '--test_filepath', help='Path to parapair file in BERT seq pair format')
    parser.add_argument('-m', '--model_path', help='Path to pre-trained/fine tuned model')
    parser.add_argument('-l', '--seq_maxlen', help='Maximum seq length, same as the model used')
    parser.add_argument('-o', '--outfile', help='Path to parapair score output file')
    args = vars(parser.parse_args())
    test_file = args['test_filepath']
    model_path = args['model_path']
    maxlen = int(args['seq_maxlen'])
    outfile = args['outfile']
    tokenizer = BertTokenizer.from_pretrained(model_path)
    pred_dict = get_similarity_scores(test_file, tokenizer, maxlen, model_path)
    print("Writing parapair score file")
    with open(outfile, 'w') as out:
        json.dump(pred_dict, out)

if __name__ == '__main__':
    main()
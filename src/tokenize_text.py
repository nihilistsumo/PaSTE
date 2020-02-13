from transformers import AutoTokenizer
import json
import argparse

def tokenize_text(test_file, model_file, out_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_file)
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
            if i % 1000 == 0:
                print(str(i) + " lines processed")
    print("Text processing done")
    print("Saving at " + out_dir)
    with open(out_dir+'/paraid.json', 'w') as outid:
        with open(out_dir+'/processed_text.json', 'w') as outtext:
            json.dump(pair_ids, outid)
            json.dump(processed_text, outtext)

def main():
    parser = argparse.ArgumentParser(description='Tokenize paratext')
    parser.add_argument('-t', '--test_filepath', help='Path to parapair file in BERT seq pair format')
    parser.add_argument('-m', '--model_path', help='Path to pre-trained/fine tuned model')
    parser.add_argument('-o', '--out_dir', help='Path to output dir to save pair ids and tokenized text')
    args = vars(parser.parse_args())
    test_file = args['test_filepath']
    model_path = args['model_path']
    outdir = args['out_dir']
    tokenize_text(test_file, model_path, outdir)

if __name__ == '__main__':
    main()
from sentence_transformers import SentenceTransformer
import argparse
import numpy as np
import spacy

# get embeddings of the whole para
def get_embeddings(paratext_file, model_name, outdir, saveid=False, batch_size=10000):
    model = SentenceTransformer(model_name)
    print("Using "+model_name+" to embed paras as a whole")
    with open(paratext_file, 'r') as pt:
        c = 0
        part = 0
        texts = []
        ids = []
        for l in pt:
            ids.append(l.split('\t')[0])
            texts.append(l.split('\t')[1])
            c += 1
            if c % batch_size == 0:
                print("Going to embed")
                embeds = model.encode(texts)
                part += 1
                print("Embedding complete, going to save part "+str(part)+", "+str(c)+" paras embedded\n")
                np.save(outdir + '/' + model_name + '-part' + str(part), embeds)
                texts = []
        print("Going to embed")
        embeds = model.encode(texts)
        part += 1
        print("Embedding complete, going to save part " + str(part) + ", " + str(c) + " paras embedded\n")
        np.save(outdir + '/' + model_name + '-part' + str(part), embeds)
        if saveid:
            print("Saving paraids file")
            ids = np.array(ids)
            np.save(outdir + '/paraids', ids)

def get_sentence_wise_embeddings(paratext_file, model_name, outdir, saveid=False, batch_size=10000):
    spacy.prefer_gpu()
    nlp = spacy.load('en_core_web_sm')
    model = SentenceTransformer(model_name)
    print("Using " + model_name + " to embed paras sentence wise")
    with open(paratext_file, 'r') as pt:
        c = 0
        part = 0
        texts = []
        ids = []
        for l in pt:
            paraid = l.split('\t')[0]
            text = l.split('\t')[1]
            text_sents = [str(s) for s in nlp(text).sents]
            for i in range(len(text_sents)):
                ids.append(paraid+'_'+str(i+1))
            texts += text_sents
            c += 1
            if c % batch_size == 0:
                print("Going to embed")
                embeds = model.encode(texts)
                part += 1
                print("Embedding complete, going to save part " + str(part) + ", " + str(c) + " paras embedded\n")
                np.save(outdir + '/' + model_name + '-part' + str(part), embeds)
                texts = []
        print("Going to embed")
        embeds = model.encode(texts)
        part += 1
        print("Embedding complete, going to save part " + str(part) + ", " + str(c) + " paras embedded\n")
        np.save(outdir + '/' + model_name + '-part' + str(part), embeds)
        if saveid:
            print("Saving paraids file")
            ids = np.array(ids)
            np.save(outdir + '/paraids_sents', ids)

def main():
    parser = argparse.ArgumentParser(description='Use sentence-transformers to embed paragraphs')
    parser.add_argument('-p', '--paratext_file', help='Path to paratext file')
    parser.add_argument('-t', '--emb_type', help='s = sentence wise emb, p = whole para emb')
    parser.add_argument('-m', '--model_name', help='Name of the model to be used for embedding')
    parser.add_argument('-o', '--outdir', help='Path to output dir')
    parser.add_argument('-i', '--save_id', type=bool, default=False, help='Save id file?')
    parser.add_argument('-b', '--batch_size', default=10000, help='Size of each batch to be encoded')
    args = vars(parser.parse_args())
    paratext_file = args['paratext_file']
    emb_type = args['emb_type']
    model_name = args['model_name']
    outdir = args['outdir']
    saveid = args['save_id']
    batch = int(args['batch_size'])
    if emb_type == 'p':
        get_embeddings(paratext_file, model_name, outdir, saveid, batch)
    elif emb_type == 's':
        get_sentence_wise_embeddings(paratext_file, model_name, outdir, saveid, batch)
    else:
        print("Wrong embedding type")

if __name__ == '__main__':
    main()
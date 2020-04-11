import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess
import smart_open
import argparse
import numpy as np

def read_paratext(fname, tokens_only=False):
    tokens = []
    ids = []
    with smart_open.open(fname, encoding="iso-8859-1") as f:
        i = 1
        for line in f:
            if tokens_only:
                tokens.append(simple_preprocess(line.split('\t')[1]))
            else:
                # For training data, add tags
                tokens.append(TaggedDocument(simple_preprocess(line.split('\t')[1]), [i]))
            ids.append(line.split('\t')[0])
            i += 1
            if i%10000 == 0:
                print(str(i)+' docs read')
    return tokens, ids

def train(paratext_file, vec_size, ep):
    train_corpus, paraids = read_paratext(paratext_file)
    model = Doc2Vec(vector_size=vec_size, min_count=2, epochs=ep)
    model.build_vocab(train_corpus)
    print("Text processed, now going to train...")
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
    print("Training complete")
    return model

def infer(paratext_file, model):
    vecs = []
    test_corpus, paraids = read_paratext(paratext_file, True)
    for i in range(len(paraids)):
        vecs.append(model.infer_vector(test_corpus[i]))
        i += 1
        if i % 10000 == 0:
            print(str(i) + ' docs inferred')
    return paraids, vecs

def main():
    parser = argparse.ArgumentParser(description='Paragraph vector')
    parser.add_argument('-pr', '--train_paratext', help='Path to train paratext file')
    parser.add_argument('-pt', '--test_paratext', help='Path to test paratext file')
    parser.add_argument('-vs', '--vec_size', type=int, help='Size of embedding vector')
    parser.add_argument('-ep', '--num_epochs', type=int, help='Num of epochs to train')
    parser.add_argument('-od', '--outdir', help='Path t output dir')
    args = vars(parser.parse_args())
    train_pt = args['train_paratext']
    test_pt = args['test_paratext']
    vec_size = args['vec_size']
    epochs = args['num_epochs']
    outdir = args['outdir']
    m = train(train_pt, vec_size, epochs)
    paraids, vecs = infer(test_pt, m)
    np.save(outdir+'/paraids.npy', paraids)
    np.save(outdir+'/doc2vec_embedding_vecs.npy', vecs)
    print("Done")

if __name__ == '__main__':
    main()
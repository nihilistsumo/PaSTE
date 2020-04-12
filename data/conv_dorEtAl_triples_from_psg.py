import csv
import nltk
from nltk import tokenize
import random
import argparse

def convert(triples_psg_csv, output_sent_csv):
    csvreader_psg = csv.reader(open(triples_psg_csv, 'r'))
    csvwriter_sent = csv.writer(open(output_sent_csv, 'w'))
    csvwriter_sent.writerow(['Article Title', 'Passage1', 'Passage2', 'Passage3', 'Article Link'])
    f = True
    i = 0
    for l in csvreader_psg:
        if f:
            f = False
            continue
        sents1 = tokenize.sent_tokenize(l[1])
        sents2 = tokenize.sent_tokenize(l[2])
        sents3 = tokenize.sent_tokenize(l[3])
        if len(sents1) > 0:
            sents1 = random.sample(sents1, 1)[0]
        else:
            sents1 = l[1]
        if len(sents2) > 0:
            sents2 = random.sample(sents2, 1)[0]
        else:
            sents2 = l[2]
        if len(sents3) > 0:
            sents3 = random.sample(sents3, 1)[0]
        else:
            sents3 = l[3]
        csvwriter_sent.writerow([l[0], sents1, sents2, sents3, l[4]])
        i += 1
        if i%10000 == 0:
            print(str(i)+' lines written')

def main():
    parser = argparse.ArgumentParser(description='Converts psg triples to sent triples for sentbert wiki sec method')
    parser.add_argument('-in', '--input_psg_triple', help='Path to input passage triples')
    parser.add_argument('-out', '--out_sent_triple', help='Path to output sent triple')
    args = vars(parser.parse_args())
    input_csv = args['input_psg_triple']
    out_csv = args['out_sent_triple']
    convert(input_csv, out_csv)

if __name__ == '__main__':
    main()
import json
import argparse

def create_book_parapairs(qrels):
    book_parapairs = {}
    rev_qrels = {}
    book_paras = {}
    with open(qrels, 'r') as qr:
        for l in qr:
            book = l.split(' ')[0].split('/')[0]
            para = l.split(' ')[2]
            rev_qrels[para] = l.split(' ')[0]
            if book not in book_paras.keys():
                book_paras[book] = [para]
            else:
                book_paras[book].append(para)
    for b in book_paras.keys():
        book_parapairs[b] = {'parapairs': [], 'labels': []}
        paras = book_paras[b]
        for i in range(len(paras) - 1):
            for j in range(i+1, len(paras)):
                p1 = paras[i]
                p2 = paras[j]
                book_parapairs[b]['parapairs'].append(p1+'#'+p2)
                if rev_qrels[p1] == rev_qrels[p2]:
                    book_parapairs[b]['labels'].append(1)
                else:
                    book_parapairs[b]['labels'].append(0)
    return book_parapairs

def create_bert_seq(parapairs, paratext_file):
    paratext = {}
    with open(paratext_file, 'r') as pt:
        for l in pt:
            paratext[l.split('\t')[0]] = l.split('\t')[1]
    lines = ['Quality\t#1 ID\t#2 ID\t#1 Text\t#2 Text']
    for b in parapairs.keys():
        pairs = parapairs[b]['parapairs']
        labels = parapairs[b]['labels']
        for i in range(len(pairs)):
            p1 = pairs[i].split('#')[0]
            p2 = pairs[i].split('#')[1]
            lines.append(str(labels[i])+'\t'+p1+'\t'+p2+'\t'+paratext[p1].strip()+'\t'+paratext[p2].strip())
    return lines

def main():
    parser = argparse.ArgumentParser(description='Create book parapair data from 4sub qrels')
    parser.add_argument('-qr', '--qrels', help='Path to 4sub qrels file')
    parser.add_argument('-pt', '--paratext', help='Path to paratext file')
    parser.add_argument('-po', '--parapair_outfile', help='Path to parapair output file')
    parser.add_argument('-bo', '--bertseq_outfile', help='Path to bert seq output file')
    args = vars(parser.parse_args())
    qrels = args['qrels']
    paratext_file = args['paratext']
    outfile = args['parapair_outfile']
    bert_outfile = args['bertseq_outfile']
    book_parapairs = create_book_parapairs(qrels)
    with open(outfile, 'w') as out:
        json.dump(book_parapairs, out)
    bertseq_lines = create_bert_seq(book_parapairs, paratext_file)
    with open(bert_outfile, 'w') as bout:
        for l in bertseq_lines:
            bout.write(l+'\n')

if __name__ == '__main__':
    main()
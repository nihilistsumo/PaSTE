import random, json, sys, argparse, csv, math

def generate_triples(page_para_dict, pagewise_hier_qrels):
    triples_data = []
    c = 0

    page_num = len(page_para_dict.keys())
    print('Triples to be generated from ' + str(page_num) + ' pages')
    for page in page_para_dict.keys():
        paras_in_page = page_para_dict[page]
        if page in pagewise_hier_qrels.keys():
            hier_qrels_for_page = pagewise_hier_qrels[page]
            for hier in hier_qrels_for_page.keys():
                simparas = hier_qrels_for_page[hier]
                if len(simparas) > 1 and len(paras_in_page) > len(simparas):
                    for i in range(len(simparas)-1):
                        for j in range(i+1, len(simparas)):
                            p1 = simparas[i]
                            p2 = simparas[j]
                            neg_paras = [p for p in paras_in_page if p not in simparas]
                            p3 = random.sample(neg_paras, 1)[0]
                            triples = [page, p1, p2, p3]
                            triples_data.append(triples)
        c += 1
        if c % 1000 == 0:
            print(str(c)+' pages done')
    return triples_data

def get_reversed_top_qrels(top_qrels_in_page):
    top_qrels_reverse = dict()
    for top in top_qrels_in_page.keys():
        for p in top_qrels_in_page[top]:
            top_qrels_reverse[p] = top
    return top_qrels_reverse

def get_pagewise_topic_qrels(qrels_file):
    pagewise_hq = dict()
    with open(qrels_file, 'r') as hq:
        for l in hq:
            hier_sec = l.split(' ')[0]
            page = hier_sec.split('/')[0]
            para = l.split(' ')[2]
            if page not in pagewise_hq.keys():
                pagewise_hq[page] = {hier_sec:[para]}
            else:
                if hier_sec in pagewise_hq[page].keys():
                    pagewise_hq[page][hier_sec].append(para)
                else:
                    pagewise_hq[page][hier_sec] = [para]
    return pagewise_hq

def main():
    parser = argparse.ArgumentParser(description='Generate pagewise discriminative triples')
    parser.add_argument('-aq', '--art_qrels', help='Path to article level qrels')
    parser.add_argument('-hq', '--hier_qrels', help='Path to hierarchical level qrels')
    parser.add_argument('-pt', '--paratext_file', help='Path to paratext file')
    parser.add_argument('-o', '--out', help='Path to output dir')
    parser.add_argument('-oo', '--out_onlyid', help='Path to output file with only ids')
    args = vars(parser.parse_args())
    art_qrels_file = args['art_qrels']
    hier_qrels_file = args['hier_qrels']
    paratext_file = args['paratext_file']
    outdir = args['out']
    onlyid_out = args['out_onlyid']

    paratext_dict = {}
    with open(paratext_file, 'r') as pt:
        for l in pt:
            paratext_dict[l.split('\t')[0]] = l.split('\t')[1]
    page_paras = {}
    with open(art_qrels_file, 'r') as aq:
        for l in aq:
            page = l.split(' ')[0]
            para = l.split(' ')[2]
            if page in page_paras.keys():
                page_paras[page].append(para)
            else:
                page_paras[page] = [para]

    hier_qrels = get_pagewise_topic_qrels(hier_qrels_file)
    triples_data = generate_triples(page_paras, hier_qrels)
    random.shuffle(triples_data)
    csvwriter_train = csv.writer(open(outdir+'/train.csv', 'w'))
    csvwriter_val = csv.writer(open(outdir + '/validation.csv', 'w'))
    csvwriter_test = csv.writer(open(outdir + '/test.csv', 'w'))

    csvwriter_train.writerow(['Article Title', 'Passage1', 'Passage2', 'Passage3', 'Article Link'])
    for i in range(len(triples_data)-200):
        t = triples_data[i]
        csvwriter_train.writerow([t[0], paratext_dict[t[1]].strip(), paratext_dict[t[2]].strip(), paratext_dict[t[3]].strip(), 'no_link'])

    csvwriter_val.writerow(['Article Title', 'Passage1', 'Passage2', 'Passage3', 'Article Link'])
    for i in range(len(triples_data) - 200, len(triples_data) - 100):
        t = triples_data[i]
        csvwriter_val.writerow([t[0], paratext_dict[t[1]].strip(), paratext_dict[t[2]].strip(), paratext_dict[t[3]].strip(), 'no_link'])

    csvwriter_test.writerow(['Article Title', 'Passage1', 'Passage2', 'Passage3', 'Article Link'])
    for i in range(len(triples_data) - 100, len(triples_data)):
        t = triples_data[i]
        csvwriter_test.writerow([t[0], paratext_dict[t[1]].strip(), paratext_dict[t[2]].strip(), paratext_dict[t[3]].strip(), 'no_link'])

    with open(onlyid_out, 'w') as out2:
        json.dump(triples_data, out2)

if __name__ == '__main__':
    main()
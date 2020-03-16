import random, json, sys, argparse, csv, math
import numpy as np

def generate_triples(page_para_dict, pagewise_hier_qrels, pagewise_top_qrels):
    triples_data = dict()
    page_num = len(page_para_dict.keys())
    print('Triples to be generated from '+str(page_num)+' pages')
    c = 0
    page_para_pages = set(page_para_dict.keys())
    top_pages = set(pagewise_top_qrels.keys())
    hier_pages = set(pagewise_hier_qrels.keys())
    common_pages = page_para_pages.intersection(top_pages.intersection(hier_pages))
    for page in common_pages:
        paras_in_page = page_para_dict[page]
        top_qrels_reversed = get_reversed_top_qrels(pagewise_top_qrels[page])
        diff_sec_in_page = set([top_qrels_reversed[p] for p in paras_in_page])
        if page in pagewise_hier_qrels.keys() and len(diff_sec_in_page) > 1:
            hier_qrels_for_page = pagewise_hier_qrels[page]
            triples_data_in_page = []
            for hier in hier_qrels_for_page.keys():
                simparas = [p for p in hier_qrels_for_page[hier] if p in paras_in_page]
                if len(simparas) > 1 and len(paras_in_page) > len(simparas):
                    for i in range(len(simparas)-1):
                        for j in range(i+1, len(simparas)):
                            p1 = simparas[i]
                            p2 = simparas[j]
                            p3 = random.sample([p for p in paras_in_page if p not in simparas], 1)[0]
                            loop = 0
                            while top_qrels_reversed[p3] == top_qrels_reversed[p1]:
                                p3 = random.sample([p for p in paras_in_page if p not in simparas], 1)[0]
                                loop += 1
                                if loop % 5000 == 0:
                                    print(loop)
                                    print(paras_in_page)
                                    print(simparas)
                                    for p in paras_in_page:
                                        print(p+': '+top_qrels_reversed[p])
                            triples = [p1, p2, p3]
                            #random.shuffle(triples)
                            #triples.append(p3)
                            triples_data_in_page.append(triples)
            if len(triples_data_in_page) > 0:
                triples_data[page] = triples_data_in_page
        c += 1
        if c % 1000 == 0:
            print(str(c)+' pages done')
    return triples_data

# This method is for both toplevel and hier
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

def get_reversed_top_qrels(top_qrels_in_page):
    top_qrels_reverse = dict()
    for top in top_qrels_in_page.keys():
        for p in top_qrels_in_page[top]:
            top_qrels_reverse[p] = top
    # with open(top_qrels_file, 'r') as tq:
    #     for l in tq:
    #         top_qrels_reverse[l.split(" ")[2]] = l.split(" ")[0]
    return top_qrels_reverse

def main():
    parser = argparse.ArgumentParser(description='Generate pagewise discriminative triples')
    parser.add_argument('-aq', '--art_qrels', help='Path to article level qrels')
    parser.add_argument('-tq', '--top_qrels', help='Path to top-level qrels')
    parser.add_argument('-hq', '--hier_qrels', help='Path to hierarchical level qrels')
    parser.add_argument('-pt', '--paratext_file', help='Path to paratext file')
    parser.add_argument('-o', '--out', help='Path to output dir')
    parser.add_argument('-oo', '--out_onlyid', help='Path to output file with only ids')
    args = vars(parser.parse_args())
    art_qrels_file = args['art_qrels']
    top_qrels_file = args['top_qrels']
    hier_qrels_file = args['hier_qrels']
    paratext_file = args['paratext_file']
    outdir = args['out']
    onlyid_out = args['out_onlyid']

    # with open(parapairs_file, 'r') as pp:
    #     parapairs = json.load(pp)
    # page_paras_from_parapairs = dict()
    # for page in parapairs.keys():
    #     pairs = parapairs[page]['parapairs']
    #     paras = set()
    #     for pair in pairs:
    #         paras.add(pair.split('_')[0])
    #         paras.add(pair.split('_')[1])
    #     page_paras_from_parapairs[page] = list(paras)

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

    top_qrels = get_pagewise_topic_qrels(top_qrels_file)
    hier_qrels = get_pagewise_topic_qrels(hier_qrels_file)
    triples_data = generate_triples(page_paras, hier_qrels, top_qrels)
    csvwriter_train = csv.writer(open(outdir+'/train.csv', 'w'))
    csvwriter_val = csv.writer(open(outdir + '/validation.csv', 'w'))
    csvwriter_test = csv.writer(open(outdir + '/test.csv', 'w'))
    total_pages_num = len(triples_data.keys())
    train_page_num = math.floor(total_pages_num * 0.9)
    val_pages_num = math.floor((total_pages_num-train_page_num)/2)
    train_pages = set(random.sample(triples_data.keys(), train_page_num))
    val_pages = set(random.sample(triples_data.keys()-train_pages, val_pages_num))
    test_pages = triples_data.keys()-(train_pages+val_pages)

    csvwriter_train.writerow(['Article Title', 'Passage1', 'Passage2', 'Passage3', 'Article Link'])
    for page in train_pages:
        for t in triples_data[page]:
            csvwriter_train.writerow([page, paratext_dict[t[0]].rstrip(), paratext_dict[t[1]].rstrip(), paratext_dict[t[2]].rstrip(), 'no_link'])

    csvwriter_val.writerow(['Article Title', 'Passage1', 'Passage2', 'Passage3', 'Article Link'])
    for page in val_pages:
        for t in triples_data[page]:
            csvwriter_val.writerow([page, paratext_dict[t[0]].rstrip(), paratext_dict[t[1]].rstrip(), paratext_dict[t[2]].rstrip(), 'no_link'])

    csvwriter_test.writerow(['Article Title', 'Passage1', 'Passage2', 'Passage3', 'Article Link'])
    for page in test_pages:
        for t in triples_data[page]:
            csvwriter_test.writerow([page, paratext_dict[t[0]].rstrip(), paratext_dict[t[1]].rstrip(), paratext_dict[t[2]].rstrip(), 'no_link'])

    with open(onlyid_out, 'w') as out2:
        json.dump(triples_data, out2)

if __name__ == '__main__':
    main()
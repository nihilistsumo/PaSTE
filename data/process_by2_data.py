import random, json

def get_page_paras(art_qrels):
    page_paras = {}
    with open(art_qrels, 'r') as aq:
        for l in aq:
            page = l.split(' ')[0]
            para = l.split(' ')[2]
            if page not in page_paras.keys():
                page_paras[page] = [para]
            else:
                page_paras[page].append(para)
    return page_paras

def get_page_sec_paras(hier_qrels):
    page_sec_paras = {}
    with open(hier_qrels, 'r') as hq:
        for l in hq:
            sec = l.split(' ')[0]
            page = sec.split('/')[0]
            para = l.split(' ')[2]
            if page not in page_sec_paras.keys():
                page_sec_paras[page] = {sec: [para]}
            elif sec not in page_sec_paras[page].keys():
                page_sec_paras[page][sec] = [para]
            else:
                page_sec_paras[page][sec].append(para)
    return page_sec_paras

def rev_hqrels(hier_qrels):
    rev_hq = {}
    with open(hier_qrels, 'r') as hq:
        for l in hq:
            rev_hq[l.split(' ')[2]] = l.split(' ')[0]
    return rev_hq

def get_paratext(paratext_file):
    paratexts = {}
    with open(paratext_file, 'r') as pt:
        for l in pt:
            paratexts[l.split('\t')[0]] = l.split('\t')[1].strip()
    return paratexts

def get_page_parapairs(page_paras, rev_hq, outfile):
    page_parapairs = {}
    for page in page_paras.keys():
        paras = page_paras[page]
        parapairs = []
        labels = []
        for i in range(len(paras)-1):
            for j in range(i+1, len(paras)):
                p1 = paras[i]
                p2 = paras[j]
                parapairs.append(p1+'_'+p2)
                if rev_hq[p1] == rev_hq[p2]:
                    labels.append(1)
                else:
                    labels.append(0)
        page_parapairs[page] = {'parapairs':parapairs, 'labels':labels}
    with open(outfile, 'w') as out:
        json.dump(page_parapairs, out)

def get_balanced_page_parapairs(page_parapairs, outfile):
    bal_page_pairs = {}
    for page in page_parapairs.keys():
        pairs = page_parapairs[page]['parapairs']
        labels = page_parapairs[page]['labels']
        pos = []
        neg = []
        for i in range(len(pairs)):
            if labels[i] == 1:
                pos.append(pairs[i])
            else:
                neg.append(pairs[i])
        neg = random.sample(neg, len(pos))
        pairs = pos + neg
        labels = [1]*len(pos) + [0]*len(neg)
        bal_page_pairs[page] = {'parapairs':pairs, 'labels':labels}
    with open(outfile, 'w') as out:
        json.dump(bal_page_pairs, out)

def write_bert_seq(page_parapairs, paratext, outfile):
    intro = 'Quality\t#1 ID\t#2 ID\t#1 Text\t#2 Text'
    lines = []
    for page in page_parapairs.keys():
        pairs = page_parapairs[page]['parapairs']
        labels = page_parapairs[page]['labels']
        for i in range(len(pairs)):
            p1 = pairs[i].split('_')[0]
            p2 = pairs[i].split('_')[1]
            lines.append(str(labels[i])+'\t'+p1+'\t'+p2+'\t'+paratext[p1]+'\t'+paratext[p2])
    random.shuffle(lines)
    with open(outfile, 'w') as out:
        out.write(intro+'\n')
        for l in lines:
            out.write(l+'\n')
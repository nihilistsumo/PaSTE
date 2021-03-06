from trec_car.read_data import *
import sys
import re
import random
import argparse
import json

def write_paratext_data(para_cbor, outfile):
    _RE_COMBINE_WHITESPACE = re.compile(r"\s+")
    c = 0
    with open(outfile, 'w') as out:
        with open(para_cbor, 'rb') as f:
            for p in iter_paragraphs(f):
                texts = [elem.text if isinstance(elem, ParaText)
                         else elem.anchor_text
                         for elem in p.bodies]
                text = ' '.join(texts)
                text = _RE_COMBINE_WHITESPACE.sub(" ", text).strip()
                out.write(p.para_id + '\t' + text + '\n')
                c += 1
                if c % 10000 == 0:
                    print(str(c) + ' paras written')

def get_score(p1score, p2score):
    MUST = 3
    SHOULD = 2
    CAN = 1
    TOPIC = 0
    NONREL = -1
    TRASH = -2
    if p1score == MUST and p2score == MUST:
        score = 1.0
    elif p1score == SHOULD and p2score == SHOULD:
        score = 0.8
    elif p1score == CAN and p2score == CAN:
        score = 0.7
    elif p1score == TOPIC and p2score == TOPIC:
        score = 0.5
    elif (p1score == MUST and p2score == SHOULD) or (p1score == SHOULD and p2score == MUST):
        score = 0.85
    elif (p1score == CAN and p2score == SHOULD) or (p1score == SHOULD and p2score == CAN):
        score = 0.75
    elif (p1score == CAN and p2score == MUST) or (p1score == MUST and p2score == CAN):
        score = 0.65
    elif (p1score == CAN and p2score == TOPIC) or (p1score == TOPIC and p2score == CAN):
        score = 0.5
    elif (p1score == TOPIC and p2score == MUST) or (p1score == MUST and p2score == TOPIC):
        score = 0.25
    elif (p1score == TOPIC and p2score == SHOULD) or (p1score == SHOULD and p2score == TOPIC):
        score = 0.2
    else:
        score = 0.0
    return score

def get_rev_hier_qrels(manual_hier_qrels):
    rev_mq = {}
    with open(manual_hier_qrels, 'r') as mq:
          for l in mq:
              sec = l.split(' ')[0]
              para = l.split(' ')[2]
              score = int(l.split(' ')[3])
              if score >= 0:
                  if para not in rev_mq.keys():
                      rev_mq[para] = {sec: score}
                  else:
                      rev_mq[para][sec] = score
    return rev_mq

def convert_manual_parapair_data(rev_mq):
    parapair_manual = {}
    all_paras = list(rev_mq.keys())
    for i in range(len(all_paras) - 1):
        for j in range(i + 1, len(all_paras)):
            p1 = all_paras[i]
            p2 = all_paras[j]
            if p1 > p2:
                temp = p1
                p1 = p2
                p2 = temp
            parapair_manual[p1 + '_' + p2] = []
            for s in rev_mq[p1].keys():
                if s in rev_mq[p2].keys():
                    parapair_manual[p1 + '_' + p2].append((rev_mq[p1][s], rev_mq[p2][s]))
                else:
                    parapair_manual[p1 + '_' + p2].append((rev_mq[p1][s], -1))
            for s in rev_mq[p2].keys():
                if s not in rev_mq[p1].keys():
                    parapair_manual[p1 + '_' + p2].append((-1, rev_mq[p2][s]))
    return parapair_manual

def get_rel_page_paras(man_qrels):
    rel_page_paras = {}
    with open(man_qrels, 'r') as mq:
        for l in mq:
            page = l.split(' ')[0].split('/')[0]
            para = l.split(' ')[2]
            score = int(l.split(' ')[3])
            if score < 0:
                continue
            else:
                if page not in rel_page_paras.keys():
                    rel_page_paras[page] = [para]
                elif para not in rel_page_paras[page]:
                    rel_page_paras[page].append(para)
    return rel_page_paras

def write_parapair_data(parapair_manual, rel_page_paras, outfile):
    parapairs = {}
    for p in rel_page_paras.keys():
        print(p)
        paras = rel_page_paras[p]
        for i in range(len(paras) - 1):
            for j in range(i + 1, len(paras)):
                p1 = paras[i]
                p2 = paras[j]
                if p1 + '_' + p2 in parapair_manual.keys():
                    dat = parapair_manual[p1 + '_' + p2]
                else:
                    dat = parapair_manual[p2 + '_' + p1]
                addit = False
                for d in dat:
                    if d[0] >= 0 and d[1] >= 0:
                        addit = True
                        break
                if not addit:
                    continue
                s = 0.0
                for d in dat:
                    s += get_score(d[0], d[1])
                s = s / len(dat)
                if s < 0.0000000001:
                    print(p + ': ' + p1 + ', ' + p2 + ', ' + str(dat))
                if p not in parapairs.keys():
                    parapairs[p] = {'parapairs': [p1 + '_' + p2], 'labels': [s]}
                else:
                    parapairs[p]['parapairs'].append(p1 + '_' + p2)
                    parapairs[p]['labels'].append(s)
    with open(outfile, 'w') as out:
        json.dump(parapairs, out)


##############
#  DEPRECATED (doesn't take care of multiple para pairs across different sections of different pages
##############
def write_query_attn_from_manual_qrels(manual_qrels, outfile):
    topic_paras = {}
    with open(manual_qrels, 'r') as mq:
        for l in mq:
            topic = l.split(' ')[0]
            para = l.split(' ')[2]
            score = int(l.split(' ')[3])
            if topic in topic_paras.keys():
                topic_paras[topic].append((para, score))
            else:
                topic_paras[topic] = [(para, score)]
    negdata = []
    posdata = []
    for t in topic_paras.keys():
        for i in range(len(topic_paras[t]) - 1):
            for j in range(i+1, len(topic_paras[t])):
                p1 = topic_paras[t][i]
                p2 = topic_paras[t][j]
                if p1>p2:
                    temp = p1
                    p1 = p2
                    p2 = temp
                score = get_score(p1, p2)
                if score == 0:
                    negdata.append(str(score) + '\t' + t.split('/')[0].split(':')[1].replace('%20', ' ') + '\t' + p1[0] + '\t' + p2[0])
                else:
                    posdata.append(str(score) + '\t' + t.split('/')[0].split(':')[1].replace('%20', ' ') + '\t' + p1[0] + '\t' + p2[0])
    negdata = random.sample(negdata, len(posdata))
    data = posdata + negdata
    random.shuffle(data)
    with open(outfile, 'w') as out:
        for d in data:
            out.write(d + '\n')

def main():
    parser = argparse.ArgumentParser(description='Write parapair data from manual qrels: score \\t p1 \\t p2')
    parser.add_argument('-mq', '--manual_qrels', help='Path to manual qrels')
    parser.add_argument('-o', '--outfile', help='Path to output file')
    args = vars(parser.parse_args())
    mq = args['manual_qrels']
    out = args['outfile']
    rev_mq = get_rev_hier_qrels(mq)
    parapair_manual = convert_manual_parapair_data(rev_mq)
    rel_page_paras = get_rel_page_paras(mq)
    write_parapair_data(parapair_manual, rel_page_paras, out)


if __name__ == '__main__':
    main()
from trec_car.read_data import *
import sys
import re
import random

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

def get_score(p1, p2):
    MUST = 3
    SHOULD = 2
    CAN = 1
    TOPIC = 0
    NONREL = -1
    TRASH = -2
    p1score = p1[1]
    p2score = p2[1]
    if p1score == MUST and p2score == MUST:
        score = 1.0
    elif p1score == SHOULD and p2 == SHOULD:
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
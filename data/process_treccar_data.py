from trec_car.read_data import *
import sys

def write_paratext_data(para_cbor, outfile):
    c = 0
    with open(outfile, 'w') as out:
        with open(para_cbor, 'rb') as f:
            for p in iter_paragraphs(f):
                texts = [elem.text if isinstance(elem, ParaText)
                         else elem.anchor_text
                         for elem in p.bodies]
                text = ' '.join(texts)
                out.write(p.para_id + '\t' + text + '\n')
                c += 1
                if c % 10000 == 0:
                    print(str(c) + ' paras written')
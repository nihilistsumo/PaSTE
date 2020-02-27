import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale
import numpy as np

class UtilTools():
    def __init__(self,):
        self.name = 'util'

    def get_page_paras_dict(self, art_qrels):
        page_para = dict()
        with open(art_qrels, 'r') as aq:
            for l in aq:
                page = l.split(' ')[0]
                para = l.split(' ')[2]
                if page in page_para.keys():
                    page_para[page].append(para)
                else:
                    page_para[page] = [para]
        return page_para

    def compare_para_emb(self, p1, p2, ids, scaled_vecs):
        fig, (ax1, ax2) = plt.subplots(2, 1)
        fig.subplots_adjust(hspace=0.5)
        plt.xlim(0, 780)
        plt.ylim(0, 1.0)
        v1 = scaled_vecs[ids.index(p1)]
        v2 = scaled_vecs[ids.index(p2)]
        ax1.plot(v1)
        ax1.plot(v2)
        ax2.plot(v1 - v2)
        plt.show()

    def show_para_emb_in_page(self, paras, ids, scaled_vecs):
        plt.xlim(0, 780)
        plt.ylim(0, 1.0)
        for p in paras:
            v = scaled_vecs[ids.index(p)]
            plt.plot(v)
        plt.show()

    def scale_emb_vecs(self, vecs):
        return minmax_scale(vecs)
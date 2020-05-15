from sklearn.metrics import adjusted_rand_score, roc_auc_score
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def get_pure_cluster(k, num_per_k):
    pure = []
    for i in range(k):
        pure += [i]*num_per_k
    return pure

def get_cand_cluster(pure, num_correct, k):
    wrong_ind = list(np.random.randint(0, len(pure), len(pure) - num_correct))
    cand = []
    for i in range(len(pure)):
        if i not in wrong_ind:
            cand.append(pure[i])
        else:
            s = random.randint(0, k)
            while s == pure[i]:
                s = random.randint(0, k)
            cand.append(s)
    return cand

def run_experiment1():
    total = 500
    mat = []
    cols = []
    for k in range(2, 25):
        pure = get_pure_cluster(k, total // k)
        dat = []
        for c in range(10, total - 10, 10):
            cand = get_cand_cluster(pure, c, k)
            dat.append(adjusted_rand_score(pure, cand))
        mat.append(dat)
        cols.append('K'+str(k))
    mat = np.transpose(mat)
    correctx = []
    for c in range(10, total - 10, 10):
        correctx.append(c)
    exp_df = pd.DataFrame(mat, correctx, cols)
    ax = sns.lineplot(data=exp_df, dashes=False)
    plt.show()

def run_experiment2(k):
    total = 500
    mat = []
    cols = []

    pure = get_pure_cluster(k, total // k)
    pure_pair = []
    for i in range(len(pure)-1):
        for j in range(i+1, len(pure)):
            if pure[i] == pure[j]:
                pure_pair.append(1)
            else:
                pure_pair.append(0)
    dat = []
    correctx = []
    for c in range(10, total - 10, 10):
        cand = get_cand_cluster(pure, c, k)
        cand_pair = []
        for i in range(len(cand) - 1):
            for j in range(i + 1, len(cand)):
                if cand[i] == cand[j]:
                    cand_pair.append(1)
                else:
                    cand_pair.append(0)
        correctx.append(roc_auc_score(pure_pair, cand_pair))
        dat.append(adjusted_rand_score(pure, cand))
    mat.append(dat)
    cols.append('K' + str(k))
    mat = np.transpose(mat)

    #exp_df = pd.DataFrame(mat, correctx, cols)
    ax = sns.lineplot(x=correctx, y=dat, dashes=False)
    plt.show()

def main():
    run_experiment2(10)

if __name__ == '__main__':
    main()
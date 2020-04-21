import numpy as np
import os
import argparse

def get_sent_embs(pid, id_dict, vecs):
    start = int(id_dict[pid][0])
    l = int(id_dict[pid][1])
    if l == 0:
        embs = np.array([np.zeros(vecs[0].shape)])
        print('zero vec')
    else:
        embs = vecs[start:start+l]
    return embs

def convert(input_sent_dir, output_dir):
    files = os.listdir(input_sent_dir)
    sent_pids = np.load(input_sent_dir+'/paraids_sents.npy')
    filename = files[1]
    if files[0] == 'paraids_sents.npy':
        sent_vecs = np.load(input_sent_dir + '/' + files[1])
    else:
        filename = files[0]
        sent_vecs = np.load(input_sent_dir + '/' + files[0])
    id_dict = {}
    for i in sent_pids:
        id_dict[i.split('\t')[0]] = (i.split('\t')[2], i.split('\t')[3])
    mean_ids = []
    mean_vecs = []
    for i in sent_pids:
        pid = i.split('\t')[0]
        embvecs = get_sent_embs(pid, id_dict, sent_vecs)
        vec = np.mean(embvecs, axis=0)
        mean_ids.append(pid)
        mean_vecs.append(vec)
    mean_ids = np.array(mean_ids)
    mean_vecs = np.array(mean_vecs)
    np.save(output_dir+'/paraids.npy', mean_ids)
    np.save(output_dir+'/'+filename, mean_vecs)

def main():
    parser = argparse.ArgumentParser(description='Convert sentwise emb vecs to mean psg vecs')
    parser.add_argument('-id', '--input_dir', help='Path to input emb dir')
    parser.add_argument('-od', '--output_dir', help='Path to output emb dir')
    args = vars(parser.parse_args())
    indir = args['input_dir']
    outdir = args['output_dir']
    convert(indir, outdir)

if __name__ == '__main__':
    main()
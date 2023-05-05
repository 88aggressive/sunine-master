import os
import numpy as np
import kaldi_io
import argparse

def eval_create_utt2spk_map(utt2spk_file):
    '''build a hash map (utt->spk)'''
    assert os.path.exists(utt2spk_file)
    print("Creating mapping dict utt2spk{}")
    utt2spk = {}
    with open(utt2spk_file) as f:
        for line in f:
            utt, spk = line.strip().split()
            utt2spk[utt] = spk
    print("Created mapping dict utt2spk{}")
    return utt2spk


def create_utt2spk_map(utt2spk_file):
    '''build a hash map (utt->spk)'''
    print("Creating mapping dict utt2spk{}")
    items = np.loadtxt(utt2spk_file, dtype=str)

    spker_list = []
    for item in items:
        spker_list.append(item[1])

    spkers = np.unique(spker_list)
    idx = 0
    spk2id = {}
    for spk in spkers:
        spk2id[spk] = idx
        idx += 1

    utt2spk = {}
    for item in items:
        utt = item[0]
        spk = item[1]
        utt2spk[utt] = spk2id[spk]

    print("Created mapping dict utt2spk{}")
    return utt2spk


def load_vector_scp(is_eval, apply_norm, scp_file, npz_file, utt2spk_file):
    '''load kaldi scp file'''
    assert(os.path.splitext(scp_file)[1] == ".scp")

    print("Loading kaldi scp file...")
    utts = []
    vecs = []
    for k, v in kaldi_io.read_vec_flt_scp(scp_file):
        utts.append(k)
        vecs.append(v)

    assert(len(utts)  == len(vecs))

    if is_eval:
        print("Loading eval data...")
        utt2spk = eval_create_utt2spk_map(utt2spk_file)
    else:
        print("Loading training data...")
        utt2spk = create_utt2spk_map(utt2spk_file)

    vectors = []
    spker_label = []
    utt_label = []
    vec_dim = len(vecs[0])
    for i in range(len(utts)):
        if apply_norm:
            vec = np.array(vecs[i])
            norm = np.linalg.norm(vec)
            vectors.append(math.sqrt(vec_dim) * vec / norm)
        else:
            vectors.append(vecs[i])
        spker_label.append(utt2spk[utts[i]])
        utt_label.append(utts[i])

    if not os.path.exists(os.path.dirname(npz_file)):
        os.makedirs(os.path.dirname(npz_file))

    np.savez(npz_file, vectors=vectors, spker_label=spker_label, utt_label=utt_label)
    print("Convert {} to {} ".format(scp_file, npz_file))


if __name__ == "__main__":
    # test case
    parser = argparse.ArgumentParser()
    parser.add_argument('--is-eval', action='store_true', default=False,
                        help='process on eval or train')
    parser.add_argument('--apply-norm', action='store_true', default=False,
                        help='if apply length normalization')
    parser.add_argument('--src-file', default="vector.scp",
                        help='src file of vector.(scp)')
    parser.add_argument('--dest-file', default="vector.npz",
                        help='dest file of vector.(npz)')
    parser.add_argument('--utt2spk-file', default="utt2spk",
                        help='mapping file between utter and spker')
    args = parser.parse_args()

    load_vector_scp(args.is_eval, args.apply_norm, args.src_file, args.dest_file, args.utt2spk_file)

    # load data and label
    vectors = np.load(args.dest_file, allow_pickle=True)['vectors']
    spker_label = np.load(args.dest_file, allow_pickle=True)['spker_label']
    utt_label = np.load(args.dest_file, allow_pickle=True)['utt_label']

    print("vectors shape: ", np.shape(vectors))

    print("spker label shape: ", np.shape(spker_label))
    print("num of spkers: ", np.shape(np.unique(spker_label)))
    print(spker_label[-1])

    print("utt label shape: ", np.shape(utt_label))
    print("num of utts: ", np.shape(np.unique(utt_label)))
    print(utt_label[-1])

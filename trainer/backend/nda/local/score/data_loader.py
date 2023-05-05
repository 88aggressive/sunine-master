import os
import numpy as np
import kaldi_io
import argparse

pi = np.array(np.pi)

def create_utt2spk_map(utt2spk_file):
    '''build a hash map (utt->spk)'''
    assert os.path.exists(utt2spk_file)
    # print("Creating mapping dict utt2spk{}")
    utt2spk = {}
    with open(utt2spk_file) as f:
        for line in f:
            utt, spk = line.strip().split()
            utt2spk[utt] = spk
    # print("Created mapping dict utt2spk{}")
    return utt2spk


def load_vector_scp(scp_file, utt2spk_file):
    '''load kaldi scp file'''
    assert(os.path.splitext(scp_file)[1] == ".scp")
    utts = []
    vecs = []
    for k, v in kaldi_io.read_vec_flt_scp(scp_file):
        utts.append(k)
        vecs.append(v)

    assert(len(utts)  == len(vecs))

    utt2spk = create_utt2spk_map(utt2spk_file)

    vectors = []
    spkers = []
    utters = []
    for i in range(len(utts)):
        vectors.append(vecs[i])
        spkers.append(utt2spk[utts[i]])
        utters.append(utts[i])
    return np.array(vectors), np.array(spkers), np.array(utters) 


def load_trials(trials_file):
    '''
    load trials file:
    <enroll-utt-id> <test-utt-id> <target|nontarge>
    '''
    assert os.path.exists(trials_file)
    enroll_id = []
    test_id = []
    target_id = []
    with open(trials_file) as f:
        for line in f:
            spk, utt, is_target = line.strip().split()
            enroll_id.append(spk)
            test_id.append(utt)
            if is_target == "target":
                target_id.append(1)
            else:
                target_id.append(0)
    return np.array(enroll_id), np.array(test_id), np.array(target_id)


def load_num_utts(num_utts):
    '''build a hash map (spk->num)'''
    assert os.path.exists(num_utts)
    spk2num = {}
    with open(num_utts) as f:
        for line in f:
            spk, num = line.strip().split()
            spk2num[spk] = int(num)
    # print("Created mapping dict spk2num{}")
    return spk2num


if __name__ == "__main__":
    # test case
    parser = argparse.ArgumentParser()
    parser.add_argument('--src-file', default="vector.scp",
                        help='src file of vector.(scp)')
    parser.add_argument('--utt2spk-file', default="utt2spk",
                        help='mapping file between utter and spker')
    args = parser.parse_args()

    vectors, spkers, utters = load_vector_scp(args.src_file, args.utt2spk_file)

    print(np.shape(vectors))
    print(vectors[0])

    print(np.shape(spkers))
    print(np.shape(np.unique(spkers)))
    print(spkers[0])

    print(np.shape(utters))
    print(np.shape(np.unique(utters)))
    print(utters[0])


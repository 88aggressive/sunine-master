import os
import numpy as np
import kaldi_io
from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from data_loader import *
from cal_eer import *
from scipy import stats
import argparse

pi = np.array(np.pi)

def nl_score(enroll_vecs, enroll_num, test_vec, SB, SW):
    '''
    normalized likelihood with uncertain means
    each enroll spk may have more than one utt(mat)
    each test utt only have one mat
    SB is the speaker between var
    SW is the speaker within var
    '''
    assert(len(enroll_vecs[0]) == len(test_vec))
    enroll_vecs = np.array(enroll_vecs, dtype=float)
    enroll_mean = np.mean(enroll_vecs, axis=0)
    test_vec = np.array(test_vec, dtype=float)

    # uk = enroll_mean * (enroll_num * SB / (enroll_num * SB + SW))
    # pk = ((test_vec - uk)**2 / (SW + SB * SW / (enroll_num * SB + SW))).sum()
    # px = (test_vec**2 / (SW + SB)).sum()

    uk = enroll_mean * (enroll_num * SB / (enroll_num * SB + SW))
    vk = SW + SB * SW / (enroll_num * SB + SW)
    pk = ((test_vec - uk)**2 / vk).sum() + np.log(2 * pi * vk).sum()
    px = (test_vec**2 / (SW + SB)).sum() + np.log(2 * pi * (SW + SB)).sum()

    score = px - pk
    return score


def get_skew_and_kurt(data):
    '''calculate skew and kurt'''
    data = data.transpose()
    print(data.shape)
    skew = []
    kurt = []
    for i in data:
        _s = stats.skew(i)
        _k = stats.kurtosis(i)
        skew.append(_s)
        kurt.append(_k)
    skew_mean = sum(skew)/len(skew)
    kurt_mean = sum(kurt)/len(kurt)
    print('skew = {}  kurt = {}'.format(skew_mean, kurt_mean))


def score_by_trials(train_npz, enroll_npz, enroll_num_utts, test_npz, test_trials, centering, apply_lda, lda_dim, score_file):
    '''
    compute back-end score by trials
    '''
    # load data
    print("Load data")
    train_vectors = np.load(train_npz)['vectors']
    train_spkers = np.load(train_npz)['spker_label']
    train_utters = np.load(train_npz)['utt_label']

    enroll_vectors = np.load(enroll_npz)['vectors']
    enroll_spkers = np.load(enroll_npz)['spker_label']
    enroll_utters = np.load(enroll_npz)['utt_label']

    test_vectors = np.load(test_npz)['vectors']
    test_spkers = np.load(test_npz)['spker_label']
    test_utters = np.load(test_npz)['utt_label']

    # centering
    if centering:
        print("Substract global mean")
        global_mean_vector = np.mean(train_vectors, axis=0)
        # print(global_mean_vector)
        train_vectors = train_vectors - global_mean_vector
        enroll_vectors = enroll_vectors - global_mean_vector
        test_vectors = test_vectors - global_mean_vector

    # apply LDA
    if apply_lda:
        print("Apply LDA, reduce dim from {} to = {}".format(np.shape(train_vectors)[1], lda_dim))
        lda = LDA(solver='svd', n_components=lda_dim)
        lda.fit(train_vectors, train_spkers)
        train_vectors = lda.transform(train_vectors)
        enroll_vectors = lda.transform(enroll_vectors)
        test_vectors = lda.transform(test_vectors)

    # compute SB and SW
    # build hashmap train_spker -> utters
    train_spk2utt = {}
    for idx in range(len(train_spkers)):
        spk = train_spkers[idx]
        if spk not in train_spk2utt:
            train_spk2utt[spk] = []
        train_spk2utt[spk].append(train_vectors[idx])

    SW = np.zeros(np.shape(train_vectors)[1], dtype=float)
    SB = []
    for key, val in train_spk2utt.items():
        vecs = np.array(val)
        SW += len(vecs) * np.var(vecs, axis=0)
        SB.append(np.mean(vecs, axis=0))

    get_skew_and_kurt(np.array(SB))
    SW = SW / len(train_vectors)
    SB = np.var(np.array(SB), axis=0)
    # print("====> SW: ")
    # print(SW)

    # print("====> SB: ")
    # print(SB)

    # build hashmap enroll_spk -> utters
    enroll_spk2utt = {}
    for idx in range(len(enroll_spkers)):
        spk = enroll_spkers[idx]
        if spk not in enroll_spk2utt:
            enroll_spk2utt[spk] = []
        enroll_spk2utt[spk].append(enroll_vectors[idx])

    # build hashmap test_utt -> utter
    test_spk2utt = {}
    for idx in range(len(test_utters)):
        label = test_utters[idx]
        test_spk2utt[label] = test_vectors[idx]

    # load trials and compute EER
    enroll_id, test_id, target_id = load_trials(test_trials)
    num_utts = load_num_utts(enroll_num_utts)
    target_scores = []
    nontarget_scores = []
    foo = open(score_file, 'w')
    for i in range(len(target_id)):
        enroll_vecs = enroll_spk2utt[enroll_id[i]]
        enroll_num = num_utts[enroll_id[i]]
        test_vec = test_spk2utt[test_id[i]]
        score = nl_score(enroll_vecs, enroll_num, test_vec, SB, 1)
        foo.write(' '.join([enroll_id[i], test_id[i], str(target_id[i]), str(score)]) + '\n')

        if target_id[i]:
            target_scores.append(score)
        else:
            nontarget_scores.append(score)

    EER, thres = compute_eer(target_scores, nontarget_scores)
    print("EER: {:.2f}% and Threshold: {:.2f}".format(EER*100.0, thres))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--train-npz', default='train/xvector.npz', help='npz file of train vectors')
    parser.add_argument(
        '--enroll-npz', default='enroll/xvector.npz', help='npz file of enroll vector')
    parser.add_argument(
        '--enroll-num-utts', default='enroll/num_utts.ark', help='mapping file of spker to utter number')
    parser.add_argument(
        '--test-npz', default='test/xvector.npz', help='npz file of test vector')
    parser.add_argument(
        '--trials', default='trials.trl', help='file of test trials')
    parser.add_argument(
        '--centering', action='store_true', default=False, help='if centralize raw data')
    parser.add_argument(
        '--apply-lda', action='store_true', default=False, help='if apply LDA dimension reduction')
    parser.add_argument(
        '--lda-dim', type=int, default=150, help='components of LDA dimension')
    parser.add_argument(
        '--score', default='score.foo', help='file of trial scores')

    args = parser.parse_args()

    score_by_trials(args.train_npz, args.enroll_npz, args.enroll_num_utts, args.test_npz, args.trials, args.centering, args.apply_lda, args.lda_dim, args.score)


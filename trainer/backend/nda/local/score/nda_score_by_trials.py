import os
import math
import numpy as np
from data_loader import *
from cal_eer import *
import argparse

pi = np.array(np.pi)


def GetNormalizationFactor(transformed_vector, num_utts, nda_SB):
  assert(num_utts > 0)
  dim = len(transformed_vector)
  inv_covar = 1.0 / (1.0 / num_utts + nda_SB)
  dot_prod = np.dot(inv_covar, transformed_vector ** 2)
  return math.sqrt(dim / dot_prod)


def TransformVector(vector, num_utts, nda_SB, simple_length_norm, normalize_length):
  dim = len(vector)
  transformed_vector = vector
  normalization_factor = 0.0
  if simple_length_norm:
    normalization_factor = math.sqrt(dim) / np.linalg.norm(transformed_vector)
  else:
    normalization_factor = GetNormalizationFactor(transformed_vector, num_utts, nda_SB)
  if normalize_length:
    transformed_vector = transformed_vector * normalization_factor
  return transformed_vector


def NLScore(enroll_vec, enroll_num, test_vec, SB, SW):
    '''
    normalized likelihood with uncertain means
    SB is the speaker between var
    SW is the speaker within var
    '''
    # uk = enroll_vec * (enroll_num * SB / (enroll_num * SB + SW))
    # pk = ((test_vec - uk)**2 / (SW + SB * SW / (enroll_num * SB + SW))).sum()
    # px = (test_vec**2 / (SW + SB)).sum()

    uk = enroll_vec * (enroll_num * SB / (enroll_num * SB + SW))
    vk = SW + SB * SW / (enroll_num * SB + SW)
    pk = ((test_vec - uk)**2 / vk).sum() + np.log(2 * pi * vk).sum()
    px = (test_vec**2 / (SW + SB)).sum() + np.log(2 * pi * (SW + SB)).sum()

    score = 0.5 * (px - pk)
    return score


def score_by_trials(enroll_npz, enroll_num_utts, test_npz, test_trials, score_file, nda_SB, simple_length_norm, normalize_length):
    '''
    compute NL scores by trials
    '''
    # load data
    print("Load data")
    enroll_vectors = np.load(enroll_npz)['vectors']
    enroll_spkers = np.load(enroll_npz)['spker_label']
    enroll_utters = np.load(enroll_npz)['utt_label']

    test_vectors = np.load(test_npz)['vectors']
    test_spkers = np.load(test_npz)['spker_label']
    test_utters = np.load(test_npz)['utt_label']

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
        enroll_vec = np.mean(np.array(enroll_vecs), axis=0)
        enroll_num = num_utts[enroll_id[i]]
        enroll_trans_vec = TransformVector(enroll_vec, enroll_num, nda_SB, simple_length_norm, normalize_length)
        test_vec = np.array(test_spk2utt[test_id[i]])
        test_trans_vec = TransformVector(test_vec, 1, nda_SB, simple_length_norm, normalize_length)
        score = NLScore(enroll_trans_vec, enroll_num, test_trans_vec, nda_SB, 1)
        foo.write(' '.join([enroll_id[i], test_id[i], str(target_id[i]), str(score)]) + '\n')

        if target_id[i]:
            target_scores.append(score)
        else:
            nontarget_scores.append(score)

    EER, thres = compute_eer(target_scores, nontarget_scores)
    print("EER: {:.3f}% and Threshold: {:.3f}".format(EER*100.0, thres))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--enroll-npz', default='enroll/xvector.npz', help='npz file of enroll vector')
    parser.add_argument(
        '--enroll-num-utts', default='enroll/num_utts.ark', help='mapping file of spker to utter number')
    parser.add_argument(
        '--test-npz', default='test/xvector.npz', help='npz file of test vector')
    parser.add_argument(
        '--trials', default='trials.trl', help='file of test trials')
    parser.add_argument(
        '--score', default='score.foo', help='file of trial scores')
    parser.add_argument(
        '--nda-SB', default='nda_SB', help='file of between-class variance')
    parser.add_argument(
        '--simple-length-norm', action='store_true', default=False, help='process simple length norm (2-norm)')
    parser.add_argument(
        '--normalize-length', action='store_true', default=False, help='process length normlization')

    args = parser.parse_args()

    nda_SB = np.loadtxt(args.nda_SB, dtype=np.float)
    
    score_by_trials(args.enroll_npz, args.enroll_num_utts, args.test_npz, args.trials, args.score, nda_SB, args.simple_length_norm, args.normalize_length)


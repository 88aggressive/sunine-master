import os
import numpy as np
import argparse
import kaldi_io
from tqdm import tqdm


def npz2kaldi(npz_file, ark_file):
    '''load npz format and save as kaldi ark format'''
    print("Loading npz file...")
    vectors = np.load(npz_file)['vectors']
    utters = np.load(npz_file)['utt_label']

    assert(len(vectors) == len(utters))

    # if not os.path.exists(os.path.dirname(ark_file)):
    #     os.makedirs(os.path.dirname(ark_file))

    pbar = tqdm(total=len(utters))
    with open(ark_file,'wb') as f:
        for i in range(len(utters)):
            kaldi_io.write_vec_flt(f, vectors[i], utters[i])
            pbar.update(1)
            pbar.set_description('generate utter {}'.format(utters[i]))
    pbar.close()
    print("Convert {} to {} ".format(npz_file, ark_file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src-file', default="vectors.npz",
                        help='src file of vectors.(npz)')
    parser.add_argument('--dest-file', default="vectors.ark",
                        help='dest file of vectors.(ark)')
    args = parser.parse_args()

    npz2kaldi(args.src_file, args.dest_file)


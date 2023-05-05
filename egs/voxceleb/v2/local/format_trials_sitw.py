#!/usr/bin/env python
# encoding: utf-8

import os
import argparse
import numpy as np

def _replace(path):
    path = path.replace("audio", "wav")
    path = path.replace("flac", "wav")
    return path

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--sitw_root', help='sitw dir', type=str, default="SITW")
    parser.add_argument('--dst_trl_path', help='output trial path', type=str, default="new.trials")
    parser.add_argument('--apply_vad', action='store_true', default=False)
    args = parser.parse_args()

    enroll_lst_path = os.path.join(args.sitw_root, "lists/enroll-core.lst")
    raw_trl_path = os.path.join(args.sitw_root, "keys/core-core.lst")

    spk2wav_mapping = {}
    enroll_lst = np.loadtxt(enroll_lst_path, str)
    for item in enroll_lst:
        spk2wav_mapping[item[0]] = item[1]
    trials = np.loadtxt(raw_trl_path, str)

    with open(args.dst_trl_path, "w") as f:
        for item in trials:
            enroll_path = os.path.join(args.sitw_root, spk2wav_mapping[item[0]])
            test_path = os.path.join(args.sitw_root, item[1])
            label = item[2]
            if label == "tgt":
                label = "1"
            else:
                label = "0"

            enroll_path = _replace(enroll_path)
            test_path = _replace(test_path)

            if args.apply_vad:
                enroll_path = enroll_path.strip("*.wav") + ".vad"
                test_path = test_path.strip("*.wav") + ".vad"

            f.write("{} {} {}\n".format(label, enroll_path, test_path))


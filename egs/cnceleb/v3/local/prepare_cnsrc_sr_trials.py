#!/usr/bin/env python
# encoding: utf-8

import os
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', help='path of cnsrc SR.dev', type=str, default="Task2_dev")
    parser.add_argument('--output_trl_path', help='path of output trial', type=str, default="data/trials/new.trl")
    args = parser.parse_args()

    enroll_dir = os.path.join(args.data_root, "target")
    pool_dir = os.path.join(args.data_root, "pool")
    enroll_wav_list = os.listdir(enroll_dir)
    pool_wav_list = os.listdir(pool_dir)    
    
    f = open(args.output_trl_path, 'w')
    for enroll_wav_name in enroll_wav_list:
        enroll_wav_path = os.path.join(enroll_dir, enroll_wav_name)
        for pool_wav_name in pool_wav_list:
            pool_wav_path = os.path.join(pool_dir, pool_wav_name)
            f.write("{} {} {}\n".format(0, enroll_wav_path, pool_wav_path))
    f.close()

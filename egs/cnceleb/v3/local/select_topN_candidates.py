#!/usr/bin/env python
# encoding: utf-8

import pandas as pd
import sys
import argparse

def pd_sort(fileName, top_N, outfile):
    header = ['spk-id', 'utt-id', 'scores']
    df = pd.read_csv(fileName, sep='\s+', names=header)
    df = df.groupby('spk-id').apply(lambda x:x.nlargest(int(top_N),'scores'))
    # print(df)

    # write to outfile
    with open(outfile, 'w') as f:
        old_spk = ' '
        for index, row in df.iterrows():
            spk = row['spk-id']
            if old_spk == ' ':
                old_spk = spk
                f.write(spk)
            elif old_spk != spk:
                old_spk = spk
                f.write('\n' + spk)
            f.write(' ' + row['utt-id'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_scores_path', help='path of the input scores', type=str, default="scores.foo")
    parser.add_argument('--top_N', help='the highest N candidates', type=int, default=10)
    parser.add_argument('--output_sr_path', help='path of the output speaker retrieval result', type=str, default="sr.topN")
    args = parser.parse_args()

    pd_sort(args.input_scores_path, args.top_N, args.output_sr_path)
    

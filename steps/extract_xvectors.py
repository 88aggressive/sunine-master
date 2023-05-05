#!/usr/bin/env python
# encoding: utf-8
# Function: Extract embeddings of wav.scp and save as Kaldi Format

import os
import numpy as np
from argparse import ArgumentParser
import torch
from scipy.io import wavfile
from pytorch_lightning import LightningModule, Trainer
from trainer import Model
from tqdm import tqdm
import kaldi_io


def loadWAV(path):
    sample_rate, audio = wavfile.read(path)
    return torch.FloatTensor(audio)

if __name__ == "__main__":
    # args
    parser = ArgumentParser()
    parser.add_argument('--wav_scp', help='utt-id wav-path', type=str, default="wav.scp")
    parser.add_argument('--ark_path', help='', type=str, default="xvector.ark")
    parser = Trainer.add_argparse_args(parser)
    parser = Model.add_model_specific_args(parser)
    args = parser.parse_args()

    args.evaluate = True
    model = Model(**vars(args))

    # pop loss Function parameter
    state_dict = torch.load(args.checkpoint_path, map_location="cpu")["state_dict"]
    loss_weights = []
    for key, value in state_dict.items():
        if "loss" in key:
            loss_weights.append(key)
    for item in loss_weights:
        state_dict.pop(item)

    # load speaker encoder state dict
    model.load_state_dict(state_dict, strict=False)

    # load wav scp
    f = open(args.wav_scp, 'r')
    lines = f.readlines()
    f.close()

    utt_data = {}
    for line in lines:
        utt_id = line.strip().split()[0]
        wav_id = line.strip().split()[1]
        utt_data[utt_id] = wav_id

    model = model.cuda()
    model.eval()
    with torch.no_grad():
        pbar = tqdm(total=len(utt_data))
        f = open(args.ark_path, 'wb')
        for utt_id, wav_id in utt_data.items():
            data = loadWAV(wav_id)
            data = data.cuda()
            xvec = model.extract_speaker_embedding(data)
            xvec = xvec.cpu().detach().numpy()
            xvec = np.squeeze(xvec, 0)
            kaldi_io.write_vec_flt(f, xvec, key=utt_id)
            pbar.update(1)
            pbar.set_description('generate utter {}'.format(utt_id))
        pbar.close()
        f.close()
    print("successfully save kaldi ark in {}".format(args.ark_path))


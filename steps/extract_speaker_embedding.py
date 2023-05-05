#!/usr/bin/env python
# encoding: utf-8

from argparse import ArgumentParser
import torch
from scipy.io import wavfile
import numpy as np
from pytorch_lightning import LightningModule, Trainer
from trainer import Model

def loadWAV(path):
    sample_rate, audio = wavfile.read(path)
    return torch.FloatTensor(audio)

if __name__ == "__main__":
    # args
    parser = ArgumentParser()
    parser.add_argument('--device', help='', type=str, default="cuda")
    parser.add_argument('--wave_path', help='', type=str, default="cuda")
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

    audio = loadWAV(args.wave_path)
    if args.device == "cuda":
        model = model.cuda()
        audio = audio.cuda()

    model = model.eval()
    with torch.no_grad():
        embedding = model.extract_speaker_embedding(audio)
    print(embedding.shape)
    print(embedding)


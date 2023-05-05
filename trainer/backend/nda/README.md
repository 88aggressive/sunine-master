# NDA (Neural discriminant analysis)
A Pytorch implementation of NDA backend scoring.

```bibtex
@article{li2020neural,
  title={Neural Discriminant Analysis for Deep Speaker Embedding},
  author={Li, Lantian and Wang, Dong and Zheng, Thomas Fang},
  journal={arXiv preprint arXiv:2005.11905},
  year={2020}
}
```

## Git repo
```base
git clone git@gitlab.com:csltstu/nda.git
```

## Dependency
Install python package
```bash
pip3 install -r requrements.txt
```

Install Kaldi
```bash
# Download Kaldi
git clone https://github.com/kaldi-asr/kaldi.git kaldi --origin upstream
cd kaldi
# Follow INSTALL file
cat INSTALL
```

## Data preparation
Data preparation from kaldi to npz.
```bash
python -u local/kaldi2npz.py
```

## Main
Recipe to train and infer NDA models.
```bash
sh run.sh
```

## Tensorboard
Monitor the training process.
```bash
tensorboard --logdir runs/*
```

## Contributor
* Lantian Li
* Email: lilt@cslt.org

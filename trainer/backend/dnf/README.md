# DNF (Discriminative normalization flow)
A Pytorch implementation of DNF embedding normalization.

The neural network structure is based on "Masked Autoregressive Flow", and the source code from [ikostrikov](https://github.com/ikostrikov/pytorch-flows/blob/master/README.md)

```bibtex
@article{cai2020deep,
  title={Deep normalization for speaker vectors},
  author={Cai, Yunqi and Li, Lantian and Abel, Andrew and Zhu, Xiaoyan and Wang, Dong},
  journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  volume={29},
  pages={733--744},
  year={2020},
  publisher={IEEE}
}
```

## Git repo
```
git clone https://github.com/Caiyq2019/DNF.git
```

## Datasets
```bash
training set: Voxceleb 
test sets: SITW, CNCeleb
```
Following this [link](https://pan.baidu.com/s/1NZXZhKbrJUk75FDD4_p6PQ) to download the dataset. 
(extraction code：8xwe)

## Run DNF
```bash
python train.py
```
The evaluation and scoring will be performed automatically during the training process.

## Other instructions
```bash
score.py : a Python implementation of the standard Kaldi cosine scoring, you can also use kaldi to make plda scoring.
tsne.py  : a visualization toolkit by t-SNE to draw the distribution of latent space. 
```

## Contributor
* Yunqi Cai
* Email: caiyq@cslt.org


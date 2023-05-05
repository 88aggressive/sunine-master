# Sunine: THU-CSLT Speaker Recognition Toolkit

<p align="center">
  <img src="http://cslt.riit.tsinghua.edu.cn/mediawiki/images/d/db/Cslt.jpg" alt="drawing" width="250"/>
</p>

Sunine is an **open-source** speaker recognition toolkit based on [PyTorch](https://pytorch.org/).

The goal is to create a **user-friendly** toolkit that can be used to easily develop **state-of-the-art speaker recognition technologies**.

> Copyright: [THU-CSLT](http://cslt.riit.tsinghua.edu.cn/) (Tsinghua University, China)  
> Apache License, Version 2.0 [LICENSE](https://gitlab.com/csltstu/sunine/-/blob/master/LICENSE)
>
> Authors : Lantian Li (lilt@cslt.org), Yang Zhang (zhangy20@mails.tsinghua.edu.cn)  
> Co-author: Dong Wang (wangdong99@mails.tsinghua.edu.cn)


---
- **Content**
  * [Features](#Features)
  * [Methodologies](#Methodologies)
    + [Data processing](#data-processing)
    + [Backbone](#backbone)
    + [Pooling](#pooling)
    + [Loss Function](#loss-function)
    + [Backend](#backend)
    + [Metric](#metric)
  * [Quick installation](#quick-installation)
  * [Recipes](#recipes)
    + [VoxCeleb](#voxceleb)
    + [CNCeleb](#cnceleb)
  * [Pretrained models](#pretrained-models)
  * [Acknowledgement](#acknowledgement)
---


## Features
+ No Dependency on [Kaldi](http://www.kaldi-asr.org/).
+ Entire recipe of neural-based speaker verification.
+ Multi-GPU training and inference. 
+ State-of-the-art speaker recognition techniques.
+ Training from scratch and fine-tuning with a pretrained model.
+ On-the-fly acoustic feature extraction and data augmentation.


## Methodologies
Sunine provides state-of-the-art speaker recognition techniques.

### Data processing
+ On-the-fly acoustic feature extraction: PreEmphasis and MelSpectrogram.
+ On-the-fly data augmentation: additive noises on [MUSAN](http://www.openslr.org/17/) and reverberation on [RIR](http://www.openslr.org/28/).

### Backbone
+ [x] [TDNN](https://ieeexplore.ieee.org/abstract/document/8461375)
+ [x] [ResNet34](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)
+ [x] [Res2Net50](https://arxiv.org/pdf/1904.01169.pdf)
+ [x] [ECAPA-TDNN](https://arxiv.org/abs/2005.07143)

### Pooling
+ [x] [Temporal Average Pooling](https://arxiv.org/pdf/1903.12058.pdf)
+ [x] [Temporal Statistics Pooling](http://www.danielpovey.com/files/2018_icassp_xvectors.pdf)
+ [x] [Self-Attentive Pooling](https://danielpovey.com/files/2018_interspeech_xvector_attention.pdf)
+ [x] [Attentive Statistics Pooling](https://arxiv.org/pdf/1803.10963.pdf)

### Loss Function
+ [x] [Softmax](https://ieeexplore.ieee.org/abstract/document/8461375)
+ [x] [AM-Softmax](https://arxiv.org/abs/1801.05599)
+ [x] [AAM-Softmax](https://arxiv.org/abs/1801.07698)
+ [x] [ARM-Softmax](https://arxiv.org/pdf/2110.09116.pdf)

### Backend
+ [x] Cosine
+ [x] [PLDA](https://link.springer.com/chapter/10.1007/11744085_41)
+ [x] Score Normalization: [S-Norm](https://www.isca-speech.org/archive/odyssey_2010/kenny10_odyssey.html), [AS-Norm](https://www.isca-speech.org/archive_v0/archive_papers/interspeech_2011/i11_2365.pdf)
+ [x] [DNF](https://arxiv.org/abs/2004.04095)
+ [x] [NDA](https://arxiv.org/abs/2005.11905)

### Metric
+ [x] Calibration: [Cllr, minCllr](https://www.sciencedirect.com/science/article/pii/S0885230805000483)
+ [x] EER
+ [x] minDCF


## Quick installation
Once you have created your Python environment (Python 3.8+), you can simply create the project and install its requirements:
```base
git clone https://gitlab.com/csltstu/sunine
cd sunine/
pip3 install requirements.txt
```


## Recipes

### VoxCeleb
[VoxCeleb](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/index.html) is a large scale audio-visual dataset of human speech.

Two recipes are provided in egs/voxceleb/{v1,v2}.
+ [v1] demonstrates a standard x-vector model using TDNN backbone with Cosine backend.
+ [v2] demonstrates a more powerful x-vector model using ResNet34 backbone with Cosine backend.

### CNCeleb
[CNCeleb](cnc.cslt.org) is a large-scale multi-genre speaker recognition dataset of Chinese celebrities.

Two recipes are provided in egs/voxceleb/{v1,v2}.
+ [v1] demonstrates a standard x-vector model using TDNN backbone with PLDA backend.
+ [v2] demonstrates a more powerful x-vector model using ResNet34 backbone with Cosine backend.


## Pretrained models
Beyond providing recipes for training the models from scratch, Sunine shares several pre-trained models. 

+ Performance

| Training Set  | Test Set  | System                               | EER(%) |
|:--------      |:--------  |:------                               |:------ |
| Vox2.Dev      | Vox1-T    | ResNet34 + AM-Softmax + TSP + Cosine | 1.627  |
| Vox2.Dev      | Vox1-H    | ResNet34 + AM-Softmax + TSP + Cosine | 2.858  |
| Vox2.Dev      | Vox1-E    | ResNet34 + AM-Softmax + TSP + Cosine | 1.688  |
| Vox2.Dev      | SITW.D    | ResNet34 + AM-Softmax + TSP + Cosine | 3.119  |
| Vox2.Dev      | SITW.E    | ResNet34 + AM-Softmax + TSP + Cosine | 3.253  |
| CNC1.T + CNC2 | CNC1.E    | ResNet34 + AM-Softmax + TSP + Cosine | 11.890 |

+ Download

| Dataset  | BaiduYunDisk | AliYun |
|:---------|:------------ |:------ |
| VoxCeleb | [Here](https://pan.baidu.com/s/1Lp4C_c4rjISKUKYhMPT3_Q) (Code: tdwt) | [Here](http://cnsrc.cslt.org/download/ckpt/voxceleb-pretrain-model.tar.gz) |
| CNCeleb  | [Here](https://pan.baidu.com/s/1sg9TwqycrP_21qzcz2uBOQ) (Code: ke47) | [Here](http://cnsrc.cslt.org/download/ckpt/cnceleb-pretrain-model.tar.gz)  |

You can extract speaker embeddings with our pre-trained model on your data.

```bash
python steps/extract_speaker_embedding.py --wave_path test.wav --nnet_type ResNet34L --pooling_type TSP --embedding_dim 256 --checkpoint_path voxceleb/ResNet34L_TSP_d256_amsoftmax_s30.0_m0.2.ckpt
```

## Acknowledgement
+ This project is supported by the National Natural Science Foundation of China (NSFC) under Grants No.61633013 and No.62171250.
+ Thanks to the relevant projects
  * [Kaldi](http://www.kaldi-asr.org/)
  * [Pytorch](https://pytorch.org/)
  * [Pytorch Lightning](https://www.pytorchlightning.ai/)
  * [SpeechBrain](https://speechbrain.github.io/)
  * [ASV-Subtools](https://github.com/Snowdar/asv-subtools)


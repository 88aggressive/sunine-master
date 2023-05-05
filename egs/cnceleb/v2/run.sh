#!/bin/bash
# Copyright   2021   Tsinghua University (Author: Lantian Li, Yang Zhang)
# Apache 2.0.

SPEAKER_TRAINER_ROOT=../../..
cnceleb1_path=/nfs/data/sid/CNC_v2.0/CN-Celeb
cnceleb2_path=/nfs/data/sid/CNC_v2.0/CN-Celeb2

nnet_type=ResNet34L
pooling_type=TSP
loss_type=amsoftmax
embedding_dim=256
scale=30.0
margin=0.1
cuda_device=0,1,2

stage=0

if [ $stage -le 0 ];then
  # flac to wav
  python3 $SPEAKER_TRAINER_ROOT/steps/flac2wav.py \
          --dataset_dir $cnceleb1_path/data \
          --speaker_level 1

  python3 $SPEAKER_TRAINER_ROOT/steps/flac2wav.py \
          --dataset_dir $cnceleb2_path/data \
          --speaker_level 1
 
  python3 $SPEAKER_TRAINER_ROOT/steps/flac2wav.py \
          --dataset_dir $cnceleb1_path/eval/enroll \
          --speaker_level 0
  
  python3 $SPEAKER_TRAINER_ROOT/steps/flac2wav.py \
          --dataset_dir $cnceleb1_path/eval/test \
          --speaker_level 0
fi


# In our experiment, we found that VAD seems useless.
# Here directly skip this stage.
if [ $stage -eq 1 ];then
  # compute VAD for each dataset
  echo Compute VAD on cnceleb1
  python3 $SPEAKER_TRAINER_ROOT/steps/compute_vad.py \
          --data_dir $cnceleb1_path/data \
          --extension wav \
          --speaker_level 1 \
          --num_jobs 40

  echo Compute VAD on cnceleb2
  python3 $SPEAKER_TRAINER_ROOT/steps/compute_vad.py \
          --data_dir $cnceleb2_path/data \
          --extension wav \
          --speaker_level 1 \
          --num_jobs 40

  echo Compute VAD on enroll
  python3 $SPEAKER_TRAINER_ROOT/steps/compute_vad.py \
          --data_dir $cnceleb1_path/eval/enroll \
          --extension wav \
          --speaker_level 0 \
          --num_jobs 40

  echo Compute VAD on test
  python3 $SPEAKER_TRAINER_ROOT/steps/compute_vad.py \
          --data_dir $cnceleb1_path/eval/test \
          --extension wav \
          --speaker_level 0 \
          --num_jobs 40
fi


if [ $stage -le 2 ];then
  # prepare data
  if [ ! -d data/wav ]; then
    mkdir -p data/wav
  fi

  mkdir -p data/wav/train
  for spk in `cat ${cnceleb1_path}/dev/dev.lst`; do
    ln -s ${cnceleb1_path}/data/${spk} data/wav/train/$spk
  done

  for spk in `cat ${cnceleb2_path}/spk.lst`; do
    ln -s ${cnceleb2_path}/data/${spk} data/wav/train/$spk
  done

  # prepare evaluation trials
  mkdir -p data/trials
  python3 local/format_trials_cnceleb.py \
          --cnceleb_root $cnceleb1_path \
          --dst_trl_path data/trials/CNC-Eval-Core.lst
fi


if [ $stage -le 3 ];then
  # prepare data for model training
  mkdir -p data
  echo Build train list
  python3 $SPEAKER_TRAINER_ROOT/steps/build_datalist.py \
          --data_dir data/wav/train \
          --extension wav \
          --speaker_level 1 \
          --data_list_path data/train_lst.csv
fi


if [ $stage -le 4 ];then
  # model training
  CUDA_VISIBLE_DEVICES=$cuda_device python3 -W ignore $SPEAKER_TRAINER_ROOT/main.py \
          --train_list_path data/train_lst.csv \
          --trials_path data/trials/CNC-Eval-Core.lst \
          --n_mels 80 \
          --max_frames 201 --min_frames 200 \
          --batch_size 256 \
          --nPerSpeaker 1 \
          --max_seg_per_spk 500 \
          --num_workers 40 \
          --max_epochs 101 \
          --loss_type $loss_type \
          --nnet_type $nnet_type \
          --pooling_type $pooling_type \
          --embedding_dim $embedding_dim \
          --learning_rate 0.01 \
          --lr_step_size 5 \
          --lr_gamma 0.9 \
          --margin $margin \
          --scale $scale \
          --eval_interval 5 \
          --eval_frames 0 \
          --scores_path tmp.foo \
          --apply_metric \
          --save_top_k 20 \
          --distributed_backend dp \
          --reload_dataloaders_every_epoch \
          --gpus 3
fi


if [ $stage -le 5 ];then
  # evaluation
  # ckpt_path=exp/*/*.ckpt

  cuda_device=0
  mkdir -p scores/

  for ckpt_path in exp/${nnet_type}_${pooling_type}_${embedding_dim}_${loss_type}_${scale}_${margin}/*.ckpt; do
    echo $ckpt_path
    echo Evaluate CNC-Eval-Core
    CUDA_VISIBLE_DEVICES=$cuda_device python3 -W ignore $SPEAKER_TRAINER_ROOT/main.py \
            --evaluate \
            --checkpoint_path $ckpt_path \
            --n_mels 80 \
            --trials_path data/trials/CNC-Eval-Core.lst \
            --scores_path scores/CNC-Eval-Core.foo \
            --apply_metric \
            --nnet_type $nnet_type \
            --pooling_type $pooling_type \
            --embedding_dim $embedding_dim \
            --scale $scale \
            --margin $margin \
            --num_workers 20 \
            --gpus 1
  done

  # reference result
  # EER: 11.873%
  # minDCF(p-target=0.01): 0.5952
  # minDCF(p-target=0.001): 0.6929
fi


if [ $stage -eq 6 ];then
  # An example of the C-P Map on CNC-Eval-Core trials against v1/.
  python3 $SPEAKER_TRAINER_ROOT/trainer/metric/compute_trial_config.py \
          --mode 1 \
          --input_scores_ref ../v1/scores/CNC-Eval-Core.foo \
          --input_scores_test scores/CNC-Eval-Core.foo \
          --scale 20 \
          --output scores/CNC-Eval-Core.cfg

  awk '{print $3}' scores/CNC-Eval-Core.cfg > scores/CNC-Eval-Core.eer.cfg
  python3 $SPEAKER_TRAINER_ROOT/trainer/metric/plot_cp_map.py \
          --mode 0 \
          --metric eer \
          --input_configs_test scores/CNC-Eval-Core.eer.cfg \
          --scale 20 \
          --savedir cpmap/

  python3 $SPEAKER_TRAINER_ROOT/trainer/metric/plot_cp_map.py \
          --mode 1 \
          --metric eer \
          --input_configs_ref ../v1/scores/CNC-Eval-Core.eer.cfg \
          --input_configs_test CNC-Eval-Core.eer.cfg \
          --scale 20 \
          --savedir cpmap/

  awk '{print $4}' scores/CNC-Eval-Core.cfg > scores/CNC-Eval-Core.dcf.cfg
  python3 $SPEAKER_TRAINER_ROOT/trainer/metric/plot_cp_map.py \
          --mode 0 \
          --metric dcf \
          --input_configs_test scores/CNC-Eval-Core.dcf.cfg \
          --scale 20 \
          --savedir cpmap/

  python3 $SPEAKER_TRAINER_ROOT/trainer/metric/plot_cp_map.py \
          --mode 1 \
          --metric dcf \
          --input_configs_ref ../v1/scores/CNC-Eval-Core.dcf.cfg \
          --input_configs_test CNC-Eval-Core.dcf.cfg \
          --scale 20 \
          --savedir cpmap/
fi

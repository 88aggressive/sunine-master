#!/bin/bash
# Copyright 2022 Tsinghua University (Author: Lantian Li, Yang Zhang, Pengqi Li)
# Apache 2.0.

SPEAKER_TRAINER_ROOT=../../..
cnceleb1_path=/nfs/data/sid/CNC_v2.0/CN-Celeb
cnceleb2_path=/nfs/data/sid/CNC_v2.0/CN-Celeb2
cnsrc_sr_dev_path=/nfs/data/sid/CNSRC/Task2_dev

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

  echo Compute VAD on cnsrc_sr_dev/target
  python3 $SPEAKER_TRAINER_ROOT/steps/compute_vad.py \
          --data_dir $cnsrc_sr_dev_path/target \
          --extension wav \
          --speaker_level 0 \
          --num_jobs 40

  echo Compute VAD on cnsrc_sr_dev/pool
  python3 $SPEAKER_TRAINER_ROOT/steps/compute_vad.py \
          --data_dir $cnsrc_sr_dev_path/pool \
          --extension wav \
          --speaker_level 0 \
          --num_jobs 40
fi


if [ $stage -le 2 ];then
  # prepare data
  if [ ! -d data/wav ];then
    mkdir -p data/wav
  fi

  mkdir -p data/wav/train
  for spk in `cat ${cnceleb1_path}/dev/dev.lst`; do
    ln -s ${cnceleb1_path}/data/${spk} data/wav/train/$spk
  done

  for spk in `cat ${cnceleb2_path}/spk.lst`; do
    ln -s ${cnceleb2_path}/data/${spk} data/wav/train/$spk
  done
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

  # prepare evaluation trials
  mkdir -p data/trials
  python3 local/prepare_cnsrc_sr_trials.py \
          --data_root $cnsrc_sr_dev_path \
          --output_trl_path data/trials/CNSRC-SR-Dev-Core.trl
fi


if [ $stage -le 4 ];then
  # model training
  CUDA_VISIBLE_DEVICES=$cuda_device python3 -W ignore $SPEAKER_TRAINER_ROOT/main.py \
          --train_list_path data/train_lst.csv \
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
          --eval_interval -1 \
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
    echo Evaluate CNSRC-SR-Dev-Core

    startTime=`date +"%Y-%m-%d %H:%M:%S"`

    CUDA_VISIBLE_DEVICES=$cuda_device python3 -W ignore $SPEAKER_TRAINER_ROOT/main.py \
            --evaluate \
            --checkpoint_path $ckpt_path \
            --n_mels 80 \
            --trials_path data/trials/CNSRC-SR-Dev-Core.trl \
            --scores_path scores/CNSRC-SR-Dev-Core.foo \
            --nnet_type $nnet_type \
            --pooling_type $pooling_type \
            --embedding_dim $embedding_dim \
            --scale $scale \
            --margin $margin \
            --num_workers 20 \
            --gpus 1

    # select top-N candidates for each target speaker
    # scores/CNSRC-SR-Dev-Core.top10 is the final output file of the SR system
    python3 local/select_topN_candidates.py \
            --input_scores_path scores/CNSRC-SR-Dev-Core.foo \
            --top_N 10 \
            --output_sr_path scores/CNSRC-SR-Dev-Core.top10

    endTime=`date +"%Y-%m-%d %H:%M:%S"`
    st=`date -d "$startTime" +%s`
    et=`date -d "$endTime" +%s`
    RunTime=$(($et-$st))
    echo "Overall Retrieval Time is $RunTime s"

    # compute mAP
    python3 $SPEAKER_TRAINER_ROOT/trainer/metric/compute_mAP.py \
            --input_sr_path scores/CNSRC-SR-Dev-Core.top10 \
            --metadata_dir $cnsrc_sr_dev_path/metadata \
            --output_sr_path scores/CNSRC-SR-Dev-Core.top10.meta
  done

  # reference result
  # mAP on SR.dev: 0.460
fi

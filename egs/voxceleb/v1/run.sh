#!/bin/bash
# Copyright   2021   Tsinghua University (Author: Lantian Li, Yang Zhang)
# Apache 2.0.

SPEAKER_TRAINER_ROOT=../../..
voxceleb1_root=/nfs/data/sid/VoxCeleb/voxceleb1/voxceleb1_wav
voxceleb2_root=/nfs/data/sid/VoxCeleb/voxceleb2/dev/aac
sitw_dev_root=/nfs/data/sid/SITW/dev
sitw_eval_root=/nfs/data/sid/SITW/eval
musan_path=/nfs/data/musan
rirs_path=/nfs/data/RIRS_NOISES

nnet_type=TDNN
pooling_type=TSP
loss_type=softmax
embedding_dim=512
scale=30.0
margin=0.2
cuda_device=1,2,3

stage=1

# In our experiment, we found that VAD seems useless.
if [ $stage -eq 0 ];then
  # compute VAD for each dataset
  echo Compute VAD $voxceleb1_root
  python3 $SPEAKER_TRAINER_ROOT/steps/compute_vad.py \
          --data_dir $voxceleb1_root \
          --extension wav \
          --speaker_level 1 \
          --num_jobs 40

  echo Compute VAD $voxceleb2_root
  python3 $SPEAKER_TRAINER_ROOT/steps/compute_vad.py \
          --data_dir $voxceleb2_root \
          --extension wav \
          --speaker_level 1 \
          --num_jobs 40

  echo Compute VAD $sitw_dev_root
  python3 $SPEAKER_TRAINER_ROOT/steps/compute_vad.py \
          --data_dir $sitw_dev_root/wav \
          --extension wav \
          --speaker_level 0 \
          --num_jobs 40

  echo Compute VAD $sitw_eval_root
  python3 $SPEAKER_TRAINER_ROOT/steps/compute_vad.py \
          --data_dir $sitw_eval_root/wav \
          --extension wav \
          --speaker_level 0 \
          --num_jobs 40
fi


if [ $stage -le 1 ];then
  # prepare data for model training
  mkdir -p data
  echo Build $voxceleb2_root list
  python3 $SPEAKER_TRAINER_ROOT/steps/build_datalist.py \
          --data_dir $voxceleb2_root \
          --extension wav \
          --speaker_level 1 \
          --data_list_path data/train_lst.csv

  echo Build $musan_path list
  python3 $SPEAKER_TRAINER_ROOT/steps/build_datalist.py \
          --data_dir $musan_path \
          --extension wav \
          --data_list_path data/musan_lst.csv

  echo Build $rirs_path list
  python3 $SPEAKER_TRAINER_ROOT/steps/build_datalist.py \
          --data_dir $rirs_path \
          --extension wav \
          --data_list_path data/rirs_lst.csv
fi


if [ $stage -le 2 ];then
  # prepare test trials for evaluation
  mkdir -p data/trials
  python3 local/format_trials_voxceleb1.py \
          --voxceleb1_root $voxceleb1_root \
          --src_trl_path $voxceleb2_root/../../List_of_trial_pairs-VoxCeleb1-Clean.txt \
          --dst_trl_path data/trials/VoxCeleb1-Clean.lst 

  python3 local/format_trials_voxceleb1.py \
          --voxceleb1_root $voxceleb1_root \
          --src_trl_path $voxceleb2_root/../../List_of_trial_pairs-VoxCeleb1-H-Clean.txt \
          --dst_trl_path data/trials/VoxCeleb1-H-Clean.lst

  python3 local/format_trials_voxceleb1.py \
          --voxceleb1_root $voxceleb1_root \
          --src_trl_path $voxceleb2_root/../../List_of_trial_pairs-VoxCeleb1-E-Clean.txt \
          --dst_trl_path data/trials/VoxCeleb1-E-Clean.lst

  python3 local/format_trials_sitw.py \
          --sitw_root $sitw_dev_root \
          --dst_trl_path data/trials/SITW-Dev-Core.lst

  python3 local/format_trials_sitw.py \
          --sitw_root $sitw_eval_root \
          --dst_trl_path data/trials/SITW-Eval-Core.lst
fi


if [ $stage -le 3 ];then
  # model training
  CUDA_VISIBLE_DEVICES=$cuda_device python3 -W ignore $SPEAKER_TRAINER_ROOT/main.py \
          --train_list_path data/train_lst.csv \
          --trials_path data/trials/VoxCeleb1-Clean.lst \
          --n_mels 80 \
          --max_frames 201 --min_frames 200 \
          --batch_size 200 \
          --nPerSpeaker 1 \
          --max_seg_per_spk 500 \
          --num_workers 40 \
          --max_epochs 51 \
          --loss_type $loss_type \
          --nnet_type $nnet_type \
          --pooling_type $pooling_type \
          --embedding_dim $embedding_dim \
          --learning_rate 0.01 \
          --lr_step_size 5 \
          --lr_gamma 0.40 \
          --margin $margin \
          --scale $scale \
          --eval_interval 5 \
          --eval_frames 0 \
          --apply_metric \
          --save_top_k 10 \
          --distributed_backend dp \
          --reload_dataloaders_every_epoch \
          --gpus 3
fi


if [ $stage -eq 4 ];then
  # evaluation
  ckpt_path=exp/*/*.ckpt

  mkdir -p scores/
  echo Evaluate VoxCeleb1-Clean
  CUDA_VISIBLE_DEVICES=$cuda_device python3 -W ignore $SPEAKER_TRAINER_ROOT/main.py \
          --evaluate \
          --checkpoint_path $ckpt_path \
          --n_mels 80 \
          --trials_path data/trials/VoxCeleb1-Clean.lst \
          --scores_path scores/VoxCeleb1-Clean.foo \
          --apply_metric \
          --nnet_type $nnet_type \
          --pooling_type $pooling_type \
          --embedding_dim $embedding_dim \
          --scale $scale \
          --margin $margin \
          --num_workers 20 \
          --gpus 1

  echo Evaluate VoxCeleb1-H-Clean
  CUDA_VISIBLE_DEVICES=$cuda_device python3 -W ignore $SPEAKER_TRAINER_ROOT/main.py \
          --evaluate \
          --checkpoint_path $ckpt_path \
          --n_mels 80 \
          --trials_path data/trials/VoxCeleb1-H-Clean.lst \
          --scores_path scores/VoxCeleb1-H-Clean.foo \
          --apply_metric \
          --nnet_type $nnet_type \
          --pooling_type $pooling_type \
          --embedding_dim $embedding_dim \
          --scale $scale \
          --margin $margin \
          --num_workers 20 \
          --gpus 1

  echo Evaluate VoxCeleb1-E-Clean
  CUDA_VISIBLE_DEVICES=$cuda_device python3 -W ignore $SPEAKER_TRAINER_ROOT/main.py \
          --evaluate \
          --checkpoint_path $ckpt_path \
          --n_mels 80 \
          --trials_path data/trials/VoxCeleb1-E-Clean.lst \
          --scores_path scores/VoxCeleb1-E-Clean.foo \
          --apply_metric \
          --nnet_type $nnet_type \
          --pooling_type $pooling_type \
          --embedding_dim $embedding_dim \
          --scale $scale \
          --margin $margin \
          --num_workers 20 \
          --gpus 1

  echo Evaluate SITW-Dev-Core
  CUDA_VISIBLE_DEVICES=$cuda_device python3 -W ignore $SPEAKER_TRAINER_ROOT/main.py \
          --evaluate \
          --checkpoint_path $ckpt_path \
          --n_mels 80 \
          --trials_path data/trials/SITW-Dev-Core.lst \
          --scores_path scores/SITW-Dev-Core.foo \
          --apply_metric \
          --nnet_type $nnet_type \
          --pooling_type $pooling_type \
          --embedding_dim $embedding_dim \
          --scale $scale \
          --margin $margin \
          --num_workers 20 \
          --gpus 1

  echo Evaluate SITW-Eval-Core
  CUDA_VISIBLE_DEVICES=$cuda_device python3 -W ignore $SPEAKER_TRAINER_ROOT/main.py \
          --evaluate \
          --checkpoint_path $ckpt_path \
          --n_mels 80 \
          --trials_path data/trials/SITW-Eval-Core.lst \
          --scores_path scores/SITW-Eval-Core.foo \
          --apply_metric \
          --nnet_type $nnet_type \
          --pooling_type $pooling_type \
          --embedding_dim $embedding_dim \
          --scale $scale \
          --margin $margin \
          --num_workers 20 \
          --gpus 1
fi


if [ $stage -eq 5 ];then
  # An example of the C-P Map on VoxCeleb1-E trials.
  python3 $SPEAKER_TRAINER_ROOT/trainer/metric/compute_trial_config.py \
          --mode 0 \
          --input_scores_test scores/VoxCeleb1-E-Clean.foo \
          --scale 20 \
          --output scores/VoxCeleb1-E-Clean.cfg

  awk '{print $3}' scores/VoxCeleb1-E-Clean.cfg > scores/VoxCeleb1-E-Clean.eer.cfg
  python3 $SPEAKER_TRAINER_ROOT/trainer/metric/plot_cp_map.py \
          --mode 0 \
          --metric eer \
          --input_configs_test scores/VoxCeleb1-E-Clean.eer.cfg \
          --scale 20 \
          --savedir cpmap/

  awk '{print $4}' scores/VoxCeleb1-E-Clean.cfg > scores/VoxCeleb1-E-Clean.dcf.cfg
  python3 $SPEAKER_TRAINER_ROOT/trainer/metric/plot_cp_map.py \
          --mode 0 \
          --metric dcf \
          --input_configs_test scores/VoxCeleb1-E-Clean.dcf.cfg \
          --scale 20 \
          --savedir cpmap/
fi


#!/bin/bash

. ./path.sh

stage=$1

# configs of data prep
min_num_utts=5

# configs of NF
flow=$2         # e.g., realnvp, linear
num_blocks=$3   # no. of blocks
num_hidden=512
num_inputs=512
batch_size=$4   # e.g., 200
epochs=5001
lr=0.0005
device=$5       # gpu id
ckpt_save_interval=50
model=xvec_${flow}_b${num_blocks}_h${num_hidden}_b${batch_size}_e${epochs}_lr${lr}
ckpt_dir=ckpt/$model
log_dir=$ckpt_dir/log


# Throw out speakers with fewer than 5 utterances.
if [ $stage -le 0 ]; then
  raw_dir=data/x-vector/train_combined_200k
  new_dir=data/x-vector/train_combined_200k_utts_${min_num_utts}

  mkdir -p $new_dir
  awk '{print $1, NF-1}' $raw_dir/spk2utt > $raw_dir/spk2num
  awk -v min_num_utts=${min_num_utts} '$2 >= min_num_utts {print $1, $2}' $raw_dir/spk2num | local/filter_scp.pl - $raw_dir/spk2utt > $new_dir/spk2utt
  local/spk2utt_to_utt2spk.pl $new_dir/spk2utt > $new_dir/utt2spk
  local/filter_scp.pl $new_dir/utt2spk $raw_dir/xvector.scp > $new_dir/xvector.scp
fi


# Convert vector.scp to vector.npz
if [ $stage -le 1 ]; then
  python -u local/kaldi2npz.py \
         --src-file data/x-vector/train_combined_200k_utts_${min_num_utts}/xvector.scp \
         --dest-file data/x-vector/train_combined_200k_utts_${min_num_utts}/xvector.npz \
         --utt2spk-file data/x-vector/train_combined_200k_utts_${min_num_utts}/utt2spk

  for sub in sitw_dev_enroll sitw_dev_test sitw_eval_enroll sitw_eval_test; do
    python -u local/kaldi2npz.py \
           --is-eval \
           --src-file data/x-vector/$sub/xvector.scp \
           --dest-file data/x-vector/$sub/xvector.npz \
           --utt2spk-file data/x-vector/$sub/utt2spk
  done
fi


# Baseline
if [ $stage -le 2 ]; then
  score_dir=scores/baseline
  mkdir -p $score_dir
  data_dir=data/x-vector
  for core in sitw_dev sitw_eval; do
    echo $core
    local/score/plda_scoring.sh xvector.scp $data_dir/train_combined_200k_utts_${min_num_utts} \
                     $data_dir/${core}_enroll $data_dir/${core}_test \
                     $data_dir/${core}_test/core-core.lst $score_dir/$core

    local/score/lda_plda_scoring.sh --lda-dim 150 --covar-factor 0.0 \
                     xvector.scp $data_dir/train_combined_200k_utts_${min_num_utts} \
                     $data_dir/${core}_enroll $data_dir/${core}_test \
                     $data_dir/${core}_test/core-core.lst $score_dir/$core

    python -u local/score/score_by_trials.py \
           --train-npz $data_dir/train_combined_200k_utts_${min_num_utts}/xvector.npz \
           --enroll-npz $data_dir/${core}_enroll/xvector.npz \
           --enroll-num-utts $data_dir/${core}_enroll/num_utts.ark \
           --test-npz $data_dir/${core}_test/xvector.npz \
           --trials $data_dir/${core}_test/core-core.lst \
           --centering \
           --score $score_dir/$core/raw_nl.foo

    python -u local/score/score_by_trials.py \
           --train-npz $data_dir/train_combined_200k_utts_${min_num_utts}/xvector.npz \
           --enroll-npz $data_dir/${core}_enroll/xvector.npz \
           --enroll-num-utts $data_dir/${core}_enroll/num_utts.ark \
           --test-npz $data_dir/${core}_test/xvector.npz \
           --trials $data_dir/${core}_test/core-core.lst \
           --centering \
           --apply-lda \
           --lda-dim 512 \
           --score $score_dir/$core/lda_512_nl.foo

    python -u local/score/score_by_trials.py \
           --train-npz $data_dir/train_combined_200k_utts_${min_num_utts}/xvector.npz \
           --enroll-npz $data_dir/${core}_enroll/xvector.npz \
           --enroll-num-utts $data_dir/${core}_enroll/num_utts.ark \
           --test-npz $data_dir/${core}_test/xvector.npz \
           --trials $data_dir/${core}_test/core-core.lst \
           --centering \
           --apply-lda \
           --lda-dim 150 \
           --score $score_dir/$core/lda_150_nl.foo
  done
fi


# NDA training
if [ $stage -le 3 ]; then
  # NDA configs
  dataset=voxceleb1
  train_data_npz=data/x-vector/train_combined_200k_utts_${min_num_utts}/xvector.npz
  echo $model
  mkdir -p $log_dir
 
  echo "Start ..."

  # This may be a bug in pyTorch... you must assign GPU device before the main.py.
  CUDA_VISIBLE_DEVICES=$device python -u nda/main.py \
         --dataset-name $dataset \
         --train-data-npz $train_data_npz \
         --flow $flow \
         --num-blocks $num_blocks \
         --num-hidden $num_hidden \
         --num-inputs $num_inputs \
         --batch-size $batch_size \
         --epochs $epochs \
         --lr $lr \
         --device 0 \
         --ckpt-save-interval $ckpt_save_interval \
         --ckpt-dir $ckpt_dir \
         --log-dir $log_dir
  echo "End ..."
fi


# NDA inference
if [ $stage -le 4 ]; then
  # NDA configs
  echo $model
  echo "Start to infer data from x space to z space and store to numpy npz"
  for ((infer_epoch=0; infer_epoch<${epochs}; infer_epoch=infer_epoch+100)); do
    echo $infer_epoch
    for sub in train_combined_200k_utts_${min_num_utts} sitw_dev_enroll sitw_dev_test sitw_eval_enroll sitw_eval_test; do
      echo $sub
      test_data_npz=data/x-vector/$sub/xvector.npz
      npz_dir=$ckpt_dir/$infer_epoch/$sub

      # This may be a bug in pyTorch... you must assign GPU device before the main.py.
      CUDA_VISIBLE_DEVICES=$device python -u nda/main.py \
             --eval \
             --test-data-npz $test_data_npz \
             --flow $flow \
             --num-blocks $num_blocks \
             --num-hidden $num_hidden \
             --num-inputs $num_inputs \
             --batch-size 2000 \
             --device 0 \
             --ckpt-dir $ckpt_dir \
             --infer-epoch $infer_epoch \
             --npz-dir $npz_dir/xvector.npz

      cp data/x-vector/$sub/{spk2utt,utt2spk,num_utts.ark} $npz_dir/
      python -u local/npz2kaldi.py \
             --src-file $npz_dir/xvector.npz \
             --dest-file $npz_dir/xvector.tmp.ark
      copy-vector ark:$npz_dir/xvector.tmp.ark ark,scp:$npz_dir/xvector.ark,$npz_dir/xvector.scp
      rm -r $npz_dir/xvector.tmp.ark
    done
  done

  echo "End ..."
fi


# Back-end Scoring
if [ $stage -le 5 ]; then
  echo $model
  for ((infer_epoch=0; infer_epoch<${epochs}; infer_epoch=infer_epoch+100)); do
    echo $infer_epoch
    score_dir=scores/$model/$infer_epoch
    mkdir -p $score_dir
    data_dir=$ckpt_dir/$infer_epoch
    for core in sitw_dev sitw_eval; do
      echo $core
      local/score/plda_scoring.sh xvector.scp $data_dir/train_combined_200k_utts_${min_num_utts} \
                       $data_dir/${core}_enroll $data_dir/${core}_test \
                       data/x-vector/${core}_test/core-core.lst $score_dir/$core

      local/score/lda_plda_scoring.sh --lda-dim 150 --covar-factor 0.0 \
                       xvector.scp $data_dir/train_combined_200k_utts_${min_num_utts} \
                       $data_dir/${core}_enroll $data_dir/${core}_test \
                       data/x-vector/${core}_test/core-core.lst $score_dir/$core

      python -u local/score/score_by_trials.py \
             --train-npz $data_dir/train_combined_200k_utts_${min_num_utts}/xvector.npz \
             --enroll-npz $data_dir/${core}_enroll/xvector.npz \
             --enroll-num-utts $data_dir/${core}_enroll/num_utts.ark \
             --test-npz $data_dir/${core}_test/xvector.npz \
             --trials data/x-vector/${core}_test/core-core.lst \
             --centering \
             --score $score_dir/$core/raw_nl.foo

      python -u local/score/score_by_trials.py \
             --train-npz $data_dir/train_combined_200k_utts_${min_num_utts}/xvector.npz \
             --enroll-npz $data_dir/${core}_enroll/xvector.npz \
             --enroll-num-utts $data_dir/${core}_enroll/num_utts.ark \
             --test-npz $data_dir/${core}_test/xvector.npz \
             --trials data/x-vector/${core}_test/core-core.lst \
             --centering \
             --apply-lda \
             --lda-dim 512 \
             --score $score_dir/$core/lda_512_nl.foo

      python -u local/score/score_by_trials.py \
             --train-npz $data_dir/train_combined_200k_utts_${min_num_utts}/xvector.npz \
             --enroll-npz $data_dir/${core}_enroll/xvector.npz \
             --enroll-num-utts $data_dir/${core}_enroll/num_utts.ark \
             --test-npz $data_dir/${core}_test/xvector.npz \
             --trials data/x-vector/${core}_test/core-core.lst \
             --centering \
             --apply-lda \
             --lda-dim 150 \
             --score $score_dir/$core/lda_150_nl.foo
    done
  done
fi


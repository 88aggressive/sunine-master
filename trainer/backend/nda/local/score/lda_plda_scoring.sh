#!/bin/bash
# Copyright 2015   David Snyder
# Apache 2.0.
#
# This script trains PLDA models and does scoring.

lda_dim=150
covar_factor=0.0
normalize_length=false
simple_length_norm=false # If true, replace the default length normalization
                         # performed in PLDA  by an alternative that
                         # normalizes the length of the iVectors to be equal
                         # to the square root of the iVector dimension.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 6 ]; then
  echo "Usage: $0 <vec-type> <dev-dir> <enroll-dir> <test-dir> <trials-file> <scores-dir>"
fi

vec_type=$1
dev_dir=$2
enroll_dir=$3
test_dir=$4
trials=$5
scores_dir=$6

mkdir -p $dev_dir/log
run.pl $dev_dir/log/compute_mean.log \
  ivector-mean scp:$dev_dir/$vec_type $dev_dir/mean.vec || exit 1;

run.pl $dev_dir/log/lda.log \
  ivector-compute-lda --total-covariance-factor=$covar_factor --dim=$lda_dim \
  "ark:ivector-subtract-global-mean scp:$dev_dir/$vec_type ark:- |" \
  ark:$dev_dir/utt2spk \
  $dev_dir/lda.mat || exit 1;

run.pl $dev_dir/log/lda_plda.log \
  ivector-compute-plda ark:$dev_dir/spk2utt \
  "ark:ivector-subtract-global-mean scp:$dev_dir/$vec_type ark:- | transform-vec $dev_dir/lda.mat ark:- ark:- |" \
  $dev_dir/lda_plda || exit 1;

mkdir -p $scores_dir/log
run.pl $scores_dir/log/lda_plda_scoring.log \
  ivector-plda-scoring --normalize-length=$normalize_length \
    --simple-length-normalization=$simple_length_norm \
    --num-utts=ark:$enroll_dir/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 $dev_dir/lda_plda - |" \
    "ark:ivector-mean ark:$enroll_dir/spk2utt scp:$enroll_dir/$vec_type ark:- | ivector-subtract-global-mean $dev_dir/mean.vec ark:- ark:- | transform-vec $dev_dir/lda.mat ark:- ark:- |" \
    "ark:ivector-subtract-global-mean $dev_dir/mean.vec scp:$test_dir/$vec_type ark:- | transform-vec $dev_dir/lda.mat ark:- ark:- |" \
    "cat '$trials' | cut -d\  --fields=1,2 |" $scores_dir/lda_plda_scores || exit 1;

eer=$(paste $trials $scores_dir/lda_plda_scores | awk '{print $6, $3}' | compute-eer - 2>/dev/null)
echo "LDA_PLDA EER: $eer%"

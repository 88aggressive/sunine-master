#!/bin/bash
# Copyright 2015   David Snyder
# Copyright 2019   Lantian Li
# Apache 2.0.
#
# This script trains LDA models and get LDA-transformed vectors.

lda_dim=512
covar_factor=0.0
echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
  echo "Usage: $0 <train-xvec-dir> <test-xvec-dir> <save-xvec-dir>"
fi

train_xvec_dir=$1
test_xvec_dir=$2
save_xvec_dir=$3

for f in $train_xvec_dir/utt2spk $train_xvec_dir/xvector.scp $test_xvec_dir/xvector.scp; do
  [ ! -f $f ] && echo "No such file $f" && exit 1;
done

mkdir -p $save_xvec_dir

# Calculate mean x-vector
ivector-mean scp:$train_xvec_dir/xvector.scp $train_xvec_dir/mean.vec || exit 1;

# LDA training
ivector-compute-lda --total-covariance-factor=$covar_factor --dim=$lda_dim \
  "ark:ivector-subtract-global-mean scp:$train_xvec_dir/xvector.scp ark:- |" \
  ark:$train_xvec_dir/utt2spk $train_xvec_dir/lda.mat || exit 1;

# LDA transformation
ivector-subtract-global-mean $train_xvec_dir/mean.vec scp:$test_xvec_dir/xvector.scp ark:- | \
 transform-vec $train_xvec_dir/lda.mat ark:- ark,scp:$save_xvec_dir/xvector.ark,$save_xvec_dir/xvector.scp || exit 1;

echo Success.

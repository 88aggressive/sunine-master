import argparse

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch DNF')

    # controller
    parser.add_argument(
        '--dataset-name',
        default='voxceleb1',
        help='dataset name')

    parser.add_argument(
        '--eval',
        action='store_true',
        default=False,
        help='process on eval or train')

    parser.add_argument(
        '--ckpt-dir',
        default='ckpt',
        help='dir to save check points')


    # model options
    parser.add_argument(
        '--flow',
        default='realnvp',
        help='flow to use: maf | realnvp | glow')

    parser.add_argument(
        '--num-blocks',
        type=int,
        default=10,
        help='number of invertible blocks (default: 10)')

    parser.add_argument(
        '--num-inputs',
        type=int,
        default=-1,
        help='number of inputs')

    parser.add_argument(
        '--num-hidden',
        type=int,
        default=256,
        help='number of hidden units')

    parser.add_argument(
        '--lda-W',
        default='lda.W',
        help='file of lda.W')

    parser.add_argument(
        '--mean',
        default='mean.vec',
        help='file of mean.vec')

    parser.add_argument(
        '--lda-SB',
        default='lda.SB',
        help='file of lda.SB')


    # training options
    parser.add_argument(
        '--train-data-npz',
        default='data/feats.npz',
        help='path of training data npz')

    parser.add_argument(
        '--batch-size',
        type=int,
        default=10000,
        help='input batch size for training (default: 10000)')

    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='number of epochs to train (default: 100)')

    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='learning rate (default: 0.001)')

    parser.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA training')

    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        help='random seed (default: 1)')

    parser.add_argument(
        '--device',
        default='0',
        help='cuda visible devices (default: 0)')

    parser.add_argument(
        '--ckpt-save-interval',
        type=int,
        default=10,
        help='how many epochs to wait before saving models')

    parser.add_argument(
        '--log-dir',
        default='log',
        help='log-dir to save training status')


    # inference options
    parser.add_argument(
        '--test-data-npz',
        default='data/feats.npz',
        help='path of infer data npz')

    parser.add_argument(
        '--infer-epoch',
        type=int,
        default=-1,
        help='index of ckpt epoch to infer (default: 100)')

    parser.add_argument(
        '--npz-dir',
        default='npz_data',
        help='infer npz dir')

    parser.add_argument(
        '--SB-file',
        default='SB',
        help='store path of SB file')

    args = parser.parse_args()

    return args


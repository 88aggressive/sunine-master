import os
import re
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import kaldi_io
import numpy as np
from tqdm import tqdm
import nda as fnn
from data_loader import *
from monitor import *
from tensorboardX import SummaryWriter

pi = torch.from_numpy(np.array(np.pi))

class trainer(object):
    def __init__(self, args):
        self.args = args
        # init cuda
        args.cuda = not args.no_cuda and torch.cuda.is_available()
        if args.cuda:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device
            self.device = torch.device("cuda:" + args.device)
            torch.cuda.manual_seed(args.seed)
        else:
            self.device = torch.device("cpu")
            torch.manual_seed(args.seed)
        print("torch device: {}".format(self.device))

        # init model
        num_inputs = args.num_inputs
        num_hidden = args.num_hidden
        num_cond_inputs = None

        act = 'relu'
        assert act in ['relu', 'sigmoid', 'tanh']

        modules = []

        # normalization flow
        assert args.flow in ['maf', 'realnvp', 'glow', 'linear', 'LU']

        if args.flow == 'glow':
            mask = torch.arange(0, num_inputs) % 2
            mask = mask.to(self.device).float()

            print("Warning: Results for GLOW are not as good as for MAF yet.")
            for _ in range(args.num_blocks):
                modules += [
                    # fnn.BatchNormFlow(num_inputs),
                    fnn.LUInvertibleMM(num_inputs),
                    fnn.CouplingLayer(
                        num_inputs, num_hidden, mask, num_cond_inputs,
                        s_act='tanh', t_act='relu')
                ]
                mask = 1 - mask

        elif args.flow == 'realnvp':
            mask = torch.arange(0, num_inputs) % 2
            mask = mask.to(self.device).float()

            for _ in range(args.num_blocks):
                modules += [
                    fnn.CouplingLayer(
                        num_inputs, num_hidden, mask, num_cond_inputs,
                        s_act='tanh', t_act='relu'),
                    # fnn.BatchNormFlow(num_inputs)
                ]
                mask = 1 - mask

        elif args.flow == 'maf':
            for _ in range(args.num_blocks):
                modules += [
                    fnn.MADE(num_inputs, num_hidden, num_cond_inputs, act=act),
                    # fnn.BatchNormFlow(num_inputs),
                    fnn.Reverse(num_inputs)
                ]

        elif args.flow == 'linear':
            for _ in range(args.num_blocks):
                modules += [
                    fnn.LinearLayer(num_inputs),
                    fnn.Tanh()
                ]
            modules += [ fnn.LinearLayer(num_inputs) ]

        elif args.flow == 'linear_init':
            W = np.loadtxt(args.lda_W, dtype=np.float32) # load W as initial params
            b = np.loadtxt(args.mean, dtype=np.float32) # load b as initial params
            for _ in range(args.num_blocks):
                modules += [
                    fnn.LinearLayer(num_inputs, W, b),
                    fnn.Tanh()
                ]
            modules += [ fnn.LinearLayer(num_inputs, W, b) ]

        elif args.flow == 'LU':
            W = np.loadtxt(args.lda_W, dtype=np.float32) # load W as initial params
            b = np.loadtxt(args.mean, dtype=np.float32) # load b as initial params
            for _ in range(args.num_blocks):
                modules += [
                    # fnn.LUInvertibleMM(num_inputs),
                    fnn.LUInvertibleMM(num_inputs, W, b),
                    fnn.Tanh()
                ]
            modules += [ fnn.LUInvertibleMM(num_inputs, W, b) ]

        self.model = fnn.FlowSequential(*modules)
        # init class variance
        self.model.set_c_var(num_inputs, self.device, True)

        # c_var = np.loadtxt(args.lda_SB, dtype=np.float32) # load lda.SB as initial params
        # c_var = np.sqrt(c_var)
        # self.model.set_c_var(c_var)

        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    module.bias.data.fill_(0)

        # init optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)

        # tensorboardX
        self.writer = SummaryWriter(comment=args.log_dir)
        self.global_step = 0


    def train(self):
        '''training process'''
        args = self.args
        kwargs = {'num_workers': 8, 'pin_memory': True}

        num_inputs = args.num_inputs

        # init dataloader
        self.dataset = train_data_loader(
            data_npz_path=args.train_data_npz, dataset_name=args.dataset_name)

        # init model
        self.reload_checkpoint()
        self.model.to(self.device)
        self.model.train()

        print(num_inputs)
        print(np.shape(self.dataset.data)[1])
        assert num_inputs == np.shape(self.dataset.data)[1]

        # main to train
        start_epoch = self.epoch_idx

        if start_epoch >= args.epochs:
            print("Training Done.")
            return

        for idx in range(start_epoch, args.epochs):
            self.epoch_idx = idx
           
            # shuffle batch
            self.train_loader = torch.utils.data.DataLoader(
                self.dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

            # loss manager
            train_avg_loss = AverageMeter()
            train_log_probs_avg_loss = AverageMeter()
            train_log_jacob_avg_loss = AverageMeter()
            train_between_class_var = AverageMeter()
            train_between_cov_concentrate = AverageMeter()
            train_within_class_var = AverageMeter()

            f_log = open(args.log_dir + os.sep + 'train_loss.log', 'a')

            # tqdm
            pbar = tqdm(total=len(self.dataset.data))
            for batch_idx, (spk) in enumerate(self.train_loader):
                spk = spk.cpu().detach().numpy()
                label = []
                for i in range(len(spk)):
                    s = spk[i]
                    num = np.shape(self.dataset.spker_data[s])[0]
                    # prepare label
                    for n in range(num):
                        label.append(s)
                    # prepare data
                    if i == 0:
                        data = torch.from_numpy(self.dataset.spker_data[s])
                    else:
                        data = torch.cat((data, torch.from_numpy(self.dataset.spker_data[s])), 0)

                label = np.array(label)
                label = torch.from_numpy(label)
                label = label.to(self.device)
                data = data.to(self.device)

                # NDA Gaussion log-likehood
                self.optimizer.zero_grad()
                z, loss, loss_probs, loss_jacob = self.model.NDA_Gaussian_log_likelihood(data, label)
                loss.backward()
                self.optimizer.step()

                train_avg_loss.update(loss.item(), 1)
                train_log_probs_avg_loss.update(loss_probs.item(), 1)
                train_log_jacob_avg_loss.update(loss_jacob.item(), 1)

                with torch.no_grad():
                    between_class_var, between_cov_concentrate, within_class_var = self.model.var_statistics(data, label)
                train_between_class_var.update(between_class_var, 1)
                train_between_cov_concentrate.update(between_cov_concentrate, 1)
                train_within_class_var.update(within_class_var, 1)

                pbar.update(data.size(0))
                # pbar.set_description('LogP(x): {:.3f} LogP(z): {:.3f} LogDet: {:.3f} SB: {:.3f} SW: {:.3f}'.format(
                #     train_avg_loss.val, train_log_probs_avg_loss.val, train_log_jacob_avg_loss.val, between_class_var, within_class_var))
                pbar.set_description('LogP(x): {:.3f} LogP(z): {:.3f} LogDet: {:.3f} BCC: {:.3f} SW: {:.3f}'.format(
                    train_avg_loss.val, train_log_probs_avg_loss.val, train_log_jacob_avg_loss.val, train_between_cov_concentrate.val, train_within_class_var.val))

                self.writer.add_scalar('LogLL', loss.item(), self.global_step)
                self.writer.add_scalar('LogP', loss_probs.item(), self.global_step)
                self.writer.add_scalar('LogDet', loss_jacob.item(), self.global_step)
                self.writer.add_scalar('Between_var', between_class_var, self.global_step)
                self.writer.add_scalar('Between_cov_concentrate', between_cov_concentrate, self.global_step)
                self.writer.add_scalar('Within_var', within_class_var, self.global_step)
                # self.writer.add_scalar('B/W ratio', between_class_var / within_class_var, self.global_step)
                self.global_step += 1

                # average loss of mini-batch training process.
                f_log.write('LogLL = {:.6f}  LogP = {:.6f}  LogDet = {:.6f}\n'.format(
                    train_avg_loss.val, train_log_probs_avg_loss.val, train_log_jacob_avg_loss.val))

                # SB and SW in this batch
                f_log.write('B_cov_concentrate = {:.6f}  B_var = {:.6f}  W_var = {:.6f}\n'.format(
                    train_between_cov_concentrate.val, train_between_class_var.val, train_within_class_var.val))

            pbar.close()

            # average loss in this epoch
            print('Epoch {} : LogLL = {:.6f}  LogP = {:.6f}  LogDet = {:.6f}'.format(
                 idx, train_avg_loss.avg, train_log_probs_avg_loss.avg, train_log_jacob_avg_loss.avg))
            print('Epoch {} : BCC = {:.6f}  B_var = {:.6f}  W_var = {:.6f}\n'.format(
                 idx, train_between_cov_concentrate.avg, train_between_class_var.avg, train_within_class_var.avg))

            f_log.write('Epoch {} : LogLL = {:.6f}  LogP = {:.6f}  LogDet = {:.6f}\n'.format(
                 idx, train_avg_loss.avg, train_log_probs_avg_loss.avg, train_log_jacob_avg_loss.avg))
            f_log.write('Epoch {} : BCC = {:.6f}  B_var = {:.6f}  W_var = {:.6f}\n'.format(
                 idx, train_between_cov_concentrate.avg, train_between_class_var.avg, train_within_class_var.avg))

            np.set_printoptions(formatter={'float': '{: 0.6f}'.format})

            c_var = self.model.c_var.cpu().detach().numpy()
            f_log.write(str(np.array(c_var**2)) + '\n')

            for module in self.model.modules():
                if hasattr(module, 'b') and module.b is not None:
                    mean_vec = module.b.cpu().detach().numpy()
                    f_log.write(str(mean_vec) + '\n')

            f_log.close()

            if self.epoch_idx % args.ckpt_save_interval == 0:
                self.save_checkpoint()

        self.save_checkpoint()
        print("Training Done.")


    def infer(self):
        '''generate z and save as npz format'''
        args = self.args
        kwargs = {'num_workers': 12, 'pin_memory': True}
        # init model
        if args.infer_epoch == -1:
            self.reload_checkpoint()
        else:
            ckpt_path = '{}/ckpt_epoch{}.pt'.format(args.ckpt_dir, args.infer_epoch)
            assert os.path.exists(ckpt_path) == True
            checkpoint_dict = torch.load(ckpt_path, map_location=self.device)
            self.model.load_state_dict(checkpoint_dict['model'])
            print("successfully reload {} [model] to infer".format(ckpt_path))

        self.model.to(self.device)
        self.model.eval()

        # init dataset
        dataset = test_data_loader(
            data_npz_path=args.test_data_npz, dataset_name=args.dataset_name)
        test_loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False, **kwargs)

        pbar = tqdm(total=len(dataset.label))
        for batch_idx, (data, label) in enumerate(test_loader):
            with torch.no_grad():
                data = data.to(self.device)
                z, _ = self.model(data)
                z = z.cpu().detach().numpy()
                if batch_idx == 0:
                    vectors = z
                else:
                    vectors = np.vstack((vectors, z))
                pbar.update(data.size(0))
                pbar.set_description('vectors.npz generating {} / {}'.format(np.shape(vectors)[0], len(dataset.label)))
        pbar.close()

        if not os.path.exists(args.npz_dir):
            os.makedirs(os.path.dirname(args.npz_dir))
        npz_path = args.npz_dir
        np.savez(npz_path, vectors=vectors, spker_label=dataset.spker_label, utt_label=dataset.utt_label)
        print("successfully save npz in {}".format(npz_path))

        # save SB==c_var**2 to file
        c_var = self.model.c_var.cpu().detach().numpy()
        np.savetxt(args.SB_file, c_var**2, fmt="%f", delimiter=" ")
        print("save SB (c_var**2) to {}".format(args.SB_file))


    def reload_checkpoint(self):
        '''check if checkpoint exists and reload the lastest checkpoint'''
        args = self.args
        self.epoch_idx = 0
        if not os.path.exists(args.ckpt_dir):
            os.makedirs(args.ckpt_dir)
            print("can not find ckpt dir, and create {} dir".format(args.ckpt_dir))
            print("start to train from epoch 0...")
        else:
            files = os.listdir(args.ckpt_dir)
            ckpts = []
            for f in files:
                if (f.endswith(".pt")):
                    ckpts.append(f)
            # load the lastest ckpt
            if (len(ckpts)):
                for ckpt in ckpts:
                    ckpt_epoch = int(re.findall(r"\d+", ckpt)[0])
                    if ckpt_epoch > self.epoch_idx:
                        self.epoch_idx = ckpt_epoch

                checkpoint_dict = torch.load(
                    '{}/ckpt_epoch{}.pt'.format(args.ckpt_dir, self.epoch_idx), map_location=self.device)

                self.model.load_state_dict(checkpoint_dict['model'])
                print("sucessfully reload ckpt_epoch{}.pt [model]".format(self.epoch_idx))

                self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
                print("sucessfully reload ckpt_epoch{}.pt [optimizer]".format(self.epoch_idx))
                # load optimizer from no-cuda to cuda
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if torch.is_tensor(v):
                            state[k] = v.cuda()

                self.epoch_idx += 1
                self.container = None
            else:
                print("start to train from epoch 0...")


    def save_checkpoint(self):
        '''save checkpoints, including model, optimizer and class means'''
        args = self.args
        if not os.path.exists(args.ckpt_dir):
            os.makedirs(args.ckpt_dir)
            print("create ckpt dir {}".format(args.ckpt_dir))

        print("Saving model to {}/ckpt_epoch{}.pt\n".format(args.ckpt_dir, self.epoch_idx))
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, '{}/ckpt_epoch{}.pt'.format(args.ckpt_dir, self.epoch_idx))


if __name__ == "__main__":
    pass

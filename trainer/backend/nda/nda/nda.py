import math
import types
import numpy as np
import scipy as sp
import scipy.linalg
import torch
import torch.nn as nn
import torch.nn.functional as F

pi = torch.from_numpy(np.array(np.pi))


def get_mask(in_features, out_features, in_flow_features, mask_type=None):
    """
    mask_type: input | None | output

    See Figure 1 for a better illustration:
    https://arxiv.org/pdf/1502.03509.pdf
    """
    if mask_type == 'input':
        in_degrees = torch.arange(in_features) % in_flow_features
    else:
        in_degrees = torch.arange(in_features) % (in_flow_features - 1)

    if mask_type == 'output':
        out_degrees = torch.arange(out_features) % in_flow_features - 1
    else:
        out_degrees = torch.arange(out_features) % (in_flow_features - 1)

    return (out_degrees.unsqueeze(-1) >= in_degrees.unsqueeze(0)).float()


class MaskedLinear(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 mask,
                 cond_in_features=None,
                 bias=True):
        super(MaskedLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        if cond_in_features is not None:
            self.cond_linear = nn.Linear(
                cond_in_features, out_features, bias=False)

        self.register_buffer('mask', mask)

    def forward(self, inputs, cond_inputs=None):
        output = F.linear(inputs, self.linear.weight * self.mask,
                          self.linear.bias)
        if cond_inputs is not None:
            output += self.cond_linear(cond_inputs)
        return output


nn.MaskedLinear = MaskedLinear


class MADESplit(nn.Module):
    """ An implementation of MADE
    (https://arxiv.org/abs/1502.03509).
    """

    def __init__(self,
                 num_inputs,
                 num_hidden,
                 num_cond_inputs=None,
                 s_act='tanh',
                 t_act='relu',
                 pre_exp_tanh=False):
        super(MADESplit, self).__init__()

        self.pre_exp_tanh = pre_exp_tanh

        activations = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh}

        input_mask = get_mask(num_inputs, num_hidden, num_inputs,
                              mask_type='input')
        hidden_mask = get_mask(num_hidden, num_hidden, num_inputs)
        output_mask = get_mask(num_hidden, num_inputs, num_inputs,
                               mask_type='output')

        act_func = activations[s_act]
        self.s_joiner = nn.MaskedLinear(num_inputs, num_hidden, input_mask,
                                        num_cond_inputs)

        self.s_trunk = nn.Sequential(act_func(),
                                     nn.MaskedLinear(num_hidden, num_hidden,
                                                     hidden_mask), act_func(),
                                     nn.MaskedLinear(num_hidden, num_inputs,
                                                     output_mask))

        act_func = activations[t_act]
        self.t_joiner = nn.MaskedLinear(num_inputs, num_hidden, input_mask,
                                        num_cond_inputs)

        self.t_trunk = nn.Sequential(act_func(),
                                     nn.MaskedLinear(num_hidden, num_hidden,
                                                     hidden_mask), act_func(),
                                     nn.MaskedLinear(num_hidden, num_inputs,
                                                     output_mask))

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            h = self.s_joiner(inputs, cond_inputs)
            m = self.s_trunk(h)

            h = self.t_joiner(inputs, cond_inputs)
            a = self.t_trunk(h)

            if self.pre_exp_tanh:
                a = torch.tanh(a)

            u = (inputs - m) * torch.exp(-a)
            return u, -a.sum(-1, keepdim=True)

        else:
            x = torch.zeros_like(inputs)
            for i_col in range(inputs.shape[1]):
                h = self.s_joiner(x, cond_inputs)
                m = self.s_trunk(h)

                h = self.t_joiner(x, cond_inputs)
                a = self.t_trunk(h)

                if self.pre_exp_tanh:
                    a = torch.tanh(a)

                x[:, i_col] = inputs[:, i_col] * torch.exp(
                    a[:, i_col]) + m[:, i_col]
            return x, -a.sum(-1, keepdim=True)


class MADE(nn.Module):
    """ An implementation of MADE
    (https://arxiv.org/abs/1502.03509).
    """

    def __init__(self,
                 num_inputs,
                 num_hidden,
                 num_cond_inputs=None,
                 act='relu',
                 pre_exp_tanh=False):
        super(MADE, self).__init__()

        activations = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh}
        act_func = activations[act]

        input_mask = get_mask(
            num_inputs, num_hidden, num_inputs, mask_type='input')
        hidden_mask = get_mask(num_hidden, num_hidden, num_inputs)
        output_mask = get_mask(
            num_hidden, num_inputs * 2, num_inputs, mask_type='output')

        self.joiner = nn.MaskedLinear(num_inputs, num_hidden, input_mask,
                                      num_cond_inputs)

        self.trunk = nn.Sequential(act_func(),
                                   nn.MaskedLinear(num_hidden, num_hidden,
                                                   hidden_mask), act_func(),
                                   nn.MaskedLinear(num_hidden, num_inputs * 2,
                                                   output_mask))

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            h = self.joiner(inputs, cond_inputs)
            m, a = self.trunk(h).chunk(2, 1)
            u = (inputs - m) * torch.exp(-a)
            return u, -a.sum(-1, keepdim=True)

        else:
            x = torch.zeros_like(inputs)
            for i_col in range(inputs.shape[1]):
                h = self.joiner(x, cond_inputs)
                m, a = self.trunk(h).chunk(2, 1)
                x[:, i_col] = inputs[:, i_col] * torch.exp(
                    a[:, i_col]) + m[:, i_col]
            return x, -a.sum(-1, keepdim=True)


class Sigmoid(nn.Module):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            s = torch.sigmoid
            return s(inputs), torch.log(s(inputs) * (1 - s(inputs))).sum(
                -1, keepdim=True)
        else:
            return torch.log(inputs /
                             (1 - inputs)), -torch.log(inputs - inputs**2).sum(
                                 -1, keepdim=True)


class Tanh(nn.Module):
    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            u  = torch.tanh(inputs)
            log_det = torch.log(1 - u**2 + 1e-40).sum(-1, keepdim=True)
            return u,  log_det
        else:
            x = 0.5 * torch.log((1 + inputs) / (1 - inputs))
            log_det = -torch.log(1 - inputs**2 + 1e-40).sum(-1, keepdim=True)
            return x,  log_det


class Logit(Sigmoid):
    def __init__(self):
        super(Logit, self).__init__()

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            return super(Logit, self).forward(inputs, 'inverse')
        else:
            return super(Logit, self).forward(inputs, 'direct')


class BatchNormFlow(nn.Module):
    """ An implementation of a batch normalization layer from
    Density estimation using Real NVP
    (https://arxiv.org/abs/1605.08803).
    """

    def __init__(self, num_inputs, momentum=0.0, eps=1e-5):
        super(BatchNormFlow, self).__init__()

        self.log_gamma = nn.Parameter(torch.zeros(num_inputs))
        self.beta = nn.Parameter(torch.zeros(num_inputs))
        self.momentum = momentum
        self.eps = eps

        self.register_buffer('running_mean', torch.zeros(num_inputs))
        self.register_buffer('running_var', torch.ones(num_inputs))

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            if self.training:
                self.batch_mean = inputs.mean(0)
                self.batch_var = (
                    inputs - self.batch_mean).pow(2).mean(0) + self.eps

                self.running_mean.mul_(self.momentum)
                self.running_var.mul_(self.momentum)

                self.running_mean.add_(self.batch_mean.data *
                                       (1 - self.momentum))
                self.running_var.add_(self.batch_var.data *
                                      (1 - self.momentum))

                mean = self.batch_mean
                var = self.batch_var
            else:
                mean = self.running_mean
                var = self.running_var

            x_hat = (inputs - mean) / var.sqrt()
            y = torch.exp(self.log_gamma) * x_hat + self.beta
            return y, (self.log_gamma - 0.5 * torch.log(var)).sum(
                -1, keepdim=True)
        else:
            if self.training:
                mean = self.batch_mean
                var = self.batch_var
            else:
                mean = self.running_mean
                var = self.running_var

            x_hat = (inputs - self.beta) / torch.exp(self.log_gamma)

            y = x_hat * var.sqrt() + mean

            return y, (-self.log_gamma + 0.5 * torch.log(var)).sum(
                -1, keepdim=True)


class ActNorm(nn.Module):
    """ An implementation of a activation normalization layer
    from Glow: Generative Flow with Invertible 1x1 Convolutions
    (https://arxiv.org/abs/1807.03039).
    """

    def __init__(self, num_inputs):
        super(ActNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(num_inputs))
        self.bias = nn.Parameter(torch.zeros(num_inputs))
        self.initialized = False

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if self.initialized == False:
            self.weight.data.copy_(torch.log(1.0 / (inputs.std(0) + 1e-12)))
            self.bias.data.copy_(inputs.mean(0))
            self.initialized = True

        if mode == 'direct':
            return (
                inputs - self.bias) * torch.exp(self.weight), self.weight.sum(
                    -1, keepdim=True).unsqueeze(0).repeat(inputs.size(0), 1)
        else:
            return inputs * torch.exp(
                -self.weight) + self.bias, -self.weight.sum(
                    -1, keepdim=True).unsqueeze(0).repeat(inputs.size(0), 1)


class InvertibleMM(nn.Module):
    """ An implementation of a invertible matrix multiplication
    layer from Glow: Generative Flow with Invertible 1x1 Convolutions
    (https://arxiv.org/abs/1807.03039).
    """

    def __init__(self, num_inputs):
        super(InvertibleMM, self).__init__()
        self.W = nn.Parameter(torch.Tensor(num_inputs, num_inputs))
        nn.init.orthogonal_(self.W)

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            return inputs @ self.W, torch.slogdet(
                self.W)[-1].unsqueeze(0).unsqueeze(0).repeat(
                    inputs.size(0), 1)
        else:
            return inputs @ torch.inverse(self.W), -torch.slogdet(
                self.W)[-1].unsqueeze(0).unsqueeze(0).repeat(
                    inputs.size(0), 1)


class LUInvertibleMM(nn.Module):
    """ An implementation of a invertible matrix multiplication
    layer from Glow: Generative Flow with Invertible 1x1 Convolutions
    (https://arxiv.org/abs/1807.03039).
    """

    # def __init__(self, num_inputs):
    def __init__(self, num_inputs, W, b):
        super(LUInvertibleMM, self).__init__()
        # self.W = torch.Tensor(num_inputs, num_inputs)
        self.W = torch.from_numpy(W)
        # self.b = nn.Parameter(torch.zeros(num_inputs))
        self.b = nn.Parameter(torch.from_numpy(b))

        # nn.init.orthogonal_(self.W)
        self.L_mask = torch.tril(torch.ones(self.W.size()), -1)
        self.U_mask = self.L_mask.t().clone()

        P, L, U = sp.linalg.lu(self.W.numpy())
        self.P = torch.from_numpy(P)
        self.L = nn.Parameter(torch.from_numpy(L))
        self.U = nn.Parameter(torch.from_numpy(U))
        S = np.diag(U)
        sign_S = np.sign(S)
        log_S = np.log(abs(S))
        self.sign_S = torch.from_numpy(sign_S)
        self.log_S = nn.Parameter(torch.from_numpy(log_S))

        self.I = torch.eye(self.L.size(0))

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if str(self.L_mask.device) != str(self.L.device):
            self.L_mask = self.L_mask.to(self.L.device)
            self.U_mask = self.U_mask.to(self.L.device)
            self.I = self.I.to(self.L.device)
            self.P = self.P.to(self.L.device)
            self.sign_S = self.sign_S.to(self.L.device)

        L = self.L * self.L_mask + self.I
        U = self.U * self.U_mask + torch.diag(
            self.sign_S * torch.exp(self.log_S))
        W = self.P @ L @ U

        if mode == 'direct':
            return inputs @ W + self.b, self.log_S.sum().unsqueeze(0).unsqueeze(
                0).repeat(inputs.size(0), 1)
        else:
            return (inputs - self.b) @ torch.inverse(
                W), -self.log_S.sum().unsqueeze(0).unsqueeze(0).repeat(
                    inputs.size(0), 1)


class Shuffle(nn.Module):
    """ An implementation of a shuffling layer from
    Density estimation using Real NVP
    (https://arxiv.org/abs/1605.08803).
    """

    def __init__(self, num_inputs):
        super(Shuffle, self).__init__()
        self.perm = np.random.permutation(num_inputs)
        self.inv_perm = np.argsort(self.perm)

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            return inputs[:, self.perm], torch.zeros(
                inputs.size(0), 1, device=inputs.device)
        else:
            return inputs[:, self.inv_perm], torch.zeros(
                inputs.size(0), 1, device=inputs.device)


class Reverse(nn.Module):
    """ An implementation of a reversing layer from
    Density estimation using Real NVP
    (https://arxiv.org/abs/1605.08803).
    """

    def __init__(self, num_inputs):
        super(Reverse, self).__init__()
        self.perm = np.array(np.arange(0, num_inputs)[::-1])
        self.inv_perm = np.argsort(self.perm)

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            return inputs[:, self.perm], torch.zeros(
                inputs.size(0), 1, device=inputs.device)
        else:
            return inputs[:, self.inv_perm], torch.zeros(
                inputs.size(0), 1, device=inputs.device)


class CouplingLayer(nn.Module):
    """ An implementation of a coupling layer
    from RealNVP (https://arxiv.org/abs/1605.08803).
    """

    def __init__(self,
                 num_inputs,
                 num_hidden,
                 mask,
                 num_cond_inputs=None,
                 s_act='tanh',
                 t_act='relu'):
        super(CouplingLayer, self).__init__()

        self.num_inputs = num_inputs
        self.mask = mask

        activations = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh}
        s_act_func = activations[s_act]
        t_act_func = activations[t_act]

        if num_cond_inputs is not None:
            total_inputs = num_inputs + num_cond_inputs
        else:
            total_inputs = num_inputs

        self.scale_net = nn.Sequential(
            nn.Linear(total_inputs, num_hidden), s_act_func(),
            nn.Linear(num_hidden, num_hidden), s_act_func(),
            nn.Linear(num_hidden, num_inputs))
        self.translate_net = nn.Sequential(
            nn.Linear(total_inputs, num_hidden), t_act_func(),
            nn.Linear(num_hidden, num_hidden), t_act_func(),
            nn.Linear(num_hidden, num_inputs))

        def init(m):
            if isinstance(m, nn.Linear):
                m.bias.data.fill_(0)
                nn.init.orthogonal_(m.weight.data)

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        mask = self.mask

        masked_inputs = inputs * mask
        if cond_inputs is not None:
            masked_inputs = torch.cat([masked_inputs, cond_inputs], -1)

        if mode == 'direct':
            log_s = self.scale_net(masked_inputs) * (1 - mask)
            t = self.translate_net(masked_inputs) * (1 - mask)
            s = torch.exp(log_s)
            return inputs * s + t, log_s.sum(-1, keepdim=True)
        else:
            log_s = self.scale_net(masked_inputs) * (1 - mask)
            t = self.translate_net(masked_inputs) * (1 - mask)
            s = torch.exp(-log_s)
            return (inputs - t) * s, -log_s.sum(-1, keepdim=True)


class LinearLayer(nn.Module):
    def __init__(self, num_inputs):
        super(LinearLayer, self).__init__()

        self.num_inputs = num_inputs
        self.tri1 = nn.Parameter(torch.eye(num_inputs, num_inputs), requires_grad=True)
        self.tri2 = nn.Parameter(torch.eye(num_inputs, num_inputs), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(num_inputs), requires_grad=True)

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            w = torch.tril(self.tri1).mm(torch.triu(self.tri2))
            u = (inputs - self.b).mm(w)
            log_det = torch.log(torch.abs(torch.diag(self.tri1) * torch.diag(self.tri2)) + torch.ones(self.num_inputs).to(inputs.device) * 1e-5).sum(-1, keepdim=True)
            return u, log_det
        else:
            w = torch.inverse(self.tri1.mm(self.tri2))
            x = inputs.mm(w) + self.b
            log_det = -torch.log(torch.abs(torch.diag(self.tri1) * torch.diag(self.tri2)) + torch.ones(self.num_inputs).to(inputs.device) * 1e-5).sum(-1, keepdim=True)
            return x, log_det


class LinearLayer_init(nn.Module):
    def __init__(self, num_inputs, W, b):
        super(LinearLayer, self).__init__()
        self.num_inputs = num_inputs

        self.W = torch.from_numpy(W.T)
        self.b = nn.Parameter(torch.zeros(num_inputs))
        # self.b = nn.Parameter(torch.from_numpy(b), requires_grad = False)
        L, U = sp.linalg.lu(self.W.numpy(), permute_l=True)
        self.tri1 = nn.Parameter(torch.from_numpy(L))
        self.tri2 = nn.Parameter(torch.from_numpy(U))

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            w = self.tri1 @ self.tri2
            u = (inputs - self.b).mm(w)
            log_det = torch.log(torch.abs(torch.diag(self.tri1) * torch.diag(self.tri2)) + torch.ones(self.num_inputs).to(inputs.device) * 1e-5).sum(-1, keepdim=True)
            return u, log_det
        else:
            w = torch.inverse(self.tri1 @ self.tri2)
            x = inputs.mm(w) + self.b
            log_det = -torch.log(torch.abs(torch.diag(self.tri1) * torch.diag(self.tri2)) + torch.ones(self.num_inputs).to(inputs.device) * 1e-5).sum(-1, keepdim=True)
            return x, log_det


class FlowSequential(nn.Sequential):
    """ A sequential container for flows.
    In addition to a forward pass it implements a backward pass and
    computes log jacobians.
    """

    def set_c_var(self, n_dim, device, required_grad = False):
        self.c_var  = nn.Parameter(torch.tensor([1.0 for i in range(n_dim)], device = device), requires_grad = required_grad)


    # def set_c_var(self, c_var):
    #     self.c_var  = nn.Parameter(torch.from_numpy(c_var))


    def print_c_var(self):
        print(self.c_var**2)


    def forward(self, inputs, cond_inputs=None, mode='direct', logdets=None):
        """ Performs a forward or backward pass for flow modules.
        Args:
            inputs: a tuple of inputs and logdets
            mode: to run direct computation or inverse
        """
        self.num_inputs = inputs.size(-1)

        if logdets is None:
            logdets = torch.zeros(inputs.size(0), 1, device=inputs.device)

        assert mode in ['direct', 'inverse']
        if mode == 'direct':
            for module in self._modules.values():
                inputs, logdet = module(inputs, cond_inputs, mode)
                logdets += logdet
        else:
            for module in reversed(self._modules.values()):
                inputs, logdet = module(inputs, cond_inputs, mode)
                logdets += logdet

        return inputs, logdets


    def NDA_Gaussian_log_likelihood(self, inputs, labels):
        device = inputs.device

        var = self.c_var ** 2

        # forward pass
        u, logdet = self(inputs)
        samples = u.shape[0]

        # num of classes
        num_class = np.unique(labels.cpu().detach().numpy())

        # NDA loss
        log_probs = 0.0
        for c in num_class:
            data_c = u[ labels == c ]
            n = data_c.shape[0]
            x1 = torch.sum(data_c ** 2, 0)
            x2 = torch.pow(torch.mean(data_c, 0), 2)
            log_post =  - 0.5 * (torch.sum( x1 - n**2 * var / ( n * var + 1.0 ) * x2 ))
            log_probs += log_post - 0.5 * torch.sum(torch.log(var)) - 0.5 * torch.sum(torch.log(n + 1 / var))
        
        loss = -(log_probs / samples + logdet.mean())
        return u, loss, log_probs / samples, logdet.mean()


    def compute_class_mean_var(self, inputs, labels):
        assert(inputs.shape[0] == labels.shape[0])

        # num of classes
        num_class = np.unique(labels.cpu().detach().numpy())

        # compute mean and variance of each class
        mean_list = []
        var_list  = []
        for c in num_class:
            inputs_c = inputs[ labels == c ]
            # means
            mean_c = torch.mean(inputs_c, dim = 0, keepdim = True)
            mean_list.append(mean_c)
            # vars
            var_c = torch.var(inputs_c, dim = 0, keepdim = True)
            var_list.append(var_c)

        class_mean = torch.cat(mean_list, 0)
        class_var  = torch.cat(var_list, 0)
        return class_mean, class_var
    

    def var_statistics(self, inputs, labels):
        u, logdet = self(inputs)

        # get the mean and variance of each class
        class_mean, class_var = self.compute_class_mean_var(u, labels)

        # compute concentration
        cov = np.cov(class_mean.cpu().detach().numpy().T)
        between_cov_concentrate = np.sum(cov.diagonal()**2) / np.sum(cov**2)
        # print("Between_class_Cov Concentration ", between_cov_concentrate)

        # compute between-class variation
        between_class_var = torch.mean(torch.var(class_mean, 0))
        # compute within-class variation 
        within_class_var = torch.mean(class_var)

        return between_class_var, between_cov_concentrate, within_class_var


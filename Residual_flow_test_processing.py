import datetime
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import pickle
import json

import models

import data_loader
import os
import lib_generation
from torchvision import transforms

device = torch.device('cuda')

import argparse

parser = argparse.ArgumentParser(description='PyTorch code: Residual flow detector test')
parser.add_argument('--cuda_index', type=int, default=0, help='index of CUDA device, default value 0')
parser.add_argument('--batch_size', type=int, default=32, metavar='N', help='batch size for data loader')
parser.add_argument('--dataroot', default='./data', help='path to dataset')
parser.add_argument('--outf', default='output', help='folder to output results')
parser.add_argument('--net_type', required=True, help='resnet | densenet')
parser.add_argument('--num_classes', default=10, help='number of classes')
parser.add_argument('--dataset', required=True, help='cifar10 | cifar100 | svhn')
parser.add_argument('--validation_src', default='IO', help='IO | FGSM (choice of validation source for hyper-parameter tuning: IO for in- and out-of-distribution, or FGSM for adverarial validation)')
args = parser.parse_args()
print('Running with CUDA {}, net type {},batch_size {}'.format(args.cuda_index, args.net_type, args.batch_size))

class RealNVP(nn.Module):
    def __init__(self, mask, num_features, length_hidden, A, A_inv, log_abs_det_A_inv):
        super(RealNVP, self).__init__()

        self.mask = nn.Parameter(mask, requires_grad=False)
        self.t = torch.nn.ModuleList([Nett(num_features, length_hidden) for _ in range(len(mask))])
        self.s = torch.nn.ModuleList([Nets(num_features, length_hidden) for _ in range(len(mask))])
        # self.bn_flow = torch.nn.ModuleList([BatchNormStats1d(num_features) for _ in range(len(mask))])
        self.perm = torch.nn.ModuleList([Permutation(num_features) for _ in range(int(len(mask)/2))])
        self.A_ = nn.Parameter(A_inv, requires_grad=False)
        self.A = nn.Parameter(A, requires_grad=False)
        self.log_abs_det_A_ = log_abs_det_A_inv

        for i in range(len(mask)):
            self.t[i].fc3.weight.data.fill_(0)
            self.t[i].fc3.bias.data.fill_(0)
            self.s[i].fc3.weight.data.fill_(0)
            self.s[i].fc3.bias.data.fill_(0)


    def g(self, z, training):
        x = z.cuda()
        zeros = torch.cuda.FloatTensor(x.shape).fill_(0)
        for i in range(len(self.t)):
            if (i % 2 > 0):
                # x, var = self.bn_flow[i].forward(x, training, inverse=True)
                x = self.perm[i].forward(x, inverse=False)

            x_ = x * self.mask[i]
            s = self.s[i](x_) * (1 - self.mask[i])
            t = self.t[i](x_) * (1 - self.mask[i])
            x =  x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)

        x = torch.mm(self.A,x.transpose(1,0)).transpose(1,0)
        return x


    def f(self, x, training):
        log_det_J, z = torch.cuda.FloatTensor(x.shape[0]).fill_(0), x.cuda()

        z = torch.mm(self.A_,z.transpose(1,0)).transpose(1,0)
        log_det_J += self.log_abs_det_A_

        for i in reversed(range(len(self.t))):

            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1 - self.mask[i])
            t = self.t[i](z_) * (1 - self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-1*s) + z_
            log_det_J -= s.sum(dim=1)

            if (i % 2 == 0):
                j = int(i/2)
                z = self.perm[j].forward(z, inverse=False)

                # if j > 0:
                #     z = self.perm[j-1].forward(z, inverse=True)
                # z, var, weights = self.bn_flow[i].forward(z, training, inverse=False)
                # log_det_J -= 0.5*torch.log(var).sum(dim=1)
                # log_det_J += torch.log(weights).sum(dim=1)

        return z, log_det_J


    def log_prob(self, x, training):
        z, logp = self.f(x.cuda(), training)
        logp_z = -0.5 * (z.size(1)) * math.log(2 * math.pi) * torch.cuda.FloatTensor(z.size(0)).fill_(1) - 0.5 * ((z ** 2).sum(dim=1))
        return logp_z + logp

    def sample(self, batchSize):
        # z = self.prior.sample((batchSize, 1))
        z = torch.cuda.FloatTensor(batchSize, 2).normal_(mean=0, std=1)
        logp = -0.5 * (z.size(1)) * math.log(2 * math.pi) * torch.cuda.FloatTensor(z.size(0)).fill_(1) - 0.5 * ((z ** 2).sum(dim=1))
        x = self.g(z, False)
        return x


class Nets(nn.Module):
    def __init__(self,num_features, length_hidden):
        super().__init__()
        # self.net = nn.Linear(num_features, num_features)
        self.fc1 = nn.Linear(num_features, int(length_hidden*num_features))
        self.fc2 = nn.Linear(int(length_hidden*num_features), int(length_hidden*num_features))
        self.fc3 = nn.Linear(int(length_hidden*num_features), num_features)
        # self.fc4 = nn.Linear(int(length_hidden * num_features), int(length_hidden*num_features))
        # self.fc5 = nn.Linear(int(length_hidden * num_features), num_features)
        # self.fc6 = nn.Linear(int(length_hidden * num_features), num_features)
        self.rescale = nn.utils.weight_norm(Rescale(num_features))

    def forward(self, x):
        # x_ = self.net(x)
        x_ = self.fc1(x)
        x_ = F.leaky_relu(x_, inplace=True)
        x_ = self.fc2(x_)
        x_ = F.leaky_relu(x_, inplace=True)
        x_ = self.fc3(x_)
        # x_ = F.leaky_relu(x_, inplace=True)
        # x_ = self.fc4(x_)
        # x_ = F.leaky_relu(x_, inplace=True)
        # x_ = self.fc5(x_)
        # x_ = F.leaky_relu(x_, inplace=True)
        # x_ = self.fc6(x_)
        x_ = self.rescale(torch.tanh(x_))
        return x_

class Nett(nn.Module):
    def __init__(self,num_features, length_hidden):
        super(Nett, self).__init__()
        self.fc1 = nn.Linear(num_features, int(length_hidden*num_features))
        self.fc2 = nn.Linear(int(length_hidden*num_features), int(length_hidden*num_features))
        self.fc3 = nn.Linear(int(length_hidden*num_features), num_features)
        # self.fc4 = nn.Linear(int(length_hidden * num_features), int(length_hidden*num_features))
        # self.fc5 = nn.Linear(int(length_hidden * num_features), num_features)
        # self.fc6 = nn.Linear(int(length_hidden * num_features), num_features)


    def forward(self, x):
        x_ = self.fc1(x)
        x_ = F.leaky_relu(x_, inplace=True)
        x_ = self.fc2(x_)
        x_ = F.leaky_relu(x_, inplace=True)
        x_ = self.fc3(x_)
        # x_ = F.leaky_relu(x_, inplace=True)
        # x_ = self.fc4(x_)
        # x_ = F.leaky_relu(x_, inplace=True)
        # x_ = self.fc5(x_)
        # x_ = F.leaky_relu(x_, inplace=True)
        # x_ = self.fc6(x_)
        return x_


class Rescale(nn.Module):
    """Per-channel rescaling. Need a proper `nn.Module` so we can wrap it
    with `torch.nn.utils.weight_norm`.
    Args:
        num_channels (int): Number of channels in the input.
    """
    def __init__(self, num_features):
        super(Rescale, self).__init__()
        self.weight = nn.Parameter(torch.ones(1, num_features))

    def forward(self, x):
        x = self.weight * x
        return x

class Nett_linear(nn.Module):
    def __init__(self,num_features):
        super(Nett_linear, self).__init__()
        self.net = nn.Linear(num_features, num_features, bias=False)

    def forward(self, x):
        x_ = self.net(x)
        return x_

class Nets_linear(nn.Module):
    def __init__(self,num_features):
        super(Nets_linear, self).__init__()
        self.net = nn.Linear(num_features, num_features)

    def forward(self, x):
        x_ = self.net(x)
        return x_


class BatchNormStats1d(nn.Module):
    """Compute BatchNorm1d normalization statistics: `mean` and `var`.
    Useful for keeping track of sum of log-determinant of Jacobians in flow models.
    Args:
        num_features (int): Number of features in the input.
        eps (float): Added to the denominator for numerical stability.
        decay (float): The value used for the running_mean and running_var computation.
            Different from conventional momentum, see `nn.BatchNorm1d` for more.
    """
    def __init__(self, num_features, eps=1e-5, decay=0.1):
        super(BatchNormStats1d, self).__init__()
        self.eps = eps

        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.weights = nn.Parameter(torch.ones(1,num_features), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(1,num_features), requires_grad=True)
        self.decay = decay
        self.init = True

    def forward(self, x, training, inverse):
        # Get mean and variance per channel
        if self.init == True:
            init_mean, init_var = x.mean(0), x.var(0)
            self.weights.data = init_var.sqrt()
            self.bias.data = init_mean
            self.running_mean = init_mean
            self.running_var = init_var
            self.init = False

        if training:
            used_mean, used_var = x.mean(0), x.var(0)
            curr_mean, curr_var = used_mean, used_var

            # Update variables
            tmp_running_mean = self.running_mean - self.decay * (self.running_mean - curr_mean)
            tmp_running_var = self.running_var - self.decay * (self.running_var - curr_var)

            self.running_mean = tmp_running_mean.detach().clone()
            self.running_var = tmp_running_var.detach().clone()

        else:
            used_mean = self.running_mean.detach().clone()
            used_var = self.running_var.detach().clone()

        # used_var += self.eps

        # Reshape
        used_mean = used_mean.view(1, x.size(1)).expand_as(x)
        used_var = used_var.view(1, x.size(1)).expand_as(x)

        used_weights = self.weights
        used_bias = self.bias

        used_weights = used_weights.view(1, x.size(1)).expand_as(x)
        used_bias = used_bias.view(1, x.size(1)).expand_as(x)
        if inverse:
            x = (x - used_bias) / used_weights
            x = x * used_var.sqrt()  + used_mean
        else:
            x = (x - used_mean) / used_var.sqrt()
            x = used_weights * x + used_bias

        return x, used_var, used_weights


class Permutation(nn.Module):
    """Permutation matrix with log determinant of zero.
    Args:
        num_channels (int): Number of channels in the input.
    """
    def __init__(self, num_features):
        super(Permutation, self).__init__()
        p = torch.randperm(num_features)
        self.register_buffer('perm', p)
        self.register_buffer('inv_perm', torch.LongTensor([(p == l).nonzero() for l in range(len(p))]))

        eye = torch.eye(num_features)
        self.register_buffer('W', eye[p, :])

    def forward(self, x, inverse = False):
        if inverse:
            x = x[:,self.inv_perm]
        else:
            x = x[:,self.perm]

        return x


def main():

    if args.validation_src == 'FGSM':
        if args.dataset == 'svhn':
            out_dist_list = ['cifar10', 'imagenet_resize', 'lsun_resize', 'FGSM']
        else:
            out_dist_list = ['svhn', 'imagenet_resize', 'lsun_resize', 'FGSM']
    else:
        if args.dataset == 'svhn':
            out_dist_list = ['cifar10', 'imagenet_resize', 'lsun_resize']
        else:
            out_dist_list = ['svhn', 'imagenet_resize', 'lsun_resize']

    outf_load = os.path.join(args.outf, args.net_type + '_' + args.dataset + 'RealNVP')
    outf = os.path.join(args.outf, args.net_type + '_' + args.dataset + 'RealNVP_magnitude')
    if os.path.isdir(outf) == False:
        os.mkdir(outf)

    # torch.cuda.manual_seed(0)
    torch.cuda.set_device(args.cuda_index)

    if args.dataset == 'cifar100':
        args.num_classes = 100
    else:
        args.num_classes = 10

    with open('feature_lists/feature_lists_{}_imagenet_resize_{}_Wlinear.pickle'.format(args.net_type, args.dataset), 'rb') as f:
        [sample_class_mean, list_features, list_features_test, list_features_out, A, A_inv, log_abs_det_A_inv] = pickle.load(f)

    pre_trained_net = args.net_type + '_' + args.dataset + '.pth'
    pre_trained_net = os.path.join('pre_trained', pre_trained_net)
    if args.net_type == 'densenet':

        if args.dataset == 'svhn':
            model = models.DenseNet3(100, int(args.num_classes))
            model.load_state_dict(torch.load(pre_trained_net, map_location="cuda:" + str(args.cuda_index)))
        else:
            model = torch.load(pre_trained_net, map_location="cpu")
            for i, (name, module) in enumerate(model._modules.items()):
                module = recursion_change_bn(model)
            for m in model.modules():
                if 'Conv' in str(type(m)):
                    setattr(m, 'padding_mode', 'zeros')
        in_transform = transforms.Compose([transforms.ToTensor(),
                                           transforms.Normalize((125.3 / 255, 123.0 / 255, 113.9 / 255),
                                                                (63.0 / 255, 62.1 / 255.0, 66.7 / 255.0)), ])
    else:
        model = models.ResNet34(num_c=args.num_classes)
        model.load_state_dict(torch.load(pre_trained_net, map_location="cuda:" + str(args.cuda_index)))
        in_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])

    print('load model: ' + args.net_type)
    model.to(device)
    model.eval()

    # load dataset
    print('load in-data: ', args.dataset)

    num_layers = len(sample_class_mean)

    for layer in range(num_layers):

        num_features = A_inv[layer].shape[0]
        half_features = int(num_features/2)
        zeros = np.zeros(half_features)
        ones = np.ones(half_features)
        right = np.concatenate((zeros, ones), axis=None)
        left = np.concatenate((ones,zeros), axis=None)

        masks = torch.from_numpy(np.array([right, left, right, left, right, left, right, left, right, left]).astype(np.float32)).cuda()
        flow = []


        # We reduce the number of neurons in the hidden layers due to GPU memory limitations (11 GB in GTX 2080Ti) - comment out this line for larger GPU memory
        length_hidden = reture_length_hidden(layer)

        A_layer = torch.tensor(A[layer])
        A_inv_layer = torch.tensor(A_inv[layer])
        log_abs_det_A_inv_layer = torch.tensor(log_abs_det_A_inv[layer])

        for i in range(args.num_classes):
            MODEL_FLOW = os.path.join(outf_load,'model_{}_layer_{}_residual_flow_{}length_hidden'.format(args.dataset, layer,length_hidden), 'flow_{}'.format(i))
            flow.append(RealNVP(masks, num_features, length_hidden, A_layer, A_inv_layer, log_abs_det_A_inv_layer))
            flow[i].load_state_dict(torch.load(MODEL_FLOW, map_location="cuda:{}".format(args.cuda_index)), strict=False)
            flow[i].to(device)
            flow[i].eval()

        sample_class_mean_layer = sample_class_mean[layer]

        m_list = [0.0, 0.01, 0.005, 0.002, 0.0014, 0.001, 0.0005]

        for magnitude in m_list:

            print('Noise: ' + str(magnitude))
            _, test_loader = data_loader.getTargetDataSet(args.dataset, args.batch_size, in_transform, args.dataroot)
            score_in = lib_generation.get_resflow_score(test_loader, model, layer, args.num_classes, args.net_type, sample_class_mean_layer, flow, magnitude)

            for out_dist in out_dist_list:
                print('load out-data: ', out_dist)

                if out_dist == 'FGSM':
                    test_loader, out_loader = data_loader.getFGSM(args.batch_size, args.dataset, args.net_type)
                    score_in = lib_generation.get_resflow_score_FGSM(test_loader, model, layer, args.num_classes, args.net_type, sample_class_mean_layer,flow, magnitude)
                    score_out = lib_generation.get_resflow_score_FGSM(out_loader, model, layer, args.num_classes, args.net_type, sample_class_mean_layer, flow, magnitude)

                else:
                    out_loader = data_loader.getNonTargetDataSet(out_dist, args.batch_size, in_transform, args.dataroot)
                    score_out = lib_generation.get_resflow_score(out_loader, model, layer, args.num_classes, args.net_type,sample_class_mean_layer, flow, magnitude)

                pram = {
                    'out_dist': out_dist,
                    'Network_type': args.net_type,
                    'Layer': layer,
                    'Batch_size': args.batch_size,
                    'cuda_index': args.cuda_index,
                    'length_hidden': length_hidden,
                    'dropout': False,
                    'weight_decay': 0,
                    'init_zeros': True,
                    'num_flows': int(len(flow[0].t)),
                    'magnitude': magnitude,
                }

                with open( os.path.join(outf,'Residual_flow_%s_%s_layer_%s_%smagnitude.txt' % (args.dataset, out_dist, layer,magnitude)), 'w') as file:
                    file.write('date: %s\n' % (datetime.datetime.now()))
                    file.write(json.dumps(pram))

                score_in = np.asarray(score_in, dtype=np.float32)
                score_out = np.asarray(score_out, dtype=np.float32)
                score_data, score_labels = lib_generation.merge_and_generate_labels(score_out, score_in)
                file_name = os.path.join(outf, 'Residual_flow_%s_%s_layer_%s_%smagnitude' % (args.dataset, out_dist, layer, magnitude))
                score_data = np.concatenate((score_data, score_labels), axis=1)
                np.savez(file_name, score_data, pram)


def recursion_change_bn(module):
    """
    Converts a model trained with pytorch0.3.x to a pytorch > 0.4.0 compatible model
    """
    if isinstance(module, torch.nn.BatchNorm2d):
        module.track_running_stats = 1
    else:
        for i, (name, module1) in enumerate(module._modules.items()):
            module1 = recursion_change_bn(module1)
    return module

def reture_length_hidden(layer):
    length_hidden = 1
    if args.net_type == 'densenet' and args.dataset == 'cifar100' and layer == 3:
        length_hidden = 0.5
    if args.net_type == 'resnet' and args.dataset == 'cifar100' and layer == 4:
        length_hidden = 0.2
    if args.net_type == 'resnet' and args.dataset == 'svhn' and layer == 4:
        length_hidden = 0.5
    return length_hidden

if __name__ == '__main__':
    main()
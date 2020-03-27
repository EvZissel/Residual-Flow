import datetime
import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import pickle
import json


import os
import lib_generation
import argparse

parser = argparse.ArgumentParser(description='PyTorch code: Residual flow detector train')
parser.add_argument('--cuda_index', type=int, default=0, help='index of CUDA device, default value 0')
parser.add_argument('--num_iter', type=int, default=1000, help='number of iterations, default value 10,000')
parser.add_argument('--batch_size', type=int, default=256, help='batch size, default value 1000')
parser.add_argument('--layer', type=int, default='0', help='features for this layer are taken')
parser.add_argument('--net_type', required=True, help='resnet | densenet')
parser.add_argument('--outf', default='output', help='folder to output results')
parser.add_argument('--lr', default=1e-5, help='learning rate')
parser.add_argument('--length_hidden', default=1, help='number of hidden neurons in s and t')
parser.add_argument('--num_classes', default=10, help='number of classes')
parser.add_argument('--dataset', required=True, help='cifar10 | cifar100 | svhn')
args = parser.parse_args()
print('Running with CUDA {} for {} iterations, net type {}, dataset {},batch_size {}, Layer {}, lr {}'.format(args.cuda_index, args.num_iter, args.net_type, args.dataset,\
                                                                                                              args.batch_size, args.layer, args.lr, args.length_hidden))

device = torch.device('cuda')

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

    if args.dataset == 'svhn':
        out_dist_list = ['cifar10', 'imagenet_resize', 'lsun_resize']
    else:
        out_dist_list = ['svhn', 'imagenet_resize', 'lsun_resize']

    outf = os.path.join(args.outf, args.net_type + '_' + args.dataset + 'RealNVP')
    if os.path.isdir(outf) == False:
        os.mkdir(outf)
    # torch.cuda.manual_seed(0)
    torch.cuda.set_device(args.cuda_index)

    if args.dataset == 'cifar100':
        args.num_classes = 100
    else:
        args.num_classes = 10

    with open(os.path.join('feature_lists','feature_lists_{}_imagenet_resize_{}_Wlinear.pickle'.format(args.net_type, args.dataset)), 'rb') as f:
        [sample_class_mean, list_features, list_features_test, list_features_out, A, A_inv, log_abs_det_A_inv] = pickle.load(f)

    num_validation = 10000
    v_size_per_class = int(num_validation/args.num_classes)
    X = []
    X_validation = []

    for i in range(args.num_classes):
        X_validation.append(list_features[args.layer][i][:v_size_per_class, :] - sample_class_mean[args.layer][i])
        X.append(list_features[args.layer][i][v_size_per_class:, :] - sample_class_mean[args.layer][i])

    train_loader_X = []
    validation_loader_X = []
    X_all = 0
    for i in range(args.num_classes):
        if i == 0:
            X_all = X[i]
        else:
            X_all = torch.cat((X_all,X[i]), 0)



    for i in range(args.num_classes):
        train_loader_X.append(torch.utils.data.DataLoader(X[i], batch_size = args.batch_size, shuffle=True))
        validation_loader_X.append(torch.utils.data.DataLoader(X_validation[i], batch_size = args.batch_size, shuffle=True))

    num_features = A_inv[args.layer].shape[0]
    num_train = X_all.size(0)

    in_features_validation_test = list_features_test[args.layer]
    in_features_validation = in_features_validation_test[:2000,:]

    out_features_validation_test = list_features_out[args.layer]
    out_features_validation = out_features_validation_test[:2000,:]

    features_validation = torch.cat((in_features_validation, out_features_validation), dim=0)
    label_validation = torch.cat((torch.cuda.FloatTensor(in_features_validation.size(0)).fill_(1), torch.cuda.FloatTensor(out_features_validation.size(0)).fill_(0)))

    half_features = int(num_features/2)
    zeros = np.zeros(half_features)
    ones = np.ones(half_features)
    right = np.concatenate((zeros, ones), axis=None)
    left = np.concatenate((ones,zeros), axis=None)

    masks = torch.from_numpy(np.array([right, left, right, left, right, left, right, left, right, left]).astype(np.float32)).cuda()
    flow = []
    optimizer = []

    A_layer = torch.tensor(A[args.layer])
    A_inv_layer = torch.tensor(A_inv[args.layer])
    log_abs_det_A_inv_layer = torch.tensor(log_abs_det_A_inv[args.layer])
    t_start = 0
    for i in range(args.num_classes):
        flow.append(RealNVP(masks, num_features, args.length_hidden, A_layer, A_inv_layer, log_abs_det_A_inv_layer))

        optimizer.append(torch.optim.Adam([p for p in flow[i].parameters() if p.requires_grad == True], lr=args.lr))


    sample_class_mean_layer = sample_class_mean[args.layer]

    loss_vec, auroc_vec, validation_vec = train(train_loader_X, flow, optimizer, args.num_iter, num_train, outf, args.layer, \
                                                args.length_hidden, validation_loader_X, num_validation, features_validation, label_validation, sample_class_mean_layer,\
                                                args.num_classes, args.dataset, t_start)

    features_test = list_features_test[args.layer]

    test_loader = torch.utils.data.DataLoader( features_test, batch_size=args.batch_size, shuffle=False)
    score_in = test(test_loader, flow, sample_class_mean_layer, args.num_classes)

    for out_dist in out_dist_list:

        with open('feature_lists/feature_lists_{}_{}_{}_Wlinear.pickle'.format(args.net_type, out_dist, args.dataset), 'rb') as f:
            [_, _, list_features_test, list_features_out,_,_,_] = pickle.load(f)

        if out_dist == 'FGSM':
            features_test = list_features_test[args.layer]

            test_loader = torch.utils.data.DataLoader(features_test, batch_size=args.batch_size, shuffle=False)
            score_in = test(test_loader, flow, sample_class_mean_layer, args.num_classes)

        features_out = list_features_out[args.layer]

        out_loader = torch.utils.data.DataLoader( features_out, batch_size=args.batch_size, shuffle=False)
        score_out = test(out_loader, flow, sample_class_mean_layer, args.num_classes)

        pram = {
            'out_dist': out_dist,
            'Network_type': args.net_type,
            'Layer': args.layer,
            'Batch_size': args.batch_size,
            'num_iter': args.num_iter,
            'cuda_index': args.cuda_index,
            'lr': args.lr,
            'lr_schedule': {1: args.lr},
            'length_hidden': args.length_hidden,
            'dropout': False,
            'weight_decay': 0,
            'init_zeros': True,
            'num_flows': int(len(flow[0].t)),
        }

        with open( os.path.join(outf,'Residual_flow_%s_%s_layer_%s_%siter_%sflows_%slength_hidden.txt' % (args.dataset, args.layer,args.num_iter,int(len(flow[0].t)), args.length_hidden)), 'w') as file:
            file.write('date: %s\n' % (datetime.datetime.now()))
            file.write(json.dumps(pram))

        score_in = np.asarray(score_in, dtype=np.float32)
        score_out = np.asarray(score_out, dtype=np.float32)
        score_data, score_labels = lib_generation.merge_and_generate_labels(score_out, score_in)
        file_name = os.path.join(outf, 'Residual_flow_%s_%s_layer_%s_%siter_%sflows_%slength_hidden' % (args.dataset, out_dist, args.layer,args.num_iter,int(len(flow[0].t)), args.length_hidden))
        score_data = np.concatenate((score_data, score_labels), axis=1)
        np.savez(file_name, score_data, loss_vec, pram, auroc_vec, validation_vec)

    std_Z_max = 0
    std_Z_min = 1
    for i in range(args.num_classes):
        std_Z = flow[i].f(X[i], training=False)[0].std(0)
        std_Z_max = max(std_Z_max, std_Z.max())
        std_Z_min = min(std_Z_min, std_Z.min())

    print('std z: \n', std_Z)
    print('std z max: \n', std_Z_max)
    print('std z min: \n', std_Z_min)


def train(train_loader_X, flow, optimizer, num_iter, num_train, outf, layer, length_hidden, validation_loader, validation_size, features_validation, label_validation, sample_class_mean, num_classes, dataset, t_start):

    loss_vec = []
    auroc_vec = []
    validation_vec = []
    data_X_first = []
    nth = 1000
    for i in range(num_classes):
        flow[i].to(device)
        flow[i].train()


        data_X_first.append(iter(train_loader_X[i]).next())
        loss_vec.append(np.empty(0))
        auroc_vec.append(np.empty(0))
        validation_vec.append(np.empty(0))

    for t in range(t_start, num_iter,1):
        running_loss = 0.0
        running_validation_loss = 0.0
        for i in range(num_classes):
            for data_X in train_loader_X[i]:
                data_X = data_X.cuda()
                loss = -flow[i].log_prob(data_X, training=True).sum()

                optimizer[i].zero_grad()
                loss.backward(retain_graph=True)
                optimizer[i].step()
                running_loss += loss

            with torch.no_grad():
                flow[i].eval()
                for validation_X in validation_loader[i]:
                    validation_X = validation_X.cuda()
                    validation_loss = -flow[i].log_prob(validation_X, training=False).sum()
                    running_validation_loss += validation_loss

                flow[i].train()

        running_loss = running_loss / num_train
        loss_vec = np.append(loss_vec, running_loss.detach().cpu())
        running_validation_loss = running_validation_loss / validation_size
        validation_vec = np.append(validation_vec, running_validation_loss.detach().cpu().numpy())

        if t % 20 == 0:
            with torch.no_grad():
                confidence_score = 0
                std_Z_max = 0
                std_Z_min = 1
                for i in range(num_classes):
                    flow[i].eval()
                    std_Z  = flow[i].f(data_X_first[i], training=False)[0].std(0)
                    std_Z_max = max(std_Z_max, std_Z.max())
                    std_Z_min = min(std_Z_min, std_Z.min())

                    batch_sample_mean = sample_class_mean[i]
                    zero_f = features_validation - batch_sample_mean
                    zero_f = zero_f.cuda()
                    score = flow[i].log_prob(zero_f, training=False)
                    if i == 0:
                        confidence_score = score.view(-1, 1)
                    else:
                        confidence_score = torch.cat((confidence_score, score.view(-1, 1)), 1)

                    flow[i].train()

                confidence_max = np.amax(confidence_score.detach().cpu().numpy(), axis=1)
                fpr, tpr, auroc, precision, recall, auprre = make_roc(confidence_max, label_validation.detach().cpu().numpy(), nth)
                auroc_vec = np.append(auroc_vec, auroc)
                print('iter %s:' % t, 'loss = %.3f' % running_loss, 'std max= %.3f' % std_Z_max,'std min= %.3f' % std_Z_min, 'validation Roc = %.3f' % (100*auroc), 'validation loss %.3f' % running_validation_loss)

        # save checkpoint with network's weights every 500 iterations
        if t % 500 == 0:
            path_dir = os.path.join(outf,'model_%s_layer_%s_residual_flow_%siter_%sflows_%slength_hidden' % (dataset, layer, t, int(len(flow[0].t)), length_hidden))
            if os.path.isdir(path_dir) == False:
                os.mkdir(path_dir)
            for i in range(num_classes):
                MODELS_PATH_flow_i = os.path.join(path_dir,'flow_%s' %(i))
                torch.save(flow[i].state_dict(), MODELS_PATH_flow_i)

    path_dir = os.path.join(outf,'model_%s_layer_%s_residual_flow_%slength_hidden' % (dataset, layer, num_iter, int(len(flow[0].t)), length_hidden))
    if os.path.isdir(path_dir) == False:
        os.mkdir(path_dir)
    for i in range(num_classes):
        MODELS_PATH_flow_i = os.path.join(path_dir, 'flow_%s' % (i))
        torch.save(flow[i].state_dict(), MODELS_PATH_flow_i)

    return loss_vec, auroc_vec, validation_vec

def test(test_loader, flow, sample_class_mean, num_classes):

    score_vec = []
    for out_features in test_loader:

        # compute score
        gaussian_score = 0
        for i in range(num_classes):
            flow[i].eval()
            batch_sample_mean = sample_class_mean[i]
            zero_f = out_features.data - batch_sample_mean
            zero_f = zero_f.cuda()
            score = flow[i].log_prob(zero_f, training=False)
            if i == 0:
                gaussian_score = score.view(-1,1)
            else:
                gaussian_score = torch.cat((gaussian_score, score.view(-1,1)), 1)

        score_vec.extend(gaussian_score.detach().cpu().numpy())

    return score_vec


def make_roc(confidence, labels, nth):
    min_thresh = np.amin(confidence)
    max_thresh = np.amax(confidence)

    thresholds = np.linspace(max_thresh, min_thresh, nth)
    # thresholds = -np.sort(-np.unique(confidence))
    thresholds = np.append(thresholds, min_thresh - 1e-3)
    thresholds = np.insert(thresholds, 0, max_thresh + 1e-3)

    fpr = np.empty(len(thresholds))
    tpr = np.empty(len(thresholds))
    precision = np.empty(len(thresholds))


    for i, th in zip(range(len(thresholds)), thresholds):
        tp = float(confidence[(labels == True) & (confidence >= th)].shape[0])
        tn = float(confidence[(labels != True) & (confidence < th)].shape[0])
        fp = float(confidence[(labels != True) & (confidence >= th)].shape[0])
        fn = float(confidence[(labels == True) & (confidence < th)].shape[0])


        fpr[i] = fp / (tn + fp) #FP from R
        tpr[i] = tp / (tp + fn) #TP from P

        if tp != 0:
            precision[i] = tp / (tp + fp)
        else:
            precision[i] = 0.0

    recall = tpr
    auroc = np.trapz(tpr, fpr)
    auprre = np.trapz(precision, recall)
    return fpr, tpr, auroc, precision, recall, auprre

if __name__ == '__main__':
    main()
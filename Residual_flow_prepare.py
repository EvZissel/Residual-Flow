
import numpy as np
import torch
from torch import nn
import pickle

import data_loader
import models
import os
import argparse

from torchvision import transforms
from torch.autograd import Variable

device = torch.device('cuda')

parser = argparse.ArgumentParser(description='PyTorch code: Residual flow detector prepare')
parser.add_argument('--batch_size', type=int, default=20, metavar='N', help='batch size for data loader')
parser.add_argument('--dataset', required=True, help='cifar10 | cifar100 | svhn')
parser.add_argument('--dataroot', default='./data', help='path to dataset')
parser.add_argument('--outf', default='output', help='folder to output results')
parser.add_argument('--num_classes', type=int, default=100, help='the # of classes')
parser.add_argument('--net_type', required=True, help='resnet | densenet')
parser.add_argument('--gpu', type=int, default=0, help='gpu index')
parser.add_argument('--validation_src', default='IO', help='IO | FGSM (choice of validation source for hyper-parameter tuning: IO for in- and out-of-distribution, or FGSM for adverarial validation)')
args = parser.parse_args()
print(args)

def main():

    if os.path.isdir('feature_lists') == False:
        os.mkdir('feature_lists')

    if args.dataset == 'cifar100':
        args.num_classes = 100
    else:
        args.num_classes = 10

    # load networks
    pre_trained_net = args.net_type + '_' + args.dataset + '.pth'
    pre_trained_net = os.path.join('pre_trained', pre_trained_net)
    if args.net_type == 'densenet':
        if args.dataset == 'svhn':
            model = models.DenseNet3(100, int(args.num_classes))
            model.load_state_dict(torch.load(pre_trained_net, map_location = "cuda:" + str(0)))
        else:
            model = torch.load(pre_trained_net, map_location="cpu")
            for i, (name, module) in enumerate(model._modules.items()):
                module = recursion_change_bn(model)
            for m in model.modules():
                if 'Conv' in str(type(m)):
                    setattr(m, 'padding_mode', 'zeros')
        in_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((125.3/255, 123.0/255, 113.9/255), (63.0/255, 62.1/255.0, 66.7/255.0)),])
    elif args.net_type  == 'resnet':

        model = models.ResNet34(num_c=args.num_classes)
        model.load_state_dict(torch.load(pre_trained_net, map_location = "cuda:" + str(0)))
        in_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])

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


    print('load model: ' + args.net_type)
    model.to(device)
    model.eval()

    # load dataset
    print('load target data: ', args.dataset)
    train_loader, test_loader = data_loader.getTargetDataSet(args.dataset, args.batch_size, in_transform, args.dataroot)

    # set information about feature extraction
    temp_x = torch.rand(2, 3, 32, 32).cuda()
    temp_x = Variable(temp_x)
    temp_list = model.feature_list(temp_x)[1]
    num_output = len(temp_list)
    feature_list = np.empty(num_output)
    count = 0
    for out in temp_list:
        feature_list[count] = out.size(1)
        count += 1

    correct, total = 0, 0
    num_output = len(feature_list)
    num_sample_per_class = np.empty(args.num_classes)
    num_sample_per_class.fill(0)
    list_features = []
    list_features_test = []
    list_features_out = []
    for i in range(num_output):
        temp_list = []
        list_features_test.append(0)
        list_features_out.append(0)
        for j in range(args.num_classes):
            temp_list.append(0)
        list_features.append(temp_list)

    for data, target in train_loader:
        total += data.size(0)
        data = data.cuda()
        data = Variable(data, volatile=True)
        output, out_features = model.feature_list(data)

        # get hidden features
        for i in range(num_output):
            out_features[i] = out_features[i].view(out_features[i].size(0), out_features[i].size(1), -1)
            out_features[i] = torch.mean(out_features[i].data, 2)

        # compute the accuracy
        pred = output.data.max(1)[1]
        equal_flag = pred.eq(target.cuda()).cpu()
        correct += equal_flag.sum()

        # construct the sample matrix
        for i in range(data.size(0)):
            label = target[i]
            if num_sample_per_class[label] == 0:
                out_count = 0
                for out in out_features:
                    list_features[out_count][label] = out[i].view(1, -1)
                    out_count += 1
            else:
                out_count = 0
                for out in out_features:
                    list_features[out_count][label] = torch.cat((list_features[out_count][label], out[i].view(1, -1)), 0)
                    out_count += 1
            num_sample_per_class[label] += 1

    sample_class_mean = []
    out_count = 0
    for num_feature in feature_list:
        temp_list = torch.Tensor(args.num_classes, int(num_feature)).cuda()
        for j in range(args.num_classes):
            temp_list[j] = torch.mean(list_features[out_count][j], 0)
        sample_class_mean.append(temp_list)
        out_count += 1

    A = []
    A_inv = []
    log_abs_det_A_inv = []
    for k in range(num_output):
        X = 0
        for i in range(args.num_classes):
            if i == 0:
                X = list_features[k][i] - sample_class_mean[k][i]
            else:
                X = torch.cat((X, list_features[k][i] - sample_class_mean[k][i]), 0)

        # find inverse
        u, s, vh = np.linalg.svd((X.cpu().numpy())/ np.sqrt(X.shape[0]), full_matrices=False)
        covariance_real = np.cov(X.cpu().numpy().T)
        valid_indx = s > 1e-5
        if (valid_indx.sum() % 2 > 0):
            valid_indx[valid_indx.sum()-1] = False
        covriance_cal = np.matmul(np.matmul(vh[valid_indx, :].transpose(), np.diag(s[valid_indx] ** 2)), vh[valid_indx, :])
        A_temp = np.matmul(vh[valid_indx, :].transpose(), np.diag(s[valid_indx]))
        A.append(A_temp)
        covriance_cal2 = np.matmul(A_temp, A_temp.transpose())
        s_inv = 1/s[valid_indx]
        A_inv_temp = np.matmul(np.diag(s_inv), vh[valid_indx, :])
        A_inv.append(A_inv_temp)
        log_abs_det_A_inv_temp = np.sum(np.log(np.abs(s_inv)))
        log_abs_det_A_inv.append(log_abs_det_A_inv_temp)

    print('\n Training Accuracy:({:.2f}%)\n'.format(100.0 * int(correct) / int(total)))

    num_sample_per_output = np.empty(num_output)
    num_sample_per_output.fill(0)
    for data, target in test_loader:

        data = data.cuda()
        data = Variable(data, volatile=True)
        output, out_features = model.feature_list(data)

        # get hidden features
        for i in range(num_output):
            out_features[i] = out_features[i].view(out_features[i].size(0), out_features[i].size(1), -1)
            out_features[i] = torch.mean(out_features[i].data, 2)

            if  num_sample_per_output[i] ==0:
                list_features_test[i] = out_features[i]
            else:
                list_features_test[i] = torch.cat((list_features_test[i], out_features[i]), 0)
            num_sample_per_output[i] += 1

    for out_dist in out_dist_list:

        if out_dist == 'FGSM':
            test_loader, out_test_loader = data_loader.getFGSM(args.batch_size, args.dataset, args.net_type)
            num_sample_per_output.fill(0)

            for data in test_loader:

                data = data.cuda()
                data = Variable(data, volatile=True)
                output, out_features = model.feature_list(data)

                # get hidden features
                for i in range(num_output):
                    out_features[i] = out_features[i].view(out_features[i].size(0), out_features[i].size(1), -1)
                    out_features[i] = torch.mean(out_features[i].data, 2)

                    if num_sample_per_output[i] == 0:
                        list_features_test[i] = out_features[i]
                    else:
                        list_features_test[i] = torch.cat((list_features_test[i], out_features[i]), 0)
                    num_sample_per_output[i] += 1

            num_sample_per_output = np.empty(num_output)
            num_sample_per_output.fill(0)

            for data in out_test_loader:
                data = data.cuda()
                data = Variable(data, requires_grad=True)
                output, out_features = model.feature_list(data)

                # get hidden features
                for i in range(num_output):
                    out_features[i] = out_features[i].view(out_features[i].size(0), out_features[i].size(1), -1)
                    out_features[i] = torch.mean(out_features[i].data, 2)

                    if num_sample_per_output[i] == 0:
                        list_features_out[i] = out_features[i]
                    else:
                        list_features_out[i] = torch.cat((list_features_out[i], out_features[i]), 0)
                    num_sample_per_output[i] += 1

            for i in range(num_output):
                sample_class_mean[i] = sample_class_mean[i].cpu()
                list_features_test[i] = list_features_test[i].cpu()
                list_features_out[i] = list_features_out[i].cpu()
                for j in range(args.num_classes):
                    list_features[i][j] = list_features[i][j].cpu()

        else:
            out_test_loader = data_loader.getNonTargetDataSet(out_dist, args.batch_size, in_transform, args.dataroot)
            num_sample_per_output.fill(0)

            for data, target in out_test_loader:

                data, target = data.cuda(), target.cuda()
                data, target = Variable(data, requires_grad=True), Variable(target)
                output, out_features = model.feature_list(data)

                # get hidden features
                for i in range(num_output):
                    out_features[i] = out_features[i].view(out_features[i].size(0), out_features[i].size(1), -1)
                    out_features[i] = torch.mean(out_features[i].data, 2)

                    if num_sample_per_output[i] == 0:
                        list_features_out[i] = out_features[i]
                    else:
                        list_features_out[i] = torch.cat((list_features_out[i], out_features[i]), 0)
                    num_sample_per_output[i] += 1

            for i in range(num_output):
                sample_class_mean[i] = sample_class_mean[i].cpu()
                list_features_test[i] = list_features_test[i].cpu()
                list_features_out[i] = list_features_out[i].cpu()
                for j in range(args.num_classes):
                    list_features[i][j] = list_features[i][j].cpu()

        file_name = os.path.join('feature_lists', 'feature_lists_{}_{}_{}_Wlinear.pickle'.format(args.net_type, out_dist, args.dataset))
        with open(file_name, 'wb') as f:
            pickle.dump([sample_class_mean, list_features, list_features_test, list_features_out, A, A_inv, log_abs_det_A_inv] , f)


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


if __name__ == '__main__':
    main()
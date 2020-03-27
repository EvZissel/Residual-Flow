"""
Created on Sun Oct 25 2018
@author: Kimin Lee
"""
from __future__ import print_function
import argparse
import torch
import data_loader
import numpy as np
import calculate_log as callog
import models
import os
import lib_generation

from torchvision import transforms
from torch.autograd import Variable

parser = argparse.ArgumentParser(description='PyTorch code: Mahalanobis detector using adversarial samples')
parser.add_argument('--batch_size', type=int, default=20, metavar='N', help='batch size for data loader')
parser.add_argument('--dataset', required=True, help='cifar10 | cifar100 | svhn')
parser.add_argument('--dataroot', default='./data', help='path to dataset')
parser.add_argument('--outf', default='./adv_output/', help='folder to output results')
parser.add_argument('--num_classes', type=int, default=10, help='the # of classes')
parser.add_argument('--net_type', required=True, help='resnet | densenet')
parser.add_argument('--gpu', type=int, default=0, help='gpu index')
args = parser.parse_args()
print(args)

def main():
    # set the path to pre-trained model and output
    pre_trained_net = './pre_trained/' + args.net_type + '_' + args.dataset + '.pth'
    args.outf = args.outf + args.net_type + '_' + args.dataset + '/'
    if os.path.isdir(args.outf) == False:
        os.mkdir(args.outf)
    torch.cuda.manual_seed(0)
    torch.cuda.set_device(args.gpu)
    # check the in-distribution dataset
    if args.dataset == 'cifar100':
        args.num_classes = 100
        
    # load networks
    if args.net_type == 'densenet':
        if args.dataset == 'svhn':
            model = models.DenseNet3(100, int(args.num_classes))
            model.load_state_dict(torch.load(pre_trained_net, map_location = "cuda:" + str(args.gpu)))
        else:
            model = torch.load(pre_trained_net, map_location = "cuda:" + str(args.gpu))
            for i, (name, module) in enumerate(model._modules.items()):
                module = recursion_change_bn(model)
            for m in model.modules():
                if 'Conv' in str(type(m)):
                    setattr(m, 'padding_mode', 'zeros')
        in_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((125.3/255, 123.0/255, 113.9/255), (63.0/255, 62.1/255.0, 66.7/255.0)),])
    elif args.net_type == 'resnet':
        model = models.ResNet34(num_c=args.num_classes)
        model.load_state_dict(torch.load(pre_trained_net, map_location = "cuda:" + str(args.gpu)))
        in_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
    model.cuda()
    print('load model: ' + args.net_type)
    
    # load dataset
    print('load target data: ', args.dataset)
    train_loader, _ = data_loader.getTargetDataSet(args.dataset, args.batch_size, in_transform, args.dataroot)
    test_clean_data = torch.load(args.outf + 'clean_data_%s_%s_%s.pth' % (args.net_type, args.dataset, 'FGSM'))
    test_adv_data = torch.load(args.outf + 'adv_data_%s_%s_%s.pth' % (args.net_type, args.dataset, 'FGSM'))
    test_noisy_data = torch.load(args.outf + 'noisy_data_%s_%s_%s.pth' % (args.net_type, args.dataset, 'FGSM'))
    test_label = torch.load(args.outf + 'label_%s_%s_%s.pth' % (args.net_type, args.dataset, 'FGSM'))

    # set information about feature extaction
    model.eval()
    temp_x = torch.rand(2,3,32,32).cuda()
    temp_x = Variable(temp_x)
    temp_list = model.feature_list(temp_x)[1]
    num_output = len(temp_list)
    feature_list = np.empty(num_output)
    count = 0
    for out in temp_list:
        feature_list[count] = out.size(1)
        count += 1
        
    print('get sample mean and covariance')
    sample_mean, precision = lib_generation.sample_estimator(model, args.num_classes, feature_list, train_loader)
    
    print('get Mahalanobis scores')
    m_list = [0.0, 0.01, 0.005, 0.002, 0.0014, 0.001, 0.0005]
    for magnitude in m_list:
        print('\nNoise: ' + str(magnitude))
        for i in range(num_output):
            M_in \
            = lib_generation.get_Mahalanobis_score_adv(model, test_clean_data, test_label, \
                                                       args.num_classes, args.outf, args.net_type, \
                                                       sample_mean, precision, i, magnitude)
            M_in = np.asarray(M_in, dtype=np.float32)
            if i == 0:
                Mahalanobis_in = M_in.reshape((M_in.shape[0], -1))
            else:
                Mahalanobis_in = np.concatenate((Mahalanobis_in, M_in.reshape((M_in.shape[0], -1))), axis=1)

        for i in range(num_output):
            M_out \
            = lib_generation.get_Mahalanobis_score_adv(model, test_adv_data, test_label, \
                                                       args.num_classes, args.outf, args.net_type, \
                                                       sample_mean, precision, i, magnitude)
            M_out = np.asarray(M_out, dtype=np.float32)
            if i == 0:
                Mahalanobis_out = M_out.reshape((M_out.shape[0], -1))
            else:
                Mahalanobis_out = np.concatenate((Mahalanobis_out, M_out.reshape((M_out.shape[0], -1))), axis=1)
                
        for i in range(num_output):
            M_noisy \
            = lib_generation.get_Mahalanobis_score_adv(model, test_noisy_data, test_label, \
                                                       args.num_classes, args.outf, args.net_type, \
                                                       sample_mean, precision, i, magnitude)
            M_noisy = np.asarray(M_noisy, dtype=np.float32)
            if i == 0:
                Mahalanobis_noisy = M_noisy.reshape((M_noisy.shape[0], -1))
            else:
                Mahalanobis_noisy = np.concatenate((Mahalanobis_noisy, M_noisy.reshape((M_noisy.shape[0], -1))), axis=1)            
        Mahalanobis_in = np.asarray(Mahalanobis_in, dtype=np.float32)
        Mahalanobis_out = np.asarray(Mahalanobis_out, dtype=np.float32)
        Mahalanobis_noisy = np.asarray(Mahalanobis_noisy, dtype=np.float32)
        Mahalanobis_pos = np.concatenate((Mahalanobis_in, Mahalanobis_noisy))

        Mahalanobis_data, Mahalanobis_labels = lib_generation.merge_and_generate_labels(Mahalanobis_out, Mahalanobis_pos)
        file_name = os.path.join(args.outf, 'Mahalanobis_%s_%s_%s.npy' % (str(magnitude), args.dataset, 'FGSM'))
        
        Mahalanobis_data = np.concatenate((Mahalanobis_data, Mahalanobis_labels), axis=1)
        np.save(file_name, Mahalanobis_data)

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


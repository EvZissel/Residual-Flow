"""
based on the code of Kimin Lee (Sun Oct 21 2018).

"""
from __future__ import print_function
import numpy as np
import os
import lib_regression
import argparse

from sklearn.linear_model import LogisticRegressionCV

parser = argparse.ArgumentParser(description='PyTorch code: Mahalanobis detector regression using adversarial samples')
parser.add_argument('--net_type', required=True, help='resnet | densenet')
args = parser.parse_args()
print(args)

def main():
    # initial setup
    dataset_list = ['cifar10', 'cifar100', 'svhn']

    score_list = ['Mahalanobis_0.0', 'Mahalanobis_0.01', 'Mahalanobis_0.005', 'Mahalanobis_0.002', 'Mahalanobis_0.0014', 'Mahalanobis_0.001', 'Mahalanobis_0.0005']
    
    # train and measure the performance of Mahalanobis detector
    list_best_results, list_best_results_index = [], []
    for dataset in dataset_list:
        print('In-distribution: ', dataset)
        outf = './output/' + args.net_type + '_' + dataset + '/'
        outf_FGSM = './adv_output/' + args.net_type + '_' + dataset + '/'
        out_list = ['svhn', 'imagenet_resize', 'lsun_resize']
        if dataset == 'svhn':
            out_list = ['cifar10', 'imagenet_resize', 'lsun_resize']

        list_best_results_out, list_best_results_index_out = [], []
        for out in out_list:
            print('Out-of-distribution: ', out)
            best_tnr, best_result, best_index = 0, 0, 0
            for score in score_list:
                total_X_FGSM, total_Y_FGSM = lib_regression.load_characteristics(score, dataset, 'FGSM', outf_FGSM)
                X_val_FGSM, Y_val_FGSM, X_test_FGSM, Y_test_FGSM = lib_regression.block_split_adv(total_X_FGSM, total_Y_FGSM)
                pivot = int(X_val_FGSM.shape[0] / 6)
                X_train_FGSM = np.concatenate((X_val_FGSM[:pivot], X_val_FGSM[2 * pivot:3 * pivot], X_val_FGSM[4 * pivot:5 * pivot]))
                Y_train_FGSM = np.concatenate((Y_val_FGSM[:pivot], Y_val_FGSM[2 * pivot:3 * pivot], Y_val_FGSM[4 * pivot:5 * pivot]))
                X_val_for_test_FGSM = np.concatenate((X_val_FGSM[pivot:2 * pivot], X_val_FGSM[3 * pivot:4 * pivot], X_val_FGSM[5 * pivot:]))
                Y_val_for_test_FGSM = np.concatenate((Y_val_FGSM[pivot:2 * pivot], Y_val_FGSM[3 * pivot:4 * pivot], Y_val_FGSM[5 * pivot:]))
                lr = LogisticRegressionCV(n_jobs=1).fit(X_train_FGSM, Y_train_FGSM)

                total_X, total_Y = lib_regression.load_characteristics(score, dataset, out, outf)
                X_val, Y_val, X_test, Y_test = lib_regression.block_split(total_X, total_Y, out)
                X_train = np.concatenate((X_val[:500], X_val[1000:1500]))
                Y_train = np.concatenate((Y_val[:500], Y_val[1000:1500]))
                X_val_for_test = np.concatenate((X_val[500:1000], X_val[1500:]))
                Y_val_for_test = np.concatenate((Y_val[500:1000], Y_val[1500:]))
                coef= lr.coef_
                bias= lr.intercept_

                results = lib_regression.detection_performance(lr, X_val_for_test_FGSM, Y_val_for_test_FGSM, outf)
                if best_tnr < results['TMP']['TNR']:
                    best_tnr = results['TMP']['TNR']
                    best_index = score
                    best_result = lib_regression.detection_performance(lr, X_test, Y_test, outf)
            list_best_results_out.append(best_result)
            list_best_results_index_out.append(best_index)
        list_best_results.append(list_best_results_out)
        list_best_results_index.append(list_best_results_index_out)
        
    # print the results
    count_in = 0
    mtypes = ['TNR', 'AUROC', 'DTACC', 'AUIN', 'AUOUT']

    for in_list in list_best_results:
        print('in_distribution: ' + dataset_list[count_in] + '==========')
        out_list = ['svhn', 'imagenet_resize', 'lsun_resize']
        if dataset_list[count_in] == 'svhn':
            out_list = ['cifar10', 'imagenet_resize', 'lsun_resize']
        count_out = 0
        for results in in_list:
            print('out_distribution: '+ out_list[count_out])
            for mtype in mtypes:
                print(' {mtype:6s}'.format(mtype=mtype), end='')
            print('\n{val:6.2f}'.format(val=100.*results['TMP']['TNR']), end='')
            print(' {val:6.2f}'.format(val=100.*results['TMP']['AUROC']), end='')
            print(' {val:6.2f}'.format(val=100.*results['TMP']['DTACC']), end='')
            print(' {val:6.2f}'.format(val=100.*results['TMP']['AUIN']), end='')
            print(' {val:6.2f}\n'.format(val=100.*results['TMP']['AUOUT']), end='')
            print('Input noise: ' + list_best_results_index[count_in][count_out])
            print('')
            count_out += 1
        count_in += 1

if __name__ == '__main__':
    main()
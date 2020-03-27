# several functions are from https://github.com/xingjunm/lid_adversarial_subspace_detection
from __future__ import print_function
import numpy as np
import os
import calculate_log as callog



def block_split(X, Y, out):
    """
    Split the data training and testing
    :return: X (data) and Y (label) for training / testing
    """
    num_samples = X.shape[0]
    if out == 'svhn':
        partition = 26032
    else:
        partition = 10000
    X_adv, Y_adv = X[:partition], Y[:partition]
    X_norm, Y_norm = X[partition: :], Y[partition: :]
    num_train = 1000

    X_train = np.concatenate((X_norm[:num_train], X_adv[:num_train]))
    Y_train = np.concatenate((Y_norm[:num_train], Y_adv[:num_train]))

    X_test = np.concatenate((X_norm[num_train:], X_adv[num_train:]))
    Y_test = np.concatenate((Y_norm[num_train:], Y_adv[num_train:]))

    return X_train, Y_train, X_test, Y_test

def block_split_RealNVP(X, Y, out):
    """
    Split the data training and testing
    :return: X (data) and Y (label) for training / testing
    """
    num_samples = X.shape[0]
    if out == 'svhn':
        partition = 26032
    else:
        partition = 10000
    X_adv, Y_adv = X[:partition], Y[:partition]
    X_norm, Y_norm = X[partition: :], Y[partition: :]
    num_train = 1000

    X_train = np.concatenate((X_norm[:num_train], X_adv[:num_train]))
    Y_train = np.concatenate((Y_norm[:num_train], Y_adv[:num_train]))

    X_test = np.concatenate((X_norm[num_train:], X_adv[num_train:]))
    Y_test = np.concatenate((Y_norm[num_train:], Y_adv[num_train:]))

    return X_train, Y_train, X_test, Y_test


def block_split_adv(X, Y):
    """
    Split the data training and testing
    :return: X (data) and Y (label) for training / testing
    """
    num_samples = X.shape[0]
    partition = int(num_samples / 3)
    X_adv, Y_adv = X[:partition], Y[:partition]
    X_norm, Y_norm = X[partition: 2*partition], Y[partition: 2*partition]
    X_noisy, Y_noisy = X[2*partition:], Y[2*partition:]
    num_train = int(partition*0.1)
    X_train = np.concatenate((X_norm[:num_train], X_noisy[:num_train], X_adv[:num_train]))
    Y_train = np.concatenate((Y_norm[:num_train], Y_noisy[:num_train], Y_adv[:num_train]))

    X_test = np.concatenate((X_norm[num_train:], X_noisy[num_train:], X_adv[num_train:]))
    Y_test = np.concatenate((Y_norm[num_train:], Y_noisy[num_train:], Y_adv[num_train:]))

    return X_train, Y_train, X_test, Y_test

def block_split_adv_RealNVP(X, Y):
    """
    Split the data training and testing
    :return: X (data) and Y (label) for training / testing
    """
    num_samples = X.shape[0]
    partition = int(num_samples / 3)
    # X_adv, Y_adv = X[:partition], Y[:partition]
    # X_norm, Y_norm = X[partition: 2*partition], Y[partition: 2*partition]
    # X_noisy, Y_noisy = X[2*partition:], Y[2*partition:]
    X_norm, Y_norm = X[:partition], Y[:partition]
    X_noisy, Y_noisy = X[partition: 2*partition], Y[partition: 2*partition]
    X_adv, Y_adv = X[2*partition:], Y[2*partition:]
    num_train = int(partition*0.1)
    X_train = np.concatenate((X_norm[:num_train], X_noisy[:num_train], X_adv[:num_train]))
    Y_train = np.concatenate((Y_norm[:num_train], Y_noisy[:num_train], Y_adv[:num_train]))

    X_test = np.concatenate((X_norm[num_train:], X_noisy[num_train:], X_adv[num_train:]))
    Y_test = np.concatenate((Y_norm[num_train:], Y_noisy[num_train:], Y_adv[num_train:]))

    return X_train, Y_train, X_test, Y_test

def detection_performance(regressor, X, Y, outf):
    """
    Measure the detection performance
    return: detection metrics
    """
    num_samples = X.shape[0]
    l1 = open('%s/confidence_TMP_In.txt'%outf, 'w')
    l2 = open('%s/confidence_TMP_Out.txt'%outf, 'w')
    y_pred = regressor.predict_proba(X)[:, 1]

    for i in range(num_samples):
        if Y[i] == 0:
            l1.write("{}\n".format(-y_pred[i]))
        else:
            l2.write("{}\n".format(-y_pred[i]))
    l1.close()
    l2.close()
    results = callog.metric(outf, ['TMP'])
    return results

def load_characteristics(score, dataset, out, outf):
    """
    Load the calculated scores
    return: data and label of input score
    """
    X, Y = None, None
    
    file_name = os.path.join(outf, "%s_%s_%s.npy" % (score, dataset, out))
    data = np.load(file_name)
    
    if X is None:
        X = data[:, :-1]
    else:
        X = np.concatenate((X, data[:, :-1]), axis=1)
    if Y is None:
        Y = data[:, -1] # labels only need to load once
         
    return X, Y


def load_characteristics_RealNVP(score, dataset, out, outf, net_type, num_classes):
    """
    Load the calculated scores
    return: data and label of input score
    """

    if net_type == 'densenet':

        file_name_L0 = os.path.join(outf, 'Residual_flow_{}_{}_layer_0_{}magnitude.npz'.format(dataset, out, score))
        file_name_L1 = os.path.join(outf, 'Residual_flow_{}_{}_layer_1_{}magnitude.npz'.format(dataset, out, score))
        file_name_L2 = os.path.join(outf, 'Residual_flow_{}_{}_layer_2_{}magnitude.npz'.format(dataset, out, score))
        file_name_L3 = os.path.join(outf, 'Residual_flow_{}_{}_layer_3_{}magnitude.npz'.format(dataset, out, score))


        data_L0 = np.load(file_name_L0)
        data_L1 = np.load(file_name_L1)
        data_L2 = np.load(file_name_L2)
        data_L3 = np.load(file_name_L3)

        RealNVP_data_L0 = data_L0['arr_0']
        RealNVP_data_L1 = data_L1['arr_0']
        RealNVP_data_L2 = data_L2['arr_0']
        RealNVP_data_L3 = data_L3['arr_0']

        RealNVP_score_L0 = RealNVP_data_L0[:, 0:num_classes]
        score_L0 = np.amax(RealNVP_score_L0, axis=1)

        RealNVP_score_L1 = RealNVP_data_L1[:, 0:num_classes]
        score_L1 = np.amax(RealNVP_score_L1, axis=1)

        RealNVP_score_L2 = RealNVP_data_L2[:, 0:num_classes]
        score_L2 = np.amax(RealNVP_score_L2, axis=1)

        RealNVP_score_L3 = RealNVP_data_L3[:, 0:num_classes]
        score_L3 = np.amax(RealNVP_score_L3, axis=1)


        X = np.stack((score_L0, score_L1, score_L2, score_L3), axis=-1)

    else:

        file_name_L0 = os.path.join(outf, 'Residual_flow_{}_{}_layer_0_{}magnitude.npz'.format(dataset, out, score))
        file_name_L1 = os.path.join(outf, 'Residual_flow_{}_{}_layer_1_{}magnitude.npz'.format(dataset, out, score))
        file_name_L2 = os.path.join(outf, 'Residual_flow_{}_{}_layer_2_{}magnitude.npz'.format(dataset, out, score))
        file_name_L3 = os.path.join(outf, 'Residual_flow_{}_{}_layer_3_{}magnitude.npz'.format(dataset, out, score))
        file_name_L4 = os.path.join(outf, 'Residual_flow_{}_{}_layer_4_{}magnitude.npz'.format(dataset, out, score))


        data_L0 = np.load(file_name_L0)
        data_L1 = np.load(file_name_L1)
        data_L2 = np.load(file_name_L2)
        data_L3 = np.load(file_name_L3)
        data_L4 = np.load(file_name_L4)
        RealNVP_data_L0 = data_L0['arr_0']
        RealNVP_data_L1 = data_L1['arr_0']
        RealNVP_data_L2 = data_L2['arr_0']
        RealNVP_data_L3 = data_L3['arr_0']
        RealNVP_data_L4 = data_L4['arr_0']

        RealNVP_score_L0 = RealNVP_data_L0[:, 0:num_classes]
        score_L0 = np.amax(RealNVP_score_L0, axis=1)

        RealNVP_score_L1 = RealNVP_data_L1[:, 0:num_classes]
        score_L1 = np.amax(RealNVP_score_L1, axis=1)

        RealNVP_score_L2 = RealNVP_data_L2[:, 0:num_classes]
        score_L2 = np.amax(RealNVP_score_L2, axis=1)

        RealNVP_score_L3 = RealNVP_data_L3[:, 0:num_classes]
        score_L3 = np.amax(RealNVP_score_L3, axis=1)

        RealNVP_score_L4 = RealNVP_data_L4[:, 0:num_classes]

        score_L4 = np.amax(RealNVP_score_L4, axis=1)

        X = np.stack((score_L0, score_L1, score_L2, score_L3, score_L4),axis=-1)

    Y = RealNVP_data_L3[:, -1]

    return X, Y

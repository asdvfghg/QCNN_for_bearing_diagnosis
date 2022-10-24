import os

import pandas as pd
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F


from Model.QCNN import QCNN
from Model.WDCNN import WDCNN
from utils.DatasetLoader import CustomTensorDataset
from utils.Preprocess import prepro

from train import random_seed

features_in_hook = []
features_out_hook = []


from matplotlib import pyplot as plt
from scipy.io import loadmat
import numpy as np
import os
from sklearn import preprocessing  # 0-1编码
from sklearn.model_selection import StratifiedShuffleSplit  # 随机划分，保证每一类比例相同

'''
Calculating Qttention for the QCNN network
'''


def hook(module, input, output):
    features_in_hook.append(input)
    features_out_hook.append(output)


def cal_mm(x, w):
    c = []
    for i in range(x.shape[0]):
        temp_x = x[i, :, :]
        temp_w = w[i, :, :]
        c.append(torch.mm(temp_x, temp_w.transpose(1, 0)))

    return torch.tensor(c)


def attention_compose(map_list, output_shape=2048, stride=1, paddings=0, kernal_size=64):
    map = torch.zeros(map_list[0].shape[0], output_shape + 2 * paddings)
    for idx, m in enumerate(map_list):
        map[:, idx * stride: idx * stride + kernal_size] += m.squeeze()
        if idx > 0:
            map[:, idx * stride: (idx - 1) * stride + kernal_size] /= 2
    return map[:, paddings:output_shape + paddings]


def attention_map(x, wr, wg, wb, br, bg, stride=1, paddings=0, kernal_size=64):
    output_shape = x.shape[2]
    if paddings != 0:
        padding = torch.zeros(1, x.shape[1], paddings)
        x = torch.cat((padding, x, padding), 2)
    n_sample = int((x.shape[2] - kernal_size) / stride + 1)
    map_list = []
    for i in range(n_sample):
        temp_x = x[:, :, i * stride:i * stride + kernal_size]
        temp_x_c = torch.repeat_interleave(temp_x, wb.shape[0], 0)
        y1 = temp_x_c * wb
        c = cal_mm(temp_x_c, wr)
        c = torch.repeat_interleave(c, wg.shape[2]).reshape(wg.shape)
        y2 = c * wg
        map_list.append(y1 + y2)
    map = attention_compose(map_list, output_shape, stride, paddings, kernal_size)

    return map

if __name__ == '__main__':
    random_seed(42)
    model_name = 'qcnn'
    model = QCNN()
    run_path = 'wandb/qcnnhit6/checkpoint.pth' # Need to be specified as the path to the model file
    best_model_dict = torch.load(run_path, map_location=torch.device('cpu'))
    model.load_state_dict(best_model_dict)
    model.eval()
    chosen_data = '0HP'
    SNR = 6
    length = 2048

    path = os.path.join('data', chosen_data)
    test_X, test_Y = prepro(d_path=path,
                             length=length,
                             number=100,
                             normal=False,
                             enc=True,
                             enc_step=28,
                             snr=SNR,
                             property='Test'
                             )

    # the raw dataset without noise
    # test_X1, test_Y1 = prepro(d_path=path,
    #                          length=length,
    #                          number=100,
    #                          normal=False,
    #                          enc=True,
    #                          enc_step=28,
    #                          snr=SNR,
    #                          property='Test',
    #                          noise=False
    #                          )

    test_X = test_X[:, np.newaxis, :]

    test_dataset = CustomTensorDataset(torch.tensor(test_X, dtype=torch.float),
                                       torch.tensor(test_Y))

    x = test_dataset.X
    y = test_dataset.y

    for name, module in model.named_children():
        if 'cnn' in name:
            module.Conv1D_1.register_forward_hook(hook)
            module.Conv1D_2.register_forward_hook(hook)
            module.Conv1D_3.register_forward_hook(hook)
            module.Conv1D_4.register_forward_hook(hook)
            module.Conv1D_5.register_forward_hook(hook)
            module.Conv1D_6.register_forward_hook(hook)


            wr1 = module.Conv1D_1.weight_r
            wg1 = module.Conv1D_1.weight_g
            wb1 = module.Conv1D_1.weight_b
            br1 = module.Conv1D_1.bias_r
            bg1 = module.Conv1D_1.bias_g

            wr2 = module.Conv1D_2.weight_r
            wg2 = module.Conv1D_2.weight_g
            wb2 = module.Conv1D_2.weight_b
            br2 = module.Conv1D_2.bias_r
            bg2 = module.Conv1D_2.bias_g

            wr3 = module.Conv1D_3.weight_r
            wg3 = module.Conv1D_3.weight_g
            wb3 = module.Conv1D_3.weight_b
            br3 = module.Conv1D_3.bias_r
            bg3 = module.Conv1D_3.bias_g

            wr4 = module.Conv1D_4.weight_r
            wg4 = module.Conv1D_4.weight_g
            wb4 = module.Conv1D_4.weight_b
            br4 = module.Conv1D_4.bias_r
            bg4 = module.Conv1D_4.bias_g

            wr5 = module.Conv1D_5.weight_r
            wg5 = module.Conv1D_5.weight_g
            wb5 = module.Conv1D_5.weight_b
            br5 = module.Conv1D_5.bias_r
            bg5 = module.Conv1D_5.bias_g

            wr6 = module.Conv1D_6.weight_r
            wg6 = module.Conv1D_6.weight_g
            wb6 = module.Conv1D_6.weight_b
            br6 = module.Conv1D_6.bias_r
            bg6 = module.Conv1D_6.bias_g
    outputs = []
    y_pre = []
    outcnn = []

    for i in range(len(y)):
        input_tensor = x[i, :, :]
        input_tensor = input_tensor[:, np.newaxis, :]
        output = attention_map(input_tensor, wr1, wg1, wb1, br1, bg1, 8, 28, 64)
        output = output.detach().numpy().squeeze()
        yhat = model(input_tensor)
        y_predict = yhat.argmax(dim=1)
        y_pre.append(y_predict.detach().numpy())
        input_tensor2 = features_out_hook[0][0, 0, :].reshape(1, 1, -1)

        wrr2 = wr2[:, 0, :].reshape(wr2.shape[0], 1, -1)
        wgg2 = wg2[:, 0, :].reshape(wg2.shape[0], 1, -1)
        wbb2 = wb2[:, 0, :].reshape(wb2.shape[0], 1, -1)
        output2 = attention_map(input_tensor2, wrr2, wgg2, wbb2, br2, bg2, 1, 1, 3)
        output2 = F.interpolate(output2.unsqueeze(0), length)
        output2 = output2.detach().numpy().squeeze()

        input_tensor3 = features_out_hook[1][0, 0, :].reshape(1, 1, -1)
        wrr3 = wr3[:, 0, :].reshape(wr3.shape[0], 1, -1)
        wgg3 = wg3[:, 0, :].reshape(wg3.shape[0], 1, -1)
        wbb3 = wb3[:, 0, :].reshape(wb3.shape[0], 1, -1)
        output3 = attention_map(input_tensor3, wrr3, wgg3, wbb3, br3, bg3, 1, 1, 3)
        input_tensor3 = F.interpolate(input_tensor3, length)
        output3 = F.interpolate(output3.unsqueeze(0), length)
        output3 = output3.detach().numpy().squeeze()

        input_tensor4 = features_out_hook[2][0, 0, :].reshape(1, 1, -1)

        wrr4 = wr4[:, 0, :].reshape(wr4.shape[0], 1, -1)
        wgg4 = wg4[:, 0, :].reshape(wg4.shape[0], 1, -1)
        wbb4 = wb4[:, 0, :].reshape(wb4.shape[0], 1, -1)
        output4 = attention_map(input_tensor4, wrr4, wgg4, wbb4, br4, bg4, 1, 1, 3)
        input_tensor4 = F.interpolate(input_tensor4, length)
        output4 = F.interpolate(output4.unsqueeze(0), length)
        output4 = output4.detach().numpy().squeeze()

        input_tensor5 = features_out_hook[3][0, 0, :].reshape(1, 1, -1)
        wrr5 = wr5[:, 0, :].reshape(wr5.shape[0], 1, -1)
        wgg5 = wg5[:, 0, :].reshape(wg5.shape[0], 1, -1)
        wbb5 = wb5[:, 0, :].reshape(wb5.shape[0], 1, -1)
        output5 = attention_map(input_tensor5, wrr5, wgg5, wbb5, br5, bg5, 1, 1, 3)
        input_tensor5 = F.interpolate(input_tensor5, length)
        output5 = F.interpolate(output5.unsqueeze(0), length)
        output5 = output5.detach().numpy().squeeze()

        input_tensor6 = features_out_hook[4][0, 0, :].reshape(1, 1, -1)
        wrr6 = wr6[:, 0, :].reshape(wr6.shape[0], 1, -1)
        wgg6 = wg6[:, 0, :].reshape(wg6.shape[0], 1, -1)
        wbb6 = wb6[:, 0, :].reshape(wb6.shape[0], 1, -1)
        output6 = attention_map(input_tensor6, wrr6, wgg6, wbb6, br6, bg6, 1, 0, 3)
        input_tensor6 = F.interpolate(input_tensor6, length)
        output6 = F.interpolate(output6.unsqueeze(0), length)
        output6 = output6.detach().numpy().squeeze()

        output = np.abs(np.gradient(output[0, :]))
        output2 = np.abs(np.gradient(output2[0, :]))
        output3 = np.abs(np.gradient(output3[0, :]))
        output4 = np.abs(np.gradient(output4[0, :]))
        output5 = np.abs(np.gradient(output5[0, :]))
        output6 = np.abs(np.gradient(output6[0, :]))

        map = [output, output2, output3, output4, output5, output6]
        map = np.array(map)
        outputs.append(map)
        print('Process X: %d' % (i))

    # Saving qttention maps of the QCNN in *.csv file
    qmap = {}
    for j in range(10):
        idx = np.argwhere(y == j)
        temp = np.array(outputs[idx[0,0]:idx[0,-1] +1])
        t = temp[1, :, :]
        for i, l in enumerate(temp):
            if i > 0:
                t = np.hstack((t, l))
        qmap[j] = t

    pd_0 = pd.DataFrame(qmap[0])
    pd_1 = pd.DataFrame(qmap[1])
    pd_2 = pd.DataFrame(qmap[2])
    pd_3 = pd.DataFrame(qmap[3])
    pd_4 = pd.DataFrame(qmap[4])
    pd_5 = pd.DataFrame(qmap[5])
    pd_6 = pd.DataFrame(qmap[6])
    pd_7 = pd.DataFrame(qmap[7])
    pd_8 = pd.DataFrame(qmap[8])
    pd_9 = pd.DataFrame(qmap[9])
    pd_10 = pd.DataFrame(y_pre)
    pd_11 = pd.DataFrame(test_Y)

    if not os.path.exists('data/qmaps'):
        os.makedirs('data/qmaps')

    pd_0.to_csv('data/qmaps/cqmaps_%s_snr_%d_class_0.csv' % (chosen_data, SNR), header=False)
    pd_1.to_csv('data/qmaps/cqmaps_%s_snr_%d_class_1.csv' % (chosen_data, SNR), header=False,)
    pd_2.to_csv('data/qmaps/cqmaps_%s_snr_%d_class_2.csv' % (chosen_data, SNR), header=False,)
    pd_3.to_csv('data/qmaps/cqmaps_%s_snr_%d_class_3.csv' % (chosen_data, SNR), header=False,)
    pd_4.to_csv('data/qmaps/cqmaps_%s_snr_%d_class_4.csv' % (chosen_data, SNR), header=False,)
    pd_5.to_csv('data/qmaps/cqmaps_%s_snr_%d_class_5.csv' % (chosen_data, SNR), header=False,)
    pd_6.to_csv('data/qmaps/cqmaps_%s_snr_%d_class_6.csv' % (chosen_data, SNR), header=False,)
    pd_7.to_csv('data/qmaps/cqmaps_%s_snr_%d_class_7.csv' % (chosen_data, SNR), header=False,)
    pd_8.to_csv('data/qmaps/cqmaps_%s_snr_%d_class_8.csv' % (chosen_data, SNR), header=False,)
    pd_9.to_csv('data/qmaps/cqmaps_%s_snr_%d_class_9.csv' % (chosen_data, SNR), header=False,)
    pd_10.to_csv('data/qmaps/predict.csv', header=False,)
    pd_11.to_csv('data/qmaps/truelabel.csv', header=False,)

    # Saving input signals in *.csv file
    input = {}
    for j in range(10):
        idx = np.argwhere(test_Y == j)
        temp = test_X[idx[0, 0]: idx[-1, 0] + 1].ravel()
        input[j] = temp

    pd_0 = pd.DataFrame(input[0])
    pd_1 = pd.DataFrame(input[1])
    pd_2 = pd.DataFrame(input[2])
    pd_3 = pd.DataFrame(input[3])
    pd_4 = pd.DataFrame(input[4])
    pd_5 = pd.DataFrame(input[5])
    pd_6 = pd.DataFrame(input[6])
    pd_7 = pd.DataFrame(input[7])
    pd_8 = pd.DataFrame(input[8])
    pd_9 = pd.DataFrame(input[9])

    if not os.path.exists('data/input'):
        os.makedirs('data/input')
    pd_0.to_csv('data/input/input_%s_snr_%d_class_0.csv' % (chosen_data, SNR), header=False)
    pd_1.to_csv('data/input/input_%s_snr_%d_class_1.csv' % (chosen_data, SNR), header=False, )
    pd_2.to_csv('data/input/input_%s_snr_%d_class_2.csv' % (chosen_data, SNR), header=False, )
    pd_3.to_csv('data/input/input_%s_snr_%d_class_3.csv' % (chosen_data, SNR), header=False, )
    pd_4.to_csv('data/input/input_%s_snr_%d_class_4.csv' % (chosen_data, SNR), header=False, )
    pd_5.to_csv('data/input/input_%s_snr_%d_class_5.csv' % (chosen_data, SNR), header=False, )
    pd_6.to_csv('data/input/input_%s_snr_%d_class_6.csv' % (chosen_data, SNR), header=False, )
    pd_7.to_csv('data/input/input_%s_snr_%d_class_7.csv' % (chosen_data, SNR), header=False, )
    pd_8.to_csv('data/input/input_%s_snr_%d_class_8.csv' % (chosen_data, SNR), header=False, )
    pd_9.to_csv('data/input/input_%s_snr_%d_class_9.csv' % (chosen_data, SNR), header=False, )

    # input2 = {}
    # for j in range(10):
    #     idx = np.argwhere(test_Y == j)
    #     temp = test_X1[idx[0, 0]: idx[-1, 0] + 1].ravel()
    #     input2[j] = temp
    #
    # pd_0 = pd.DataFrame(input2[0])
    # pd_1 = pd.DataFrame(input2[1])
    # pd_2 = pd.DataFrame(input2[2])
    # pd_3 = pd.DataFrame(input2[3])
    # pd_4 = pd.DataFrame(input2[4])
    # pd_5 = pd.DataFrame(input2[5])
    # pd_6 = pd.DataFrame(input2[6])
    # pd_7 = pd.DataFrame(input2[7])
    # pd_8 = pd.DataFrame(input2[8])
    # pd_9 = pd.DataFrame(input2[9])
    #
    # pd_0.to_csv('data/input/rawinput_%s_snr_%d_class_0.csv' % (chosen_data, SNR), header=False)
    # pd_1.to_csv('data/input/rawinput_%s_snr_%d_class_1.csv' % (chosen_data, SNR), header=False, )
    # pd_2.to_csv('data/input/rawinput_%s_snr_%d_class_2.csv' % (chosen_data, SNR), header=False, )
    # pd_3.to_csv('data/input/rawinput_%s_snr_%d_class_3.csv' % (chosen_data, SNR), header=False, )
    # pd_4.to_csv('data/input/rawinput_%s_snr_%d_class_4.csv' % (chosen_data, SNR), header=False, )
    # pd_5.to_csv('data/input/rawinput_%s_snr_%d_class_5.csv' % (chosen_data, SNR), header=False, )
    # pd_6.to_csv('data/input/rawinput_%s_snr_%d_class_6.csv' % (chosen_data, SNR), header=False, )
    # pd_7.to_csv('data/input/rawinput_%s_snr_%d_class_7.csv' % (chosen_data, SNR), header=False, )
    # pd_8.to_csv('data/input/rawinput_%s_snr_%d_class_8.csv' % (chosen_data, SNR), header=False, )
    # pd_9.to_csv('data/input/rawinput_%s_snr_%d_class_9.csv' % (chosen_data, SNR), header=False, )
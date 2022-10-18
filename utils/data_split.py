

from matplotlib import pyplot as plt
from scipy.io import loadmat, savemat
import numpy as np
import os
from sklearn import preprocessing  # 0-1编码
from sklearn.model_selection import StratifiedShuffleSplit  # 随机划分，保证每一类比例相同

'''
Splitting the data set and generating noisy data
'''

def wgn(x, snr):
    snr = 10 ** (snr / 10.0)
    xpower = np.sum(np.absolute(x) ** 2, axis=0) / x.shape[0]
    npower = xpower / snr
    return np.random.standard_normal(x.shape) * np.sqrt(npower)


def add_noise(data, snr_num):
    rand_data = wgn(data, snr_num)
    n_data = data + rand_data
    return n_data, rand_data

def preprocess(d_path, noise=False, snr=0):
    filenames = os.listdir(d_path)

    def capture():
        files = {}
        for i in filenames:
            file_path = os.path.join(d_path, i)
            file = loadmat(file_path)
            file_keys = file.keys()
            for key in file_keys:
                if 'DE' in key:
                    files[i] = file[key].ravel()
        return files

    def slice_enc(data, noise, snr):

        noised_data_dict1 = {}
        noise_dict1 = {}
        noised_data_dict2 = {}
        noise_dict2 = {}
        for key, val in data.items():
            slice_data = val
            train_data= slice_data[:len(slice_data) * 2 // 3]
            test_data = slice_data[len(slice_data) * 2 // 3:]
            if noise:
                noised_data1, white_noise1 = add_noise(train_data, snr)
                noised_data2, white_noise2 = add_noise(test_data, snr)

            noised_data_dict1[key] = noised_data1
            noise_dict1[key] = white_noise1
            noised_data_dict2[key] = noised_data2
            noise_dict2[key] = white_noise2
        return noised_data_dict1, noise_dict1, noised_data_dict2, noise_dict2

    def save_mat(d_path, noised_data_dict1, noised_data_dict2, snr):
        path1 = d_path + '_TrainNoised_' + str(snr)
        if not os.path.exists(path1):
            os.mkdir(path1)
        for k, dat in noised_data_dict1.items():
            savemat(os.path.join(path1, k), {'DE': dat})

        path1 = d_path + '_TestNoised_' + str(snr)
        if not os.path.exists(path1):
            os.mkdir(path1)
        for k, dat in noised_data_dict2.items():
            savemat(os.path.join(path1, k), {'DE': dat})

    # 从所有.mat文件中读取出数据的字典
    data = capture()
    noised_data_dict1, noise_dict1, noised_data_dict2, noise_dict2 = slice_enc(data, noise, snr)
    save_mat(d_path, noised_data_dict1, noised_data_dict2, snr)


if __name__ == "__main__":
    path = '../data/0HP' # change dataset file folders to split data
    preprocess(d_path=path,
                noise=True,
                snr=0)

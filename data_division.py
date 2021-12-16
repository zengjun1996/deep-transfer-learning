"""
此部分代码用于将数据集划分为训练集、验证集以及测试集，并进行相应的处理
"""

import h5py
import numpy as np

envir = 'CDL-B'  # 可以修改为CDL-C、CDL-D、CDL-E
path = envir + '.mat'
data = h5py.File(path, 'r')
data = data['mimo_downlink']

print("训练集加载中")
tem = data[:, :, :, :, 0:4000]  # (32, 2, 14, 72, 4000)
real = tem['real'].reshape(32, 28, 72, 4000).transpose()
imag = tem['imag'].reshape(32, 28, 72, 4000).transpose()
data_train = np.empty(shape=(4000, 72, 28, 32, 2), dtype=np.float32)
data_train[:, :, :, :, 0] = real
data_train[:, :, :, :, 1] = imag
file1 = h5py.File('data_train.mat', 'w')
file1.create_dataset('data_train', data=data_train)

print("验证集加载中")
tem = data[:, :, :, :, 4000:5000]  # (32, 2, 14, 72, 1000)
real = tem['real'].reshape(32, 28, 72, 1000).transpose()
imag = tem['imag'].reshape(32, 28, 72, 1000).transpose()
data_val = np.empty(shape=(1000, 72, 28, 32, 2), dtype=np.float32)
data_val[:, :, :, :, 0] = real
data_val[:, :, :, :, 1] = imag
file2 = h5py.File('data_val.mat', 'w')
file2.create_dataset('data_val', data=data_val)

print("测试集加载中")
tem = data[:, :, :, :, 5000:10000]  # (32, 2, 14, 72, 5000)
real = tem['real'].reshape(32, 28, 72, 5000).transpose()
imag = tem['imag'].reshape(32, 28, 72, 5000).transpose()
data_test = np.empty(shape=(5000, 72, 28, 32, 2), dtype=np.float32)
data_test[:, :, :, :, 0] = real
data_test[:, :, :, :, 1] = imag
file3 = h5py.File('data_test.mat', 'w')
file3.create_dataset('data_test', data=data_test)

print('{}数据集划分完成'.format(envir))

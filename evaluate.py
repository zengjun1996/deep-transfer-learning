"""
此部分代码用于对迁移学习得到的模型进行评估
"""

from keras.models import load_model
import numpy as np
import h5py
import os
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
session = tf.Session(config=tf_config)

envir = 'CDL_B'  # 可以修改为CDL-C、CDL-D、CDL-E
filepath = envir + '/data_test.mat'
compression_ratio = '1_8'
epoch = 50

model = load_model('model_1_8.h5')  # 此处可以选择已经训练好的迁移学习模型
data = h5py.File(filepath, 'r')
data_test = data['data_test']

data_predict = model.predict(data_test)
loss_testset = model.evaluate(data_test, data_test)
print('测试集损失为: {}'.format(loss_testset))

# 计算NMSE以及余弦相似度rho
data_test_real = np.reshape(data_test[:, :, :, :, 0], (len(data_test), -1))
data_test_imag = np.reshape(data_test[:, :, :, :, 1], (len(data_test), -1))
data_test_C = data_test_real + 1j*data_test_imag    # (5000, 72*28*32)

data_predict_real = np.reshape(data_predict[:, :, :, :, 0], (len(data_predict), -1))
data_predict_imag = np.reshape(data_predict[:, :, :, :, 1], (len(data_predict), -1))
data_predict_C = data_predict_real + 1j*data_predict_imag    # (5000, 72*28*32)

n1 = np.sqrt(np.sum(abs(np.conj(data_test_C)*data_test_C), axis=1))  # (5000, 1)
n2 = np.sqrt(np.sum(abs(np.conj(data_predict_C)*data_predict_C), axis=1))  # (5000, 1)
aa = np.sum(abs(np.conj(data_test_C)*data_predict_C), axis=1)
rho = np.mean(aa/(n1*n2))
power = np.sum(abs(data_test_C)**2, axis=1)
mse = np.sum(abs(data_test_C-data_predict_C)**2, axis=1)
print('在{}环境中'.format(envir))
print('当压缩率为:{}'.format(compression_ratio))
print('epoch为: {}'.format(epoch))
print('NMSE为:{}'.format(10*np.log10(np.mean(mse/power))))
print('余弦相似度为:{}'.format(rho))

"""
此部分代码用于加载预训练模型进行迁移学习训练
"""

from keras.models import load_model
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.optimizers import Adam
import h5py
import tensorflow as tf
import numpy as np
import time
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
session = tf.Session(config=tf_config)

envir = 'CDL_B'  # 可以修改为CDL-C、CDL-D、CDL-E
compression_ratio = '1_8'  # 选择不同压缩率下的预训练模型进行迁移学习，可以选择1/8,1/64,1/128以及1/256的模型
# 加载训练集、验证集以及测试集
filepath1 = envir + '/data_train.mat'
filepath2 = envir + '/data_val.mat'
filepath3 = envir + '/data_test.mat'

model = load_model('model_1_8.h5')
# for i in range(18):    # 将模型的某些层进行冻结，只训练部分网络
#     model.layers[i].trainable = False
adam = Adam(lr=0.0001)
model.compile(optimizer=adam, loss='mse', metrics=['accuracy'])
# print(model.summary())

print("训练集加载中")
data = h5py.File(filepath1, 'r')  # (4000, 72, 28, 32, 2)
data_train = data['data_train'][0:4000, :, :, :, :]

print("验证集加载中")
data = h5py.File(filepath2, 'r')  # (1000, 72, 28, 32, 2)
data_val = data['data_val']

print('训练集与验证集加载完毕')

initial_loss_train = model.evaluate(data_train, data_train)
print('训练集初始损失为: {}'.format(initial_loss_train))

path = 'F_' + envir + '_' + compression_ratio + '_{}'.format(time.strftime('%m_%d_%H_%M'))
tensorboard = TensorBoard(log_dir='logs/{}'.format(path))
string = 'saved_models/F_model_{}_{}'.format(envir, compression_ratio)
path1 = string+'_{epoch:03d}.h5'
checkpoint = ModelCheckpoint(filepath=path1, monitor='val_loss', period=10, verbose=1, save_best_only=True)
time_start = time.perf_counter()
record = model.fit(data_train, data_train,
                   epochs=200,
                   batch_size=50,
                   shuffle='batch',
                   validation_data=(data_val, data_val),
                   callbacks=[tensorboard, checkpoint])

time_end = time.perf_counter()
print('模型训练时间: {:.2f}s'.format(time_end-time_start))

print("测试集加载中")
data = h5py.File(filepath3, 'r')  # (5000, 72, 28, 32, 2)
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
print('NMSE为:{}'.format(10*np.log10(np.mean(mse/power))))
print('余弦相似度为:{}'.format(rho))

# loss = []
# for i in range(11):
#     loss.append(record.history['loss'][50*i])
# np.savetxt('loss.csv', loss)

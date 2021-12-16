import tensorflow as tf
from keras.layers import Input, add, LeakyReLU, Conv3DTranspose, Conv3D
from keras.models import Model
from keras.optimizers import adam
from keras import backend as K
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, TensorBoard
import numpy as np
import h5py
import time
import os
import math

# 指定GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
K.set_session(tf.Session(config=config))


def scheduler(epoch):
    # 每隔50个epoch，学习率减小为原来的0.8
    if epoch % 50 == 0 and epoch != 0:
        lr = K.get_value(autoencoder.optimizer.lr)
        K.set_value(autoencoder.optimizer.lr, lr * 0.8)
        print("lr changed to {}".format(lr * 0.8))
    return K.get_value(autoencoder.optimizer.lr)


def network_3d(y, residualnum):

    def add_common_layers(y):
        y = LeakyReLU()(y)
        return y

    def residual_block_decoded(y):
        shortcut = y

        y = Conv3D(8, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(y)
        y = add_common_layers(y)

        y = Conv3D(16, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(y)
        y = add_common_layers(y)

        y = Conv3D(2, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(y)
        y = add_common_layers(y)

        y = add([shortcut, y])

        y = LeakyReLU()(y)

        return y

    # 提取特征
    y = Conv3D(2, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(y)
    y = add_common_layers(y)
    y = Conv3D(8, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(y)
    y = add_common_layers(y)
    y = Conv3D(16, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(y)
    y = add_common_layers(y)
    y = Conv3D(32, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(y)
    y = add_common_layers(y)
    y = Conv3D(16, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(y)
    y = add_common_layers(y)
    y = Conv3D(8, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(y)
    y = add_common_layers(y)
    y = Conv3D(2, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(y)
    y = add_common_layers(y)

    # 压缩
    y = Conv3D(2, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same')(y)
    y = add_common_layers(y)
    # y = Conv3D(2, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same')(y)
    # y = add_common_layers(y)
    # y = Conv3D(2, kernel_size=(3, 3, 3), strides=(2, 1, 1), padding='same')(y)
    # y = add_common_layers(y)
    # y = Conv3D(2, kernel_size=(3, 3, 3), strides=(2, 1, 2), padding='same')(y)
    # y = add_common_layers(y)
    # 传输，视为完美传输

    # 解压缩
    y = Conv3DTranspose(2, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same')(y)
    y = add_common_layers(y)
    # y = Conv3DTranspose(2, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same')(y)
    # y = add_common_layers(y)
    # y = Conv3DTranspose(2, kernel_size=(3, 3, 3), strides=(2, 1, 1), padding='same')(y)
    # y = add_common_layers(y)
    # y = Conv3DTranspose(2, kernel_size=(3, 3, 3), strides=(2, 2, 2), padding='same')(y)
    # y = add_common_layers(y)

    # 恢复
    for i in range(residualnum):
        y = residual_block_decoded(y)

    return y


img_depth = 72
img_height = 28
img_width = 32
img_channels = 2
img_total = img_depth * img_height * img_width * img_channels
residual_num = 3
compression_ratio = '1_8'

# 搭建3d模型
image_tensor = Input(shape=(img_depth, img_height, img_width, img_channels))
network_output = network_3d(image_tensor, residual_num)
autoencoder = Model(inputs=[image_tensor], outputs=[network_output])
reduce_lr = LearningRateScheduler(scheduler)
ADAM = adam(lr=0.001)
autoencoder.compile(optimizer=ADAM, loss='mse', metrics=['accuracy'])
print(autoencoder.summary())

# 加载数据
csi_dl_train = h5py.File("data_train_B.mat", 'r')['data_train']
csi_dl_val = h5py.File("data_val_B.mat", 'r')['data_val']


# 训练和评估模型
path = 'CDL_B_1_8' + '_{}'.format(time.strftime('%m_%d_%H_%M'))
tensorboard = TensorBoard(log_dir='logs/{}'.format(path))
string = 'saved_models/model_{}_CDL_B'.format(compression_ratio)
path1 = string+'_{epoch:03d}.h5'
checkpoint = ModelCheckpoint(filepath=path1, monitor='val_loss', period=10, verbose=1, save_best_only=True)
tStart_ = time.time()
autoencoder.fit(csi_dl_train, csi_dl_train,
                epochs=200,
                validation_data=(csi_dl_val, csi_dl_val),
                batch_size=50,
                shuffle='batch',
                callbacks=[tensorboard, reduce_lr, checkpoint])
tEnd_ = time.time()
print("training process cost %f sec" % (tEnd_-tStart_))


# calculating NMSE
csi_dl_test = h5py.File("data_test_B.mat", 'r')['data_test']
csi_dl_test_hat = autoencoder.predict(csi_dl_test)
csi_dl_test_real = np.reshape(csi_dl_test[:, :, :, :, 0], (len(csi_dl_test), -1))
csi_dl_test_imag = np.reshape(csi_dl_test[:, :, :, :, 1], (len(csi_dl_test), -1))
csi_dl_test_C = csi_dl_test_real + 1j*csi_dl_test_imag
csi_dl_test_hat_real = np.reshape(csi_dl_test_hat[:, :, :, :, 0], (len(csi_dl_test_hat), -1))
csi_dl_test_hat_imag = np.reshape(csi_dl_test_hat[:, :, :, :, 1], (len(csi_dl_test_hat), -1))
csi_dl_test_hat_C = csi_dl_test_hat_real + 1j*csi_dl_test_hat_imag
n1 = np.sqrt(np.sum(abs(np.conj(csi_dl_test_C)*csi_dl_test_C), axis=1))
n2 = np.sqrt(np.sum(abs(np.conj(csi_dl_test_hat_C)*csi_dl_test_hat_C), axis=1))
aa = np.sum(abs(np.conj(csi_dl_test_C)*csi_dl_test_hat_C), axis=1)
rho = np.mean(aa/(n1*n2))
power = np.sum(abs(csi_dl_test_C)**2, axis=1)
mse = np.sum(abs(csi_dl_test_C-csi_dl_test_hat_C)**2, axis=1)
print("When Compression ratio is", compression_ratio)
print("NMSE is ", 10*math.log10(np.mean(mse/power)))
print("Correlation is ", rho)

# Testing data
tStart = time.time()
loss = autoencoder.evaluate(csi_dl_test, csi_dl_test)
tEnd = time.time()
print("testing process cost %f sec" % (tEnd - tStart))
print('test loss:', loss)

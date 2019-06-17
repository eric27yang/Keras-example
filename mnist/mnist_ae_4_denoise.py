# stacked auto encoder
# 基于卷积神经网络，实现图像去噪

import keras.backend as K
from keras.datasets import mnist
from keras.layers import Input,Dense,Conv2D,MaxPooling2D,UpSampling2D
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard

epochs=5
batch_size=128


# load data
# 数据格式为通道在前
(x_train,_),(x_test,_)=mnist.load_data()

x_train=x_train.reshape((len(x_train),28,28,1))
x_test=x_test.reshape((len(x_test),28,28,1))
x_train=x_train.astype('float32')
x_test=x_test.astype('float32')
x_train/=255
x_test/=255
print('#Get train samples',x_train.shape[0])
print('#Get test samples',x_test.shape[0])

# 添加噪声
noise_factor=0.3
# 从均值为0,标准差为1的正态分布中采样噪声
x_train_noise=x_train+noise_factor*np.random.normal(loc=0.0,scale=1.0,size=x_train.shape)
x_test_noise=x_test+noise_factor*np.random.normal(loc=0.0,scale=1.0,size=x_test.shape)
# 将数据限制在0-1的范围
x_train_noise = np.clip(x_train_noise, 0., 1.)
x_test_noise = np.clip(x_test_noise, 0., 1.)
# 可视化污染后的图片
n = 10
plt.figure(figsize=(20, 2))
for i in range(n):
    ax = plt.subplot(1, n, i+1)
    plt.imshow(x_test_noise[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
# 隐层神经元个数，即code维度
encoding_dim=32

# ----------输入层
input_img=Input(shape=(28,28,1))
# ----------编码层
# 为了提高重建质量，在每一层使用更多的卷积核
x=Conv2D(32,(3,3),activation='relu',padding='same')(input_img)
x=MaxPooling2D((2,2),padding='same')(x)
x=Conv2D(32,(3,3),activation='relu',padding='same')(x)
encoded=MaxPooling2D((2,2),padding='same')(x)
# ----------解码层
x=Conv2D(32,(3,3),activation='relu',padding='same')(encoded)
x=UpSampling2D((2,2))(x)
x=Conv2D(32,(3,3),activation='relu',padding='same')(x)
x=UpSampling2D((2,2))(x)

# ----------输出层
decoded=Conv2D(1,(3,3),activation='sigmoid',padding='same')(x)

# 自编码器模型定义
autoencoder=Model(input_img,decoded)

autoencoder.summary()

# 编译
autoencoder.compile(optimizer='adam',loss='binary_crossentropy')

# 训练
# 在终端中，输入tensorboard --logdir=/tmp/autoencoder，打开tensorboard服务器
# 加入回调
autoencoder.fit(x_train_noise,
                x_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(x_test_noise,x_test),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

# encode and decode some imgs
decoded_imgs=autoencoder.predict(x_test_noise)

# 可视化
# 可视化几个图
n=10
plt.figure(figsize=(20,4))# figsize指定总的图像大小
for i in range(n):
    # 绘制原图
    # subplot(numRows, numCols, plotNum)
    ax=plt.subplot(2,n,i+1)
    # 原本的一维向量展成28×28图像
    plt.imshow(x_test_noise[i].reshape(28,28))
    plt.gray()
    # 隐藏坐标轴
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # 绘制重建图
    ax=plt.subplot(2,n,i+1+n)
    plt.imshow(decoded_imgs[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()

# epoch=20，loss=0.0783






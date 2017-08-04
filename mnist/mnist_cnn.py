# refer to keras/examples/mnist_cnn.py
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dropout,Flatten,Dense
from keras.losses import categorical_crossentropy
from keras import backend as K

batch_size=128
num_classes=10
epochs=5

# data,shuffled and split between train and test sets
(x_train,y_train),(x_test,y_test)=mnist.load_data()

# input shape
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
    x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)
    input_shape = (1, 28, 28)
else:
    # 后端使用tensorflow时，即tf模式下，
    # 会将100张RGB三通道的16*32彩色图表示为(100,16,32,3)，
    # 第一个维度是样本维，表示样本的数目，
    # 第二和第三个维度是高和宽，
    # 最后一个维度是通道维，表示颜色通道数
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)

x_train=x_train.astype('float32')
x_test=x_test.astype('float32')
# 归一化
x_train/=255
x_test/=255
# convert class vectors to binary class matrices
y_train=to_categorical(y_train,10)
y_test=to_categorical(y_test,10)

print('#Get',x_train.shape[0],'train examples')
print('#Get',x_test.shape[0],'test examples')

# network structure
model=Sequential()

# ----------卷积层
# 当使用该层为第一层时，应提供input_shape参数
# filters代表卷积核的数目，kernel代表卷积核的大小
model.add(Conv2D(filters=32,
                 kernel_size=(3,3),
                 activation='relu',
                 input_shape=input_shape))
# ----------卷积层
model.add(Conv2D(filters=64,
                 kernel_size=(3,3),
                 activation='relu'))
# ----------池化层
# pool_size代表池化范围
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
# ----------Flatten层
# 把多维输入进行一维化，常用在卷积层到全连接层的过渡
model.add(Flatten())
# ----------全连接层
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
# ----------输出层
model.add(Dense(10,activation='softmax'))

model.summary()

# compile
model.compile(loss=categorical_crossentropy,optimizer='adam',metrics=['accuracy'])

# fit
model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,verbose=1)
score=model.evaluate(x_test,y_test,verbose=0)
print('#Test loss:',score[0])
print('#Test accuracy:',score[1])

# Get acc=0.9831,total loss=0.0697 in epoch 3


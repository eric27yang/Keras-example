# refer to keras/examples/mnist_mlp.py

from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.datasets import mnist
from keras.utils import np_utils

batch_size=100
epochs=5

# mnist data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# 纵坐标每一个训练样本是28×28的图像，横坐标是样本个数
x_train = x_train.reshape(60000, 28*28)
x_test = x_test.reshape(10000, 28*28)
# label共有0-9类，因为softmax要求是binary class matrics，所以进行转化
y_train=np_utils.to_categorical(y_train,10)
y_test=np_utils.to_categorical(y_test,10)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# 归一化
x_train /= 255
x_test /= 255
print("#Get",x_train.shape[0], 'train samples')
print('#Get',x_test.shape[0], 'test samples')

# network structure
model=Sequential()
model.add(Dense(units=500,input_dim=28*28))
model.add(Activation('sigmoid'))
model.add(Dense(units=500))
model.add(Activation('sigmoid'))
model.add(Dense(units=10))
model.add(Activation('softmax'))

model.summary()

# how to train
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# train
model.fit(x_train,
          y_train,
          batch_size=batch_size,
          epochs=epochs)

# evaluation
score=model.evaluate(x_test,y_test)
print('#Total loss on Testing Set',score[0])
print('#Accuracy of Testing Set',score[1])
result=model.predict(x_test)

# Get acc=0.9724,total loss=0.0866


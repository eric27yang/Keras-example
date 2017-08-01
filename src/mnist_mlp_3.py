# refer to keras/examples/mnist_mlp.py
# compared to 2
# --use early stopping

from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
from keras.datasets import mnist
from keras.utils import np_utils
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping

batch_size=128
epochs=20

# mnist data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# 纵坐标每一个训练样本是28×28的图像，横坐标是样本个数
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
# label共有0-9类，因为softmax要求是binary class matrics，所以进行转化
y_train=np_utils.to_categorical(y_train,10)
y_test=np_utils.to_categorical(y_test,10)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print("#Get",x_train.shape[0], 'train samples')
print('#Get',x_test.shape[0], 'test samples')

# network structure
model=Sequential()
model.add(Dense(units=512,
                activation="relu",
                input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(units=512,
                activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(units=10,
                activation="softmax"))

model.summary()

# how to train
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

# early stopping
early_stopping=EarlyStopping(monitor="val_loss",patience=2)

# train
model.fit(x_train,
          y_train,
          batch_size=batch_size,
          validation_split=0.2,
          callbacks=[early_stopping],
          epochs=epochs)

# evaluation
score=model.evaluate(x_test,y_test)
print('#Total loss on Testing Set',score[0])
print('#Accuracy of Testing Set',score[1])
result=model.predict(x_test)

# Get acc=0.979,total loss=0.08903


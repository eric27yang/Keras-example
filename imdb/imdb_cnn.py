from keras.models import Sequential
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.layers import Dense,Dropout,Embedding,Conv1D,GlobalMaxPooling1D


batch_size=20
epochs=5
max_features=5000
maxlen=400
embedding_dims=50

# imdb数据集：来自IMDB的影评，被标注为正面/负面两种评价
# num_words:要考虑的最常见的单词数
(x_train,y_train),(x_test,y_test)=imdb.load_data(num_words=max_features)
print('#Get train seq',len(x_train))
print('#Get test seq',len(x_test))

print('#Pad sequences')
# 将序列填充为同样长度
# maxlen：最大序列长度，任何长度大于此值的序列将被截断
# X_train和X_test：序列的列表，每个序列都是词下标的列表。
# 单词的下标基于它在数据集中出现的频率标定，例如整数3所编码的词为数据集中第3常出现的词
# 如果指定了num_words，则序列中可能的最大下标为num_words-1。
# 如果指定了maxlen，则序列的最大可能长度为maxlen
x_train=sequence.pad_sequences(x_train,maxlen=maxlen)
x_test=sequence.pad_sequences(x_test,maxlen=maxlen)
print('#x_train shape:',x_train.shape)
print('#x_test shape:',x_test.shape)

# build model
model=Sequential()
# -------------嵌入层
# 将词下标转化为向量
# ”词向量”（词嵌入）是将一类将词的语义映射到向量空间中去的自然语言处理技术。
# 向量之间的距离（例如，任意两个向量之间的L2范式距离或更常用的余弦距离）一定程度上表征了的词之间的语义关系
# 原理：参考https://keras-cn.readthedocs.io/en/latest/blog/word_embedding/
# 一个Embedding层的输入应该是一系列的整数序列，比如一个2D的输入，它的shape值为(samples, indices)，
# 也就是一个samples行，indeces列的矩阵。每一次的batch训练的输入应该被padded成相同大小
# （尽管Embedding层有能力处理不定长序列，如果你不指定数列长度这一参数） dim).
# 所有的序列中的整数都将被对应的词向量矩阵（比如训练好的GloVe词向量或word2vec词向量）中对应的列（也就是它的词向量）代替,
# 比如序列[1,2]将被序列[词向量[1],词向量[2]]代替。这样，输入一个2D张量后，我们可以得到一个3D张量。
model.add(Embedding(max_features,embedding_dims,input_length=maxlen))
model.add(Dropout(0.2))
# -------------卷积层
model.add(Conv1D(filters=250,kernel_size=3,padding='valid',activation='relu',strides=1))
# -------------池化层
model.add(GlobalMaxPooling1D())
# ------------全连接层
model.add(Dense(250,activation='relu'))
model.add(Dropout(0.2))
# ------------输出层
model.add(Dense(1,activation='sigmoid'))

# complie
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

# train
model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs)
# test
score=model.evaluate(x_test,y_test,verbose=0)
print('#Test loss:',score[0])
print('#Test accuracy:',score[1])

# Get epoch 2, acc=0.905, total loss=0.233
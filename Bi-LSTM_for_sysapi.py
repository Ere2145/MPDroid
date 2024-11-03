import numpy as np
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import Bidirectional, LSTM
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.metrics import Precision,Recall


# 加载数据
def load_data(path):
    data = []
    for file in os.listdir(path):
        if file.endswith('.npy'):
            data.append(np.load(os.path.join(path, file)))
    return data

if __name__=='__main__':
    mal_data = load_data('./sys_api_vec/mal')
    bni_data = load_data('./sys_api_vec/bni')

    # 创建标签
    mal_labels = np.ones(len(mal_data))
    bni_labels = np.zeros(len(bni_data))

    # 合并数据
    data = mal_data + bni_data
    labels = np.concatenate((mal_labels, bni_labels))

    # 获取最大长度
    maxlen = 2048

    # 对数据进行填充或截断
    data = pad_sequences(data, maxlen=maxlen, dtype='float32', padding='post', truncating='post')

    # 划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    # 创建嵌入模型
    embedding_model = Sequential()
    embedding_model.add(Embedding(maxlen, output_dim=256))
    embedding_model.add(Bidirectional(LSTM(64)))

    # 创建分类模型
    classification_model = Sequential()
    classification_model.add(Dropout(0.5))
    classification_model.add(Dense(1, activation='sigmoid'))

    # 连接嵌入模型和分类模型
    model = Sequential()
    model.add(embedding_model)
    model.add(classification_model)

    opt=tf.keras.optimizers.Adam(learning_rate=0.001)

    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy',Precision(),Recall()])

    # 训练模型
    history=model.fit(x_train, y_train, batch_size=16, epochs=1)

    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train'], loc='upper right')
    plt.savefig('./figs/Bi-LSTM_loss_10.png')

    # 测试模型
    score = model.evaluate(x_test, y_test, batch_size=16)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

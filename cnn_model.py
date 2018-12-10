# coding: utf-8

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import tensorflow as tf
import setting
import load_data
import os

class cnn_model:

    def __init__(self,setting):
        self.setting = setting
    

    def My_Accuracy(self, y_true, y_pred):
        """
        输入：以 one-hot vector 格式的真实标签和预测标签
        功能：评价模型性能
        输出：预测的准确度
        """
        
        res = tf.argmax(y_true,axis=1)
        pre = tf.argmax(y_pred,axis=1)
        acc = tf.cast(tf.equal(res, pre),tf.float32)
        
        return tf.reduce_mean(acc)

    def model(self):
        """
        定义卷积神经网络模型
        """
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='softmax'))

        opt = keras.optimizers.Adam(lr=self.setting.learning_rate, beta_1=self.setting.beta_1, beta_2=self.setting.beta_2, epsilon=None, decay=0.0, amsgrad=False)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=[self.My_Accuracy])
        return model


    def train(self):
        """
        训练模型
        """
        fashion_mnist_model = self.model()
        # 加载已有模型
        if os.path.exists(self.setting.save_model):
            fashion_mnist_model.load_weights(self.setting.save_model)

        train_x, train_y, _, _ = load_data.Load_And_Preprocess_Data(self.setting)
        fashion_mnist_model.fit(train_x, train_y, batch_size=self.setting.batch_size, epochs=self.setting.epochs)
        
        # 存储训练好的模型
        fashion_mnist_model.save(self.setting.save_model)


    def test(self):
        """
        通过模型预测测试集标签，输出准确率
        """
        _, _, test_x, test_y = load_data.Load_And_Preprocess_Data(setting)
        fashion_mnist_model = self.model()
        fashion_mnist_model.load_weights(self.setting.save_model)
        _,acc = fashion_mnist_model.evaluate(test_x, test_y, batch_size=32)
        print 'predict accuracy in test data is %.2f%%'%(acc*100)


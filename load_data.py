# coding: utf-8

import pandas as pd
import numpy as np
import keras

import setting


def Load_And_Preprocess_Data(setting):
    """
    导入训练集和测试集
    提取训练集和测试集的数据和标签
    将训练集图片由（1，784）转化为（28，28，1）
    并将标签转化为 one-hot vector格式
    输出 train_x, train_y, test_x, test_y
    """
    # 导入训练集和测试集
    train_df = pd.read_csv(setting.train_data)
    test_df = pd.read_csv(setting.test_data)

    # 提取图像和标签
    train_y = train_df.label
    train_x = train_df.iloc[:,1:]
    test_y = test_df.label
    test_x = test_df.iloc[:,1:]

    # 将训练集(60000,784) 转为(60000,28,28,1)，即将每个训练样本由原来的一维向量转化为(28,28,1)的图片
    train_x = np.reshape(np.array(train_x),[-1,28,28,1])
    test_x = np.reshape(np.array(test_x),[-1,28,28,1])

    # 将标签转化为 one-hot vector格式
    train_y = keras.utils.to_categorical(train_y, num_classes=10)
    test_y = keras.utils.to_categorical(test_y, num_classes=10)
    
    return train_x, train_y, test_x, test_y







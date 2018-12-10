# coding: utf-8


# 训练集和测试集路径
train_data = 'fashionmnist/fashion-mnist_train.csv'
test_data = 'fashionmnist/fashion-mnist_test.csv'

# 模型训练迭代次数
epochs = 20

# 每次参加训练的数据数目
batch_size=32

# 训练还是测试，默认为测试模式
mode = '-test'

# 保存模型
save_model = 'fashion_mnist_model.h5'

# 模型训练超参数
# 使用Adamy优化损失函数
learning_rate = 0.001
beta_1=0.9
beta_2=0.999
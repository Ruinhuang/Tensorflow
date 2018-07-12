import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

def nn():
    # 随机在-0.5到0.5之间生成200个等差数据点
    x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]
    noise = np.random.normal(0, 0.02, x_data.shape)
    # y = x^2 再加上noise值
    y_data = np.square(x_data) + noise

    # 定义两个placeholder(占位符);大小为n行1列
    x = tf.placeholder(tf.float32, [None, 1])
    y = tf.placeholder(tf.float32, [None, 1])

    # 定义神经网络中间层,链接输入层与输出层
    # 随机生成中间层的weights,中间层为10个神经元
    weights_L1 = tf.Variable(tf.random_normal([1, 10]))
    b1 = tf.Variable(tf.zeros([1, 10]))
    y1 = tf.matmul(x, weights_L1) + b1
    # 中间层的输出;激活函数为tanh
    L1 = tf.nn.tanh(y1)

    # 神经网络的输出层
    weights_L2 = tf.Variable(tf.random_normal([10, 1]))
    b2 = tf.Variable(tf.zeros([1, 1]))
    y2 = tf.matmul(L1, weights_L2) + b2
    prediction = tf.nn.tanh(y2)

    # 代价函数
    loss = tf.reduce_mean(tf.square(y - prediction))

    # 使用梯度下降训练,最小化代价函数
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    with tf.Session() as sess:
        # 变量初始化
        sess.run(tf.global_variables_initializer())
        # 训练2000次
        for _ in range(2000):
            # 将x_data和y_data传入x,y中
            sess.run(train_step, feed_dict={x: x_data, y: y_data})

        # 测试
        prediction_value = sess.run(prediction, feed_dict={x: x_data})

        # 画图
        plt.figure()
        plt.scatter(x_data, y_data)
        plt.plot(x_data, prediction_value, 'r-', lw=5)
        plt.show()

def mnist():

    # 载入数据集
    mnist = input_data.read_data_sets('data', one_hot=True)
    # 分批次,每个批次大小
    batch_size = 100
    # 计算一共有几个批次
    n_batch = mnist.train.num_examples // batch_size

    # 定义两个placeholder
    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])

    # 创建一个没有中间层的神经网络
    w = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    prediction = tf.nn.softmax(tf.matmul(x, w) + b)

    # 二次loss函数
    loss = tf.reduce_mean(tf.square(y - prediction))

    # 梯度下降
    train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

    # 初始化变量
    init = tf.global_variables_initializer()

    # 预测结果，结果存放在一个bool列表中
    # argmax函数返回的是最大值的下标(所在的位置)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
    # 准确率
    # cast函数是将correct_prediction类型由bool改变为float32,True->1.0,False->0.0
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(21):
            for batch in range(n_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})

            acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
            print("Iter: " + str(epoch) + ", Test Accuracy: " + str(acc))








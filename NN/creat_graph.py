import tensorflow as tf
import numpy as np

# 使用numpy生成100个随机点
x_data = np.random.rand(100)
# 真实的y值
y_data = x_data * 0.1 + 0.2


# 构造一个线性模型
b = tf.Variable(0.)
w = tf.Variable(0.)
y_pred = w * x_data + b

# 利用最小二乘法和梯度下降更新求解w和b
# 真实值减去预测值的差值的平方，再求均值得到代价函数
loss = tf.reduce_mean(tf.square(y_data - y_pred))

# 梯度下降优化器
optimizer = tf.train.GradientDescentOptimizer(0.2)
# 最小化代价函数loss
train = optimizer.minimize(loss)

# 初始化变量,tf中变量必须初始化
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for step in range(201):
        sess.run(train)
        if step % 10 == 0:
            print(step, sess.run([w, b]))

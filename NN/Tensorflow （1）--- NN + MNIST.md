# Tensorflow （1）--- NN + MNIST

## 基本概念

- 使用图(graphs)来表示计算任务；图中的节点称之为op(operation);一个op获得0个或多个Tensor，执行计算，产生0个或多个Tensor；图必须在Session里被启动
- 在被称之为会话(session)的上下文(context)中执行图
- 使用tensor表示数据，可以看作是一个n维的数组或列表
- 通过变量(variable)维护状态
- 使用feed和fetch可以为任意的操作赋值或者从中获取数据

![](https://www.z4a.net/images/2018/07/12/2.png)

## 示例

- 简单的示例

```python
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

```

- 结果：![](https://www.z4a.net/images/2018/07/12/3.png)
- 可以从结果看出，w不断地训练后可接近0.1,b可接近0.2

## 神经网络-回归

- 搭建一个神经网络，拟合y = x^2

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

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

```

- 结果：![](https://www.z4a.net/images/2018/07/12/6.png)

## MNIST 手写识别

### MNIST数据集

- 下载地址http://yann.lecun.com/exdb/mnist/

- 每张图片包含28 * 28个像素，将此数组展开为一个向量，长度是28 * 28 = 784
- 在MNIST数据集中mnist.train.images 是一个形状为[60000,784]的张量
- 图片里的每个像素的强度值介于0-1之间

![](https://www.z4a.net/images/2018/07/12/4.png)

- 标签是0-9的数字，我们要把标签转化为‘one-hot vectors’向量；比如标签0为([1,0,0,0,0,0,0,0,0,0]),标签3为([0,0,0,1,0,0,0,0,0,0])
- 因此，labels是一个[60000,10]的数字矩阵

### 神经网络构建

- 构建一个简单的没有隐藏层的神经网络

![](https://www.z4a.net/images/2018/07/12/5-2.png)

- 使用Softmax函数
- $$ softmax\left( x\right) _{i}=\dfrac {\exp \left( x_{i}\right) }{\sum _{j}\exp \left( x_{j}\right) } $$

  1. 例如手写识别输出[1,5,3] 
  2. e^1=2.718; e^5=148.413; e^3 = 20.086
  3. p1 = e^1 / (e^1 + e^5 + e^3) = 0.016; p2 = e^5 / (e^1 + e^5 + e^ 3) = 0.867；p3 = e^3 / (e^1 + e^5 + e^3) = 0.117

### 代码

```python
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

```

- 结果：![](https://www.z4a.net/images/2018/07/12/7f923a10c050717ab.png)
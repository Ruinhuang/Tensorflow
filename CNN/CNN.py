import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 载入数据集
mnist = input_data.read_data_sets("C:\\Code\\tensorflow\\basic_tutorial\\nn\\data", one_hot=True)

# 每个批次大小
batch_size = 100
# 计算一共有多少批次
n_batch = mnist.train.num_examples // batch_size


# 参数概要
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        # 平均值
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            # square平方；sqrt开根号
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        # 标准差
        tf.summary.scalar('stddev', stddev)
        # 最大值
        tf.summary.scalar('max', tf.reduce_max(var))
        # 最小值
        tf.summary.scalar('min', tf.reduce_min(var))
        # 直方图
        tf.summary.histogram('histogram', var)

# 初始化权值
def weight_variable(shape, name):
    # 生成一个截断的正态分布
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

# 初始化偏置值
def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

# 卷积层
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# 池化层
def max_pool_2x2(x):
    # ksize [1, x, y, 1]
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 命名空间
with tf.name_scope('input'):
    # 定义两个placeholder
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y = tf.placeholder(tf.float32, [None, 10], name='y-input')
    with tf.name_scope('x_image'):
        # 改变x的格式转为4d的向量[batch, in_height, in_width, in_channels]
        # -1代表先不考虑输入的图片例子多少这个维度，因为我们输入的图片是黑白的，因此channel是1，如果是RGB图像，那么channel就是3
        x_image = tf.reshape(x, [-1, 28, 28, 1], name='x-image')

with tf.name_scope('Conv1'):
    # 初始化第一个卷积层的权值和偏置
    with tf.name_scope('W_conv1'):
        # 5*5的采样窗口，32个卷积核从一个平面抽取特征
        W_conv1 = weight_variable([5, 5, 1, 32], name='W_conv1')
    with tf.name_scope('b_conv1'):
        # 每一个卷积核一个偏置值
        b_conv1 = bias_variable([32], name='b_conv1')

    # 把x_image和权值向量进行卷积，再加上偏置，然后用relu函数激活
    with tf.name_scope('conv2d_1'):
        conv2d_1 = conv2d(x_image, W_conv1) + b_conv1
    with tf.name_scope('relu'):
        h_conv1 = tf.nn.relu(conv2d_1)

    # 进行max-pooling
    with tf.name_scope('h_pool1'):
        h_pool1 = max_pool_2x2(h_conv1)

with tf.name_scope('Conv2'):
    # 初始化第二个卷积层的权值和偏置
    with tf.name_scope('W_conv2'):
        # 5*5的采样窗口，64个卷积核从32个平面抽取特征
        W_conv2 = weight_variable([5, 5, 32, 64], name='W_conv2')
    with tf.name_scope('b_conv2'):
        b_conv2 = bias_variable([64], name='b_conv2')

    # 将h_pool1和权值向量进行卷积，再加上偏置值，然后用relu函数激活
    with tf.name_scope('conv2d_2'):
        conv2d_2 = conv2d(h_pool1, W_conv2) + b_conv2
    with tf.name_scope('relu'):
        h_conv2 = tf.nn.relu(conv2d_2)
    with tf.name_scope('h_pool2'):
        h_pool2 = max_pool_2x2(h_conv2)

# 28*28的图片第一次卷积之后还是28*28，第一次池化后变成14*14
# 第二次卷积之后为14*14，第二次池化之后变为7*7
# 两次卷积池化之后得到64张7*7的平面

with tf.name_scope('fc1'):
    # 初始化第一个全连接层的权值
    with tf.name_scope('W_fc1'):
        W_fc1 = weight_variable([7*7*64, 1024], name='W_fc1')
    with tf.name_scope('b_fc1'):
        b_fc1 = bias_variable([1024], name='b_fc1')

    # 将池化层2的输出扁平化为1维
    with tf.name_scope('h_pool2_flat'):
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64], name='h_pool2_flat')

    # 第一个全连接层的输出
    with tf.name_scope('wx_plus_b1'):
        wx_plus_b1 = tf.matmul(h_pool2_flat, W_fc1) + b_fc1
    with tf.name_scope('relu'):
        h_fc1 = tf.nn.relu(wx_plus_b1)

    # keep_prob表示神经元的输出概率
    with tf.name_scope('keep_prob'):
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    with tf.name_scope('h_fc1_drop'):
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob, name='h_fc1_drop')

with tf.name_scope('fc2'):
    # 初始化第二个全连接层
    with tf.name_scope('W_fc2'):
        W_fc2 = weight_variable([1024, 10], name='W_fc2')
    with tf.name_scope('b_fc2'):
        b_fc2 = bias_variable([10], name='b_fc2')
    with tf.name_scope('wx_plus_b2'):
        wx_plus_b2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    with tf.name_scope('softmax'):
        # 计算输出
        prediction = tf.nn.softmax(wx_plus_b2)

# 交叉熵代价函数
with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction), name='cross_entropy')
    tf.summary.scalar('cross_entropy', cross_entropy)

# 使用Adam进行优化
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# 准确率
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        # 结果存放在一个bool列表中
        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    with tf.name_scope('accuracy'):
        # 准确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

# 合并所有的summary
merged = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter('logs/train', sess.graph)
    test_writer = tf.summary.FileWriter('logs/test', sess.graph)
    for i in range(1001):
        # 训练模型
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.5})
        # 记录训练集计算的参数
        summary = sess.run(merged, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})
        train_writer.add_summary(summary, i)
        # 计算测试集计算的参数
        batch_xs, batch_ys = mnist.test.next_batch(batch_size)
        summary = sess.run(merged, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.0})
        test_writer.add_summary(summary, i)

        if i % 100 == 0:
            test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
            train_acc = sess.run(accuracy, feed_dict={x: mnist.train.images[:10000], y: mnist.train.labels[:10000],
                                                      keep_prob: 1.0})
            print("Iter " + str(i) + ", Testing Accuracy=" + str(test_acc) + ", Training Accuracy=" + str(train_acc))

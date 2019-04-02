import tensorflow as tf
import ssl
ssl._create_default_https_context = ssl._create_unverified_context# 解决ssl证书错误问题
import input_data
# 导入训练数据

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 训练参数设置
learning_rate = 0.01
training_epochs = 50
batch_size = 100
display_step = 1

# 定义tf的输入
x = tf.placeholder(tf.float32,[None,784])# 根据mnist数据集而来，28*28=784
y = tf.placeholder(tf.float32,[None,10])# 0-9个数字，一共10个类别

# 设置权重
W = tf.Variable(tf.zeros([784,10]))# 一个784*10的矩阵，初始值都为0
b = tf.Variable(tf.zeros([10]))

# 构造模型 softmax函数可以将一个任意的K维度向量压缩到另一个K维向量中，使得每一个元素都在0-1之间，而且和为1
# Softmax = soft+max 不是最大值，是按照概率的"最大值"，但是其他的值也可以取的到

pred = tf.nn.softmax(tf.matmul(x,W)+b)#矩阵的乘法 matrix multiply
# Minimize error using cross entropy 交叉熵
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    # k开始循环训练
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # loop for all batchs
        for i in range(total_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                          y: batch_ys})
            avg_cost += c/total_batch
            # Display logs per epoch step
        if (epoch + 1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

    print("Optimization Finished!")
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

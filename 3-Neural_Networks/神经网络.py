import input_data
import tensorflow as tf
import numpy as np


mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# 设置学习参数
learning_rate = 7
num_steps = 500
batch_size = 128
display_step = 100

# 神经网络的参数
n_hidden_1 = 256
n_hidden_2 = 256
num_input = 784  # 代表输入的维数
num_classes = 10  # 代表10个类别，分别是0...9

# tf的输入
X = tf.placeholder("float", [None, num_input])  # None代表任意数量
Y = tf.placeholder("float", [None, num_classes])

weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}

# 定义模型
def neural_net(x):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])

    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])

    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']

    return out_layer


# 建造模型
logits = neural_net(X)
prediction = tf.nn.softmax(logits)

# 定义损失和目标
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# 评价模型
correct_pred = tf.equal(tf.arg_max(prediction, 1), tf.arg_max(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 初始化所有的模型
init = tf.global_variables_initializer()

# 开始训练
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)
    for step in range(1, num_steps+ 1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        #   进行优化
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))
    print("优化完成")

    print("测试准确性", sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))



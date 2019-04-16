import tensorflow as tf
import numpy as np
import csv
import sys
# todo:控制输入，分为三部分。
csv_file = csv.reader(open("train_set.csv", "r"))
csv.field_size_limit(sys.maxsize)
headerrow = next(csv_file)
print(headerrow)
# todo:目标是将数据分为两部分，一个是二维的article 和 word seg ，另外一个是一维度的class
def get_next_line_array():
    line = str(next(csv_file))
    line = line.replace("'", "").replace("[", "").replace("]", "").replace("\"", "")
    array = line.split(",")
    return array



# 训练参数设置
learning_rate = 0.01
training_epochs = 50000000
batch_size = 100
display_step = 1
# 定义tf的输入
x = tf.placeholder(tf.float32, [None, 190])
y = tf.placeholder(tf.float32, [None, 19])
# 设置权重
W = tf.Variable(tf.zeros([190, 19]))
b = tf.Variable(tf.zeros([19]))


pred = tf.nn.softmax(tf.matmul(x, W)+b)  # 矩阵的乘法 matrix multiply
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
        #total_batch = int(mnist.train.num_examples/batch_size)
        # loop for all batchs
        for i in range(batch_size):
            array = get_next_line_array()

            words = array[1]
            print(len(words))

            words = words.replace("\"", "").replace("'", "").split(" ")
            if len(words) > 191:
                words = words[1:191]
                # 将words类型转换为float的数组
                words = list(map(float, words))
                new_words = []
                for j in words:
                    new_words.append(j)
                new_words = np.reshape(new_words, (1, 190))
                batch_xs = new_words

                index = int(array[3].replace("\"", "").replace("'", "").replace(" ", ""))
                classess = np.zeros([19])
                classess[index-1] = 1
                classess = np.reshape(classess, (1, 19))
                batch_ys = classess
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                              y: batch_ys})
                avg_cost += c/batch_size
                # Display logs per epoch step
            if (epoch + 1) % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

    print("Optimization Finished!")
    # correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # # Calculate accuracy
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

import tensorflow as tf
import numpy as np
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import input_data

# 设置参数表示训练阶段和测试阶段
batch_size = 256
test_size = 512
# 定义图像的参数 这里是mnist的数据集，所以是28*28
img_size = 28
# 将数值0-9都设置一个类
num_class = 10


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape,stddev=0.01))# 产生一个随机的权重 random_normal 函数从符合正态分布的某些数值中取出来一个特定的值。shape：输出张量的形状，stddev 标准差

def model(X,w,w2,w3,w4,w_o,p_keep_conv,p_keep_hidden):
    conv1 = tf.nn.conv2d(X,w,strides=[1,1,1,1],padding="SAME")# tf.nn.conv2d进行卷积操作,对于每一个维度都设置称为1。padding是设置图像边界被0填充，保证输出的大小一致
    conv1_a = tf.nn.relu(conv1)# 传递给一个relu层，计算每个像素的x的max（x，0）函数，为公式增加一些非线性的东西，让模型可以学习出来更加复杂的函数
    conv1 = tf.nn.max_pool(conv1_a,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")# 这是一个2*2的最大池化操作，使用2*2的窗，选择每个窗中的最大值，然后移动两个两素，接着进入下一个窗
    conv1 = tf.nn.dropout(conv1,p_keep_conv)# 为了减少过拟合，使用dropout函数
    # 至此，第一个卷积层结束，然后按照同样的方法定义第二个和第三个卷积层
    conv2 = tf.nn.conv2d(conv1, w2, strides=[1,1,1,1],
                         padding="SAME")  # tf.nn.conv2d进行卷积操作,对于每一个维度都设置称为1。padding是设置图像边界被0填充，保证输出的大小一致
    conv2_a = tf.nn.relu(conv2)  # 传递给一个relu层，计算每个像素的x的max（x，0）函数，为公式增加一些非线性的东西，让模型可以学习出来更加复杂的函数

    conv2 = tf.nn.max_pool(conv2_a,ksize=[1,2,2,1],strides=[1, 2, 2, 1],
                           padding="SAME")  # 这是一个2*2的最大池化操作，使用2*2的窗，选择每个窗中的最大值，然后移动两个两素，接着进入下一个窗
    conv2 = tf.nn.dropout(conv2, p_keep_conv)  # 为了减少过拟合，使用dropout函数

    conv3 = tf.nn.conv2d(conv2, w3, strides=[1, 1, 1, 1],
                         padding="SAME")  # tf.nn.conv2d进行卷积操作,对于每一个维度都设置称为1。padding是设置图像边界被0填充，保证输出的大小一致
    conv3 = tf.nn.relu(conv3)  # 传递给一个relu层，计算每个像素的x的max（x，0）函数，为公式增加一些非线性的东西，让模型可以学习出来更加复杂的函数
    # 然后加入两个全联接层
    FC_layer = tf.nn.max_pool(conv3,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
    FC_layer = tf.reshape(FC_layer,[-1,w4.get_shape().as_list()[0]])# -1代表着任意的行数，（根据列自适应）
    FC_layer = tf.nn.dropout(FC_layer,p_keep_conv)
    # 输出层接受全联接层为输入，并接受权重张量w4
    output_layer = tf.nn.relu(tf.matmul(FC_layer,w4))
    output_layer = tf.nn.dropout(output_layer,p_keep_hidden)
    result = tf.matmul(output_layer,w_o)# result 是一个10维的向量，用于确定输出图像是10类中的哪一个
    return result

# 导入训练数据

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
trX , trY , TeX, TeY = mnist.train.images , mnist.train.labels , mnist.test.images , mnist.test.labels
# 输入集需要随着形状发生变化
trX = trX.reshape(-1,img_size,img_size,1)#第一个
TeX = TeX.reshape(-1,img_size,img_size,1)

# 定义输入和输出
X = tf.placeholder("float",[None,img_size,img_size,1],"X")
Y = tf.placeholder("float",[None,num_class],"Y")




# 构建第一层
w = init_weights([3,3,1,32])# 由输入张量的每一个小子集卷积而来，维度是3*3*1 ，32是这一层的特征值的数量

w2 = init_weights([3,3,32,64])# 输入数量为32，第二层的每一个神经元都是由第一个卷积层中的3*3*32个神经元卷积来的，64是输出特征数量

w3 = init_weights([3,3,64,128])# 第三个卷积层由3*3*64而来，输出的特征值的数量为128

# 第四层为全链接层
w4 = init_weights([128*4*4,625])

# 输出层
w_o = init_weights([625,num_class])# 输出是类的数量

p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")

py_x = model(X,w,w2,w3,w4,w_o,p_keep_conv,p_keep_hidden)
Y_ = tf.nn.softmax_cross_entropy_with_logits(logits=py_x,labels=Y)
# 定义代价函数
cost = tf.reduce_mean(Y_)
optimizer = tf.train.RMSPropOptimizer(0.001 , 0.9).minimize(cost)

# 是输出维度中最大值的索引
predict_op = tf.argmax(py_x,1)



with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(100): # 取第一批
        training_batch = zip(range(0,len(trX),batch_size),range(batch_size,len(trX)+1,batch_size))
        for start ,end in training_batch:
            sess.run(optimizer,feed_dict={X:trX[start:end],
                                          Y:trY[start:end],
                                          p_keep_conv:0.8,p_keep_hidden:0.5})
        test_indices = np.arange(len(TeX))
        np.random.shuffle(test_indices)
        test_indices = test_indices[0:test_size]
        print(i,np.mean(np.argmax(TeY[test_indices],axis=1)==sess.run(predict_op,feed_dict={X:TeX[test_indices],
                                                                                                Y:TeY[test_indices],
                                                                                                p_keep_conv:1.0,
                                                                                                p_keep_hidden:1.0})))




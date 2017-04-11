'''
A Convolutional Network implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Import MNIST data
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
train_data = np.load('/home/trantrunghieu/lv/project/demo/train_data.npy')
test_data = np.load('/home/trantrunghieu/lv/project/demo/eval_data.npy')
check_data = np.load('/home/trantrunghieu/lv/project/demo/test_data.npy')

# Parameters
learning_rate = 0.001
training_iters = 20000
batch_size = 128
display_step = 10

# Network Parameters
n_input = 128*128 # MNIST data input (img shape: 28*28)
n_classes = 14 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

def next_batch(num, data):
    """
    Return a total of `num` samples from the array `data`. 
    """
    idx = np.arange(0, len(data))  # get all possible indexes
    np.random.shuffle(idx)  # shuffle indexes
    idx = idx[0:num]  # use only `num` random indexes
    data_shuffle = [data[i] for i in idx]  # get list of `num` random samples
    data_shuffle = np.asarray(data_shuffle)  # get back numpy array

    return data_shuffle



# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='VALID')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='VALID')


# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 128, 128, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    # Max Pooling (down-sampling)
    conv3 = maxpool2d(conv3, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([11, 11, 1, 20])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([8, 8, 20, 40])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc3': tf.Variable(tf.random_normal([5, 5, 40, 80])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([11*11*80, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([20])),
    'bc2': tf.Variable(tf.random_normal([40])),
    'bc3': tf.Variable(tf.random_normal([80])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# dataset1 = next_batch(batch_size,train_data)
# x = np.array([i[0] for i in dataset1])

# x = tf.cast(x,tf.float32)

# print("Before: ",x.shape)

# x = tf.reshape(x, shape=[-1, 128, 128, 1])

# print("After: ",x.shape)

# print("----Conv1----")
# conv1 = conv2d(x, weights['wc1'], biases['bc1'])

# print("Conv: ",conv1.shape)

# conv1 = maxpool2d(conv1, k=2)
# print("Maxpool1: ",conv1.shape)

# print("----Conv2----")
# # Convolution Layer
# conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
# print("conv2: ",conv2.shape)
# # Max Pooling (down-sampling)
# conv2 = maxpool2d(conv2, k=2)
# print("Maxpool: ",conv2.shape)


# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

saver = tf.train.Saver()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        # batch_x, batch_y = mnist.train.next_batch(batch_size)
        dataset = next_batch(batch_size,train_data)
        batch_x = np.array([i[0] for i in dataset])
        batch_y = [i[1] for i in dataset]
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       keep_prob: dropout})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y,
                                                              keep_prob: 1.})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")
    saver.save(sess,"conv_net_model")
    test = next_batch(batch_size,test_data)

    # Calculate accuracy for 256 mnist test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: np.array([i[0] for i in test]),
                                      y: [i[1] for i in test],
                                      keep_prob: 1.}))
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

from create_data import *

train_data = create_train_data()
print(len(train_data))
valid_data = create_valid_data()
print(len(valid_data))
test_data = create_test_data()
print(len(test_data))


#load data
# train_data = np.load('/home/ngoc/luanvan/lv/project/demo/train_data.npy')
# test_data = np.load('/home/ngoc/luanvan/lv/project/demo/eval_data.npy')
# check_data = np.load('/home/ngoc/luanvan/lv/project/demo/test_data.npy')
print("Loaded dataset!!!")

# Parameters
learning_rate = 0.001
training_iters = 20000
batch_size = 128
display_step = 1
training_epochs = 25

# Network Parameters
n_input = 128*128 # data input (img shape: 128*128)
n_classes = 17 # total classes (19 faces)
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)
print("initialize parameters!!!")
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
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=3):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 128, 128, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=3)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=3)

    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    # Max Pooling (down-sampling)
    conv3 = maxpool2d(conv3, k=3)

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
    'wc1': tf.Variable(tf.random_normal([11, 11, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([8, 8, 32, 64])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc3': tf.Variable(tf.random_normal([5, 5, 64, 128])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([5*5*128, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bc3': tf.Variable(tf.random_normal([128])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}
print("Initialize weights and bias!!!")

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
print("Initialize all variables!!!")
saver = tf.train.Saver()
print("Start train model!!!")
# Launch the graph


with tf.Session() as sess:
    sess.run(init)
    step = 1
    total_batch = int(len(train_data)/batch_size)
    print("Total batch:",total_batch)
    # Keep training until reach max iterations
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(len(train_data)/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            dataset = next_batch(batch_size,train_data)
            batch_x = np.array([i[0] for i in dataset])
            batch_y = [i[1] for i in dataset]
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c, acc = sess.run([optimizer, cost, accuracy], feed_dict={x: batch_x,y: batch_y,keep_prob: dropout})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print ("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost),"Accuracy= ", \
                  "{:.5f}".format(acc))
    print ("Optimization Finished!")
    saver.save(sess,"net_model_v1")
    
    # Test model
    # correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # # Calculate accuracy
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    
    print ("Testing Accuracy:", accuracy.eval({x: np.array([i[0] for i in valid_data]), y: [i[1] for i in valid_data],keep_prob: 1.}))
    # print ("Accuracy:", sess.run(accuracy,feed_dict={x: np.array([i[0] for i in valid_data]), y: [i[1] for i in valid_data],keep_prob: 1.}))





# with tf.Session() as sess:
#     sess.run(init)
#     step = 1
#     # Keep training until reach max iterations
#     while step * batch_size < training_iters:
#         # batch_x, batch_y = mnist.train.next_batch(batch_size)
#         dataset = next_batch(batch_size,train_data)
#         batch_x = np.array([i[0] for i in dataset])
#         batch_y = [i[1] for i in dataset]
#         # Run optimization op (backprop)
#         sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
#                                        keep_prob: dropout})
#         if step % display_step == 0:
#             # Calculate batch loss and accuracy
#             loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
#                                                               y: batch_y,
#                                                               keep_prob: 1.})
#             print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
#                   "{:.6f}".format(loss) + ", Training Accuracy= " + \
#                   "{:.5f}".format(acc))
#         step += 1
#     print("Optimization Finished!")
#     saver.save(sess,"conv_net_model")
#     test = next_batch(batch_size,test_data)

#     # Calculate accuracy for 128 test images
#     print("Testing Accuracy:", \
#         sess.run(accuracy, feed_dict={x: np.array([i[0] for i in test]),
#                                       y: [i[1] for i in test],
#                                       keep_prob: 1.}))

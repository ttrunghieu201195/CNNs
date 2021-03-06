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

#load data
from create_data import *

# train_data = create_train_data()
# print(len(train_data))
valid_data = create_valid_data()
print(len(valid_data))
test_data = create_test_data()
print(len(test_data))
print(type(test_data))
check_data = test_data_random()
print(len(check_data))
print("Loaded dataset!!!")

IMG_SIZE = 128

test_dir = '/home/manhthe/project/dulieu/check/'
list_img = [os.path.join(test_dir,folder) for folder in
                 os.listdir(test_dir) if not folder.startswith('.')]


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


saver = tf.train.Saver()


fig=plt.figure()
num = 1

test = next_batch(12,test_data)
# check = next_batch(12,lgiang_data)
with tf.Session() as sess:
    saver.restore(sess, "/home/manhthe/project/demo/code/net_model_v1")
    print("Model restored.") 
    print('Initialized')

    print ("Testing Accuracy:", sess.run(accuracy,feed_dict={x: np.array([i[0] for i in valid_data]), y: [i[1] for i in valid_data],keep_prob: 1.}))
    # test = next_batch(12,check)
    # for idx in range(10):
    # print("Pred:", sess.run(tf.argmax(pred,1), feed_dict={x:np.array([i[0] for i in check]), keep_prob: 1.}))
    # print("Pred:", sess.run(tf.argmax(pred,1), feed_dict={x:img, keep_prob: 1.}))
    # # print("Label:", sess.run(tf.argmax(y,1), feed_dict={y: [i[1] for i in test]}))
    # print("------------------------------------------------------------")

    # check = next_batch(12,hieu_data)
    # dataset = next_batch(batch_size,eval_data)
    # check = np.array([i[0] for i in dataset])
    labels = ['ngoc', 'nguyen', 'ky', 'truong', 'phuc', 'duc', 'huyen', 'thinh', 'thuan', 'duyen', 'hieu', 'lgiang', 'giang', 'my', 'btran', 'tram', 'manh']
    # check labels of dataset
    for num,data in enumerate(check_data):
        y=fig.add_subplot(5,5,num)
        img = np.reshape(data[0],(-1,128))

        
        data[0] = data[0].reshape(-1,128*128)        

        label = sess.run(tf.argmax(pred,1), feed_dict={x:np.array(data[0]), keep_prob: 1.})
        print(label)
        # print(np.argmax(data[1]))

        str_label = labels[label[0]]


        y.imshow(img,cmap='gray')
        plt.title(str_label)
        y.axes.get_xaxis().set_visible(False)
        y.axes.get_yaxis().set_visible(False)
    plt.show()



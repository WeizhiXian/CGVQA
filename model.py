import tensorflow as tf
import cv2
import os
import numpy as np
import sys
import h5py
from keras.layers import *
import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def conv_bn_relu_block(input,
            n_channl=8,
            kernel_size=[3,3],
            strides=[1,1],
            padding='same',
            dilation_rate=1,
            activation=None,
            is_training=True,
            is_bn=True,
            name='conv'):
    if is_bn:
        padding=padding.upper()
        conv=tf.layers.conv2d(input,
                            n_channl,
                            kernel_size,
                            strides=strides,
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                            padding=padding,
                            dilation_rate=dilation_rate,
                            activation=None,
                            name=name)
        conv_norm=tf.layers.batch_normalization(conv, training=is_training)
        if activation==None:
            conv=tf.nn.relu(conv_norm)
            # conv=tf.nn.relu6(conv_norm)
        else:
            conv=tf.nn.sigmoid(conv_norm)
    else:
        padding=padding.upper()
        conv=tf.layers.conv2d(input,
                            n_channl,
                            kernel_size,
                            strides=strides,
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                            padding=padding,
                            dilation_rate=dilation_rate,
                            activation=None,
                            name=name)
        if activation==None:
            conv=tf.nn.relu(conv)
            # conv=tf.nn.relu6(conv)
        else:
            conv=tf.nn.sigmoid(conv)
    return conv

def max_pooling_block(input,pool_size=(2,2),strides=[2,2]):
    max_conv=tf.layers.max_pooling2d(input,pool_size=pool_size,strides=strides)
    return max_conv

def resize_tensor(input_tensor, input_tensor_target):
    h = K.int_shape(input_tensor_target)[1]
    w = K.int_shape(input_tensor_target)[2]
    x = Lambda(lambda x: tf.image.resize_bilinear(input_tensor, size=(h, w)))(input_tensor)

    return K.eval(x)

def branch_one(input,keep_prob=1.0,is_training=True,is_upscale=True,is_bn=True):
  
    conv1=conv_bn_relu_block(input,n_channl=32,kernel_size=[5,5],is_training=is_training,is_bn=is_bn,name='c1')
    p1=max_pooling_block(conv1)

    conv2=conv_bn_relu_block(p1,n_channl=64,kernel_size=[5,5],is_training=is_training,is_bn=is_bn,name='c2')
    p2=max_pooling_block(conv2)

    conv3=conv_bn_relu_block(p2,n_channl=128,kernel_size=[3,3],is_training=is_training,is_bn=is_bn,name='c3')
    p3=max_pooling_block(conv3)

    conv4=conv_bn_relu_block(p3,n_channl=128,kernel_size=[3,3],is_training=is_training,is_bn=is_bn,name='c4')
    p4=max_pooling_block(conv4)

    fc1 = tf.reshape(p4,[-1, 6*6*128])
    w_fc1 = tf.Variable(tf.random_normal([6*6*128, 1024], stddev=0.01), name='w_fc1')
    out_fc1 = tf.nn.relu(tf.add(tf.matmul(fc1, w_fc1), tf.constant(0.01, shape=[1024])))

    w_fc2 = tf.Variable(tf.random_normal([1024, 512], stddev=0.01), name='w_fc2')
    out_fc2 = tf.nn.relu(tf.add(tf.matmul(out_fc1, w_fc2), tf.constant(0.01, shape=[512])))

    w_fc3 = tf.Variable(tf.random_normal([512, 12], stddev=0.01), name='w_fc3')
    out_fc3 = tf.nn.sigmoid(tf.add(tf.matmul(out_fc2, w_fc3), tf.constant(0.01, shape=[12])))

    return out_fc3

def My_Net(input1,input2,keep_prob=1.0,is_training=True,is_upscale=True,is_bn=True):
    result1 = branch_one(input1)
    matmul_result = tf.multiply(result1, input2)

    w_fc4 = tf.Variable(tf.random_normal([12, 100], stddev=0.01), name='w_fc4')
    out_fc4 = tf.nn.relu(tf.add(tf.matmul(matmul_result, w_fc4), tf.constant(0.01, shape=[100])))

    w_fc5 = tf.Variable(tf.random_normal([100, 40], stddev=0.01), name='w_fc5')
    out_fc5 = tf.nn.relu(tf.add(tf.matmul(out_fc4, w_fc5), tf.constant(0.01, shape=[40])))

    w_fc6 = tf.Variable(tf.random_normal([40, 20], stddev=0.01), name='w_fc6')
    out_fc6 = tf.nn.relu(tf.add(tf.matmul(out_fc5, w_fc6), tf.constant(0.01, shape=[20])))
    
    w_fc7 = tf.Variable(tf.random_normal([20, 10], stddev=0.01), name='w_fc7')
    out_fc7 = tf.nn.relu(tf.add(tf.matmul(out_fc6, w_fc7), tf.constant(0.01, shape=[10])))
    
    w_fc8 = tf.Variable(tf.random_normal([10, 5], stddev=0.01), name='w_fc7')
    out_fc8 = tf.nn.relu(tf.add(tf.matmul(out_fc7, w_fc8), tf.constant(0.01, shape=[5])))

    logits= tf.layers.dense(inputs=out_fc8, units=1, activation=None)

    return logits

if __name__ == "__main__":
    input_image1 = tf.placeholder(tf.float32, [None, 100, 100, 3], name = 'input_image1')
    input_image2 = tf.placeholder(tf.float32, [None, 12], name = 'input_image2')
    out = My_Net(input_image1, input_image2)
    print(out)

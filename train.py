import tensorflow as tf
import numpy as np
import pandas as pd
import scipy.misc as misc
import glob
import os
import random
import tqdm
import time
import cv2
from model import *
from csvTools import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto(allow_soft_placement=False, log_device_placement=False) 
config.gpu_options.allow_growth = True 
session = tf.Session(config=config)


def normalazation(image_array):
    max = image_array.max()
    min = image_array.min()
    image_array = (image_array-min)/(max-min)  # float cannot apply the compute,or array error will occur
    avg = image_array.mean()
    image_array = image_array-avg
    return image_array   # a bug here, a array must be returned,directly appling function did't work

def load_image(batch_file, train_image_path):
    images_list = []
    feature_list = []
    labels_list = []
    # print(batch_file)
    for item in batch_file:
        image_ID = item[0]
        image_path = train_image_path + image_ID + '.jpg'
        image_load = cv2.imread(image_path)
        image_load = normalazation(image_load)
        images_list.append(image_load)

        feature_item = item[1:-1]
        feature_list.append(feature_item)

        label_item = item[-1]
        labels_list.append(label_item)

    images_array = np.array(images_list)
    features_array = np.array(feature_list)
    labels_array = np.array(labels_list)
    labels_array = labels_array[:, np.newaxis]

    return images_array, features_array, labels_array

def train(train_image_path, train_feature_label, batch_size=16, epoch=120, learning_rate=0.0001, keep_prob=1.0):

    input_image = tf.placeholder(tf.float32, [None, 100, 100, 3], name='input_image')
    input_feature = tf.placeholder(tf.float32, [None, 12], name='input_feature')

    label = tf.placeholder(tf.float32, [None, 1], name='label')
    pre = My_Net(input_image, input_feature)

    train_number = len(train_feature_label)
    epoch_step = train_number/batch_size
    if epoch_step % 1:
        epoch_step = int(epoch_step) + 1

    saver = tf.train.Saver()  # default to save all variable,save mode or restore from path
    global_step = tf.constant(0, name='global_step')
    learning_rate = tf.train.exponential_decay(learning_rate, global_step, epoch_step*15, 0.95, staircase=True, name='learning_rate')
    loss = tf.reduce_mean(tf.square(label-pre))

    train_step = tf.train.AdamOptimizer(learning_rate, 0.9).minimize(loss)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_writer = tf.summary.FileWriter('F:/MM/fully_tensorboard/', sess.graph)
        steps = 0
        for i in tqdm.tqdm(range(epoch)):
            loss_train = 0
            epoch_start = time.time()
            # the data will be shuffled by every epoch
            random.shuffle(train_feature_label)
            for t in range(epoch_step):
                steps += 1
                batch_files = train_feature_label[t*batch_size:(t+1)*batch_size]
                batch_image, batch_feature, batch_label = load_image(batch_files, train_image_path)
                feed_dict = {input_image: batch_image, input_feature: batch_feature, label:batch_label}
                sess.run(train_step, feed_dict=feed_dict)
                net_loss = sess.run([loss], feed_dict=feed_dict)
                loss_train += net_loss[0]
            saver.save(sess, 'F:/MM/fully_ckpt/fully', steps)
            average_loss = loss_train / epoch_step
            lnrt = sess.run(learning_rate, feed_dict={global_step: steps})
            print('loss is ', average_loss)
            print('learning rate is ', lnrt)


if __name__ == "__main__":
    train_image_path = './Dataset/'
    train_label_path = './feature_label.csv'
    train_feature_label = readCSV(train_label_path)

    train(train_image_path, train_feature_label)

import tensorflow as tf
import numpy as np
import pandas as pd
import scipy.misc as misc
import glob
# import pydicom
import os
import random
import tqdm
import time
import cv2
from model import *
from csvTools import *


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# sess = tf.Session()
X1 = None  # input
X2 = None
yhat = None  # output
prob_keep = None

def normalazation(image_array):
    max = image_array.max()
    min = image_array.min()
    image_array = (image_array-min)/(max-min)  # float cannot apply the compute,or array error will occur
    avg = image_array.mean()
    image_array = image_array-avg
    return image_array   # a bug here, a array must be returned,directly appling function did't work

def load_image(test_batch_file, test_image_path):
    images_list = []
    feature_list = []
    labels_list = []

    image_ID = test_batch_file[0]
    image_path = test_image_path + image_ID + '.jpg'
    image_load = cv2.imread(image_path)
    image_load = normalazation(image_load)
    images_list.append(image_load)

    feature_item = test_batch_file[1:-1]
    feature_list.append(feature_item)

    label_item = test_batch_file[-1]
    labels_list.append(label_item)

    images_array = np.array(images_list)
    features_array = np.array(feature_list)
    labels_array = np.array(labels_list)
    labels_array = labels_array[:, np.newaxis]

    return images_array, features_array, labels_array

def load_model():
    """
        Loading the pre-trained model and parameters.
    """
    global X1, X2, yhat
    modelpath = './fully_ckpt/'
    saver = tf.train.import_meta_graph(modelpath + 'fully-960.meta')
    saver.restore(sess, tf.train.latest_checkpoint(modelpath))
    graph = tf.get_default_graph()
    # for tensor_name in tf.contrib.graph_editor.get_tensors(graph):
    #     print(str(tensor_name))
    #     file_tensor.write(str(tensor_name))
    #     file_tensor.write('\n')
    X1 = graph.get_tensor_by_name("input_image:0")
    X2 = graph.get_tensor_by_name("input_feature:0")
    yhat = graph.get_tensor_by_name("dense/BiasAdd:0")
    print('Successfully load the pre-trained model!')


def predict(test_image_path, test_feature_label):
    """
        Convert data to Numpy array which has a shape of (-1, 41, 41, 41 3).
        Test a single example.
        Arg:
                txtdata: Array in C.
        Returns:
            Three coordinates of a face normal.
    """
    global X1, X2, yhat
    test_results_csv = open('test_results.csv', 'w+')
    test_results_writer = csv.writer(test_results_csv)
    header = ['number', 'predict_value', 'real_label']
    test_results_writer.writerow(header)
    loss_total = 0
    for i in range(len(test_feature_label)):
        batch_image, batch_feature, batch_label = load_image(test_feature_label[i], test_image_path)
        output = sess.run(yhat, feed_dict={X1: batch_image, X2: batch_feature})  # (-1, 3)
        test_results_writer.writerow([test_feature_label[i][0], str(output), str(batch_label)])
        out_value = float(output[0][0])
        label_value = float(batch_label[0][0])
        loss = np.square((out_value-label_value))
        loss_total += loss
    print('final MSE loss is, ', loss_total)

if __name__ == "__main__":
    load_model()
    test_image_path = './test_set_demo/'
    test_label_path = './test_feature_label_demo.csv'
    test_feature_label = readCSV(test_label_path)

    predict(test_image_path, test_feature_label)
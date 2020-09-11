from skimage import io,transform
import tensorflow as tf
import numpy as np

#randomly capture 5 frames from the test video
path1 = '/CNN_Tensorflow/testvideo/1.jpg'
path2 = '/CNN_Tensorflow/testvideo/2.jpg'
path3 = '/CNN_Tensorflow/testvideo/3.jpg'
path4 = '/CNN_Tensorflow/testvideo/4.jpg'
path5 = '/CNN_Tensorflow/testvideo/5.jpg'

content_dict = {0:'Character',1:'Item',2:'MOBA',3:'Scenery',4:'Special'}
 
w=100
h=100
c=3

def read_one_image(path):
    img = io.imread(path)
    img = transform.resize(img,(w,h))
    return np.asarray(img)

with tf.Session() as sess:
    data = []
    data1 = read_one_image(path1)
    data2 = read_one_image(path2)
    data3 = read_one_image(path3)
    data4 = read_one_image(path4)
    data5 = read_one_image(path5)
    data.append(data1)
    data.append(data2)
    data.append(data3)
    data.append(data4)
    data.append(data5)

    saver = tf.train.import_meta_graph('/CNN_Tensorflow/model/model.ckpt.meta')
    saver.restore(sess,tf.train.latest_checkpoint('/CNN_Tensorflow/model'))

    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    feed_dict = {x:data}

    logits = graph.get_tensor_by_name("logits_eval:0")
    classification_result = sess.run(logits,feed_dict)

#print prediction matrix
    print(classification_result)
    print(tf.argmax(classification_result,1).eval())
    output = []
    output = tf.argmax(classification_result,1).eval()
    for i in range(len(output)):
        print("The",i+1,"frame belongs to: "+content_dict[output[i]])
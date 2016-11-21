#!/usr/bin/env python


from __future__ import print_function
import tensorflow as tf
import numpy as np
from skimage.transform import resize
from scipy.misc import imread
from scipy import misc
import scipy.ndimage.morphology as morph
from numpy import array
import sys

batch_size = 100
test_data_size = 50
img_size = 28
max_accuracy=0
text_file = open("4d_conv.txt", "w")


#list_string=["data/train/0.png","data/train/1.png","data/train/200.png"]


list_string=[]
for i in range(1,len(sys.argv)):
    list_string.append(sys.argv[i])



def initialize_weights(weight_shape):
    return tf.Variable(tf.random_normal(weight_shape, stddev=0.01))

def convert_labels_to_one_hot(labels, classes=104):
    num_labels = labels.shape[0]
    offset = np.arange(num_labels) * classes
    labels_converted = np.zeros((num_labels, classes))
    labels_converted.flat[offset + labels.ravel()] = 1
    return labels_converted

def dilate_image(image):
    return  misc.imresize(morph.binary_dilation(255.0 - image, iterations=3),(img_size,img_size))

def batch_norm(x, output_number, training_phase, scope='bn'):
    with tf.variable_scope(scope):
        beta = tf.Variable(tf.constant(0.0, shape=[output_number]),name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[output_number]),name='gamma', trainable=True)
        mean_of_batch, variance_of_batch = tf.nn.moments(x, [0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([mean_of_batch,variance_of_batch])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(mean_of_batch), tf.identity(variance_of_batch)

        mean, variance = tf.cond(training_phase,
                            mean_var_with_update,
                            lambda: (ema.average(mean_of_batch), ema.average(variance_of_batch)))
        normalized = tf.nn.batch_normalization(x, mean, variance, beta, gamma, 0.001)
    return normalized

def model(X, w, w2, w3, w4,w5, w_o, prob_conv, prob_hidden,training_phase):
    
    layer_1_a = tf.nn.relu(batch_norm(tf.nn.conv2d(X, w,                      
                        strides=[1, 1, 1, 1], padding='SAME'),32,training_phase))
    layer_1 = tf.nn.max_pool(layer_1_a, ksize=[1, 2, 2, 1],              
                        strides=[1, 2, 2, 1], padding='SAME')
    layer_1 = tf.nn.dropout(layer_1, prob_conv)

    layer_2_a = tf.nn.relu(batch_norm(tf.nn.conv2d(layer_1, w2,                    
                        strides=[1, 1, 1, 1], padding='SAME'),64,training_phase))
    layer_2 = tf.nn.max_pool(layer_2_a, ksize=[1, 2, 2, 1],             
                        strides=[1, 2, 2, 1], padding='SAME')
    layer_2 = tf.nn.dropout(layer_2, prob_conv)
    
    layer_3_a = tf.nn.relu(batch_norm(tf.nn.conv2d(layer_2, w3,                    
                        strides=[1, 1, 1, 1], padding='SAME'),128,training_phase))
    layer_3 = tf.nn.max_pool(layer_3_a, ksize=[1, 2, 2, 1],             
                        strides=[1, 2, 2, 1], padding='SAME')
    layer_3 = tf.nn.dropout(layer_3, prob_conv)

    layer_4_a = tf.nn.relu(batch_norm(tf.nn.conv2d(layer_3, w4,                     
                        strides=[1, 1, 1, 1], padding='SAME'),256,training_phase))
    layer_4 = tf.nn.max_pool(layer_4_a, ksize=[1, 2, 2, 1],              
                        strides=[1, 2, 2, 1], padding='SAME')
    layer_4 = tf.reshape(layer_4, [-1, w5.get_shape().as_list()[0]])    
    layer_4 = tf.nn.dropout(layer_4, prob_conv)


    layer_5 = tf.nn.relu(tf.matmul(layer_4, w5))
    layer_5 = tf.nn.dropout(layer_5, prob_hidden)

    pyx = tf.matmul(layer_5, w_out)
    return pyx

def get_probability(list_string):
    images_train=[]
    for i in list_string:
        image_train=imread(i)
        images_train.append(misc.imresize(image_train,(img_size,img_size))/255.0)
    #print(images_train)
    training_X=array(images_train)  
    X_all= training_X.reshape(-1, 28, 28, 1)
                                                
    #print(np.argmax(X_all,axis=1))
    test_indices = np.arange(len(list_string))
    test_indices = test_indices[0:len(list_string)]

    X_all=np.float32(X_all)
    py_x = model(X_all, w, w2, w3, w4,w5, w_out, prob_conv, prob_hidden,training_phase)
    return py_x


training_phase = tf.placeholder(tf.bool, name='training_phase')
X = tf.placeholder("float", [None, 28, 28, 1])
Y = tf.placeholder("float", [None, 104])

w = initialize_weights([3, 3, 1, 32])       
w2 = initialize_weights([3, 3, 32, 64])     
w3 = initialize_weights([3, 3, 64, 128]) 
w4=initialize_weights([3,3,128,256]) 
w5 = initialize_weights([64 * 4 * 4, 625]) 
w_out = initialize_weights([625, 104])  


prob_conv = tf.placeholder("float")
prob_hidden = tf.placeholder("float")

predict=get_probability(list_string)


with tf.Session() as sess:
    
    tf.initialize_all_variables().run()
    saver = tf.train.Saver()
    saver.restore(sess, "model/model0.92.ckpt")       
    prediction=sess.run(predict, feed_dict={prob_conv: 1.0,prob_hidden: 1.0,training_phase: False})
    max_total=0
    for i in range(0,len(prediction)):
        max_val=max(abs(prediction[i]))
        max_total=max(max_total,max_val)


    for i in range(0,len(prediction)):
        max_val=max(prediction[i])+max_total
        print("Class of image "+str(i+1)+":")
	print(np.argmax(prediction[i]))
        #print("Probabilities:")
	#for j in range(0,len(prediction[0])):
            #print((prediction[i][j]+max_total)/max_total,end="")
            #print(' ', end="")
                     

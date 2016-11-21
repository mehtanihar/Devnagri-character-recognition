#!/usr/bin/env python



import tensorflow as tf
import numpy as np
from skimage.transform import resize
from scipy.misc import imread
from scipy import misc
import scipy.ndimage.morphology as morph

batch_size = 100
test_data_size = 50
img_size = 28
max_accuracy=0
text_file = open("3_conv.txt", "w")

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

def model(X, w, w2, w3, w4, w_o, prob_conv, prob_hidden,training_phase):
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
    layer_3 = tf.reshape(layer_3, [-1, w4.get_shape().as_list()[0]])    
    layer_3 = tf.nn.dropout(layer_3, prob_conv)

    layer_4 = tf.nn.relu(tf.matmul(layer_3, w4))
    layer_4 = tf.nn.dropout(layer_4, prob_hidden)

    pyx = tf.matmul(layer_4, w_out)
    return pyx
with open('./train/labels.txt') as f:
    labels_t = [[int(x) for x in line.split()] for line in f]
from numpy  import array
labels_train=array(labels_t)

with open('./valid/labels.txt') as f:
    labels_v = [[int(x) for x in line.split()] for line in f]
labels_valid=array(labels_v)
labels_train1=labels_train[0:5000]
labels_valid1=labels_valid[0:1800]


images_train=[]
for count in range(0,5000):
    images_train.append(dilate_image(imread("./train/"+str(count)+".png")))
images_valid=[]
for count in range(0,1800):
    images_valid.append(dilate_image(imread("./valid/"+str(count)+".png")))


training_X=array(images_train)
training_Y1=labels_train1
testing_X=array(images_valid)
testing_Y1=labels_valid1
training_Y=convert_labels_to_one_hot(training_Y1)
testing_Y=convert_labels_to_one_hot(testing_Y1)
training_X = training_X.reshape(-1, 28, 28, 1)  
testing_X = testing_X.reshape(-1, 28, 28, 1)

training_phase = tf.placeholder(tf.bool, name='training_phase')
X = tf.placeholder("float", [None, 28, 28, 1])
Y = tf.placeholder("float", [None, 104])

w = initialize_weights([3, 3, 1, 32])       
w2 = initialize_weights([3, 3, 32, 64])     
w3 = initialize_weights([3, 3, 64, 128])  
w4 = initialize_weights([128 * 4 * 4, 625]) 
w_out = initialize_weights([625, 104])  



prob_conv = tf.placeholder("float")
prob_hidden = tf.placeholder("float")
py_x = model(X, w, w2, w3, w4, w_out, prob_conv, prob_hidden,training_phase)


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
train_op = tf.train.RMSPropOptimizer(0.003, 0.9).minimize(cost)
predict_op = tf.argmax(py_x, 1)


with tf.Session() as sess:

    tf.initialize_all_variables().run()
    

    
 

   

    for i in range(100):
        training_batch = zip(range(0, len(training_X), batch_size),
                             range(batch_size, len(training_X)+1, batch_size))

        for start, end in training_batch:
            sess.run(train_op, feed_dict={X: training_X[start:end], Y: training_Y[start:end],
                                          prob_conv: 0.8, prob_hidden: 0.5,training_phase: True})
	


        test_indices = np.arange(len(testing_X))
        np.random.shuffle(test_indices)
        test_indices = test_indices[0:test_data_size]

        accuracy=np.mean(np.argmax(testing_Y[test_indices], axis=1) ==
                         sess.run(predict_op, feed_dict={X: testing_X[test_indices],
                                                         Y: testing_Y[test_indices],
                                                         prob_conv: 1.0,
                                                         prob_hidden: 1.0,
                                                         training_phase: False}))
	print(i,accuracy)
	text_file.write(str(i)+":"+str(accuracy))
	if(accuracy>max_accuracy):
		max_accuracy=accuracy
		
	
		
	print("max_accuracy="+str(max_accuracy))
	text_file.write(str(i)+":"+str(accuracy))



import numpy as np
import tensorflow as tf
from lib.networks import *
from lib.musicalPiece import *
from lib.dataset import *
from lib.file_handle import *
import os

n_steps   = 25
n_inputs  = 127
n_neurons = 150
n_layers = 2
n_outputs = 127
batch_size = 500
n_epochs = 20

learning_rate = 0.001

init = tf.global_variables_initializer()
dir_path = os.path.dirname(os.path.realpath(__file__))
saver = tf.train.import_meta_graph(dir_path + "\log\LSTM_layer_2_model_20180505145915.ckpt.meta")

#Set up dataset
data = Dataset("Dataset 1","bach","Just the one hot vectors, with no pre-processing","28/5/2018")
# data.make_dataset(trainingSplit=0.9)
# data.save("dataset1.pickle")
data.load("bach.pickle")

#Split input and targets. (I'll make a function to do that, also idk if this is the right way)
train_x = data.train[0:-1]
train_y = data.train[1:]
test_x = data.test[0:-1]
test_y = data.test[1:]
seed = data.test[0:25]


with tf.Session() as sess:
    #init.run()
    #saver = tf.train.import_meta_graph(dir_path + "\log\LSTM_layer_2_model_20180505145915.ckpt.meta")
    #saver.restore(sess,dir_path + "\log\LSTM_layer_2_model_20180505145915.ckpt")

    #CHRIS TRAINED
    saver.restore(sess,dir_path + "\log\LSTM_layer_2_model_20180505145915.ckpt")

    # for op in tf.get_default_graph().get_operations():
    #     print(op.name)

    X = tf.get_default_graph().get_tensor_by_name("inputs:0")
    logits = tf.get_default_graph().get_tensor_by_name("outputs/kernel/Assign:0")
    #logits = tf.get_default_graph().get_tensor_by_name("outputs")

    sequence = seed
    for i in range(500):
        batch_x = np.array(sequence[-n_steps:]).reshape(1,n_steps,127)
        prediction = sess.run(logits,feed_dict={X: batch_x})
        binary_prediction = sess.run(tf.to_int32(prediction[149,:] > 0.1))

        sequence.append(binary_prediction)

    print(sequence)

    save_var(dir_path+"\generated_sequence.pickle",sequence)

'''
outputs/kernel/Initializer/random_uniform/shape
outputs/kernel/Initializer/random_uniform/min
outputs/kernel/Initializer/random_uniform/max
outputs/kernel/Initializer/random_uniform/RandomUniform
outputs/kernel/Initializer/random_uniform/sub
outputs/kernel/Initializer/random_uniform/mul
outputs/kernel/Initializer/random_uniform
outputs/kernel
outputs/kernel/Assign
outputs/kernel/read
outputs/bias/Initializer/zeros
outputs/bias
outputs/bias/Assign
outputs/bias/read
outputs/MatMul
outputs/BiasAdd
'''

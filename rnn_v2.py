import numpy as np
import tensorflow as tf
from lib.networks import *
from lib.musicalPiece import *
from lib.dataset import *
from lib.file_handle import *
import os
from datetime import datetime

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)


tf.reset_default_graph()

directory = os.path.dirname(os.path.realpath(__file__))
path = directory + "\log\model.ckpt"
n_steps   = 25
n_inputs  = 127
n_neurons = 150
n_outputs = 127
batch_size = 150
n_epochs = 10

learning_rate = 0.001

batch_training = tf.placeholder_with_default(False, shape=(), name='training')

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs], name='inputs')
Y = tf.placeholder(tf.float32, [None, n_outputs], name='target')

cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

logits_before_bn = tf.layers.dense(states, n_outputs, name='outputs')
logits = tf.layers.batch_normalization(logits_before_bn, training=batch_training, momentum=0.9)

#xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y, logits=logits)
loss = tf.reduce_mean(tf.square(logits - Y), name='mse')
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

#Set up dataset
data = Dataset("Dataset 1","bach","Just the one hot vectors, with no pre-processing","28/5/2018")
data.load("bach.pickle")

#Split input and targets. (I'll make a function to do that, also idk if this is the right way)
raw_x_train = np.array(data.train[0:-1])
raw_y_train = np.array(data.train[25:])
raw_x_test = np.array(data.test[0:-1])
raw_y_test = np.array(data.test[25:])
x_train, y_train, n_batches = batch_maker2(raw_x_train, raw_y_train, n_steps, batch_size)

init = tf.global_variables_initializer()
saver = tf.train.Saver()
mse_summary = tf.summary.scalar('MSE', mse)
file_writer= tf.summary.FileWriter(logdir, tf.get_default_graph())

with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(n_epochs):
        for batch in range(n_batches):
            run_count = epoch*batch + batch
            X_batch, Y_batch = x_train[batch, :], y_train[batch, :]
            sess.run(training_op, feed_dict={X: X_batch, Y: Y_batch})
            if run_count%50 == 0:
                summary_str = mse_summary.eval(feed_dict={X: X_batch, Y: Y_batch})
                file_writer.add_summary(summary_str, run_count)
                mse = loss.eval(feed_dict={X: X_batch, Y: Y_batch})
                print(epoch, '\tMSE:', mse)
    save_path = saver.save(sess, path)
    
    file_writer.close()

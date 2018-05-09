import numpy as np
import tensorflow as tf
from lib.networks import *
from lib.musicalPiece import *
from lib.dataset import *
from lib.file_handle import *
import os
from datetime import datetime
import argparse

def main(args):

    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "tf_logs"
    logdir = "{}/run-{}/".format(root_logdir, now)

    tf.reset_default_graph()

    directory = os.path.dirname(os.path.realpath(__file__))
    path = directory + "\log\LSTM_layer_2_model_{}.ckpt".format(now)

    #Set up parameters (can put those in a config file and parse them.)
    n_epochs = int(args.num_epochs)
    batch_size = int(args.batch_size)
    n_steps   = 25
    n_inputs  = 127
    n_neurons = 150
    n_layers = int(args.num_layers)
    n_outputs = 127
    learning_rate = 0.001

    batch_training = tf.placeholder_with_default(False, shape=(), name='training')

    X = tf.placeholder(tf.float32, [None, n_steps, n_inputs], name='inputs')
    Y = tf.placeholder(tf.float32, [None, n_outputs], name='target')

    layers = [tf.contrib.rnn.LSTMCell(num_units=n_neurons, activation = tf.nn.relu, use_peepholes=True) for layers in range(n_layers)]
    multi_layer_cell = tf.contrib.rnn.MultiRNNCell(layers)
    outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype = tf.float32)

    #logits_before_bn = tf.layers.dense(outputs[:,(n_steps-1),:], n_outputs)
    logits_before_bn = tf.layers.dense(states[-1][1], n_outputs)
    logits = tf.layers.batch_normalization(logits_before_bn, training=batch_training, momentum=0.9,name='outputs')

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
    x_test, y_test, _ = batch_maker2(raw_x_test, raw_y_test, n_steps, raw_x_test[:,0].size)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    mse_summary = tf.summary.scalar('MSE', loss)
    file_writer= tf.summary.FileWriter(logdir, tf.get_default_graph())

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(n_epochs):
            for batch in range(n_batches):
                print('Epoch {} of {}, batch {} of {}'.format(epoch, n_epochs, batch, n_batches))
                run_count = epoch*batch + batch
                X_batch, Y_batch = x_train[batch, :], y_train[batch, :]
                sess.run(training_op, feed_dict={X: X_batch, Y: Y_batch})
                if run_count%50 == 0:
                    summary_str = mse_summary.eval(feed_dict={X: X_batch, Y: Y_batch})
                    file_writer.add_summary(summary_str, run_count)
    #                mse = loss.eval(feed_dict={X: X_batch, Y: Y_batch})
    #                print(epoch, '\tMSE:', mse)
            loss_train = loss.eval(feed_dict={X: X_batch, Y: Y_batch})
            loss_test = loss.eval(feed_dict={X: x_test[0,:], Y: y_test[0,:]})
            print(epoch, 'Loss_train:', loss_train, '\tLoss_test:', loss_test)
        save_path = saver.save(sess, path)

        file_writer.close()

if __name__ == "__main__":
    #Set up argument parser
	parser = argparse.ArgumentParser(description="Train an amazing neural network that makes music stuffs.")
	group = parser.add_mutually_exclusive_group()

	#Add parser arguments
	group.add_argument("-e", "--num_epochs", action="store",default = 100, nargs='?', help='Set num of epochs')
	group.add_argument("-b", "--batch_size", action="store",default = 500, nargs='?', help='Set batch size')
	group.add_argument("-s", "--num_layers", action="store", default = 2, nargs='?', help='Set num of layers')

	#Parse arguments and start main
	args = parser.parse_args()
	main(args)

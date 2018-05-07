import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from tqdm import tqdm
from pprint import pprint
import os

def set_up_placeholders(n_inputs,n_steps,n_outputs):

    tf.reset_default_graph()

    # Height by Width
    input_ = tf.placeholder('float',[None,n_steps,n_inputs],name='input')
    target_ = tf.placeholder('float',[None,n_steps,n_outputs],name='target')

    return input_,target_

def rnn_model(input_,rnn_size,n_outputs):

    cell = tf.contrib.rnn.BasicRNNCell(num_units=rnn_size, activation=tf.nn.relu) #hiddel layer for 1 time step
    outputs, states = tf.nn.dynamic_rnn(cell, input_, dtype=tf.float32) # Use hidden layer for n_steps timesteps
    l1 = tf.layers.dense(outputs, 180)
    logits = tf.layers.dense(l1, n_outputs, name='predictions')

    return cell, outputs, states, logits

def batch_maker2(x, y, n_steps, batch_size):
    x, y = np.array(x), np.array(y)
    n_batches = int(x[:,0].size//batch_size)
    print(n_batches)
    n_inputs = x[0, :].size

    X_batch = np.empty([n_batches, batch_size, n_steps, n_inputs])
    print(X_batch.shape)
    Y_batch = np.empty([n_batches, batch_size, n_inputs])
    print(Y_batch.shape)

    for batch in range(n_batches):
        for count_size in range(batch_size):
            x_start = (batch*batch_size)+count_size
            try:
                X_batch[batch, count_size, :] = x[x_start: x_start+n_steps, :]
            except:
                X_batch[batch, count_size, :] = np.zeros_like(X_batch[batch, count_size, :])
            try:
                Y_batch[batch, count_size, :] = y[(batch*batch_size)+count_size, :]
            except:
                Y_batch[batch, count_size, :] = np.zeros_like(Y_batch[batch, count_size, :])
    return X_batch, Y_batch, n_batches

def seed_maker(train_x,train_y,n_steps,batch_size):

    batches_x = list()
    batches_y = list()

    j=0
    step_x = []
    step_y = []
    while j < batch_size:
        step_start = j;
        step_end = j + n_steps

        try:
            step_x.append(train_x[step_start:step_end])
            step_y.append(train_y[step_start:step_end])
        except:
            continue

        j += n_steps

    batches_x = batches_x + step_x
    batches_y = batches_y + step_x

    return batches_x,batches_y

def batch_maker(train_x,train_y,n_steps,batch_size):
    i=0
    batches_x = []
    batches_y = []
    while i < len(train_x):

        j=0
        step_x = []
        step_y = []
        while j < batch_size:
            step_start = j;
            step_end = j + n_steps

            try:
                step_x.append(train_x[step_start:step_end])
                step_y.append(train_y[step_start:step_end])
            except:
                continue

            j += n_steps

        batches_x.append(step_x)
        batches_y.append(step_x)
        i += batch_size

    return batches_x,batches_y

def train_v2(x,y,logits,train_x, train_y, test_x, test_y,n_steps,batch_size,num_epochs,learning_rate=0.001):

    xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=y)
    cost = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    training_op = optimizer.minimize(cost)
    prediction = tf.nn.softmax(logits)

    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    init = tf.global_variables_initializer() #initialise parameters
    saver = tf.train.Saver()
    save_path = "/tmp/test.ckpt"

    with tf.Session() as sess:
        init.run()

        print("Making Batches...")
        batches_x,batches_y = batch_maker(train_x,train_y,n_steps,batch_size)

        print("Training for",num_epochs,"epochs...")
        for epoch in tqdm(range(num_epochs)):
            n_batches = int(len(train_x)/batch_size)
            i = 0

            while i < n_batches:
                batch_x = batches_x[i]
                batch_y = batches_y[i]
                sess.run(training_op, feed_dict = {x: batch_x,y: batch_y})
                i += 1

#            acc_train = accuracy.eval(feed_dict = {x: batch_x,y: batch_y})
#            print('Epoch', epoch, 'completed. Accuracy:',acc_train)

        batches_x,batches_y = batch_maker(test_x,test_y,n_steps,batch_size)
        print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: batch_x,y: batch_y}))
        saver.save(sess, save_path=save_path)

    return save_path

def train(x,y,logits,train_x, train_y, test_x, test_y,n_steps,batch_size,num_epochs,learning_rate=0.001):
    xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=y)
    cost = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    init = tf.global_variables_initializer() #initialise parameters
    saver = tf.train.Saver()

    with tf.Session() as sess:
        init.run()

        print("Making Batches...")
        batches_x,batches_y = batch_maker(train_x,train_y,n_steps,batch_size)

        print("Training for",num_epochs,"epochs...")
        for epoch in tqdm(range(num_epochs)):
            epoch_loss = 0

            n_batches = int(len(train_x)/batch_size)
            i = 0
            while i < n_batches:
                batch_x = batches_x[i]
                batch_y = batches_y[i]

                _, c = sess.run([optimizer,cost],feed_dict = {x: batch_x,y: batch_y})
                epoch_loss += c

                i += 1


#            print('Epoch', epoch, 'completed. loss:',epoch_loss)

        batches_x,batches_y = batch_maker(test_x,test_y,n_steps,batch_size)

        # Save the variables to disk.
        dir_path = os.path.dirname(os.path.realpath(__file__))
        save_path = saver.save(sess, dir_path + "/model/rnn.ckpt")
        print("Model saved in path: %s" % save_path)


        prediction = tf.nn.softmax(logits)
        correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: batch_x,y: batch_y}))


def multilayer_perceptron_model(input_,n_input_nodes,n_nodes_hl1,n_nodes_hl2,n_nodes_hl3,n_classes):
    hidden_layer_1 = {'weights': tf.Variable(tf.random_normal([n_input_nodes,n_nodes_hl1])),
    'biases':  tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_layer_2 = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2])),
    'biases':  tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_layer_3 = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_hl3])),
    'biases':  tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3,n_classes])),
    'biases':  tf.Variable(tf.random_normal([n_classes]))}


    l1 = tf.add(tf.matmul(input_,hidden_layer_1['weights']),hidden_layer_1['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_layer_2['weights']),hidden_layer_2['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,hidden_layer_3['weights']),hidden_layer_3['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3,output_layer['weights']) + output_layer['biases']

    return output

def run(length,n_steps,dir_to_load):

    dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    saver = tf.train.import_meta_graph(dir_path + dir_to_load)
    x = tf.get_default_graph().get_tensor_by_name("inputs:0")
    logits = tf.get_default_graph().get_tensor_by_name("logits:0")


    with tf.Session() as sess:
        #batch_x, batch_y = seed_maker(seed,seed,n_steps,1)

        step = [0.]*127
        print(step)
        sequence = [step]*n_steps
        print(sequence)

        for i in range(length):
                batch_x = np.array(sequence[-n_steps:]).reshape(1,n_steps,127)
                print("Running session...")
                prediction = sess.run(logits,feed_dict={x: batch_x})
                sequence.append(prediction[0,-1,0])
        print(sequence)

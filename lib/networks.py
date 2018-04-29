import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

def set_up_placeholders(n_inputs,n_steps,n_outputs):

    tf.reset_default_graph()

    # Height by Width
    input_ = tf.placeholder('float',[None,n_steps,n_inputs],name='input')
    target_ = tf.placeholder('float',[None,n_steps,n_outputs],name='target')

    return input_,target_

def rnn_model(input_,rnn_size,n_outputs):

    # RNN output node weights and biases
    weights = {
        'out': tf.Variable(tf.random_normal([rnn_size, n_outputs]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([n_outputs]))
    }

    cell = tf.contrib.rnn.BasicRNNCell(num_units=rnn_size, activation=tf.nn.relu) #hiddel layer for 1 time step
    outputs, states = tf.nn.dynamic_rnn(cell, input_, dtype=tf.float32) # Use hidden layer for n_steps timesteps
    logits = tf.layers.dense(outputs, n_outputs, name='predictions')

    return logits

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

            step_x.append(train_x[step_start:step_end])
            step_y.append(train_y[step_start:step_end])

            j += n_steps

        batches_x.append(step_x)
        batches_y.append(step_x)
        i += batch_size

    return batches_x,batches_y

def train(x,y,logits,train_x, train_y, test_x, test_y,n_steps,batch_size,num_epochs,learning_rate=0.001):
    xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=y)
    cost = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        print("Making Batches")
        batches_x,batches_y = batch_maker(train_x,train_y,n_steps,batch_size)

        i = 0
        for epoch in range(num_epochs):
            print('Epoch', epoch, ' started out of', num_epochs)
            epoch_loss = 0

            n_batches = len(train_x)/batch_size
            while i < n_batches:
                print('Batch', i, ' started out of', n_batches)
                batch_x = batches_x[i]
                batch_y = batches_y[i]

                _, c = sess.run([optimizer,cost],feed_dict = {x: batch_x,y: batch_y})
                epoch_loss += c

                i += 1

            print('Epoch', epoch, 'completed out of', num_epochs, 'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(logits,1),tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        print('Accuracy:', accuracy.eval({x:test_x,y:test_y}))

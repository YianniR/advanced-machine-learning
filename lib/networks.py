import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

def set_up_placeholders(n_steps,n_inputs,n_outputs):

    tf.reset_default_graph()

    # Height by Width
    input_ = tf.placeholder('float',[None,n_steps,n_inputs])
    target_ = tf.placeholder('float',[None])

    return input_,target_

def rnn_model(input_,rnn_size,n_classes):
    basic_cell = rnn.BasicRNNCell(num_units=rnn_size)
    outputs, states = tf.nn.dynamic_rnn(basic_cell, input_, dtype=tf.float32)
    logits = tf.layers.dense(states, n_classes)

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

def train(x,y,logits,train_x, train_y, test_x, test_y,batch_size,num_epochs,learning_rate=0.001):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=y)
    cost = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(num_epochs):
            epoch_loss = 0

            i = 0;
            while i < len(train_x):
                start = i;
                end = i + batch_size

                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])

                _, c = sess.run([optimizer,cost],feed_dict = {x: batch_x,y: batch_y})
                epoch_loss += c

                i += batch_size

            print('Epoch', epoch, 'completed out of', num_epochs, 'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(logits,1),tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        print('Accuracy:', accuracy.eval({x:test_x,y:test_y}))

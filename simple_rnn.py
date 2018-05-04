import tensorflow as tf
import numpy as np
from lib.musicalPiece import *
from lib.dataset import *
from lib.file_handle import *

# =============================================================================
# Construction Phase
# =============================================================================
tf.reset_default_graph()

n_steps = 30
n_inputs = 127
n_neurons = 150 # 1 hidden layer with 150 neurons
n_outputs = 127
learning_rate = 0.001

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None])

cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu) #hiddel layer for 1 time step
outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32) # Use hidden layer for n_steps timesteps

logits = tf.layers.dense(states, n_outputs, name='predictions')
xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits)
loss = tf.reduce_mean(xentropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)
correct = tf.nn.in_top_k(logits, tf.cast(y, tf.int32), 4)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()

# =============================================================================
# Load Data
# =============================================================================
#Set up dataset
data = Dataset("Dataset 1","bach","Just the one hot vectors, with no pre-processing","28/5/2018")
data.load("Bach.pickle")

#Split input and targets. (I'll make a function to do that, also idk if this is the right way)
train_x = np.array(data.train[0:-1])
train_y = np.array(data.train[1:])
test_x = np.array(data.test[0:-1])
test_y = np.array(data.test[1:])

print("Train Length:" + str(len(data.train)))
print("Test Length :" + str(len(data.test)))
# =============================================================================
# Execution phase
# =============================================================================
n_epochs = 5
batch_size = 150

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(train_x[:,0].size // batch_size):
            X_batch, y_batch = train
            
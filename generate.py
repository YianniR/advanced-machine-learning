import numpy as np
import tensorflow as tf
from lib.networks import *
from lib.musicalPiece import *
from lib.dataset import *
from lib.file_handle import *
from music21 import *
import os

n_steps   = 25
n_inputs  = 127
n_neurons = 150
n_layers = 2
n_outputs = 127
batch_size = 500
n_epochs = 20

learning_rate = 0.001

dir_path = os.path.dirname(os.path.realpath(__file__))

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

# saver = tf.train.import_meta_graph(dir_path + "\log\LSTM_layer_2_model_20180507194433.ckpt.meta")

#Set up dataset
data = Dataset("Dataset 1","bach","Just the one hot vectors, with no pre-processing","28/5/2018")
data.load("bach.pickle")

#Split input and targets. (I'll make a function to do that, also idk if this is the right way)
train_x = data.train[0:-1]
train_y = data.train[1:]
test_x = data.test[0:-1]
test_y = data.test[1:]
seed = data.test[0:25]

saver = tf.train.Saver()

with tf.Session() as sess:
    #saver = tf.train.import_meta_graph(dir_path + "\log\LSTM_layer_2_model_20180507194433.ckpt.meta")
    #saver.restore(sess,dir_path + "\log\LSTM_layer_2_model_20180507194433.ckpt")

    #CHRIS TRAINED
    saver.restore(sess,dir_path + "\log\LSTM_layer_2_model_20180507194433.ckpt")

    # for op in tf.get_default_graph().get_operations():
    #     print(op.name)

    # X = tf.get_default_graph().get_tensor_by_name("inputs:0")
    # logits = tf.get_default_graph().get_tensor_by_name("outputs/beta/Adam/Assign:0")

    sequence = seed
    step = [0]*127
    sequence = [step]*n_steps
    for i in range(100):
        batch_x = np.array(sequence[-n_steps:]).reshape(1,n_steps,127)
        prediction = np.array(sess.run(logits,feed_dict={X: batch_x}))
        binary_prediction = np.array(sess.run(tf.to_int32(prediction> 0.01)))
        #sequence.append(prediction[0,:])
        sequence.append(binary_prediction[0,:])

    print("Sequence: ",sequence)

    save_var(dir_path+"\generated_sequence.pickle",sequence)


def stream_from_piano_roll(pianoroll, divisor):
    #go through each time step  in the piano roll. if there is a note there, search for the end and get duration (filling zeros behind it)
    #if it's a rest, look for first column with next note in and make a rest of that duration
    song_stream = stream.Stream()
    idx = 0
    while idx < len(pianoroll): #t is the time step, represented by a list
        t = pianoroll[idx]
        position = idx/divisor
        pitch = next((i for i, x in enumerate(t) if x==1), None) #get the position of the non-zero element
        if pitch == None: #rest or old note
            pitch = next((i for i, x in enumerate(t) if x==2), None)

        if pitch == None: #this is a rest
            #look for the end of the rest
            tmp_idx = idx+1
            while tmp_idx < len(pianoroll) and sum(pianoroll[tmp_idx]) == 0:
                tmp_idx += 1

            d = (tmp_idx - idx)/divisor

            #make a rest
            r = note.Rest()
            r.duration.quarterLength = d #duration is set. propogates through
            r.offset = position
            song_stream.append(r)

            idx = tmp_idx #skip all the rests
        else: #a value of 1 at the pitch means its a new note, a alue of 2 means its an old note
            #look for the end of the note
            if t[pitch] == 1: #it's a new note
                pianoroll[idx][pitch] = 2 #we have dealt with this note
                tmp_idx = idx+1
                while tmp_idx < len(pianoroll) and pianoroll[tmp_idx][pitch] == 1:
                    pianoroll[tmp_idx][pitch] = 2 #we have dealt with this note
                    tmp_idx += 1

                d = (tmp_idx - idx)/divisor

                #make a note
                n = note.Note()
                n.pitch.midi = pitch #pitch is set. propogates through to other properties
                n.duration.quarterLength = d #duration is set. propogates through
                n.offset = position
                song_stream.append(n)

                #here, we do not move onto the next time step in case we have simultaneous notes

            else: #it's an old note
                idx += 1 #move onto next time step


    return song_stream

song = stream_from_piano_roll(sequence, 8)
#song.show()

#make a file, then close immediately
f= open(os.path.expanduser(dir_path+'/generated_200.mid'),"w+")
f.close() #this is only done to ensure that the file exists

fp = song.write('midi', fp=os.path.expanduser(dir_path+'/generated_200.mid')) #save as a midi file

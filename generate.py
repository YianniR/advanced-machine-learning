import numpy as np
import tensorflow as tf
from lib.networks import *
from lib.musicalPiece import *
from lib.dataset import *
from lib.file_handle import *
from music21 import *
import os
import argparse

def stream_from_piano_roll(pianoroll, divisor):
    #ensure that we have all values in the piano roll at 0 or 1
    for t in pianoroll:
        for p in t:
            if p<1:
                p = 0;
            else:
                p = 1;



    #go through each time step  in the piano roll. if there is a note there, search for the end and get duration (filling zeros behind it)
    #if it's a rest, look for first column with next note in and make a rest of that duration
    song_stream = stream.Stream()
    idx = 0
    while idx < len(pianoroll): #t is the time step, represented by a list
        t = pianoroll[idx]
        position = idx/divisor
        pitch = next((i for i, x in enumerate(t) if x==1), None) #get the position of the non-zero element
        if pitch == None: #rest or old note
            pitch = next((i for i, x in enumerate(t) if x==2), None) #check for an old note

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
        else: #a value of 1 at the pitch means its a new note, a value of 2 means its an old note
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
                song_stream.insert(n)

                #here, we do not move onto the next time step in case we have simultaneous notes

            else: #it's an old note
                idx += 1 #move onto next time step


    return song_stream

#use the previous threshold output from the function as the next min_threshold
def threshold_to_one_hot(continuous_piano_roll, min_threshold=0):
    sorted_pr_idx = sorted(range(len(continuous_piano_roll)), key=lambda k: continuous_piano_roll[k]) #largest last - gives the sorted indexes

    first  = continuous_piano_roll[sorted_pr_idx[-1]]
    second = continuous_piano_roll[sorted_pr_idx[-2]]
    third  = continuous_piano_roll[sorted_pr_idx[-3]]

    thresholded_piano_roll = np.zeros(128)
    if(first > min_threshold):
        thresholded_piano_roll[sorted_pr_idx[-1]] = 1
        if(second > 0.9*first):
            thresholded_piano_roll[sorted_pr_idx[-2]] = 1
            if(third> 0.9*first):
                thresholded_piano_roll[sorted_pr_idx[-3]] = 1

    return {'pianoroll':thresholded_piano_roll, 'nextthreshold':third}

def main(args):

    #Set up parameters (can put those in a config file and parse them.)
    input_file = args.input_file
    output_file = args.output_file
    length = int(args.length)
    n_steps = 25
    n_inputs = 127
    n_neurons = 150
    n_layers = 2
    n_outputs = 127
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
        saver.restore(sess,dir_path + input_file)
        # sequence = seed
        step = [0]*127
        sequence = [step]*n_steps
        for i in range(length):
            batch_x = np.array(sequence[-n_steps:]).reshape(1,n_steps,127)
            prediction = np.array(sess.run(logits,feed_dict={X: batch_x}))
            binary_prediction = np.array(sess.run(tf.to_int32(prediction> 0.31)))
            # sequence.append(prediction[0,:])
            sequence.append(binary_prediction[0,:])

    print("Sequence: ",sequence)
    #save_var(dir_path+"\generated_sequence.pickle",sequence)

    # sequence = threshold_to_one_hot(sequence,0.07)
    song = stream_from_piano_roll(sequence, 1)

    #make a file, then close immediately
    f= open(os.path.expanduser(dir_path+"/"+output_file),"w+")
    f.close() #this is only done to ensure that the file exists

    fp = song.write('midi', fp=os.path.expanduser(dir_path+"/"+output_file)) #save as a midi file

if __name__ == "__main__":
    #Set up argument parser
	parser = argparse.ArgumentParser(description="Run an amazing neural network that makes music stuffs.")
	group = parser.add_argument_group('group')

	#Add parser arguments

	group.add_argument("-i", "--input_file", action="store", default = "\log\LSTM_2000_epochs.ckpt", help='Input dir to run')
	group.add_argument("-o", "--output_file", action="store", default = "generated.mid", help='Output Dir')
	group.add_argument("-l", "--length", action="store", default = 150, help='Length of song')

	#Parse arguments and start main
	args = parser.parse_args()
	main(args)

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
    step = [0.]*127
    sequence = [step]*n_steps
    for i in range(100):
        batch_x = np.array(sequence[-n_steps:]).reshape(1,n_steps,127)
        prediction = sess.run(logits,feed_dict={X: batch_x})
        binary_prediction = sess.run(tf.to_int32(prediction[149,:] > 0.1))

        sequence.append(binary_prediction)

    print(sequence)

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
f= open(os.path.expanduser(dir_path+'/generated.mid'),"w+")
f.close() #this is only done to ensure that the file exists

fp = song.write('midi', fp=os.path.expanduser(dir_path+'/generated.mid')) #save as a midi file

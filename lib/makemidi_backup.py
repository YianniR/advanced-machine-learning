"""
File to create a set of training data from the corpus
"""
from pre_processing import *
from reverse_processing import *
from file_handle import *
from music21 import *
import os

"""
#THIS SECTION IS USED TO GENERATE A TEST
song = parse_corpus('bach/bwv66.6')

note_sequence = flatted_single_part(song,0) #returns a sequence of notes with offsets
note_sequence.show()
pianoroll = flat_to_piano_roll(note_sequence, 8); #8 is the length of a crotchet
"""

#load the generated piano roll from a file
generated_pianoroll = load_var('~/Desktop/generated_sequence.pickle')

song = stream_from_piano_roll(generated_pianoroll, 1)
#song.show()

#make a file, then close immediately
f= open(os.path.expanduser('~/Desktop/generated.mid'),"w+")
f.close() #this is only done to ensure that the file exists

fp = song.write('midi', fp=os.path.expanduser('~/Desktop/generated.mid')) #save as a midi file
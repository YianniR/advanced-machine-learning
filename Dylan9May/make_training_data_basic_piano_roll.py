"""
File to create a set of training data from the corpus
"""
from pre_processing import *
from file_handle import *
from music21 import *
import os

bachBundle = corpus.getComposer('bach') #get all bach pieces
print(bachBundle)

for idx, corpus_file in enumerate(bachBundle):
    song = parse_corpus(corpus_file)
    note_sequence = flatted_single_part(song,0) #returns a sequence of notes with offsets
    pianoroll = flat_to_piano_roll(note_sequence, 8); #8 is the length of a crotchet
    
    save_var("~/Desktop/pianoroll/%d.hex" %idx, pianoroll)
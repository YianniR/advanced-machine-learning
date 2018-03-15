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
    note_sequence_single = remove_simultaneous_notes(note_sequence)
    note_sequence_single = remove_rests_and_fill_time(note_sequence_single) #fixes issues with rests not being taken care of properly
    filled_durations = stretch_duration_to_fill_time(note_sequence_single)
    
    durations = durations_to_integers(filled_durations, 4)
    pitches = flat_to_midi_pitches(note_sequence_single)
    
    one_hot_pitches = list_to_one_hot(pitches)
    one_hot_duration = list_to_one_hot(durations) #durations include a 0th row, but will never actually have anything in there (unless i suppose it's a tiny ass note)
    
    if one_hot_pitches != -1 and one_hot_duration != -1:
        save_var("~/Desktop/pitches/%d.hex" %idx, one_hot_pitches)
        save_var("~/Desktop/durations/%d.hex" %idx, one_hot_duration)


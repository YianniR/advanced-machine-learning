'''
Includes functions that aid to the preprocessing of data,
such as parsing, converting formats, filtering and normalizing.
'''

from music21 import converter, instrument, note, chord, stream
import numpy as np

def play_notes(notes):
    midi_stream = stream.Stream(notes)
    midi_stream.show('midi')

def parse_midi_file(filename):
    midi = converter.parse(filename)
    return midi.flat.notes

def load_data():
    # TODO: Read a bunch of midi files on memmory and train.

def add_to_one_hot(vector,element):
    vector[element] += 1
    return vector

def song_to_one_hot(note_sequence):
    song = list()
    one_hot_step = np.zeros(127)

    #Loop over whole song. Element can be a note or a chord.
    #If the offset changes, add the one hot step to the song list.
    previus_offset = 0.0
    for element in note_sequence:
        if element.offset != previus_offset:
            previus_offset = element.offset
            song.append(one_hot_step)
            one_hot_step = np.zeros(127)

        #If a note is found, add it to the one hot step
        if isinstance(element, note.Note):
            add_to_one_hot(one_hot_step,element.pitch.midi)

        #If a chord is found, add every note of the chord.
        elif isinstance(element, chord.Chord):
            for chord_note in element.pitches:
                add_to_one_hot(one_hot_step,chord_note.midi)

    return song

def batch_producer():
    # TODO: Make train and test sets, with inputs and targets as one hot feature vectors.

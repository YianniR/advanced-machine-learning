from music21 import converter, instrument, note, chord, stream, corpus
import numpy as np
import pickle
import os

class Data(object):

    def __init__(self, n_steps, step_size, n_notes):
        self.inputs = np.zeros((n_steps, step_size, n_notes))
        self.targets = np.zeros((n_steps, step_size, n_notes))

class MusicalPiece(object):

    def __init__(self,name="Sample Song"):
        self.name = name
        self.path = ""
        self.file_extension = ""
        self.filename = ""
        self.full_music21_stream = stream.Stream()
        self.one_hot_vector_sequence = []

    def play(self):
        # midi_stream = stream.Stream(self.full_music21_stream)
        # midi_stream.show('midi')
        self.full_music21_stream.show('text')

    def load_song(self,infile):
        path, file_extension = os.path.splitext(infile)
        self.path = path
        self.file_extension = file_extension
        self.filename = path + file_extension

        print(self.path)

        if self.file_extension == ".midi":
            self.parse_midi_file()
        else:
            self.parse_corpus()

    def parse_midi_file(self):
        self.full_music21_stream = converter.parse(self.filename)

    def parse_corpus(self):
        self.full_music21_stream = corpus.parse(self.filename)

    def save_song_as_midi(self,path):
        s = stream.Stream(self.full_music21_stream)
        path =  self.path + ".midi"
        fp = s.write('midi', fp=path)

    def make_pitches_one_hot(self,n_notes=127,keep_chords = False):
        one_hot_step = np.zeros(n_notes)

        #Loop over whole song. Element can be a note or a chord.
        #If the offset changes, add the one hot step to the song list.
        previous_offset = 0.0
        for element in self.full_music21_stream.flat.notes:
            if element.offset != previous_offset:
                previous_offset = element.offset
                self.one_hot_vector_sequence.append(one_hot_step)
                one_hot_step = np.zeros(n_notes)

            #If a note is found, add it to the one hot step
            if isinstance(element, note.Note):
                one_hot_step[element.pitch.midi] = 1#element.beatDuration.quarterLength

            #If a chord is found, add every note of the chord.
            elif keep_chords is True:
                if isinstance(element, chord.Chord):
                    for chord_note in element.pitches:
                        one_hot_step[chord_note.midi] = 1#element.beatDuration.quarterLength


    def make_targets(self, step_size, n_notes=127):
        #Each song is a batch
        self.make_pitches_one_hot(n_notes,keep_chords=True)

        n_steps = int(len(self.one_hot_vector_sequence)/step_size)
        self.training_data = Data(n_steps, step_size, n_notes)

        for i in range(0, n_steps):
            X_sequence = self.one_hot_vector_sequence[i*step_size:(i+1)*step_size]
            y_sequence = self.one_hot_vector_sequence[i*step_size+1:(i+1)*step_size+1]

            self.training_data.inputs[i] = X_sequence
            self.training_data.targets[i] = y_sequence

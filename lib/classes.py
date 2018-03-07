from music21 import converter, instrument, note, chord, stream
import numpy as np
import pickle
import os

class Data(object):

    def __init__(self, n_steps, step_size, n_notes):
        self.inputs = np.zeros((n_steps, step_size, n_notes))
        self.targets = np.zeros((n_steps, step_size, n_notes))

class Song(object):
    name = ""
    note_sequence = []
    one_hot_sequence = []

    def __init__(self, path):
        filename, file_extension = os.path.splitext(path)
        self.name = filename
        if file_extension is "midi":
            self.parse_midi_file()
        elif file_extension is "pickle":
            print("PICKLE RICKKK")
            self.load_song_from_pickle()

    def parse_midi_file(self):
        self.note_sequence = converter.parse(self.name).flat.notes

    def play(self):
        midi_stream = stream.Stream(self.note_sequence)
        midi_stream.show('midi')

    def song_to_one_hot(self,n_notes):
        one_hot_step = np.zeros(n_notes)

        #Loop over whole song. Element can be a note or a chord.
        #If the offset changes, add the one hot step to the song list.
        previus_offset = 0.0
        for element in self.note_sequence:
            if element.offset != previus_offset:
                previus_offset = element.offset
                self.one_hot_sequence.append(one_hot_step)
                one_hot_step = np.zeros(n_notes)

            #If a note is found, add it to the one hot step
            if isinstance(element, note.Note):
                one_hot_step[element.pitch.midi] += 1

            #If a chord is found, add every note of the chord.
            elif isinstance(element, chord.Chord):
                for chord_note in element.pitches:
                    one_hot_step[chord_note.midi] += 1

    def make_targets(self, step_size, n_notes=127):
        #Each song is a batch
        self.song_to_one_hot(n_notes)

        n_steps = int(len(self.one_hot_sequence)/step_size)
        self.data = Data(n_steps, step_size, n_notes)

        for i in range(0, n_steps):
            X_sequence = self.one_hot_sequence[i*step_size:(i+1)*step_size]
            y_sequence = self.one_hot_sequence[i*step_size+1:(i+1)*step_size+1]

            self.data.inputs[i] = X_sequence
            self.data.targets[i] = y_sequence

    def save_song_as_pickle(self):
        filename =  self.name + ".pickle"
        with open(filename, 'wb') as output_file:
            pickle.dump(self.data, output_file, protocol=pickle.HIGHEST_PROTOCOL)

    def load_song_from_pickle(self):
        with open(self.name, "rb") as input_file:
            self.data = pickle.load(input_file)

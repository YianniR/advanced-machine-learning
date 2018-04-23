from music21 import converter, instrument, note, chord, stream, corpus, freezeThaw
import numpy as np
from random import shuffle
from lib.musicalPiece import *
from lib.file_handle import *

class Dataset(object):

    def __init__(self,dataset_name,composer,description,date_made):
        self.dataset_name = dataset_name
        self.composer = composer
        self.description = description
        self.date_made = date_made
        self.train = list()
        self.test = list()

    def make_dataset(self,trainingSplit):
        print("Making Dataset!")

        bundle = corpus.getComposer(self.composer,fileExtensions='mxl') #get all bach pieces
        shuffle(bundle)#Shuffle to make dataset less likely to create a bias
        bundle = bundle[:10]

        tr_bundle = bundle[:int(len(bundle)*trainingSplit)]
        ts_bundle = bundle[int(len(bundle)*trainingSplit)+1:]

        for idx, corpus_file_name in enumerate(tr_bundle):
            piece = MusicalPiece()
            piece.load_song(corpus_file_name)

            piece.make_pitches_one_hot(keep_chords=False)
            self.train.append(piece)

        for idx, corpus_file_name in enumerate(ts_bundle):
            piece = MusicalPiece()
            piece.load_song(corpus_file_name)
            piece.make_pitches_one_hot(keep_chords=False)
            self.test.append(piece)

        print("Dataset complete!")

    def save(self,filename):

        for idx, piece in enumerate(self.train):
            sf = freezeThaw.StreamFreezer(self.train[idx].full_music21_stream)
            sf.setupSerializationScaffold()

        for idx, piece in enumerate(self.test):
            sf = freezeThaw.StreamFreezer(self.test[idx].full_music21_stream)
            sf.setupSerializationScaffold()

        save_var(filename,self)
        print("Dataset saved!")


    def load(self,filename):
        self.train = load_var(filename).train
        self.test = load_var(filename).test

        sf =freezeThaw.StreamThawer()

        for idx, piece in enumerate(self.train):
            sf.teardownSerializationScaffold(self.train[idx].full_music21_stream)

        for idx, piece in enumerate(self.test):
            sf.teardownSerializationScaffold(self.test[idx].full_music21_stream)

        print("Dataset loaded!")

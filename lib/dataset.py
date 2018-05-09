from music21 import converter, instrument, note, chord, stream, corpus, freezeThaw
import numpy as np
from random import shuffle
from lib.musicalPiece import *
from lib.file_handle import *
'''
This library allows for the creation of .pickle files which will include datasets
of training and test data, made from MusicalPieces.
'''

class Dataset(object):

    def __init__(self,dataset_name,composer,description,date_made):
        self.dataset_name = dataset_name
        self.composer = composer
        self.description = description
        self.date_made = date_made
        self.train = list()
        self.test = list()

    def make_dataset(self,trainingSplit):
        #Make another function to create datasets from midi files in a given folder
        print("Making Dataset!")

        bundle = corpus.getComposer(self.composer,fileExtensions='mxl') #get all bach pieces
        print(len(bundle))
        shuffle(bundle)#Shuffle to make dataset less likely to create a bias
        #bundle = bundle[:9] #Limit dataset size for testing

        tr_bundle = bundle[:int(len(bundle)*trainingSplit)]
        ts_bundle = bundle[int(len(bundle)*trainingSplit)+1:]

        for idx, corpus_file_name in enumerate(tr_bundle):
            piece = MusicalPiece()
            piece.load_song(corpus_file_name)

            transpositions = [0,4,-4,12,-12] #oriignal, up and down a major third, and up and down an octave
            for trans_idx in range(len(transpositions)):
                trans = transpositions[trans_idx]
                piece.make_pitches_one_hot(keep_chords=True,transposition=trans)

            self.train = self.train + piece.one_hot_vector_sequence

        for idx, corpus_file_name in enumerate(ts_bundle):
            piece = MusicalPiece()
            piece.load_song(corpus_file_name)

            transpositions = [0,4,-4,12,-12]; #oriignal, up and down a major third, and up and down an octave
            for trans_idx in range(len(transpositions)):
                trans = transpositions[trans_idx];
                piece.make_pitches_one_hot(keep_chords=True,transposition=trans)

            self.test = self.test + piece.one_hot_vector_sequence

        print("Dataset complete!")

    def save(self,filename):

        save_var(filename,self)
        print("Dataset saved!")


    def load(self,filename):
        self.train = load_var(filename).train
        self.test = load_var(filename).test

        print("Dataset loaded!")

'''
Includes functions that aid to the preprocessing of data,
such as parsing, converting formats, filtering and normalizing.
'''

from music21 import *
import numpy as np
from math import ceil
import operator

def play_notes(notes):
    midi_stream = stream.Stream(notes)
    midi_stream.show('midi')

def parse_midi_file(filename):
    midi = converter.parse(filename)
    return midi

def parse_corpus(filename):
    
    full_piece = corpus.parse(filename)
    return full_piece

def flatted_single_part(full_piece, part_number):
    return full_piece.parts[part_number].flat.notesAndRests

def flat_to_midi_pitches(no_rests):
    pitches = list() #not finished as it does not take into account the time signature
    for el in no_rests:
        pitches.append(el.pitch.midi)
    return pitches

#remove the rests from the stave, and fill the gaps with the previous note
def remove_rests_and_fill_time(flat_note_vector):
    no_rests = stream.Stream()
    for idx, el in enumerate(flat_note_vector):
        if el.isNote:
            no_rests.append(el)
            
        elif el.isRest:
            if len(no_rests) > 1:
                no_rests[-1].duration.quarterLength = no_rests[-1].duration.quarterLength + el.duration.quarterLength
            
        else:
            raise ValueError("Element is not a note or rest...")
            return -1    
    
    return no_rests

#returns just the durations of the stretched notes which fill the spaces left by rests
def stretch_duration_to_fill_time(no_rests): #remove rests in this function
    time_durations = list()
    
    
    for i in range(0,len(no_rests)-1): #there is a bug here that causes rests to act inccorectly. fi by checking that all elements are notes before hand
        time_durations.append( no_rests[i+1].offset - no_rests[i].offset)
        
    #final duration remains the same
    time_durations.append( no_rests[-1].duration.quarterLength)
    return time_durations

def durations_to_integers(float_durations, divisor): #divisor is the smallest size of note allowed (1 for crotches, 2 for quaver etc.)
    int_durations = np.array(float_durations)*divisor
    return int_durations.astype(int)

def remove_simultaneous_notes(flat_note_vector):
    flat_note_vector = flat_note_vector.stripTies()
    corrected = stream.Stream()
    previous_offset = flat_note_vector[0].offset + 1
    for el in flat_note_vector:
        if el.offset != previous_offset:
            if el.isChord:
                corrected.append( note.Note( el.root() ))
            elif el.isNote:
                corrected.append(el)
            elif el.isRest: 
                corrected.append(el)
            else:
                raise ValueError("Element is not a note, chord, or rest...")
                return -1

        
        previous_offset = el.offset
    return corrected #still keep rests

#returns a 128 wide, X long matrix of one hots from the input list
def list_to_one_hot(note_values):
    one_hot_full = list()

    for el in note_values:
        if el > 127:
            print('Duration/Pitch is out of bounds of size of 1-hot matrix. Skipping file')
            return -1
            
            
        one_hot_step = np.zeros(128)
        one_hot_step[el] = 1
        one_hot_full.append(one_hot_step)

    return one_hot_full

#creates (for example), a 1 hot where the position is the pitch, and the value of the '1' is the duration
    # inputs are lists of integers
    #not necessarily the difference
def pitch_duration_matrix(position_representation, value_representation):
    #create a matrix with the position of a value dictating a variable, and the value of that element also dictating a variable
    one_hot_full = list()

    if len(position_representation) != len(value_representation):
        raise ValueError("Arrays must have the same size")
        return -1

    if (min(value_representation) < 1):
        raise ValueError("Value must not be less than 1. (Zero is bad)")
        return -1

    for i in range(0, len(position_representation)):
        one_hot_step = np.zeros(128)
        one_hot_step[position_representation[i]] = value_representation[i]
        one_hot_full.append(one_hot_step)

    return one_hot_full


def difference_in_list(ip_list):
    diff = list()
    diff.append(0)
    diff = diff + list(map(operator.sub,ip_list[1:], ip_list[:-1]))
    return diff

#puts a 1 in the position of the one-hot difference
def list_to_one_hot_differences(note_values):
    max_difference = 128
    one_hot_full = list()
    differences = difference_in_list(note_values)

    for el in differences:
        one_hot_step = np.zeros((max_difference-1)*2 +1)
        one_hot_step[(el*-1) + max_difference - 1] = 1 #centers a difference of 0 @ idx 127
        one_hot_full.append(one_hot_step)

    return one_hot_full


#puts a value (not necessarily 1) in the position of the one-hot difference
    #
def list_to_combined_differences(position,value):
    max_difference = 128
    one_hot_full = list()
    differences = difference_in_list(position)

    for i in range(0,len(differences)):
        one_hot_step = np.zeros((max_difference-1)*2 +1)
        one_hot_step[(differences[i] * -1) + max_difference - 1] = value[i] #centers a difference of 0 @ idx 127
        one_hot_full.append(one_hot_step)

    return one_hot_full


#function to create a piano roll from flat part (takes int durations as returned by durations_to_integers)
def flat_to_piano_roll(flat, divisor):
    #first of all, make the piano roll of all zeros
    pianoroll = list()
    for i in range(0,ceil(flat.duration.quarterLength*divisor)):
        pianoroll.append(np.zeros(128))
    
    
    for idx, el in enumerate(flat):
        #get the duration and the position of it
        if el.isNote: 
            d = ceil(el.duration.quarterLength * divisor) #d is the quantity of time steps that the note fills
            p = ceil(el.offset * divisor) #p is the starting position of the note
            for t in range(p,p+d): #fill ones in for the relevant places
                pianoroll[t][el.pitch.midi] = 1;
            
        elif not el.isRest:
            print("Element is not a note or rest... Probably no need to worry")
            #return -1    
            
    
    return pianoroll
    
#function to create a piano roll from flat part, where a rest has a note value

    
    
    
    
    
    
    
    
    
    
    
    
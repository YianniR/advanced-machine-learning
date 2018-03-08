'''
Includes functions that aid to the preprocessing of data,
such as parsing, converting formats, filtering and normalizing.
'''

from music21 import * 
import numpy as np
import operator

def play_notes(notes):
    midi_stream = stream.Stream(notes)
    midi_stream.show('midi')

def parse_midi_file(filename):
    midi = converter.parse(filename)
    return midi.flat.notes

def parse_corpus(filename):
    
    full_piece = corpus.parse(filename)
    return full_piece

def flatted_single_part(full_piece, part_number):
    return full_piece.parts[part_number].flat.notes

def flat_to_midi_pitches(flat_note_vector):
    pitches = list() #not finished as it does not take into account the time signature
    for el in flat_note_vector:
        pitches.append(el.pitch.midi)
    return pitches

def stretch_duration_to_fill_time(flat_note_vector):
    time_durations = list()
    for i in range(0,len(flat_note_vector)-1):
        time_durations.append( flat_note_vector[i+1].offset - flat_note_vector[i].offset)
    #final duration remains the same
    time_durations.append( flat_note_vector[-1].duration.quarterLength)
    return time_durations

def durations_to_integers(float_durations, divisor): #divisor is the smallest size of note allowed (1 for crotches, 2 for quaver etc.)
    int_durations = np.array(float_durations)*divisor
    return int_durations.astype(int)
    
def remove_simultaneous_notes(flat_note_vector):
    corrected = stream.Stream()
    corrected.append(flat_note_vector[0])
    previous_offset = flat_note_vector[0].offset
    for el in flat_note_vector[1:]:
        if el.offset != previous_offset:
            corrected.append(el)
        previous_offset = el.offset       
    return corrected

def list_to_one_hot(note_values):
    one_hot_full = list()
    
    for el in note_values:
        one_hot_step = np.zeros(128)
        one_hot_step[el] = 1
        one_hot_full.append(one_hot_step)
    
    return one_hot_full

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

def list_to_one_hot_differences(note_values,max_difference):
    one_hot_full = list()
    differences = difference_in_list(note_values)
    
    for el in differences:
        one_hot_step = np.zeros((max_difference-1)*2 +1)
        one_hot_step[(el*-1) + max_difference - 1] = 1 #centers a difference of 0 @ idx 127
        one_hot_full.append(one_hot_step)
    
    return one_hot_full

    
def list_to_combined_differences(position,max_difference,value):
    one_hot_full = list()
    differences = difference_in_list(position)
    
    for i in range(0,len(differences)):
        one_hot_step = np.zeros((max_difference-1)*2 +1)
        one_hot_step[(differences[i] * -1) + max_difference - 1] = value[i] #centers a difference of 0 @ idx 127
        one_hot_full.append(one_hot_step)
    
    return one_hot_full
    
    
    
'''
Functions to reconstruct a music file from pitches and durations in any form
'''

from music21 import * 
import numpy as np
import operator
    
def ensure_single_line_in_one_hot(one_hot,correct): #if correct is true, the one hot will be corrected, otherwise an error will be returned
    corrected = list()
    for idx, el in enumerate(one_hot):
        hits = np.count_nonzero(el)
        if hits > 1:
            if correct == 1:
                #remove entries
                temp_idx = 0
                while np.count_nonzero(el) > 1:
                    el[temp_idx] = 0
                    temp_idx += 1
                
                corrected.append(el)
                        
            else:
                raise ValueError("Simultaneous notes in this one_hot. To correct, pass 1 as second argument")
                return -1
            
        elif hits == 0:
            #remove this row
            if correct == 0:
                raise ValueError("No Notes found in this one-hot column. To correct, pass 1 as second argument")
                return -1
        else:
            corrected.append(el)
            
    return corrected

def song_from_pitches_durations(int_pitches, int_durations,divisor):
    position = 0.0
    song_stream = stream.Stream()
    for p,d in zip(int_pitches,int_durations):
        d_c = d/divisor
        n = note.Note()
        n.pitch.midi = p #pitch is set. propogates through to other properties
        n.duration.quarterLength = d_c #duration is set. propogates through
        n.offset = position #set the position to come directly after previous note ends
        position += d_c
        song_stream.append(n)
    return song_stream

def stream_from_one_hot_pitch_duration(one_hot_pitches, one_hot_durations,divisor): #takes two one hot matrices
    int_pitches = list()
    int_durations = list()
    
    for p,d in zip(one_hot_pitches,one_hot_durations):
        int_pitches.append(p.tolist().index(1))
        int_durations.append(d.tolist().index(1))
    
    song = song_from_pitches_durations(int_pitches,int_durations,divisor)
    
    return song
    
    
def stream_from_positional_value_representation(data,divisor,position): #takes two one hot matrices
    int_pos = list()
    int_val = list()
    
    standard = position == 'pitch' #set position to 'pitch' to make it work normally. says that the pitch is represented by the position of the duration value
    
    for el in data:
        tmp_pos = [i for i,v in enumerate(el) if v > 0]
        tmp_val = el[tmp_pos].item(0)
        
        int_pos += tmp_pos
        int_val.append(tmp_val)
    
    if standard:
        song = song_from_pitches_durations(int_pos,int_val,divisor)
    else:
        song = song_from_pitches_durations(int_val,int_pos,divisor)
    
    return song

def stream_from_one_hot_differences(diff_pitch,diff_duration,divisor,transposition,start_duration): #takes two one hot matrices
    start_duration = start_duration * divisor
    int_pitches = list()
    int_duration = list()
    int_pitches.append(transposition)
    int_duration.append(start_duration)
    
    for el in diff_pitch[1:]:
        tmp = int_pitches[-1]-(el.tolist().index(1)-127)
        if tmp < 0:
            raise ValueError("Negative Pitch is not allowed. Increase transposition")
            return -1
        if tmp > 127:
            raise ValueError("Pitch over 127 is not allowed. Decrease transposition")
            return -1
        int_pitches.append(tmp)
    
    for el in diff_duration[1:]:
        tmp = int_duration[-1]-(el.tolist().index(1)-127)
        if tmp < 0:
            raise ValueError("Negative Duration is not allowed. Increase starting duration")
            return -1
        if tmp > 127:
            raise ValueError("Duration over 127 is not allowed. Decrease starting duration")
            return -1
        int_duration.append(tmp)
        
    song = song_from_pitches_durations(int_pitches,int_duration,divisor)
    
    return song

def stream_from_combined_differences(data,divisor,start_pos_val,position): #takes two one hot matrices
    int_pos = list()
    int_val = list()
    
    standard = position == 'pitch'
    if standard == 0:
        start_pos_val = start_pos_val * divisor
    
    prev_pos = start_pos_val
    for el in data:
        tmp_pos = [i for i,v in enumerate(el) if v > 0]
        tmp_val = el[tmp_pos].item(0)
        
        tmp_pos = tmp_pos[0]
        tmp_pos = prev_pos - (tmp_pos - 127)
        
        prev_pos = tmp_pos
        
        if tmp_pos < 0:
            raise ValueError("Negative Value from position is not allowed. Increase starting position")
            return -1
        if tmp_pos >127:
            raise ValueError("Value over 127 is not allowed. Decrease starting position")
            return -1
        
        
        int_pos.append(tmp_pos)
        int_val.append(tmp_val)
    
    if standard:
        song = song_from_pitches_durations(int_pos,int_val,divisor)
    else:
        song = song_from_pitches_durations(int_val,int_pos,divisor)
    
    return song

def stream_from_piano_roll(pianoroll, divisor):
    #go through each time step  in the piano roll. if there is a note there, search for the end and get duration (filling zeros behind it)
    #if it's a rest, look for first column with next note in and make a rest of that duration
    song_stream = stream.Stream()
    idx = 0
    while idx < len(pianoroll): #t is the time step, represented by a list
        t = pianoroll[idx]
        position = idx/divisor
        pitch = next((i for i, x in enumerate(t) if x==1), None) #get the position of the non-zero element
        if pitch == None: #rest or old note
            pitch = next((i for i, x in enumerate(t) if x==2), None)
            
        if pitch == None: #this is a rest
            #look for the end of the rest
            tmp_idx = idx+1
            while tmp_idx < len(pianoroll) and sum(pianoroll[tmp_idx]) == 0:
                tmp_idx += 1
                
            d = (tmp_idx - idx)/divisor
            
            #make a rest
            r = note.Rest()
            r.duration.quarterLength = d #duration is set. propogates through
            r.offset = position 
            song_stream.append(r)
            
            idx = tmp_idx #skip all the rests
        else: #a value of 1 at the pitch means its a new note, a alue of 2 means its an old note
            #look for the end of the note
            if t[pitch] == 1: #it's a new note
                pianoroll[idx][pitch] = 2 #we have dealt with this note
                tmp_idx = idx+1
                while tmp_idx < len(pianoroll) and pianoroll[tmp_idx][pitch] == 1:
                    pianoroll[tmp_idx][pitch] = 2 #we have dealt with this note
                    tmp_idx += 1
                    
                d = (tmp_idx - idx)/divisor
                
                #make a note
                n = note.Note()
                n.pitch.midi = pitch #pitch is set. propogates through to other properties
                n.duration.quarterLength = d #duration is set. propogates through
                n.offset = position 
                song_stream.append(n)
                    
                #here, we do not move onto the next time step in case we have simultaneous notes
                
            else: #it's an old note
                idx += 1 #move onto next time step
            
            
    return song_stream
    
    
    
    
    
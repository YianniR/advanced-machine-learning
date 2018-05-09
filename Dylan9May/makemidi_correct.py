from music21 import *
import os

#load the generated piano roll from a file
generated_pianoroll = #variable here

song = stream_from_piano_roll(generated_pianoroll, 1)
#song.show()

#make a file, then close immediately
f= open(os.path.expanduser('~/Desktop/generated.mid'),"w+")
f.close() #this is only done to ensure that the file exists

fp = song.write('midi', fp=os.path.expanduser('~/Desktop/generated.mid')) #save as a midi file


def stream_from_piano_roll(pianoroll, divisor):
    #ensure that we have all values in the piano roll at 0 or 1
    for t in pianoroll:
        for p in t:
            if p<1:
                p = 0;
            else:
                p = 1;
        
    
    
    #go through each time step  in the piano roll. if there is a note there, search for the end and get duration (filling zeros behind it)
    #if it's a rest, look for first column with next note in and make a rest of that duration
    song_stream = stream.Stream()
    idx = 0
    while idx < len(pianoroll): #t is the time step, represented by a list
        t = pianoroll[idx]
        position = idx/divisor
        pitch = next((i for i, x in enumerate(t) if x==1), None) #get the position of the non-zero element
        if pitch == None: #rest or old note
            pitch = next((i for i, x in enumerate(t) if x==2), None) #check for an old note
            
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
        else: #a value of 1 at the pitch means its a new note, a value of 2 means its an old note
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
                song_stream.insert(n)
                    
                #here, we do not move onto the next time step in case we have simultaneous notes
                
            else: #it's an old note
                idx += 1 #move onto next time step
            
            
    return song_stream
    
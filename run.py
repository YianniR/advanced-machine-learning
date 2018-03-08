from pre_processing import *

song_name = 'bach/bwv431.mxl'
    
note_sequence = flatted_single_part(parse_corpus(song_name),1) #returns a sequence of notes with offsets
note_sequence_single = remove_simultaneous_notes(note_sequence)

filled_durations = stretch_duration_to_fill_time(note_sequence_single)
durations = durations_to_integers(filled_durations, 8)
#note_sequence.show()
#note_sequence_single.show()

#get midi values
pitches = flat_to_midi_pitches(note_sequence_single)
#print(pitches)

#need to write programs to take pitches and durations and output different representaitons 
#1 hot, simpleduration in pitch etc.

one_hot_pitches = list_to_one_hot(pitches)
one_hot_duration = list_to_one_hot(durations) #durations include a 0th row, but will never actually have anything in there (unless i suppose it's a tiny ass note)

matrix_representation = pitch_duration_matrix(pitches, durations) #pitches represented by position in the array, durations are the values at those positions

# now to make a difference representation
one_hot_pitches_difference = list_to_one_hot_differences(pitches,128)
one_hot_duration_difference = list_to_one_hot_differences(durations,128)


combined_differences_matrix = list_to_combined_differences(pitches,128,durations)
from pre_processing import *
from reverse_processing import *

song_name = 'bach/bwv431.mxl'
# we are losing ties - so notes appear shorter than they should - need to fix that
note_sequence = flatted_single_part(parse_corpus(song_name),1) #returns a sequence of notes with offsets
note_sequence_single = remove_simultaneous_notes(note_sequence)

filled_durations = stretch_duration_to_fill_time(note_sequence_single)
durations = durations_to_integers(filled_durations, 8)
#note_sequence.show()
note_sequence_single.show()

#get midi values
pitches = flat_to_midi_pitches(note_sequence_single)
#print(pitches)

#need to write programs to take pitches and durations and output different representaitons 
#1 hot, simpleduration in pitch etc.

one_hot_pitches = list_to_one_hot(pitches)
one_hot_duration = list_to_one_hot(durations) #durations include a 0th row, but will never actually have anything in there (unless i suppose it's a tiny ass note)

matrix_representation = pitch_duration_matrix(pitches, durations) #pitches represented by position in the array, durations are the values at those positions

# now to make a difference representation
one_hot_pitches_difference = list_to_one_hot_differences(pitches)
one_hot_duration_difference = list_to_one_hot_differences(durations)


combined_differences_matrix = list_to_combined_differences(pitches,durations)



##### REVERESE PROCESSING

incorrect_one_hot_pitches = np.add(one_hot_pitches,one_hot_duration)

fixed_one_hot = ensure_single_line_in_one_hot(incorrect_one_hot_pitches,1) #removes simultaneous - pass 0 as second argument to return errors instead


song = stream_from_one_hot_pitch_duration(one_hot_pitches,one_hot_duration,8)
#song.show() #this is a recreation of the note_sequence

song = stream_from_positional_value_representation(matrix_representation ,8,'pitch')#pitch or duration - which one is represented by position?

#with the difference represenation, you must specify the starting value of whatever is being represented with differences.
song = stream_from_one_hot_differences(one_hot_pitches_difference,one_hot_duration_difference,8,60,1.5) #pitches,duration,divisor,transposition,beginning duration
#song = stream_from_combined_differences(combined_differences_matrix,8)

song = stream_from_combined_differences(combined_differences_matrix,8,60,'pitch')


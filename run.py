from preprocessing import parse_midi_file, song_to_one_hot, play_notes

def main():
    note_sequence = parse_midi_file('data\Around The World - Chorus.midi')
    one_hot_sequence = song_to_one_hot(note_sequence)
    play_notes(note_sequence)

if __name__ == "__main__":
    main()

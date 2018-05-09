import numpy as np
import os
import binascii

directory = os.path.dirname(os.path.realpath(__file__))
path = directory + "\lib\mackey_glass_1000.hex"

print (path)

def read_hex_to_int(filename, text = False):
    with open(filename, 'rb') as f:
        data_file = binascii.hexlify(f.read())
    num_array = np.empty(int(len(data_file)/2), dtype=int)
    if text == True:
        char_array = np.empty_like(num_array, dtype=str)
    index = 0
    for i in range(0, len(data_file), 2):
        integer = int(data_file[i:i+2], 16)
        num_array[index] = integer
        if text == True:
            char_array[index] = chr(integer)
        index += 1
    if text == True:
        return num_array, char_array
    else:
        return num_array

data = read_hex_to_int(path)
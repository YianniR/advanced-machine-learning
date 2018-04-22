"""
Save and load variables from files
"""

import pickle
import os

def save_var(file_name, variable):
    with open(os.path.expanduser(file_name), 'wb') as f:
        pickle.dump(variable, f)

def load_var(file_name):
    with open(os.path.expanduser(file_name), 'rb') as f:
        obj = pickle.load( f )
        
    return obj


import numpy as np
import tensorflow as tf
from lib.networks import *
from lib.musicalPiece import *
from lib.dataset import *
from lib.file_handle import *

#Set up dataset
data = Dataset("Dataset 1","bach","Just the one hot vectors, with no pre-processing","28/5/2018")
data.load("bach.pickle")

#Split input and targets. (I'll make a function to do that, also idk if this is the right way)
train_x = np.array(data.train[0:-1])
train_y = np.array(data.train[25:])
test_x = np.array(data.test[0:-1])
test_y = np.array(data.test[25:])

x_batch, y_batch, n_batches = batch_maker2(train_x, train_y, 25, 150)
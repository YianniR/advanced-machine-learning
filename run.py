from lib.networks import *
from lib.musicalPiece import *
from lib.dataset import *
from lib.file_handle import *

from pprint import pprint

def main():

    #Set up parameters (can put those in a config file and parse them.)
    batch_size = 200
    n_steps   = 30
    n_inputs  = 127
    n_neurons = 150
    n_outputs = 127
    learning_rate = 0.001
    num_epochs = 10

    #Set up placeholders
    input_,target_ = set_up_placeholders(n_inputs,n_steps,n_outputs)

    #Set up dataset
    data = Dataset("Dataset 1","bach","Just the one hot vectors, with no pre-processing","28/5/2018")
    # data.make_dataset(trainingSplit=0.9)
    # data.save("dataset1.pickle")
    data.load("bach.pickle")

    #Split input and targets. (I'll make a function to do that, also idk if this is the right way)
    train_x = data.train[0:-1]
    train_y = data.train[1:]
    test_x = data.test[0:-1]
    test_y = data.test[1:]



    print("Train Length:" + str(len(data.train)))
    print("Test Length :" + str(len(data.test)))

    # #Setup model and train it
    #prediction = multilayer_perceptron_model(input_,n_inputs,150,100,75,n_outputs)
    prediction = rnn_model(input_,n_steps,n_outputs)
    train(input_,target_,prediction,train_x, train_y, test_x, test_y,n_steps,batch_size,num_epochs)

if __name__ == "__main__":
    main()

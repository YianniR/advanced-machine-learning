from lib.networks import *
from lib.musicalPiece import *
from lib.dataset import *
from lib.file_handle import *

def main():

    #Set up parameters (can put those in a config file and parse them.)
    n_steps   = 10
    n_inputs  = 127
    n_outputs = 10
    learning_rate = 0.001
    batch_size = 1
    num_epochs = 2

    #Set up placeholders
    input_,target_ = set_up_placeholders(n_steps,n_inputs,n_neurons,n_outputs,learning_rate,batch_size)

    #Set up dataset
    data = Dataset("Dataset 1","bach","Description","22/5/2018")
    data.make_dataset(trainingSplit=0.9)
    data.save("dataset1.pickle")
    # data.load("dataset1.pickle")

    #Split input and targets. (I'll make a function to do that, also idk if this is the right way)
    train_x = data.train[0:-1]
    train_y = data.train[1:]
    test_x = data.test[0:-1]
    test_y = data.test[1:]

    #Setup model and train it
    prediction = rnn_model(input_,n_neurons,n_outputs)
    train(input_,target_,prediction,train_x, train_y, test_x, test_y,batch_size,num_epochs)

if __name__ == "__main__":
    main()

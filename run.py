#from lib.networks import *
from lib.musicalPiece import *
from lib.dataset import *
from lib.file_handle import *

def main():
    #
    # n_steps   = 28
    # n_inputs  = 127
    # n_neurons = 150
    # n_outputs = 10
    # learning_rate = 0.001
    # batch_size = 100
    #
    # input_,target_ = set_up_placeholders(n_steps,n_inputs,n_neurons,n_outputs,learning_rate,batch_size)


    data = Dataset("Dataset 1","bach","Description","22/5/2018")

    # data.make_dataset(trainingSplit=0.9)
    # data.train[1].play()
    # data.save("dataset1.pickle")
    # data.train[1].play()

    data.load("dataset1.pickle")
    print(data.train[1].filename)

    # prediction = rnn_model(input_,n_neurons,n_outputs)
    # train(input_,target,prediction,train_x, train_y, test_x, test_y,batch_size,num_epochs)


if __name__ == "__main__":
    main()

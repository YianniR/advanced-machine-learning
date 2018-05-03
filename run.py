from lib.networks import *
from lib.musicalPiece import *
from lib.dataset import *
from lib.file_handle import *
import argparse

def main(args):

    #Set up parameters (can put those in a config file and parse them.)
    batch_size = int(args.batch_size)
    n_steps   = int(args.num_steps)
    n_inputs  = 127
    n_neurons = 150
    n_outputs = 127
    learning_rate = 0.001
    n_epochs = int(args.num_epochs)

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
    seed = test_y[0:n_steps]

    print("Train Length:" + str(len(data.train)))
    print("Test Length :" + str(len(data.test)))

    # #Setup model and train it
    #prediction = multilayer_perceptron_model(input_,n_inputs,150,100,75,n_outputs)
    logits = rnn_model(input_,n_steps,n_outputs)
    train(input_,target_,logits,train_x, train_y, test_x, test_y,n_steps,batch_size,n_epochs)
    run(input_,target_,seed,logits,n_steps,batch_size)

if __name__ == "__main__":
    #Set up argument parser
	parser = argparse.ArgumentParser(description="Run an amazing neural network that makes music stuffs.")
	group = parser.add_mutually_exclusive_group()

	#Add parser arguments
	group.add_argument("-e", "--num_epochs", action="store",default = 10, nargs='?', help='Set num of epochs')
	group.add_argument("-b", "--batch_size", action="store",default = 200, nargs='?', help='Set batch size')
	group.add_argument("-s", "--num_steps", action="store", default = 30, nargs='?', help='Set num of steps')

	#Parse arguments and start main
	args = parser.parse_args()
	main(args)

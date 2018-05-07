from lib.networks import *
from lib.musicalPiece import *
from lib.dataset import *
from lib.file_handle import *
import argparse

def main(args):

    input_dir = args.input_dir

    #Set up parameters (can put those in a config file and parse them.)
    batch_size = int(args.batch_size)
    n_steps   = int(args.num_steps)
    n_inputs  = 127
    n_neurons = 150
    n_layers = 2
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

    print("Train Length:" + str(len(data.train)))
    print("Test Length :" + str(len(data.test)))

    # #Setup model and train it

    cell, outputs, states, logits = rnn_model(input_,n_steps,n_outputs)
    if not input_dir:
        train(input_,target_,logits,train_x, train_y, test_x, test_y,n_steps,batch_size,n_epochs)
    else:
        run(10,n_steps,input_dir)

if __name__ == "__main__":
    #Set up argument parser
	parser = argparse.ArgumentParser(description="Run an amazing neural network that makes music stuffs.")
	group = parser.add_mutually_exclusive_group()

	#Add parser arguments
	group.add_argument("-e", "--num_epochs", action="store",default = 100, nargs='?', help='Set num of epochs')
	group.add_argument("-b", "--batch_size", action="store",default = 500, nargs='?', help='Set batch size')
	group.add_argument("-s", "--num_steps", action="store", default = 25, nargs='?', help='Set num of steps')
	group.add_argument("-i", "--input_dir", action="store", default = "", nargs='?', help='Input dir to run')

	#Parse arguments and start main
	args = parser.parse_args()
	main(args)

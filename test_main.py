import argparse
from remove_space import remove_space
from seq2point_test import Tester
from matplotlib import pyplot as plt
import pandas as pd

# Allows a model to be tested from the terminal.

# You need to input your test data directory
test_directory="~/mingjun/research/housedata/refit/kettle/kettle_test_H2.csv"

parser = argparse.ArgumentParser(description="Train a pruned neural network for energy disaggregation. ")

parser.add_argument("--appliance_name", type=remove_space, default="kettle", help="The name of the appliance to perform disaggregation with. Default is kettle. Available are: kettle, fridge, dishwasher, microwave. ")
parser.add_argument("--batch_size", type=int, default="1000", help="The batch size to use when training the network. Default is 1000. ")
parser.add_argument("--crop", type=int, default="10000", help="The number of rows of the dataset to take training data from. Default is 10000. ")
parser.add_argument("--algorithm", type=remove_space, default="seq2point", help="The pruning algorithm of the model to test. Default is none. ")
parser.add_argument("--network_type", type=remove_space, default="", help="The seq2point architecture to use. Only use if you do not want to use the standard architecture. Available are: default, dropout, reduced, and reduced_dropout. ")
parser.add_argument("--input_window_length", type=int, default="599", help="Number of input data points to network. Default is 599. ")
parser.add_argument("--test_directory", type=str, default=test_directory, help="The dir for training data. ")

arguments = parser.parse_args()

# You need to provide the trained model
saved_model_dir = "saved_models/" + arguments.appliance_name + "_" + arguments.algorithm + "_model.h5"

# The logs including results will be recorded to this log file
log_file_dir = "saved_models/" + arguments.appliance_name + "_" + arguments.algorithm + "_" + arguments.network_type + ".log"


def do1(name_app):
    """
    docstring
    """
    file_directory = "./__data__/exm" + '/'.join([r'j', name_app, name_app])

    args = {
        'appliance': name_app, 
        'algorithm': 'seq2point',
        'crop': 1267000,
        'batch_size': 130, 
        'network_type': "",
        'test_directory': file_directory + "_test_S(14, 15, 16).csv",
        'saved_model_dir': file_directory + "_" + 'seq2point' + "_model.h5",
        'log_file_dir': file_directory + "_" + 'seq2point' + "_.log", 
        'input_window_length': 599,
    }
    tester = Tester(**args)

    # tester = Tester(arguments.appliance_name, arguments.algorithm, arguments.crop, 
    #                 arguments.batch_size, arguments.network_type,
    #                 arguments.test_directory, saved_model_dir, log_file_dir,
    #                 arguments.input_window_length
    #                 )
    get = tester.test_model()
    return get

if __name__ == "__main__":
    
    for app in ('Fridge', ):
        history, target = do1(app)
        # plt.plot(target)
        # plt.plot(history)
        # plt.show()
        x = pd.DataFrame(history)
        x.to_csv('h_Fridge_h.csv', header = 0, index=0)
        y = pd.DataFrame(target)
        y.to_csv('h_Fridge_t.csv', header = 0, index=0)
        print(f'{type(history)}')
        print('\t ' + f'{x.size=}')
        print('\t ' + str(x.size/1266760))

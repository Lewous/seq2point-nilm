import argparse
from remove_space import remove_space
from seq2point_train import Trainer
# Allows a model to be trained from the terminal.

training_directory="~/mingjun/research/housedata/refit/kettle/kettle_training_.csv"
validation_directory="~/mingjun/research/housedata/refit/kettle/kettle_validation_.csv"

parser = argparse.ArgumentParser(description="Train sequence-to-point learning for energy disaggregation. ")

parser.add_argument("--appliance_name", type=remove_space, default="kettle", help="The name of the appliance to train the network with. Default is kettle. Available are: kettle, fridge, washing machine, dishwasher, and microwave. ")
parser.add_argument("--batch_size", type=int, default="1000", help="The batch size to use when training the network. Default is 1000. ")
parser.add_argument("--crop", type=int, default="1000", help="The number of rows of the dataset to take training data from. Default is 10000. ")
#parser.add_argument("--pruning_algorithm", type=remove_space, default="default", help="The pruning algorithm that the network will train with. Default is none. Available are: spp, entropic, threshold. ")
parser.add_argument("--network_type", type=remove_space, default="seq2point", help="The seq2point architecture to use. ")
parser.add_argument("--epochs", type=int, default="2", help="Number of epochs. Default is 10. ")
parser.add_argument("--input_window_length", type=int, default="599", help="Number of input data points to network. Default is 599.")
parser.add_argument("--validation_frequency", type=int, default="1", help="How often to validate model. Default is 1. ")
parser.add_argument("--training_directory", type=str, default=training_directory, help="The dir for training data. ")
parser.add_argument("--validation_directory", type=str, default=validation_directory, help="The dir for validation data. ")

arguments = parser.parse_args()

# Need to provide the trained model
save_model_dir = "saved_models/" + arguments.appliance_name + "_" + arguments.network_type + "_model.h5"


def do1(name_app):
    """
    docstring
    """

    # global n_app
    global arguments

    # file_directory = "./__data__/exm" + '/'.join([str(ind), n_app, n_app])
    file_directory = "./__data__/exm" + '/'.join([r'j', name_app, name_app])

    args = {
        'appliance': name_app, 
        'batch_size': 1000, 
        'crop': 100000, 
        'network_type': 'seq2point', 
        'training_directory': file_directory + "_training_.csv",
        'validation_directory': file_directory + "_valid_S(9,).csv",
        'save_model_dir': file_directory + "_" + 'seq2point' + "_model.h5",
        'epochs': 10, 
        'imput_window_length': 599, 

    }

    trainer = Trainer(**args)
    get = trainer.train_model()

    return get

if __name__ == "__main__":
    
    print(do1('Fridge'))
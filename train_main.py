import argparse
from remove_space import remove_space
from seq2point_train import Trainer
# Allows a model to be trained from the terminal.

n_app = 'freezer'

parser = argparse.ArgumentParser(description="Train sequence-to-point learning for energy disaggregation. ")

parser.add_argument("--appliance_name", type=remove_space, default=n_app,
                    help="The name of the appliance to train the network with. Default is kettle. Available are: kettle, fridge, washing machine, dishwasher, and microwave. ")
parser.add_argument("--batch_size", type=int, default="1000",
                    help="The batch size to use when training the network. Default is 1000. ")
parser.add_argument("--crop", type=int, default="1000",
                    help="The number of rows of the dataset to take training data from. Default is 10000. ")
parser.add_argument("--network_type", type=remove_space, default="seq2point",
                    help="The seq2point architecture to use. ")
parser.add_argument("--epochs", type=int, default="2",
                    help="Number of epochs. Default is 10. ")
parser.add_argument("--input_window_length", type=int, default="599",
                    help="Number of input data points to network. Default is 599.")
parser.add_argument("--validation_frequency", type=int, default="1",
                    help="How often to validate model. Default is 1. ")

arguments = parser.parse_args()


def do1(ind):
    """
    docstring
    """

    global n_app
    global arguments

    file_directory = "./__data__/exm" + '/'.join([str(ind), n_app, n_app])
    training_directory = file_directory + "_training_.csv"
    validation_directory = file_directory + "_valid_.csv"

    # Need to provide the trained model
    save_model_dir = file_directory + "_" + arguments.network_type + "_model.h5"

    trainer = Trainer(n_app, arguments.batch_size, arguments.crop, arguments.network_type,
                      training_directory, validation_directory, save_model_dir,
                    epochs = arguments.epochs, input_window_length = arguments.input_window_length,
                    validation_frequency = arguments.validation_frequency)
    trainer.train_model()

    return 0


if __name__ == "__main__":

    for ind in (0, ):
        do1(ind)

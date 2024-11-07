__author__ = "Suraj Kumar"
__copyright__ = "Suraj Kumar 2024"
__version__ = "1.0.0"
__license__ = "MIT"

import os
import json
import time
import math
import matplotlib.pyplot as plt
from core.data_processor import DataLoader
from core.model import Model

def plot_results(predicted_data, true_data):
    """Function to plot predicted data against true data."""
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()

def plot_results_multiple(predicted_data, true_data, prediction_len):
    """Function to plot multiple predicted sequences on the same graph."""
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    for i, data in enumerate(predicted_data):
        # Pad the list of predictions to shift it correctly in the graph
        padding = [None for _ in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()

def main():
    """Main function to load data, train the model, and plot predictions."""
    # Load configuration from the config.json file
    configs = json.load(open('config.json', 'r'))
    
    # Ensure the directory for saving models exists
    if not os.path.exists(configs['model']['save_dir']):
        os.makedirs(configs['model']['save_dir'])
    
    # Load the data using the DataLoader class
    data = DataLoader(
        os.path.join('data', configs['data']['filename']),
        configs['data']['train_test_split'],
        configs['data']['columns']
    )

    # Initialize the model and build it using the configurations
    model = Model()
    model.build_model(configs)
    
    # Get training data
    x, y = data.get_train_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise']
    )

    # Train the model using out-of-memory generative training
    steps_per_epoch = math.ceil((data.len_train - configs['data']['sequence_length']) / configs['training']['batch_size'])
    model.train_generator(
        data_gen=data.generate_train_batch(
            seq_len=configs['data']['sequence_length'],
            batch_size=configs['training']['batch_size'],
            normalise=configs['data']['normalise']
        ),
        epochs=configs['training']['epochs'],
        batch_size=configs['training']['batch_size'],
        steps_per_epoch=steps_per_epoch,
        save_dir=configs['model']['save_dir']
    )

    # Get test data
    x_test, y_test = data.get_test_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise']
    )

    # Generate predictions for multiple steps
    predictions = model.predict_sequences_multiple(x_test, configs['data']['sequence_length'], configs['data']['sequence_length'])

    # Plot results: Predicted data vs True data
    plot_results_multiple(predictions, y_test, configs['data']['sequence_length'])

if __name__ == '__main__':
    main()

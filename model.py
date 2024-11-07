import os
import math
import numpy as np
import datetime as dt
from numpy import newaxis
from core.utils import Timer
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

class Model():
    """A class for building and inference with an LSTM model"""

    def __init__(self):
        self.model = Sequential()

    def load_model(self, filepath):
        """Load the model from a specified file."""
        print(f'[Model] Loading model from file {filepath}')
        self.model = load_model(filepath)

    def build_model(self, configs):
        """Build the LSTM model based on the given configurations."""
        timer = Timer()
        timer.start()

        for layer in configs['model']['layers']:
            neurons = layer.get('neurons', None)
            dropout_rate = layer.get('rate', None)
            activation = layer.get('activation', None)
            return_seq = layer.get('return_seq', None)
            input_timesteps = layer.get('input_timesteps', None)
            input_dim = layer.get('input_dim', None)

            if layer['type'] == 'dense':
                self.model.add(Dense(neurons, activation=activation))
            elif layer['type'] == 'lstm':
                self.model.add(LSTM(neurons, input_shape=(input_timesteps, input_dim), return_sequences=return_seq))
            elif layer['type'] == 'dropout':
                self.model.add(Dropout(dropout_rate))

        self.model.compile(loss=configs['model']['loss'], optimizer=configs['model']['optimizer'])

        print('[Model] Model Compiled')
        timer.stop()

    def train(self, x, y, epochs, batch_size, save_dir):
        """Train the model with the given data and configuration."""
        timer = Timer()
        timer.start()
        print(f'[Model] Training Started')
        print(f'[Model] {epochs} epochs, {batch_size} batch size')

        save_fname = os.path.join(save_dir, f'{dt.datetime.now().strftime("%d%m%Y-%H%M%S")}-e{epochs}.h5')
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=2),
            ModelCheckpoint(filepath=save_fname, monitor='val_loss', save_best_only=True)
        ]
        self.model.fit(
            x,
            y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,  # Added validation split for early stopping
            callbacks=callbacks
        )

        print(f'[Model] Training Completed. Model saved as {save_fname}')
        timer.stop()

    def train_generator(self, data_gen, epochs, batch_size, steps_per_epoch, save_dir):
        """Train the model using a data generator."""
        timer = Timer()
        timer.start()
        print(f'[Model] Training Started')
        print(f'[Model] {epochs} epochs, {batch_size} batch size, {steps_per_epoch} batches per epoch')

        save_fname = os.path.join(save_dir, f'{dt.datetime.now().strftime("%d%m%Y-%H%M%S")}-e{epochs}.h5')
        callbacks = [
            ModelCheckpoint(filepath=save_fname, monitor='loss', save_best_only=True)
        ]

        self.model.fit(
            data_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            callbacks=callbacks,
            workers=1
        )

        print(f'[Model] Training Completed. Model saved as {save_fname}')
        timer.stop()

    def predict_point_by_point(self, data):
        """Predict one step ahead each time for each timestep."""
        print('[Model] Predicting Point-by-Point...')
        predicted = self.model.predict(data)
        return np.reshape(predicted, (predicted.size,))

    def predict_sequences_multiple(self, data, window_size, prediction_len):
        """Predict sequences of multiple timesteps."""
        print('[Model] Predicting Sequences Multiple...')
        prediction_seqs = []
        for i in range(int(len(data)/prediction_len)):
            curr_frame = data[i*prediction_len]
            predicted = []
            for j in range(prediction_len):
                predicted.append(self.model.predict(curr_frame[newaxis,:,:])[0,0])
                curr_frame = curr_frame[1:]
                curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
            prediction_seqs.append(predicted)
        return prediction_seqs

    def predict_sequence_full(self, data, window_size):
        """Predict sequences one step at a time, updating the window."""
        print('[Model] Predicting Sequences Full...')
        curr_frame = data[0]
        predicted = []
        for i in range(len(data)):
            predicted.append(self.model.predict(curr_frame[newaxis,:,:])[0,0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
        return predicted

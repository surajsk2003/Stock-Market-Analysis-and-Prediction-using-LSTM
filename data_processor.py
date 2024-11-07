import math
import numpy as np
import pandas as pd

class DataLoader():
    """A class for loading and transforming data for the LSTM model"""

    def __init__(self, filename, split, cols):
        """
        Initialize the DataLoader class with the CSV file, split ratio, and columns to load.

        Arguments:
        filename : str : Path to the CSV file containing data.
        split : float : Ratio for splitting data into training and testing sets (e.g., 0.8 for 80% train).
        cols : list : List of columns to be used for the LSTM model.
        """
        dataframe = pd.read_csv(filename)
        i_split = int(len(dataframe) * split)
        self.data_train = dataframe.get(cols).values[:i_split]
        self.data_test  = dataframe.get(cols).values[i_split:]
        self.len_train  = len(self.data_train)
        self.len_test   = len(self.data_test)
        self.len_train_windows = None

    def get_test_data(self, seq_len, normalise):
        """
        Create x, y test data windows.

        Arguments:
        seq_len : int : The length of the sequence to consider for each prediction.
        normalise : bool : Whether to normalize the data or not.

        Returns:
        x, y : ndarray : Input features and labels for testing.
        """
        data_windows = []
        for i in range(self.len_test - seq_len):
            data_windows.append(self.data_test[i:i+seq_len])

        data_windows = np.array(data_windows).astype(float)
        data_windows = self.normalise_windows(data_windows, single_window=False) if normalise else data_windows

        x = data_windows[:, :-1]
        y = data_windows[:, -1, [0]]
        return x, y

    def get_train_data(self, seq_len, normalise):
        """
        Create x, y train data windows.

        Arguments:
        seq_len : int : The length of the sequence to consider for each prediction.
        normalise : bool : Whether to normalize the data or not.

        Returns:
        x, y : ndarray : Input features and labels for training.
        """
        data_x = []
        data_y = []
        for i in range(self.len_train - seq_len):
            x, y = self._next_window(i, seq_len, normalise)
            data_x.append(x)
            data_y.append(y)
        return np.array(data_x), np.array(data_y)

    def generate_train_batch(self, seq_len, batch_size, normalise):
        """
        Yield a generator of training data in batches.

        Arguments:
        seq_len : int : The length of the sequence to consider for each prediction.
        batch_size : int : The number of samples in each batch.
        normalise : bool : Whether to normalize the data or not.

        Yields:
        x_batch, y_batch : ndarray : Batches of input features and labels for training.
        """
        i = 0
        while i < (self.len_train - seq_len):
            x_batch = []
            y_batch = []
            for b in range(batch_size):
                if i >= (self.len_train - seq_len):
                    # stop-condition for a smaller final batch if data doesn't divide evenly
                    yield np.array(x_batch), np.array(y_batch)
                    i = 0
                x, y = self._next_window(i, seq_len, normalise)
                x_batch.append(x)
                y_batch.append(y)
                i += 1
            yield np.array(x_batch), np.array(y_batch)

    def _next_window(self, i, seq_len, normalise):
        """
        Generates the next data window from the given index location.

        Arguments:
        i : int : The index at which to start the window.
        seq_len : int : The length of the sequence to consider.
        normalise : bool : Whether to normalize the data or not.

        Returns:
        x, y : ndarray : Input features and labels for the next window.
        """
        window = self.data_train[i:i+seq_len]
        window = self.normalise_windows(window, single_window=True)[0] if normalise else window
        x = window[:-1]
        y = window[-1, [0]]
        return x, y

    def normalise_windows(self, window_data, single_window=False):
        """
        Normalizes a window of data with a base value of zero.

        Arguments:
        window_data : ndarray : The data window to normalize.
        single_window : bool : Whether to normalize a single window or multiple windows.

        Returns:
        normalised_data : ndarray : Normalized data windows.
        """
        normalised_data = []
        window_data = [window_data] if single_window else window_data
        for window in window_data:
            normalised_window = []
            for col_i in range(window.shape[1]):
                normalised_col = [((float(p) / float(window[0, col_i])) - 1) for p in window[:, col_i]]
                normalised_window.append(normalised_col)
            normalised_window = np.array(normalised_window).T  # Reshape and transpose array back into original format
            normalised_data.append(normalised_window)
        return np.array(normalised_data)

import torch
import numpy as np

class TimeSeriesProcessor:
    def __init__(self, input_size=10, steps_ahead=1):
        """
        Initializes the processor with the `input_size` number of lags and prediction horizon.
        """
        self.input_size = input_size
        self.steps_ahead = steps_ahead

    def prepare_data(self, series):
        """
        Prepares lagged inputs and targets for supervised learning.
        """
        X = torch.stack([series[i:-(self.input_size - i)] for i in range(self.input_size)], dim=1)
        y = torch.stack(
            [series[i + self.input_size:i + self.input_size + self.steps_ahead] for i in range(len(series) - self.input_size - self.steps_ahead + 1)],
            dim=0
        )
        return X[:len(y)], y

    def reserve_out_of_sample(self, series, steps):
        """
        Splits the series into a training set and an out-of-sample test set.
        """
        out_of_sample_series = series[-steps:]
        training_series = series[:-steps]
        return training_series, out_of_sample_series

    def prepare_forecast_input(self, series):
        """
        Prepares the input for forecasting based on the last `input_size` values of a series.
        """
        return series[-self.input_size:].view(1, -1).float()

    def calculate_error(self, predicted, actual):
        """
        Calculates the Mean Squared Error (MSE) between predicted and actual values.
        """
        return np.mean((predicted - actual) ** 2)

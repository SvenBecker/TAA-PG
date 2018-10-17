import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def get_flatten(array):
    tri = []
    top = array[:-array.shape[1]]
    z = array.shape[1] - 1
    for i in range(array.shape[1] - 1):
        tri.append(array[-(1 + i)][:z])
        z -= 1
    tri = np.concatenate(tri, axis=0)
    return np.concatenate((top.flatten(), tri.flatten()))


class PrepData:
    def __init__(
            self,
            horizon=10,
            window_size=100,
            split=0.85,
            nb_assets=10,
            log_return=True
    ):

        self.horizon = horizon
        self.nb_assets = nb_assets
        self.window_size = window_size
        self.split = split
        self.log_return = log_return

    def get_mean_return(self, assets):
        # the asset returns have to be on the first x columns
        # returns the mean of the future (horizon) returns
        return np.array([np.mean(assets[i:i + self.horizon], axis=0)
                         for i in range(len(assets) - self.horizon)])

    def get_log_return(self, assets):
        # returns the cumulative log return after horizon time steps
        log_returns = np.log(assets + 1)
        return np.array([np.sum(log_returns[i:i + self.horizon], axis=0)
                         for i in range(len(assets) - self.horizon)])

    def get_data(self, file):
        data = pd.read_csv(file, sep=r',|;', engine='python')
        data = np.array(data)[:, 1:]
        return np.array(data, dtype='float64')

    def get_scaler(self, train, standardize=False, feature_range=(0, 1)):
        if standardize:
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler(feature_range=feature_range)
        return scaler.fit(train)

    def reshape_data(self, x, y, scaler=None):
        y = y[1:]
        x = scaler.transform(x[: -(1 + self.horizon)])
        xdata, ydata = [], []
        for step in range(x.shape[0] - self.window_size):
            xdata.append(x[0 + step: self.window_size + step])
            ydata.append(y[self.window_size + step: self.window_size + step + 1])
        xdata = np.array(xdata)
        ydata = np.array(ydata).reshape((x.shape[0] - self.window_size, y.shape[1]))
        return xdata, ydata

    def split_data(self, x, y):
        train_x, train_y = x[: int(x.shape[0] * self.split)], y[: int(x.shape[0] * self.split)]
        test_x, test_y = x[len(train_x):], y[len(train_y):]
        return [train_x, train_y, test_x, test_y]

    def run_default(self, file):
        # run_default is specified for the prediction.py file
        data = self.get_data(file)
        if self.log_return:
            label = self.get_log_return(data[:, 0:self.nb_assets])
        else:
            label = self.get_mean_return(data[:, 0:self.nb_assets])
        train = data[0:int(self.split * label.shape[0]), :]
        scaler = self.get_scaler(train)
        x, y = self.reshape_data(data, label, scaler=scaler)
        return self.split_data(x, y), scaler


if __name__ == '__main__':
    # just for testing
    d = PrepData(window_size=100, split=0.9, horizon=10)
    data, scaler = d.run_default('../env/data/environment.csv')
    print(data[1])
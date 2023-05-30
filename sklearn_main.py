
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, Matern, RBF
from tqdm import tqdm
from data_processor import DataPreprocessor, DataLoaderGtorch
from utils import plot_result
import numpy as np

if __name__ == '__main__':
    file_path, file_name = './data/', 'BA'
    preprocessed_data = DataPreprocessor(file_path, file_name, '2008-Q3', '2016-Q4')
    dataloader = DataLoaderGtorch(preprocessed_data, 'Open', 0.8, "numpy")
    train_x, train_y, test_x, test_y = dataloader.retrieve_data()

    # concate train_x and test_x
    x = np.concatenate((train_x, test_x), axis=0)
    y = np.concatenate((train_y, test_y), axis=0)

    # array.reshape
    train_x = train_x.reshape(-1, 1)
    train_y = train_y.reshape(-1, 1)
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)

    # kernel = 1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0), nu=1.5)
    kernel = DotProduct()
    gpr = GaussianProcessRegressor(
        kernel = kernel, 
        random_state=0,
        alpha = 0.1
        ).fit(train_x, train_y)
    print(gpr.score(train_x, train_y))

    # also calculate the confidence interval
    pred_y, y_std = gpr.predict(x, return_std=True)
    # calculate the confidence bound
    lower_bound = pred_y - y_std
    upper_bound = pred_y + y_std

    plot_result(x, y, pred_y, lower_bound, upper_bound, len(train_x))


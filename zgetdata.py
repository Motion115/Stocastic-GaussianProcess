import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

class StockDataset(Dataset):
    def __init__(self, file_path, file_name, window_size, dataset_type, split_ratio=0.8):
        file = pd.read_csv(os.path.join(file_path, file_name))
        self.target = file['Close'].values
        self.date = file['Date'].values
        # drop Close and Date for file, create data
        # self.data = file.drop(columns=['Close', 'Date']).values
        # generate a list of length from 0 to len(file) - 1
        self.data = [i for i in range(len(file))]
        self.window_size = window_size
        
        split_point = int(self.__len__() * split_ratio)
        if dataset_type == 'train':
            self.data = self.data[10000:split_point]
            self.target = self.target[10000:split_point]
        elif dataset_type == 'test':
            self.data = self.data[split_point:]
            self.target = self.target[split_point:]
        else:
            raise ValueError('dataset_type should be either train or test')


    def __len__(self) -> int:
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        #x = self.data[idx: idx + self.window_size]
        x = self.data[idx]
        x = np.array(x, dtype=np.float32)
        y = self.target[idx+self.window_size]
        y = np.array(y, dtype=np.float32)
        # x, y to tensor
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        return x, y

def get_tensor_dataset(file_path, file_name, window_size, batch_size):
    train_set = StockDataset(file_path, file_name, window_size, "train")
    test_set = StockDataset(file_path, file_name, window_size, "test")
    X_train = torch.stack([train_set[i][0] for i in range(train_set.__len__())])
    y_train = torch.stack([train_set[i][1] for i in range(train_set.__len__())])
    X_test = torch.stack([test_set[i][0] for i in range(test_set.__len__())])
    y_test = torch.stack([test_set[i][1] for i in range(test_set.__len__())])
    return X_train, y_train, X_test, y_test
    

if __name__ == "__main__":
    file_path = "./data/"
    file_name = "BA.csv"
    get_tensor_dataset(file_path, file_name, 10, 1)
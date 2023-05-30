import pandas as pd
import numpy as np
import torch

class DataPreprocessor:
    '''
    The smallest grain for data retrival should be in quarters, 
    i.e. we can start from any quarter of any year, to start a standardized stock prediction process.

    Parameters: 
        - file_path: relative or absolute path
        - file_name: must be a csv file
        - start_quarter: <year>-Q<quarter>, e.g. 2018-Q3
        - end_quarter: <year>-Q<quarter>, e.g. 2018-Q3, this quarter will be included
        - mode: 'centerize' or 'absolute'
    '''
    def __init__(self, file_path: str, file_name: str, report = False):
        data, trade_day, start_year, end_year = self.read_raw_data(file_path, file_name, report)
        self.data = data
        self.trade_day = trade_day
        self.start_year = start_year
        self.end_year = end_year

    def read_raw_data(self, file_path, file_name, report):
        raw_data = pd.read_csv(file_path + file_name + '.csv')
        # the columns for raw_data are ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        # Task1: the 'Date' column's storage is not aligned with 3NF of database
        # split it to 3 columns of year, month, day
        raw_data['Year'] = pd.DatetimeIndex(raw_data['Date']).year
        raw_data['Month'] = pd.DatetimeIndex(raw_data['Date']).month
        raw_data['Day'] = pd.DatetimeIndex(raw_data['Date']).day
        # delete the 'Date' column
        raw_data = raw_data.drop(['Date'], axis=1)

        # Task2: for easier querying, we need to map month to quarters, using pandas "map" function
        # create a dictionary for mapping
        month_to_quarter = {1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2,
                            7: 3, 8: 3, 9: 3, 10: 4, 11: 4, 12: 4}
        # map the 'Month' column to 'Quarter' column
        raw_data['Quarter'] = raw_data['Month'].map(month_to_quarter)
        
        # Task3: theoretically, for each quarter, we have 63 trading days.
        # However, we are not sure about the exact number, so we need to find
        # the smallest number of trading days for each quarter from the entire dataset
        # and use this number as the standard for all quarters
        # create a dictionary for storing the smallest number of trading days for each quarter
        min_trading_day = 100
        # Note that we don't consider the last quarter of the last year
        for year in range(raw_data['Year'].min(), raw_data['Year'].max()):
            for quarter in range(1, 5):
                # get the number of trading days for each quarter
                trading_days = raw_data.loc[(raw_data['Year'] == year) & (raw_data['Quarter'] == quarter)].shape[0]
                # update the smallest number of trading days
                if trading_days < min_trading_day:
                    min_trading_day = trading_days
        print("The smallest number of trading days for each quarter is: ", min_trading_day)
        # Then we can truncate each quarter to the smallest number of trading days
        # create a new dataframe for storing the truncated data
        truncated_data = pd.DataFrame(columns=['Year', 'Quarter', 'Month', 'Day', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])
        # truncate each quarter to the smallest number of trading days
        for year in range(raw_data['Year'].min(), raw_data['Year'].max()):
            for quarter in range(1, 5):
                # to aviod breaking the contigious property of data, we first randomly choose the index that we want
                random_index = np.random.choice(raw_data.loc[(raw_data['Year'] == year) & (raw_data['Quarter'] == quarter)].index, min_trading_day, replace=False)
                # sort the index
                random_index.sort()
                # get the truncated data by random_index
                truncated_data = pd.concat([truncated_data, raw_data.loc[random_index]])
        
        '''
        From here on, the data is in clean slate, with the following properties:
        1. The 'Date' column is split into 'Year', 'Month', 'Day' columns
        2. The 'Month' column is mapped to 'Quarter' column
        3. Each quarter has the same number of trading days
        4. The data is sorted by 'Year', 'Quarter', 'Month', 'Day' columns
        5. Every year is a full year
        '''
        # give a report on the preprocessed data
        if report:
            print("--- Data preorocess report ---")
            print("Trading days for each quarter is:", min_trading_day)
            print("Original data items: {}, Truncated data items:{}".format(raw_data.shape[0],truncated_data.shape[0]))
            print("------------------------------")

        # check if the year is continuous
        for year in range(truncated_data['Year'].min(), truncated_data['Year'].max()):
            if year not in truncated_data['Year'].unique():
                # raise an error
                raise ValueError("Year {} is not in the data".format(year))

        # return the truncated data, min_trading_day, start_year, end_year
        return truncated_data, min_trading_day, truncated_data['Year'].min(), truncated_data['Year'].max()


class DataLoaderGtorch:
    '''
    Parameters:
        - preprocessor: the data from class DataPreprocessor
        - predictive_target: the target that we want to predict
        - test_cases: the ratio of training data to testing data (ratio is training data percentage)
        - target_dtype: the data type of the target, 'torch' or 'numpy'
    
    Attributes:
        - X_train: the training data
        - y_train: the training target
        - X_test: the testing data
        - y_test: the testing target
        - date_train: the date of training data (use for visualization)
        - date_test: the date of testing data (use for visualization)
    '''
    def __init__(self, preprocessor, start_quarter: str, end_quarter: str, predictive_target: str, mode: str = 'centerize', test_cases: int = 5, target_dtype: str = 'torch', report = False):
        self.preprocessor = preprocessor
        self.data = self.preprocessor.data
        self.normalized_data = self.get_data(start_quarter, end_quarter, mode)
        self.duration = len(self.normalized_data)
        def to_torch(data):
            return torch.from_numpy(data).float()
        X_train, y_train, X_test, y_test, date_train, date_test = self.get_column_traintest_split(predictive_target, test_cases)
        if target_dtype == 'torch':
            self.X_train = to_torch(X_train)
            self.y_train = to_torch(y_train)
            self.X_test = to_torch(X_test)
            self.y_test = to_torch(y_test)
        else:
            self.X_train = X_train
            self.y_train = y_train
            self.X_test = X_test
            self.y_test = y_test
        self.date_train = date_train
        self.date_test = date_test
        if report:
            # print dataloader report
            print("--- Dataloader report ---")
            print("Time series data type is: ", target_dtype)
            print("Training data shape: ", self.X_train.shape)
            print("Testing data shape: ", self.X_test.shape)
            print("-------------------------")
    
    def get_data(self, start, end, mode):
        data_portion = self.get_data_portion(start, end)
        if mode == 'centerize':
            returned_data = self.centerize_data(data_portion)
        elif mode == 'absolute':
            returned_data = data_portion
        else:
            raise ValueError('mode can only be centerize or absolute')
        return returned_data
    
    def centerize_data(self, data_portion):
        # transform data_portion's each column
        # we only do the transformation on part of the columns, i.e. 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'
        for col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
            # get the benchmark value, which is the mean of the data in the columns
            benchmark = data_portion[col].mean()
            # use map function to apply the transformation
            data_portion[col] = data_portion[col].map(lambda x: (x - benchmark) / benchmark)
        return data_portion

    # input should be like 2018-Q3
    def get_data_portion(self, start:str, end:str):
        def check_input(in_str: str):
            # parse start and end, with format of year(4 digits) + quarter(1 digit)
            year = int(in_str[:4])
            padding = in_str[4:6]
            quarter = int(in_str[6:])
            if padding != '-Q' or quarter not in [1, 2, 3, 4] or year not in range(self.preprocessor.start_year, self.preprocessor.end_year + 1):
                raise ValueError('Input format error, please check your input.')
            else:
                return year, quarter
        user_start_year, user_start_quarter = check_input(start)
        user_end_year, user_end_quarter = check_input(end)
        def check_effectiveness(start_year, start_quarter, end_year, end_quarter):
            if start_year > end_year:
                raise ValueError('Start year should be earlier than end year.')
            elif start_year == end_year:
                if start_quarter > end_quarter:
                    raise ValueError('Start quarter should be earlier than end quarter.')
                else:
                    return True
            else:
                return True
        if check_effectiveness(user_start_year, user_start_quarter, user_end_year, user_end_quarter):
            # get the index of the first entry that matches the start year and quarter
            start_index = self.data[(self.data['Year'] == user_start_year) & (self.data['Quarter'] == user_start_quarter)].index.tolist()[0]
            # get the index of the last entry that matches the end year and quarter
            end_index = self.data[(self.data['Year'] == user_end_year) & (self.data['Quarter'] == user_end_quarter)].index.tolist()[-1]
            # get a copy of the data between start_index and end_index
            data_portion = self.data.loc[start_index:end_index]
            # refresh index
            data_portion = data_portion.reset_index(drop=True)
            return data_portion
        else:
            raise ValueError('Input does not meet common sense!')
    
    def retrieve_data(self):
        return self.X_train, self.y_train, self.X_test, self.y_test

    
    def get_column_traintest_split(self, predictive_target, overlook_days):
        # check if the predictive_target is a column name in the data
        if predictive_target not in self.normalized_data.columns[4:]:
            raise ValueError("The predictive_target is not in the data!")
        data = self.normalized_data
        # get time_variable and y respectively
        # time_variable is the index of the data, which is a list
        time_vairable = data.index.values
        # y is the predictive_target
        y = data[predictive_target]

        # concatenate year, month, day to a string
        actual_dates = []
        for year, month, day in zip(data['Year'], data['Month'], data['Day']):
            actual_dates.append(str(year) + '-' + str(month) + '-' + str(day))
        
        # get split
        split_pos = int(len(time_vairable) - overlook_days)
        # get train and test
        train_time_variable = np.array(time_vairable[:split_pos])
        train_y = np.array(y[:split_pos])
        test_time_variable = np.array(time_vairable[split_pos:])
        test_y = np.array(y[split_pos:])
        # also split the dates
        train_actual_dates = actual_dates[:split_pos]
        test_actual_dates = actual_dates[split_pos:]
        # return the values
        return train_time_variable, train_y, test_time_variable, test_y, train_actual_dates, test_actual_dates
        
if __name__ == '__main__':
    file_path = './data/'
    file_name = 'BA'
    preprocessed_data = DataPreprocessor(file_path, file_name)
    dataloader = DataLoaderGtorch(preprocessed_data,'2012-Q4', '2012-Q4', 'Close', report=True)

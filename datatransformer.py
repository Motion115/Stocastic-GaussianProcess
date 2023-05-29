import pandas as pd
import numpy as np

class DataTransformer:
    '''
    The objective of DataTransformer is to read raw data from csv file,
    and transform it into a dictionary based, fixed length, year-wise data.

    The smallest grain for data retrival should be in quarters, 
    i.e. we can start from any quarter of any year, to start a standardized stock prediction process.
    '''
    def __init__(self, file_path: str, file_name: str):
        data, trade_day, start_year, end_year = self.read_raw_data(file_path, file_name)
        self.data = data
        self.trade_day = trade_day
        self.start_year = start_year
        self.end_year = end_year
    
    def get_data(self, start, end):
        data_portion = self.get_data_portion(start, end)
        # transform data_portion's each column
        self.to_varience_data(data_portion)
    
    def to_varience_data(self, data_portion):
        # transform data_portion's each column
        # use the first entry as the base, and calculate the varience of each entry
        # we only do the transformation on part of the columns, i.e. 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'
        benchmark = data_portion.iloc[0]
        for col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
            # use map function to apply the transformation
            data_portion[col] = data_portion[col].map(lambda x: x - benchmark[col])
        print(data_portion)
        return data_portion

    # input should be like 2018-Q3
    def get_data_portion(self, start:str, end:str):
        def check_input(in_str: str):
            # parse start and end, with format of year(4 digits) + quarter(1 digit)
            year = int(in_str[:4])
            padding = in_str[4:6]
            quarter = int(in_str[6:])
            if padding != '-Q' or quarter not in [1, 2, 3, 4] or year not in range(self.start_year, self.end_year + 1):
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
            return data_portion
        else:
            raise ValueError('Input does not meet common sense!')

    def read_raw_data(self, file_path, file_name):
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

if __name__ == '__main__':
    file_path = './data/'
    file_name = 'BA'
    data_transformer = DataTransformer(file_path, file_name)
    data_transformer.get_data('2010-Q3', '2012-Q4')

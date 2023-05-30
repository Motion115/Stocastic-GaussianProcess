from gp_operations import train_eval
from data_processor import DataPreprocessor, DataLoaderGtorch
import pandas as pd
from tqdm import tqdm

if __name__ == '__main__':
    file_path, file_name = './data/', 'BA'
    preprocessed_data = DataPreprocessor(file_path, file_name)
    # get preprocessed_data start and end year
    start_year = preprocessed_data.start_year
    end_year = preprocessed_data.end_year
    print(start_year, end_year)
    output = []
    for year in tqdm(range(start_year, end_year + 1)):
        for quarter_id in range(1, 5):
            quarter = str(year) + "-Q" + str(quarter_id)
            dataloader = DataLoaderGtorch(preprocessed_data, quarter, quarter, 'Close', test_cases=3)
            mae, mse, r2 = train_eval(dataloader, file_name, quarter, iters=200)
            output.append([year, quarter_id, mae, mse, r2])
    
    # use pandas to save the output, add column names
    output = pd.DataFrame(output, columns=['year', 'quarter', 'mae', 'mse', 'r2'])
    output.to_csv('./result/' + file_name + '-close-RQ3.csv', index=False)


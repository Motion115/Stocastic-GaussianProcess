from gp_operations import train_eval
from data_processor import DataPreprocessor, DataLoaderGtorch
import pandas as pd
from tqdm import tqdm

# combined: 48s
# linear: 26s
# rq: 44s
# rbf: 37s
# matern: 35s

if __name__ == '__main__':
    file_path, file_name = './data/', 'BA'
    kernel_option = 'Matern'
    preprocessed_data = DataPreprocessor(file_path, file_name)
    # get preprocessed_data start and end year
    start_year = preprocessed_data.start_year
    end_year = preprocessed_data.end_year
    start_year = 2005
    output = []
    for year in tqdm(range(start_year, end_year + 1)):
        for quarter_id in range(1, 5):
            quarter = str(year) + "-Q" + str(quarter_id)
            dataloader = DataLoaderGtorch(preprocessed_data, quarter, quarter, 'Adj Close', test_cases=5)
            mae, mse, r2 = train_eval(dataloader, file_name, quarter, kernel_option=kernel_option, iters=100)
            output.append([year, quarter_id, mae, mse, r2])
    
    # use pandas to save the output, add column names
    output = pd.DataFrame(output, columns=['year', 'quarter', 'mae', 'mse', 'r2'])
    output.to_csv('./result/' + file_name + "-" + kernel_option + '-closeAdj-RQ5.csv', index=False)


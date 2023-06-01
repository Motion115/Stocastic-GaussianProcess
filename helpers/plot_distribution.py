from data_processor import DataPreprocessor, DataLoaderGtorch
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy import stats

def plot_normal_distribution(y, title):
    # set the center as 0, the left and right equal length
    # set x range
    x = pd.Series(y)
    x.plot(kind='kde', style='b')
    # plot a centerline at 0
    plt.axvline(x.mean(), color='r', linestyle='--')
    # add title
    plt.title(title)
    # plt.show()
    plt.savefig('./img/BA-density/' + title + '.png')
    plt.close()


if __name__ == '__main__':
    file_path, file_name = './data/', 'BA'
    preprocessed_data = DataPreprocessor(file_path, file_name)
    # get preprocessed_data start and end year
    start_year = preprocessed_data.start_year
    end_year = preprocessed_data.end_year
    start_year = 2005
    output = []
    normal = []
    all_y = pd.Series()
    for year in tqdm(range(start_year, end_year + 1)):
        for quarter_id in range(1, 5):
            quarter = str(year) + "-Q" + str(quarter_id)
            dataloader = DataLoaderGtorch(preprocessed_data, quarter, quarter, 'Adj Close', test_cases=5)
            train_x, train_y, test_x, test_y = dataloader.retrieve_data()
            # concat train_y, test_y
            train_y = train_y.reshape(-1)
            test_y = test_y.reshape(-1)
            train_y = pd.Series(train_y)
            test_y = pd.Series(test_y)
            # use concat to combine train_y and test_y
            train_y = pd.concat([train_y, test_y])
            # plot the distribution
            plot_normal_distribution(train_y, quarter)
            if stats.normaltest(train_y)[1] > 0.05:
                # print(quarter + " normal distribution")
                normal.append(quarter)
            # add to all_y
            all_y = pd.concat([all_y, train_y])
    # plot the distribution of all_y
    plot_normal_distribution(all_y, 'All')
    # judge whether it is normal distribution
    if stats.normaltest(all_y)[1] > 0.05:
        print('Normal distribution')
    # normal to csv
    normal = pd.DataFrame(normal, columns=['quarter'])
    normal.to_csv('./result/' + file_name + '-normal.csv', index=False)


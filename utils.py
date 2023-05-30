import matplotlib.pyplot as plt
import numpy as np

def plot_result(x, y, pred_y, lower_bound, upper_bound, start_of_pred):
    '''
    Note: parameters all in numpy array
    '''
    # Initialize plot
    f, ax = plt.subplots(1, 1, figsize=(4, 3))
    # Plot test data as blue line
    ax.plot(x, y, 'g')
    # Plot predictive means as blue line
    ax.plot(x, pred_y, 'b')
    x = x.reshape(-1)
    ax.fill_between(x, lower_bound, upper_bound, alpha=0.5)
    ax.legend(['Observed Data', 'Mean', 'Confidence'])
    # add a vertical line to indicate the start of prediction
    ax.axvline(x=start_of_pred, color='r', linestyle='--')
    plt.show()
import math
import torch
import gpytorch
from matplotlib import pyplot as plt
from tqdm import tqdm
from data_processor import DataPreprocessor, DataLoaderGtorch
from utils import plot_result, plot_result_simple
from deprecated_models import ExactGPModel, SpectralMixtureGPModel

def train(temp_x, temp_y, device, next_x):
    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(temp_x, temp_y, likelihood)
    model = model.to(device)
    likelihood = likelihood.to(device)

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters
    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    training_iter = 50
    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(temp_x)
        # Calc loss and backprop gradients
        loss = -mll(output, temp_y)
        loss.backward()
        # print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.item()))
        optimizer.step()
    # get a prediction
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(next_x))
        mean = observed_pred.mean
    return mean
    
if __name__ == '__main__':
    file_path, file_name = './data/', 'BA'
    preprocessed_data = DataPreprocessor(file_path, file_name, '2010-Q4', '2010-Q4')
    dataloader = DataLoaderGtorch(preprocessed_data, 'Open', 0.8, "torch")
    train_x, train_y, test_x, test_y = dataloader.retrieve_data()
    train_length = len(train_x)

    # check if cuda is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_x, train_y = train_x.to(device), train_y.to(device)

    # next_x is the first of the unseen data
    for i in tqdm(range(0, len(test_x))):
        # select next_x as the ith entry, but in a torch tensor
        next_x = test_x[i].reshape(1, -1).to(device)
        mean = train(train_x, train_y, device, next_x)
        # add the next_x and mean to train_x and train_y
        # reshape next_x back to 1D
        next_x = next_x.reshape(-1)
        train_x = torch.cat((train_x, next_x), dim=0)
        train_y = torch.cat((train_y, mean), dim=0)

    
    # Plot the results
    with torch.no_grad():
        # detach the tensor from GPU to CPU
        x = train_x.cpu().numpy()
        y = train_y.cpu().numpy()
        test_y = test_y.cpu().numpy()
        plot_result_simple(x, y, test_y, train_length)

'''
    # Get into evaluation (predictive posterior) mode
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(next_x))
        mean = observed_pred.mean
        # lower, upper = observed_pred.confidence_region()
    # delete current model
    del model
    '''



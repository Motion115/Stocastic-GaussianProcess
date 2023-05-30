import math
import torch
import gpytorch
from matplotlib import pyplot as plt
from tqdm import tqdm
from data_processor import DataPreprocessor, DataLoaderGtorch
from utils import plot_result

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RQKernel()
            )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
if __name__ == '__main__':
    file_path, file_name = './data/', 'BA'
    preprocessed_data = DataPreprocessor(file_path, file_name, '2016-Q1', '2016-Q4')
    dataloader = DataLoaderGtorch(preprocessed_data, 'Open', 0.8, "torch")
    train_x, train_y, test_x, test_y = dataloader.retrieve_data()
    # concatenate train_x and test_x in torch
    x = torch.cat((train_x, test_x), dim=0)
    y = torch.cat((train_y, test_y), dim=0)
    
    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood)

    # check if cuda is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    likelihood = likelihood.to(device)
    train_x, train_y = train_x.to(device), train_y.to(device)
    x, y = x.to(device), y.to(device)

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    training_iter = 50
    for i in tqdm(range(training_iter)):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()
        # print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.item()))
        optimizer.step()
    
    # Get into evaluation (predictive posterior) mode
    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(x))
        mean = observed_pred.mean
        lower, upper = observed_pred.confidence_region()
        # print('Test MAE: {}'.format(torch.mean(torch.abs(mean - test_y))))
    
    # Plot the results
    with torch.no_grad():
        # detach the tensor from GPU to CPU
        x = x.cpu().numpy()
        y = y.cpu().numpy()
        mean = mean.cpu().numpy()
        lower = lower.cpu().numpy()
        upper = upper.cpu().numpy()
        plot_result(x, y, mean, lower, upper, len(train_x))



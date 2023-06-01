import math
import torch
import gpytorch
from matplotlib import pyplot as plt
from tqdm import tqdm
from data_processor import DataPreprocessor, DataLoaderGtorch
from utils import save_plot

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel_option):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        kernel_options = {
            'RQ': gpytorch.kernels.RQKernel(),
            'Matern': gpytorch.kernels.MaternKernel(),
            'RBF': gpytorch.kernels.RBFKernel(),
            'Linear': gpytorch.kernels.LinearKernel(),
            'Combined': gpytorch.kernels.RQKernel() + gpytorch.kernels.MaternKernel(),
            'Spectural': gpytorch.kernels.SpectralMixtureKernel(num_mixtures=30),
            'Arc': gpytorch.kernels.ArcKernel(gpytorch.kernels.RBFKernel())
        }
        self.mean_module = gpytorch.means.ConstantMean()
        if kernel_option != 'Spectural':
            self.covar_module = gpytorch.kernels.ScaleKernel(
                kernel_options[kernel_option]        
            )
        else:
            self.covar_module = kernel_options[kernel_option]

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def train_eval(dataloader, file_name, start_period, kernel_option, iters = 100):  
    duration = dataloader.duration
    train_x, train_y, test_x, test_y = dataloader.retrieve_data()
    predict_length = test_x.shape[0]

    # concatenate train_x and test_x in torch
    x = torch.cat((train_x, test_x), dim=0)
    y = torch.cat((train_y, test_y), dim=0)
    
    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood, kernel_option)

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

    training_iter = iters
    for i in tqdm(range(training_iter)):
    #for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()
    
    # Get into evaluation (predictive posterior) mode
    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(x))
        mean = observed_pred.mean
        lower, upper = observed_pred.confidence_region()
        
    test_y = test_y.to(device)
    # evaluate the model
    # metrics include: MAE, MSE, R-squared
    mae = torch.mean(torch.abs(mean[-predict_length:] - test_y))
    mse = torch.mean((mean[-predict_length:] - test_y) ** 2)
    r2 = 1 - mse / torch.var(test_y)
    
    # target directory for image is
    target_dir = "./img/" + file_name + "/" + str(predict_length) + "-" + start_period + ".png"
    title = file_name
    subtitle =  start_period + " (" + str(duration - predict_length) + " + " + str(predict_length) + " days)"
    
    # Plot the results
    with torch.no_grad():
        # detach the tensor from GPU to CPU
        x = x.cpu().numpy()
        y = y.cpu().numpy()
        mean = mean.cpu().numpy()
        lower = lower.cpu().numpy()
        upper = upper.cpu().numpy()
        save_plot(x, y, mean, lower, upper, len(train_x), target_dir, title, subtitle)

    # metrics drop to CPU, then turn to float
    mae = mae.cpu().numpy().item()
    mse = mse.cpu().numpy().item()
    r2 = r2.cpu().numpy().item()
    return mae, mse, r2

    
if __name__ == '__main__':
    file_path, file_name = './data/', 'BA'
    quarter_s, quarter_e = "2012-Q4", "2012-Q4"
    preprocessed_data = DataPreprocessor(file_path, file_name)
    dataloader = DataLoaderGtorch(preprocessed_data, quarter_s, quarter_e, 'Adj Close', test_cases=5)
    mae, mse, r2 = train_eval(dataloader, file_name, quarter_s, kernel_option='Matern', iters=150)
    print(mae, mse, r2)
    
    




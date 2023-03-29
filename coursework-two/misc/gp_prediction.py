# numeric imports
import torch
import numpy as np

# gp imports
import gpytorch as gp

# data imports
import pandas as pd

# plotting imports
import matplotlib.pyplot as plt
import matplotlib.figure as figure

# utility imports
from utilities.processing_utility import get_moving_average

# misc imports
import datetime
from tqdm import tqdm

# typing imports
from typing import List, Tuple, Dict


class LinearGP(gp.models.ExactGP):
    """  GP with Linear Mean """
    
    def __init__(self, inputs, targets, likelihood):
        super(LinearGP, self).__init__(inputs, targets, likelihood)
        self.mean_module = gp.means.LinearMean(input_size=torch.tensor(1 if len(inputs.shape) == 1 else inputs.shape[-1]))
        self.covar_module = gp.kernels.ScaleKernel(gp.kernels.RBFKernel())
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gp.distributions.MultivariateNormal(mean_x, covar_x)
    
    def get_posterior(self, test_inputs, confidenceregion = False):
        self.eval()
        with torch.no_grad(), gp.settings.fast_pred_var():
            f_preds = self(test_inputs)

        if confidenceregion:
            lower, upper = f_preds.confidence_region()
            return f_preds, lower, upper
        
        return f_preds

    def get_posterior_predictive(self, test_inputs, confidenceregion = False):
        # get posterior
        self.eval()
        with torch.no_grad(), gp.settings.fast_pred_var():
            observed_pred = self.likelihood(self(test_inputs))

        if confidenceregion:
            lower, upper = observed_pred.confidence_region()
            return observed_pred, lower, upper
        
        return observed_pred
    

def optimise_marginal_likelihood(
    model : LinearGP,
    n_iter : int = 10000,
    )-> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """ 
    Finds the optimal hyperparameters for a Linear GP model by optimising the marginal likelihood.

    Arguments:
    ----------
    inputs          : {torch.Tensor}
                        > x training data to condition on.
    targets         : {torch.Tensor}
                        > y training data to condition on.
    n_iter          : {int}
                        > number of iterations to run the optimisation for.

    Returns:
    ----------
    lengthscale_hat : {torch.Tensor}
                        > optimal lengthscale.
    noise_hat       : {torch.Tensor}
                        > optimal noise.
    weights_hat     : {torch.Tensor}
                        > optimal weights.
    bias_hat        : {torch.Tensor}
                        > optimal bias.
    """
    print_freq = int(n_iter / 10)

    # get model components
    likelihood = model.likelihood
    inputs = model.train_inputs[0].squeeze()
    targets = model.train_targets

    # put into training mode
    model.train()
    likelihood.train()

    # optimiser
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gp.mlls.ExactMarginalLogLikelihood(likelihood, model)

    print('='*100)
    print(f"{'Iteration':^24}|{'Loss':^24}|{'Lengthscale':^24}|{'Noise':^24}")
    print('='*100)
    for i in range(n_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(inputs)
        # Calc loss and backprop gradients
        loss = -mll(output, targets)
        loss.backward()

        if (i + 1) % print_freq == 0:
            print(f"{i+1:^24}|{loss.item():^24}|{model.covar_module.base_kernel.lengthscale.item():^24.5f}|{model.likelihood.noise.item():^24.5f}")
            print('-'*100)

        optimizer.step()

    # get lengthscale, noise and weights
    lengthscal_hat = model.covar_module.base_kernel.lengthscale.item()
    noise_hat = model.likelihood.noise.item()
    weights_hat = getattr(model.mean_module, "weights").detach()
    bias_hat = getattr(model.mean_module, "bias").detach()

    return lengthscal_hat, noise_hat, weights_hat, bias_hat


def get_linear_gp(
    inputs : torch.Tensor,
    targets : torch.Tensor,
    lengthscale : float,
    noise : float,
    weights : torch.Tensor,
    bias : torch.Tensor = None,
    ) -> gp.models.ExactGP:
    """
    Consturcts a Linear GP model with Gaussian likelihood

    Arguments:
    ----------
    inputs         : {np.ndarray}
                        > x training data to condition on.
    targets         : {np.ndarray}
                        > y training data to condition on.
    length_scale    : {float}
                        > lengthscale of the RBF kernel.
    noise           : {float}
                        > noise of the Gaussian likelihood.
    weights         : {torch.Tensor}
                        > weights of the linear mean.
    bias            : {torch.Tensor}
                        > bias of the linear mean.
    
    Returns:
    ----------
    linear_gp       : {LinearGP}
                        > Linear GP model.
    """
    # instantiate model
    likelihood = gp.likelihoods.GaussianLikelihood()
    linear_gp = LinearGP(inputs, targets, likelihood)

    # plug optimised parameters into model
    linear_gp.covar_module.base_kernel.lengthscale = lengthscale
    linear_gp.likelihood.noise = noise
    linear_gp.mean_module.weights.data = weights

    if bias is not None:
        linear_gp.mean_module.bias.data = bias
    else:
        linear_gp.mean_module.bias.data = targets[0]

    return linear_gp


def get_x_star( # BLELONGS IN PROCESSING UTILS #
    inputs : torch.Tensor,
    ) -> torch.Tensor:
    """
    Constructs a tensor of x_star values to sample from GP posterior.

    Arguments:
    ----------
    x_train : {torch.Tensor}
                > x training data conditioned on in posterior.
    
    Returns:
    ----------
    x_star  : {torch.Tensor}
                > x_star values to sample from posterior.
    """
    # get sample range
    N = len(inputs)
    min_range = torch.min(inputs)
    max_range = torch.max(inputs)
    sample_range = torch.linspace(min_range, max_range, N*4)

    # get x_star
    x_star = torch.cat((sample_range, inputs))

    # remove duplicates
    x_star = torch.unique(x_star)

    # sort x_star
    x_star = torch.sort(x_star)[0]

    return x_star


def get_linear_gp_posterior(
    lineargpmodel : gp.models.ExactGP,
    inputs : torch.Tensor,
    ) -> gp.distributions.MultivariateNormal:
    """
    Constructs a MultivariateNormal posterior distribution of dim(x_star) to sample f_preds.

    Arguments:
    ----------
    gp_model            : {gpytorch.models.ExactGP}
                            > GP model.
    x_star              : {torch.Tensor}
                            > inputs values to sample from posterior.
                            -> (governs the dimensionality of the posterior)

    Returns:
    ----------
    linear_gp_posterior : {gpytorch.distributions.MultivariateNormal}
                            > MultivariateNormal posterior distribution conditioned on data.
    """
    # get posterior
    lineargpmodel.eval()
    with torch.no_grad(), gp.settings.fast_pred_var():
        f_preds = lineargpmodel(inputs)

    return f_preds


def get_gp_posterior_predictive(
    gp_model : gp.models.ExactGP,
    likelihood : gp.likelihoods.GaussianLikelihood,
    x_star : torch.Tensor,
    ) -> gp.distributions.MultivariateNormal:
    """
    Constructs a MultivariateNormal posterior distribution of dim(x_star) to sample f_preds.

    Arguments:
    ----------
    gp_model    : {gpytorch.models.ExactGP}
                    > GP model.
    x_star      : {torch.Tensor}
                    > x_star values to sample from posterior.

    Returns:
    ----------
    posterior   : {gpytorch.distributions.MultivariateNormal}
                    > MultivariateNormal posterior distribution conditioned on data.
    """
    # get posterior
    gp_model.eval()
    with torch.no_grad(), gp.settings.fast_pred_var():
        observed_pred = likelihood(gp_model(x_star))

    return observed_pred


def plot_sequential_fit(
    inputs : torch.Tensor,
    targets : torch.Tensor,
    lengthscale : torch.Tensor,
    noise : torch.Tensor,
    weights : torch.Tensor,
    bias : bool = None,
    ) -> List[figure.Figure]:
    """ 
    Constructs a list of plots for sequential fit of a Linear GP model.

    Arguments:
    ----------
    inputs  : {torch.Tensor}
                > x training data to condition on.
    targets : {torch.Tensor}
                > y training data to condition on.
    lengthscale : {torch.Tensor}
                > lengthscale of the RBF kernel.
    noise : {torch.Tensor}
                > noise of the Gaussian likelihood.
    weights : {torch.Tensor}
                > weights of the linear mean.
    bias : {torch.Tensor}
                > bias of the linear mean.
    
    Returns:
    ----------
    figures : {List[figure.Figure]}
                > list of figures.
    """
    # data
    x_star = torch.linspace(inputs[0], inputs[-1], len(inputs)*4)
    intercept = targets[0] if bias is None else bias

    # set up figure
    figures = []

    # sequential prediction
    for i in range(len(inputs[:-1])):

        # train points
        target_train_points = targets[:i+1]
        input_train_points = inputs[:i+1]

        # try and fit a GP
        n_gp = get_linear_gp(
                input_train_points, 
                target_train_points, 
                lengthscale, 
                noise,
                weights, 
                intercept)
        
        f_preds, lower, upper = n_gp.get_posterior(x_star, True)

        # make plot
        fig = plt.figure()

        plt.plot(input_train_points, target_train_points, marker = 'x', color = 'black', alpha = 1)
        plt.plot(x_star, f_preds.mean, color='red', alpha=1)
        plt.fill_between(x_star, lower, upper, color='red', alpha=.1)

        for i in range(25):
            plt.plot(x_star, f_preds.sample(), color='red', alpha=.1)

        # append figure
        figures.append(fig)

        plt.close()

    return figures


def plot_sequential_pred(
    inputs : torch.Tensor,
    targets : torch.Tensor,
    lengthscale : torch.Tensor,
    noise : torch.Tensor,
    weights : torch.Tensor,
    bias : bool = None,
    ) -> List[figure.Figure]:
    """ 
    Constructs a list of plots for sequential prediction of a Linear GP model.

    Arguments:
    ----------
    inputs  : {torch.Tensor}
                > x training data to condition on.
    targets : {torch.Tensor}
                > y training data to condition on.
    lengthscale : {torch.Tensor}
                > lengthscale of the RBF kernel.
    noise : {torch.Tensor}
                > noise of the Gaussian likelihood.
    weights : {torch.Tensor}
                > weights of the linear mean.
    bias : {torch.Tensor}
                > bias of the linear mean.
    
    Returns:
    ----------
    figures : {List[figure.Figure]}
                > list of figures.
    """
    # intercept term
    intercept = targets[0] if bias is None else bias
    
    # set up figure
    figures = []

    # sequential prediction
    for i in range(len(inputs[:-1])):
        
        # test point
        input_test_point = inputs[i].unsqueeze(-1)
        target_test_point = targets[i]
        
        # train points
        input_train_points = inputs[:i]
        target_train_points = targets[:i]

        # fit a GP to ths historica data
        n_gp = get_linear_gp(
                input_train_points, 
                target_train_points, 
                lengthscale, 
                noise,
                weights, 
                intercept)
        
        train_f_preds, train_lower, train_upper = n_gp.get_posterior(input_train_points, True)

        # get predictive
        test_f_preds, test_lower, test_upper = n_gp.get_posterior_predictive(input_test_point, True)

        # make plot
        fig = plt.figure()

        # training
        plt.plot(input_train_points, target_train_points, color = 'black', alpha = 1)
        
        plt.plot(input_train_points, train_f_preds.mean, color='red', alpha=1)
        plt.fill_between(input_train_points, train_lower, train_upper, color='red', alpha=.1)

        # testing
        plt.plot(input_test_point, target_test_point, marker = 'x', color = 'black', alpha = 1)
        plt.plot(input_test_point, test_f_preds.mean, marker = 'x', color = 'steelblue', alpha = 1)
        plt.fill_between(input_test_point, test_lower, test_upper, color='steelblue', alpha=1)

        # append figure
        figures.append(fig)

        plt.close()

    return figures

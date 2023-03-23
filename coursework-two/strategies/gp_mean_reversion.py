# numeric imports
import torch
import numpy as np

# gp imports
import gpytorch as gp
from GP import *

# data imports
import pandas as pd

# plotting imports
import matplotlib.pyplot as plt

# utility imports
from utilities.processing_utility import get_moving_average

# misc imports
import datetime
from tqdm import tqdm

# typing imports
from typing import List, Tuple, Dict


class ExactGPModel(gp.models.ExactGP):
    """  Closed form solution to GP regression """
    
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gp.means.ConstantMean()
        self.covar_module = gp.kernels.ScaleKernel(gp.kernels.RBFKernel())
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gp.distributions.MultivariateNormal(mean_x, covar_x)
    

def get_train_test_split(
        series : torch.Tensor,
        split_index : int,
        ) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Splits a series into a training and test set given a split index.

    Arguments:
    ----------
    series      : {torch.Tensor}
                    > A series to split.
    split_index : {int}
                    > The index to split the series at.
    
    Returns:
    ----------
    train_set   : {torch.Tensor}
                    > The training set.
    test_set    : {torch.Tensor}
                    > The test set.
    """
    # split data
    train_set = series[:split_index]
    test_set = series[split_index:]

    return train_set, test_set


def optimise_marginal_likelihood(
    inputs : torch.Tensor,
    targets : torch.Tensor,
    n_iter : int = 100,
    )-> Tuple[torch.Tensor, torch.Tensor]:
    """ 
    Finds the optimal lengthscale and noise parameters for a GP model by optmisising the marginal likelihood.

    Arguments:
    ----------
    x_train         : {torch.Tensor}
                        > x training data to condition on.
    y_train         : {torch.Tensor}
                        > y training data to condition on.
    n_iter          : {int}
                        > number of iterations to run the optimisation for.

    Returns:
    ----------
    lengthscale_hat : {torch.Tensor}
                        > optimal lengthscale.
    noise_hat       : {torch.Tensor}
                        > optimal noise.
    """
    # construct model
    likelihood = gp.likelihoods.GaussianLikelihood()
    model = ExactGPModel(inputs, targets, likelihood)

    # put into training mode
    model.train()
    likelihood.train()

    # optimiser
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Includes GaussianLikelihood parameters

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

        if (i + 1) % 10 == 0:
            print(f"{i+1:^24}|{loss.item():^24}|{model.covar_module.base_kernel.lengthscale.item():^24.5f}|{model.likelihood.noise.item():^24.5f}")
            print('-'*100)

        optimizer.step()

    # get optimal parameters
    lengthscale_hat = model.covar_module.base_kernel.lengthscale.item()
    noise_hat = model.likelihood.noise.item()

    return lengthscale_hat, noise_hat


def get_gp(
    x_train : torch.Tensor,
    y_train : torch.Tensor,
    length_scale : float = 2.5,
    noise : float = 0.5,
    ) -> gp.models.ExactGP:
    """
    Consturcts a GP model with Gaussian likelihood

    Arguments:
    ----------
    x_train         : {np.ndarray}
                        > x training data to condition on.
    y_train         : {np.ndarray}
                        > y training data to condition on.
    length_scale    : {float}
                        > lengthscale of the RBF kernel.
    noise           : {float}
                        > noise of the Gaussian likelihood.
    
    Returns:
    ----------
    gp_model        : {gpytorch.models.ExactGP}
                        > GP model.
    """
    # instantiate model
    likelihood = gp.likelihoods.GaussianLikelihood()
    model = ExactGPModel(x_train, y_train, likelihood)

    # update lengthscale and noise
    model.covar_module.base_kernel.lengthscale = torch.tensor(length_scale)
    model.likelihood.noise = torch.tensor(noise)

    return model


def get_x_star(
    x_train : torch.Tensor,
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
    N = len(x_train)
    min_range = torch.min(x_train)
    max_range = torch.max(x_train)
    sample_range = torch.linspace(min_range, max_range, N*4)

    # get x_star
    x_star = torch.cat((sample_range, x_train))

    # remove duplicates
    x_star = torch.unique(x_star)

    # sort x_star
    x_star = torch.sort(x_star)[0]

    return x_star


def get_gp_posterior(
    gp_model : gp.models.ExactGP,
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
        f_preds = gp_model(x_star)

    return f_preds


def get_x_star_idx(
    x_train_point : torch.Tensor,
    x_star : torch.Tensor,
    ) -> int:
    """
    Finds the index of a point in x_star.

    Arguments:
    ----------
    x_train_point   : {torch.Tensor}
                        > A point in x_train.
    x_star          : {torch.Tensor}
                        > x_star values to sample from posterior.

    Returns:
    ----------
    x_star_idx      : {int}
                        > The index of x_train_point in x_star.
    """
    # get x_star index
    x_star_index = np.where(x_star == x_train_point)[0][0]

    return x_star_index


def gaussian_process_mean_reversion_strategy(
    price_series : torch.Tensor,
    daily_excess_returns_series : torch.Tensor,
    x_series : torch.Tensor,
    length_scale : float = 2.5,
    noise : float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int], List[int]]:
    """
    pass
    """
    # initialise No. of shares & cash
    w = np.zeros(np.shape(price_series))
    cash = np.zeros(np.shape(price_series))
    cash[0] = 1.

    # get x_star
    x_star = get_x_star(x_series)

    # save long and short indices
    long_idx = []
    short_idx = []

   # backtest strategy
    for i, (p, r) in tqdm(enumerate(zip(price_series[:-1], daily_excess_returns_series[:-1]))):

        # get current values
        x = x_series[i]
        
        # get GP for the current data
        gp_model = get_gp(x_train = x_series[:i+1], 
                          y_train = daily_excess_returns_series[:i+1], 
                          length_scale = length_scale, 
                          noise = noise,)

        # get posterior
        gp_posterior = get_gp_posterior(gp_model, x_star, )

        # get confidence region
        gp_lower, gp_upper = gp_posterior.confidence_region()

        # get x_star index
        x_star_index = get_x_star_idx(x, x_star)

        # get strategy metrics
        mean = gp_posterior.mean[x_star_index]
        upper_cci = gp_upper[x_star_index]
        lower_cci = gp_lower[x_star_index]

        # if we are inside the bollinger band -> hold
        if (r >= lower_cci) and (r <= upper_cci):
            w[i+1] = w[i]
            cash[i+1] = cash[i]
            continue
        
        # if we are below the bollinger band -> buy
        if r < lower_cci:
            w[i+1] = cash[i] / p + w[i]
            cash[i+1] = 0
            long_idx.append(i)
            continue
        
        # if we are above the bollinger band - sell
        if r > upper_cci:
            cash[i+1] = w[i] * p + cash[i]
            w[i+1] = 0
            short_idx.append(i)
            continue

    # strategy returns
    strategy_returns = (price_series * w) + cash

    # from collections import namedtuple
    # Data = namedtuple('Data', ['strategy_returns', 'w', 'cash', 'long_x', 'short_x', 'gp_posterior', 'gp_lower', 'gp_upper'])
    # useful_stuff = Data(strategy_returns, w, cash, long_x, short_x, gp_posterior, gp_lower, gp_upper)

    return strategy_returns, w, cash, long_idx, short_idx, gp_posterior


def plot_strategy(
    plot_axis : plt.Axes,
    normalised_daily_excess_returns : np.ndarray,
    x_series : torch.Tensor,
    posterior : gp.distributions.MultivariateNormal,
    long_idx : List[int],
    short_idx : List[int],
    ) -> None:
    """
    Plots the gaussian process mean reversion strategy.
    (i.e plots buy and sell calls with uncertainty interval)

    Arguments:
    ----------
    plot_axis                   : {plt.Axes}
                                    > The axis to plot on.
    normalised_daily_excess_returns : {np.ndarray}
                                        > Normalised daily excess returns.
    x_series                    : {torch.Tensor}
                                    > x values.
    posterior                   : {gp.distributions.MultivariateNormal}
                                    > Posterior distribution.
    long_idx                    : {List[int]}
                                    > Indices of long calls.
    short_idx                   : {List[int]}
                                    > Indices of short calls.   

    Returns:
    ----------
    None
    """
    # get x_star
    x_star = get_x_star(x_series)

    # get confidence region
    gp_lower, gp_upper = posterior.confidence_region()

    # plot daily excess returns
    plot_axis.plot(x_series, normalised_daily_excess_returns, color = 'black', lw = 0.8)

    # plot moving average
    plot_axis.plot(x_star, posterior.mean, color = 'steelblue', lw = 1.)

    # plot boillinger bands
    plot_axis.fill_between(x_star, gp_lower, gp_upper, color = 'steelblue', alpha = 0.1)

    # plot buy calls
    plot_axis.scatter(x_series[long_idx], normalised_daily_excess_returns[long_idx], marker = 'x', color = 'green')

    # plot sell calls
    plot_axis.scatter(x_series[short_idx], normalised_daily_excess_returns[short_idx], marker = 'x', color = 'red')




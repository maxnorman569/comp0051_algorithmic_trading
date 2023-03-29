# numeric imports
import torch
import numpy as np

# gp imports
import gpytorch as gp

# plotting imports
import matplotlib.pyplot as plt

# typing imports
from typing import List, Tuple, Dict


class ConstantGP(gp.models.ExactGP):
    """  Closed form solution to GP regression """
    
    def __init__(self, train_x, train_y, likelihood):
        super(ConstantGP, self).__init__(train_x, train_y, likelihood)
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
    model = ConstantGP(inputs, targets, likelihood)

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


def get_constant_gp(
    x_train : torch.Tensor,
    y_train : torch.Tensor,
    length_scale : float = 2.5,
    noise : float = 0.5,
    ) -> ConstantGP:
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
    model = ConstantGP(x_train, y_train, likelihood)

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
    gp_model : ConstantGP,
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


def get_gp_mean_reversion_signal(
        X : np.ndarray,
        R : np.ndarray,
        lengthscale : float,
        noise : float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Constructs a signal for a constant GP mean reversion trading strategy.

    Arguments:
    ----------
    X           : {np.ndarray}
                    > x training data to condition on.
    R           : {np.ndarray}
                    > y training data to condition on.
    lengthscale : {float}
                    > lengthscale of the RBF kernel.
    noise       : {float}

    Returns:
    ----------
    signal      : {np.ndarray}
                    > Signal for trading strategy.
    buy_mask    : {np.ndarray}
                    > Mask for buy signals.
    sell_mask   : {np.ndarray}
                    > Mask for sell signals.
    """
    buy_mask = torch.zeros_like(X)
    sell_mask = torch.zeros_like(X)

    buy_delta = torch.zeros_like(X)
    sell_delta = torch.zeros_like(X)

    signal = torch.zeros_like(X)

    for i, (x, r) in enumerate(zip(X, R)):

        if i == 0:
            continue

        constant_gp = get_constant_gp(x_train=X[:i], y_train=R[:i], length_scale=lengthscale, noise=noise)
        f_preds = get_gp_posterior(constant_gp, x.unsqueeze(-1))
        lower, upper = f_preds.confidence_region()

        if (r > lower[-1]) and (r < upper[-1]):

            if (signal[i-1] > 0) and (r > f_preds.mean[-1]):
                signal[i] = 0.

            elif (signal[i-1] < 0) and (r < f_preds.mean[-1]):
                signal[i] = 0.

            else:
                signal[i] = signal[i-1]

        if r <= lower[-1]:
            buy_mask[i] = 1
            delta = torch.clamp(r - lower[-1], min=-1, max=None).item()
            buy_delta[i] = delta
            signal[i] = delta
        
        if r >= upper[-1]:
            sell_mask[i] = 1
            delta = torch.clamp(r - upper[-1], min=None, max=1).item()
            sell_delta[i] = delta
            signal[i] = delta

    return signal, buy_mask, sell_mask


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




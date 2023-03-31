import torch
import gpytorch as gp

from utilities.processing_utility import get_rsi, get_macd

from typing import Tuple


def get_augmented_data(x, p):
    macd_train = torch.tensor(get_macd(p).to_numpy(), dtype=torch.float32)
    rsi_train = torch.tensor(get_rsi(p).to_numpy(), dtype=torch.float32)
    data = torch.cat((x.unsqueeze(1), macd_train, rsi_train), dim = 1)
    return data


class ConstantGP(gp.models.ExactGP):
    """  GP with Linear Mean """
    
    def __init__(self, inputs, targets, likelihood):
        super(ConstantGP, self).__init__(inputs, targets, likelihood)
        self.mean_module = gp.means.constant_mean.ConstantMean()
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
    X : torch.tensor,
    P : torch.tensor,
    n_iter : int = 100,
    )-> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """ 
    Finds the optimal hyperparameters for a Linear GP model by optimising the marginal likelihood.

    Arguments:
    ----------
    X               : {torch.tensor}
                        > input data.
    P               : {torch.tensor}
                        > output data.
    n_iter          : {int}
                        > number of iterations to run the optimisation for.

    Returns:
    ----------
    lengthscale_hat : {torch.Tensor}
                        > optimal lengthscale.
    noise_hat       : {torch.Tensor}
                        > optimal noise.
    """
    print_freq = int(n_iter / 10)

    # get model components
    likelihood = gp.likelihoods.GaussianLikelihood()
    model = ConstantGP(X, P, likelihood)

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
        output = model(X)
        # Calc loss and backprop gradients
        loss = -mll(output, P)
        loss.backward()

        if (i + 1) % print_freq == 0:
            print(f"{i+1:^24}|{loss.item():^24}|{model.covar_module.base_kernel.lengthscale.item():^24.5f}|{model.likelihood.noise.item():^24.5f}")
            print('-'*100)

        optimizer.step()

    # get lengthscale, noise and weights
    lengthscal_hat = model.covar_module.base_kernel.lengthscale.item()
    noise_hat = model.likelihood.noise.item()

    return lengthscal_hat, noise_hat
    

def get_gp_prediction_signal(
    X : torch.Tensor,
    P : torch.Tensor,
    lengthscale : float = 2.5,
    noise : float = .5,
    window : int = 20,
    ) -> torch.Tensor:
    """
    Get the signal for a GP cross over strategy.

    Arguments:
    ----------
    X           : {torch.Tensor}
                    > The time series of the data.
    P           : {torch.Tensor}
                    > The time series of the prices.

    Returns:
    ----------
    signal      : {torch.Tensor}
                    > The signal for the GP cross over strategy.
    buy_mask    : {torch.Tensor}
                    > The mask for the buy signal.
    sell_mask   : {torch.Tensor}
    """
    # setp
    buy_mask = torch.zeros_like(P)
    sell_mask = torch.zeros_like(P)

    signal = torch.zeros_like(P)   

    rolling_std = P.unfold(0, window, 1).std(dim=1)
    signal_weights = (rolling_std / rolling_std.max())*0.98

    # sequential prediction
    for i, (x, p) in enumerate(zip(X, P)):     
        
        if i <= window:
            signal[i] = 0
            continue

        # fit a GP to ths historica data
        n_start = i - window
        n_end = i 
        window_gp = ConstantGP(X[n_start:n_end], P[n_start:n_end], gp.likelihoods.GaussianLikelihood())

        window_gp.mean_module.constant = torch.mean(P[n_start:n_end])
        window_gp.covar_module.base_kernel.lengthscale = lengthscale
        window_gp.likelihood.noise = noise

        if i + 1 < len(X):
            f_test, upper, lower = window_gp.get_posterior_predictive(X[n_end+1].view(1,-1), True) # wrong way around, should be lower, upper
            f_test = f_test.mean.item()

        if (upper > p) and (p > lower):
            signal[i] = 0.5

        elif (f_test > p) and (p > upper):
            buy_mask[i] = 1
            signal[i] = signal_weights[i-window]

        elif (f_test < p) and (p < lower):
            sell_mask[i] = 1
            signal[i] = -signal_weights[i-window]

        else:
            signal[i] = 0

    return signal, buy_mask, sell_mask
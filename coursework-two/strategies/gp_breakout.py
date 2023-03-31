# numeric imports
import torch
import gpytorch as gp

# typing imports
from typing import Tuple


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

def get_gp_breakout_signal(
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
    buy_mask = torch.zeros_like(X)
    sell_mask = torch.zeros_like(X)

    signal = torch.zeros_like(X)   

    upper_cci, lower_cci = torch.zeros_like(X), torch.zeros_like(X)
    gp_map = torch.zeros_like(X)
    gp_pred = torch.zeros_like(X)

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

        f_preds, lower, upper = window_gp.get_posterior(X[n_start:n_end], True)

        if i + 1 < len(X):
            f_test = window_gp.get_posterior_predictive(X[n_end+1].unsqueeze(-1), False).mean

        # upper_cci[i] = upper[-1]
        # lower_cci[i] = lower[-1]
        # gp_map[i] = f_preds.mean[-1].item()
        # gp_pred[i] = f_test.item()

        # if (p > lower[-1]) and (p < upper[-1]):
            
        #     if (signal[i-1] > 0) and (p > f_preds.mean[-1].item()):
        #         signal[i] = signal[i-1]

        #     elif (signal[i-1] < 0) and (p < f_preds.mean[-1].item()):
        #         signal[i] = signal[i-1]

        #     else:
        #         signal[i] = 0
        
        # # buy
        # elif p >= upper[-1]:
        #     buy_mask[i] = 1
        #     signal[i] = p - upper[-1]

        # # sell
        # elif p <= lower[-1]:
        #     sell_mask[i] = 1
        #     signal[i] = p - lower[-1]

        if f_test > p:
            signal[i] = f_test - p

        elif f_test < p:
            signal[i] = f_test - p

        else:
            signal[i] = 0

    return signal, buy_mask, sell_mask, gp_map, lower_cci, upper_cci, gp_pred

    # # norm scale the signal
    # scaled_signal = signal / torch.std(signal)
    # scaled_signal = torch.clamp(scaled_signal, -2, 2)
    
    # return scaled_signal / 2, buy_mask, sell_mask, gp_map, lower_cci, upper_cci, gp_pred
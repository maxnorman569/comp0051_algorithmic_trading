import torch
import gpytorch as gp

class ExactGPModel(gp.models.ExactGP):
    """  Closed form solution to GP regression"""
    
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gp.means.ConstantMean()
        self.covar_module = gp.kernels.ScaleKernel(gp.kernels.RBFKernel())
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gp.distributions.MultivariateNormal(mean_x, covar_x)
    

def get_gp(
        x_train : torch.Tensor, 
        y_train : torch.Tensor,
        length_scale : float = 2.5,
        epsilon : float = 0.5
        ) -> gp.models.ExactGP:
    """
    Consturcts a GP model with Gaussian likelihood

    Arguments:
    ----------
    x_train     : {np.ndarray}
                    > x training data to condition on.
    y_train     : {np.ndarray}
                    > y training data to condition on.
    length_scale: {float}
                    > lengthscale of the RBF kernel.
    epsilon     : {float}
                    > noise of the Gaussian likelihood.
    
    Returns:
    --------
    gp_model   : {gpytorch.models.ExactGP}
                    > GP model.
    """
    # instantiate model
    model = ExactGPModel(x_train, y_train)

    # update lengthscale and noise
    model.covar_module.base_kernel.lengthscale = torch.tensor(length_scale)
    model.likelihood.noise = torch.tensor(epsilon)

    return model
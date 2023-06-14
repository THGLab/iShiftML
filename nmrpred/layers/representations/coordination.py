import torch
import torch.nn as nn


class FermiDirac(nn.Module):
    """
    The Fermi-Dirac distribution with simplified hyperparameters.

    Parameters
    ----------
    mu: flaot
        equivalent to the total chemical potential

    sigma: float
        equivalent to the 1/kT

    """
    def __init__(self,
                 mu=1.0,
                 sigma=3.0,
                 trainable=False,
                 device=None
                ):
        super(FermiDirac, self).__init__()
        mu = torch.tensor(mu,
                            dtype=torch.float32,
                            device=device)
        sigma = torch.tensor(sigma,
                              dtype=torch.float32,
                              device=device)

        if trainable:
            self.mu = nn.Parameter(mu)
            self.sigma = nn.Parameter(sigma)
        else:
            self.register_buffer("mu", mu)
            self.register_buffer("sigma", sigma)

    def forward(self, inputs):
        """

        Parameters
        ----------
        inputs: torch.tensor
            This is a 1D tensor of distances

        Returns
        -------
        torch.tensor: The F-D distribution

        """
        return 1.0/(1.0+torch.exp(self.sigma * (inputs - self.mu)))
import torch
from torch import nn


class PoolPolar(nn.Module):
    """

    """
    def __init__(self, pooling_type, dim):
        self.pooling_type = pooling_type
        self.dim = dim
        super(PoolPolar, self).__init__()

    def forward(self, x):
        if self.pooling_type == 'mean':
            n = x.shape[self.dim]
            x = torch.sum(x, dim=self.dim)  # (B,A,G,G)

            n *= x.shape[self.dim]
            x = torch.sum(x, dim=self.dim)  # (B,A,G)

            x = x / n

        elif self.pooling_type == 'mean_dim':
            x = torch.mean(x, dim=self.dim)    #(B,A,G,G)
            x = torch.mean(x, dim=self.dim)    #(B,A,G)

        elif self.pooling_type == 'sum':
            x = torch.sum(x, dim=self.dim)    #(B,A,G,G)
            x = torch.sum(x, dim=self.dim)    #(B,A,G)

        elif self.pooling_type == 'max':
            x, _ = torch.max(x, dim=self.dim)  #(B,A,G,G)
            x, _ = torch.max(x, dim=self.dim)  #(B,A,G)

        return x

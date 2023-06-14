import torch
from torch import nn
from torch.nn.init import xavier_uniform_
from torch.nn.init import zeros_

from nmrpred.layers import BatchRenorm3d

class Conv3D(nn.Conv3d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding,
                 bias=True,
                 activation=nn.ReLU(),
                 weight_init=xavier_uniform_,
                 bias_init=zeros_,
                 dropout=False,
                 norm=False):
        self.weight_init = weight_init
        self.bias_init = bias_init
        super(Conv3D, self).__init__(in_channels,
                                     out_channels,
                                     kernel_size=kernel_size,
                                     padding=padding,
                                     bias=bias)
        self.activation = activation
        # initialize linear layer y = xW^T + b

        self.dropout = dropout
        if dropout:
            self.dropout = nn.Dropout(dropout)

        self.norm = norm
        if norm:
            self.norm = nn.BatchNorm3d(
                num_features=out_channels,
                affine=True,
                track_running_stats=True
            )  #, momentum=0.99, eps=0.001) # momentum and eps are based on Keras default values

    def reset_parameters(self):
        self.weight_init(self.weight)
        if self.bias is not None:
            self.bias_init(self.bias)

    def forward(self, x):
        # compute linear layer y = xW^T + b
        x = super(Conv3D, self).forward(x)

        # batch normalization
        if self.norm:
            x = self.norm(x)

        # add activation function
        if self.activation:
            x = self.activation(x)

        # dropout
        if self.dropout:
            x = self.dropout(x)

        return x


class GaussianFilter(nn.Module):
    def __init__(
        self, grid_length=10, grid_size=20, trainable=False
    ):
        super(GaussianFilter, self).__init__()
        # compute offset and width of Gaussian functions
        half_length = grid_length / 2.
        offsets = torch.linspace(-half_length, half_length, grid_size)
        grids = torch.meshgrid([offsets, offsets, offsets])
        grid = torch.stack(grids, dim=-1)   # shape: G,G,G,3
        widths = torch.FloatTensor((offsets[1] - offsets[0]) * torch.ones_like(grid)) # shape: G,G,G,3

        if trainable:
            self.register_parameter('widths', nn.Parameter(widths))
            self.register_parameter('grid', nn.Parameter(grid))
        else:
            self.register_buffer('widths', widths)
            self.register_buffer('grid', grid)

    def forward(self, distance_vector):
        """Compute smeared-gaussian distance values.

        Parameters
        ----------
        distance_vector: torch.Tensor
        interatomic distance vectors of (B, A, N, 3) shape.

        Returns
        -------
        torch.Tensor: layer output of (B, A, N, G, G, G, 3) shape.

        """
        return gaussian_smearing_3d(
            distance_vector, self.grid, self.widths)

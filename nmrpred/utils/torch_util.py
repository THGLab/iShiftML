import torch
from torch import nn
from torch.nn import Conv3d, Linear
from torch.nn.modules.batchnorm import BatchNorm1d, BatchNorm3d
from torch.nn.modules.dropout import Dropout


def generate_grid(grid_length, grid_size, device):
    """
    The main function to create the mesh grid for voxel representation.
    equivalent with torch.stack(torch.meshgrid(), dim=-1)

    Parameters
    ----------
    grid_length: float or int
        The length of the edge of the grid box.

    grid_size: int
        number of vertices on each edge of the grid box.

    Returns
    -------
    torch.Tensor: The vertices of the grid box with shape: (G,G,G); G=grid_size

    """
    length = grid_length / 2.
    ticks = torch.linspace(-length, length, grid_size, device=device)
    x = ticks.repeat(grid_size, 1).repeat(grid_size,
                                          1).t().contiguous().view(-1)
    y = ticks.repeat(grid_size, grid_size).t().contiguous().view(-1)
    z = ticks.repeat(grid_size**2)
    grid = torch.stack([x, y, z], dim=-1).view(grid_size, grid_size, grid_size,
                                               3)
    return grid

class ConvOps(nn.Module):
    """
    Parameters
    ----------
    in_channels: int
        number of input channels

    out_channels: int
        number of output channels

    filter_size: int
        kernel size of the filter
    """
    def __init__(self, in_channels, out_channels, filter_size):
        super().__init__()
        padding_size = (filter_size - 1) // 2
        self.conv_layer = Conv3d(in_channels, out_channels, filter_size, 
                                 padding=padding_size)
        self.norm = BatchNorm3d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.norm(x)
        x = self.activation(x)
        return x

class FCOps(nn.Module):
    """
    Parameters
    ----------
    in_channels: int
        number of input channels

    out_channels: int
        number of output channels

    dropout: float
        dropout probability for the current layer
    """
    def __init__(self, in_channels, out_channels, dropout, use_batchnorm=True):
        super().__init__()
        self.fc = Linear(in_channels, out_channels)
        self.use_batchnorm = use_batchnorm
        if use_batchnorm:
            self.norm = BatchNorm1d(out_channels)
        self.activation = nn.ReLU()
        self.dropout = Dropout(dropout)

    def forward(self, x):
        x = self.fc(x)
        if self.use_batchnorm:
            x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x

def set_param(curr_mod, name, param):
    if '.' in name:
        n = name.split('.')
        module_name = n[0]
        rest = '.'.join(n[1:])
        for name, mod in curr_mod.named_children():
            if module_name == name:
                set_param(mod, rest, param)
                break
    else:
        delattr(curr_mod, name)
        setattr(curr_mod, name, param)

def set_model_parameters(model, parameter_dict):
    for item in parameter_dict:
        set_param(model, item, parameter_dict[item])
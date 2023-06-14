import torch
from torch import nn


class PadWrap(nn.Module):
    r"""
    equivalanet with numpy pad (wrap)
    Only good for the polar coordinates:
    first and second dimenstion will be padded with zero values
    third (last) dimension will be wrapped padded

    Parameters
    ----------
    pad: tuple
    dim: list or int
        the dimension of input tensor that needs to be wrapped

    """

    def __init__(self, pad, dim):
        super(PadWrap, self).__init__()

        self.padzero = nn.ConstantPad3d((0,0,pad[1],pad[1],pad[0],pad[0]), 0.0) # pad first and second dimension (front&back and top&bottom)
        self.pad = pad
        self.dim = dim

    def forward(self, x):
        """

        Parameters
        ----------
        x: torch.tensor

        Returns
        -------
        torch.tensor

        """

        if isinstance(self.dim, int):
            self.dim = [self.dim]

        for d in self.dim:
            if d >= len(x.shape):
                raise IndexError(f"dim {d} out of range")

            pad = self.pad[-1]  # forced to act only on the last dimension: can be generalized

            idx = tuple(slice(0, None if s != d else pad, 1) for s in range(len(x.shape)))
            x = torch.cat([x, x[idx]], dim=d)

            idx = tuple(slice(None if s != d else -2 * pad, None if s != d else -pad, 1) for s in range(len(x.shape)))
            x = torch.cat([x[idx], x], dim=d)

        # pad r and theta axis with zero values
        x = self.padzero(x)

        return x
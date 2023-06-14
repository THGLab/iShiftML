from torch import nn


class Crop3D(nn.Module):
    """
    This layer crops 3d box with shape (batch_size, channel_size, dim1, dim2, dim3)
    and returns new batch of 3d boxes with shape (batch_size, channel_size, ind:-ind, ind:-ind, ind:-ind), where
    ind is the number of voxels/pixels that should be removed from each side.

    Parameters
    ----------
    ind: int
        This layer does the cropping from ind to -ind

    """
    def __init__(self, ind):
        super(Crop3D, self).__init__()
        self.ind = ind

    def forward(self, x):
        """

        Parameters
        ----------
        x: torch.Tensor
            A tensor with shape (batch_size, channel_size, dim1, dim2, dim3)

        Returns
        -------
        torch.Tensor
            A tensor with shape (batch_size, channel_size, ind:-ind, ind:-ind, ind:-ind)
        """
        x = x[:, :, self.ind:-self.ind, self.ind:-self.ind, self.ind:-self.ind]

        return x

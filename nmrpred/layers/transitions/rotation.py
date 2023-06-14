import numpy as np
import torch
from torch import nn


class EulerRotation(nn.Module):
    r"""Rotate vectors with euler rotation matrix.


    Parameters
    ----------
    trainable: bool, optional (default: False)

    """
    def __init__(self, pi=None, trainable=False):
        super(EulerRotation, self).__init__()

        self.trainable = trainable
        if trainable:
            t = nn.Parameter(torch.ones(3))
            self.register_parameter('t', t)

        if pi is None:
            self.pi = torch.tensor(np.pi)
        else:
            self.pi = pi

    def forward(self, data, distance_vector=None):
        """Compute layer output.

        Parameters
        ----------
        data: BatchDataset
            input data.

        distance_vector: torch.tensor

        Returns
        -------
        torch.Tensor: layer output.

        """
        if self.trainable:
            t_clamped = self.t.tanh()  # range (-1,1)
            thetas = t_clamped * self.pi / 2
            R_x = torch.tensor([[1, 0, 0], 
                            [0, torch.cos(thetas[0]), -torch.sin(thetas[0])],
                            [0, torch.sin(thetas[0]), torch.cos(thetas[0])]])
            R_y = torch.tensor([[torch.cos(thetas[1]), 0, torch.sin(thetas[1])],
                                [0, 1, 0],
                                [-torch.sin(thetas[1]), 0, torch.cos(thetas[1])]])
            R_z = torch.tensor([[torch.cos(thetas[2]), -torch.sin(thetas[2]), 0],
                                [torch.sin(thetas[2]), torch.cos(thetas[2]), 0],
                                [0, 0, 1]])
            
            R_xy = torch.matmul(R_y, R_x)
            RM = torch.matmul(R_z, R_xy)
            
        else:
            RM = data.RM

        if distance_vector is None:
            X = torch.matmul(data.R, RM)
        else:
            X = torch.matmul(distance_vector, RM)

        return X, RM

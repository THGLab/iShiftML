import torch
from torch import nn
import numpy as np

from .cutoff import CosineCutoff, PolynomialCutoff


def gaussian_smearing_3d(distance_vector, grid, widths):
    r"""

    Parameters
    ----------
    distance_vector: torch.Tensor
        This is the :math:`X` in the gaussian equation.

    grid: torch.Tensor
        This is the :math:`\mu` in the gaussian equation.

    widths: torch.Tensor
        This is the standard deviation :math:`\sigma` of gaussian distributions.

    Returns
    -------
    torch.Tensor: the gaussian distributions with shape: B,A,N,G,G,G

    """
    diff = grid[None, None,
           None, :, :, :, :] - distance_vector[:, :, :, None,
                               None, None, :]   # B,A,N,G,G,G,3
    diff = torch.norm(diff, 2, -1)        # B,A,N,G,G,G
    widths = torch.mean(widths, dim=-1)   # G,G,G

    coeff = -0.5 / torch.pow(widths, 2)
    gauss= torch.exp(coeff * torch.pow(diff, 2))  #B,A,N,G,G,G

    return gauss


class GaussianSmearing3D(nn.Module):
    r"""Smear layer using a set of Gaussian functions.

    Parameters
    ----------
    grid_length: float, optional (default: 10)
        length of the grid box, :math:`L`.
    grid_size: int, optional (default: 20)
        total number of Gaussian functions, :math:`N_g`.
    trainable: bool, optional (default: False)
        If True, widths and offset of Gaussian functions
        are adjusted during training process.

    """

    def __init__(
        self, grid_length=10, grid_size=20, trainable=False
    ):
        super(GaussianSmearing3D, self).__init__()
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
        torch.Tensor: layer output of (B, A, N, G, G, G) shape.

        """
        return gaussian_smearing_3d(
            distance_vector, self.grid, self.widths)


class GaussianSmearing3DPolar(nn.Module):
    r"""Smear layer using a set of Gaussian functions.

    Parameters
    ----------
    cutoff: float, optional (default: 10.0)
        cutoff radius, :math:`r`.

    grid_size: tuple of int, optional (default: (20,20,30))
        total number of Gaussian functions along theta, phi, and r.

    trainable: bool, optional (default: False)
        If True, widths and offset of Gaussian functions
        are adjusted during training process.

    """

    def __init__(
        self, cutoff=10.0, grid_size=(20, 20, 30), trainable=False
    ):
        super(GaussianSmearing3DPolar, self).__init__()
        # compute offset and width of Gaussian functions
        offsets0 = torch.linspace(0, cutoff, grid_size[0])
        offsets1 = torch.linspace(0, np.pi, grid_size[1])
        offsets2 = torch.linspace(-np.pi, np.pi, grid_size[2])
        grids = torch.meshgrid([offsets0, offsets1, offsets2])
        grid = torch.stack(grids, dim=-1)   # shape: G,G,G,3

        widths0 = torch.FloatTensor((offsets0[1] - offsets0[0]) * torch.ones_like(grids[0])) # shape: G,G,G
        widths1 = torch.FloatTensor((offsets1[1] - offsets1[0]) * torch.ones_like(grids[1])) # shape: G,G,G
        widths2 = torch.FloatTensor((offsets2[1] - offsets2[0]) * torch.ones_like(grids[2])) # shape: G,G,G
        widths = torch.stack([widths0, widths1, widths2], dim=-1)   # shape: G,G,G,3

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
        torch.Tensor: layer output of (B, A, N, G, G, G) shape.

        """
        return gaussian_smearing_3d(
            distance_vector, self.grid, self.widths)


class ManyBodyVoxel(nn.Module):
    """
    The implementation of voxel representation as described by Liu et al.
    Todo: Reference

    Parameters
    ----------
    mode: str, optional(default: None)
        Here is a list of available modes:
        - 'channelize': channels for atom types and grid lengths are stacked on top of each other.
            returns a tensor of shape (B, A, C, G, G, G);
            C=len(atom_types) * len(grid_length)

        - 'embed_elemental': embedded atom types weigh channels that are stacked by grid lengths
            returns a tensor of shape (B, A, C, G, G, G);
            C=len(grid_length)

        - 'split_elemental': channels are separated by atom types while stacked by grid lengths
            returns a list of len(atom_types) tensors of shape (B, A, C, G, G, G);
            C = len(grid_length)

    atom_types: list
        list of unique atomic numbers available to channelize representations.
        Each atomic number must be an integer.

    grid_length: list or float
        The length of the edge of the grid box.
        The grid_length must be passed as a list of one or more values.

    grid_size: list or int
        number of vertices on each edge of the grid box.

    sigma: torch.tensor
    the standard deviation of gaussian distribution.

    trainable_sigma: bool, optional (default: False)
        if True, sigma will be tuned during training.

    return_atombox: bool, optional (default: False)
        if True the ruturn tensor will be of shape (B,A,N,G,G,G)
        i.e., one box per atoms in the atomic environment.
    """
    def __init__(self,
                 mode='channelize',
                 voxel_valtype='norm',
                 smearing_type='cartesian',
                 atom_types=(1,8),
                 grid_length=(6,10),
                 grid_size=16,
                 cutoff_net='cosine',
                 trainable_sigma=False,
                 return_atombox=False):

        super(ManyBodyVoxel, self).__init__()

        self.mode = mode
        self.voxel_valtype = voxel_valtype
        self.atom_types = atom_types
        self.grid_size = grid_size
        self.return_atombox = return_atombox

        if smearing_type == 'cartesian':
            self.gaussian = nn.ModuleList([
                GaussianSmearing3D(l, grid_size, trainable_sigma)
                for l in grid_length
            ])

            # cutoffs
            if cutoff_net == 'poly':
                self.cutoffs = nn.ModuleList([
                    PolynomialCutoff(0.5 * l, p=9) for l in grid_length
                ])
            elif cutoff_net=='cosine':
                self.cutoffs = nn.ModuleList([
                    CosineCutoff(0.5 * l) for l in grid_length
                ])
            elif cutoff_net is None:
                self.cutoffs = None

        elif smearing_type == 'polar':
            self.gaussian = nn.ModuleList([
                GaussianSmearing3DPolar(l, grid_size, trainable_sigma)
                for l in grid_length
            ])

            # cutoff
            if cutoff_net == 'poly':
                self.cutoffs = nn.ModuleList([
                    PolynomialCutoff(l, p=9) for l in grid_length
                    ])
            elif cutoff_net == 'cosine':
                self.cutoffs = nn.ModuleList([
                    CosineCutoff(l) for l in grid_length
                    ])
            elif cutoff_net is None:
                self.cutoffs = None

    def forward(self, distance_vector, distances, atomic_numbers, neighbors,
                neighbor_mask=None):
        """
        The main function to compute the voxel representation.

        Parameters
        ----------
        distance_vector: torch.Tensor
            A tensor of distance_vector with shape: (B, A, N, 3)

        atomic_numbers: torch.Tensor
            A tensor of atomic numbers for atoms in atomic environments.
            shape: (B, A)

        Returns
        -------
        torch.Tensor: voxel representation with shape: (A, C, G, G, G)
            where C refers to number of channels and is computed by (n_atom_types * n_grid_length)
            and G is exactly the value of grid_size parameter.
            Todo: we can support channels (C) at first or end, similar to the Conv layer.
        """
        # Construct auxiliary index vector
        n_batch = distance_vector.size()[0]
        idx_m = torch.arange(n_batch, device=distance_vector.device, dtype=torch.long)[
                :, None, None
                ]

        boxes = []
        for inet, gaussian in enumerate(self.gaussian):

            # voxel repr
            box = gaussian(distance_vector)   # B,A,N,G,G,G

            # how to give a single value to the voxels?
            # if self.voxel_valtype == 'norm':
            #     box = torch.norm(box, 2, -1)
            # elif self.voxel_valtype == 'mean':
            #     box = torch.mean(box, dim=-1)
            # elif self.voxel_valtype == 'r':
            #     box = box.select(-1, 2)

            # cutoff
            # distances = torch.norm(distance_vector, 2, -1)
            if self.cutoffs is not None:
                cutoff_network = self.cutoffs[inet]
                C = cutoff_network(distances)   # B,A,N
                box = box * C[:, :, :, None, None, None]

            if self.return_atombox:
                boxes.append(box)
            else:
                # get atomic numbers of all neighbors
                ZN = atomic_numbers[idx_m, neighbors[:, :, :]]
                ZN = ZN * neighbor_mask

                for z in self.atom_types:
                    z_box = torch.zeros_like(box)
                    z_box[ZN == z] = box[ZN == z]

                    # add neighbors visualizations
                    z_box = torch.sum(z_box, dim=2) # shape: B,A,G,G,G

                    boxes.append(z_box)

        if self.return_atombox:
            return boxes  # list of tensors with shape: B,A,N,G,G,G
        else:
            boxes = torch.stack(boxes, dim=2)  # shape: B,A,C,G,G,G
            return boxes


class RadialBesselLayer(nn.Module):
    r"""Radial Bessel functions based on the work by DimeNet team #todo: cite properly

    Args:
        n_radials (int, optional): total number of radial functions, :math:`N_g`.
        cutoff (float, optional): cutoff, :math:`\mu_{r_c}`

    """

    def __init__(
        self, n_radial=16, cutoff=5.0, device=None
    ):
        super(RadialBesselLayer, self).__init__()
        self.inv_cutoff = 1/cutoff
        self.frequencies = nn.Parameter(torch.tensor(np.arange(1,n_radial+1) * np.pi, device=device), requires_grad=False)

    def forward(self, distances):
        """Compute smeared-gaussian distance values.

        Args:
            distances (torch.Tensor): interatomic distance values of
                (N_b x N_at x N_nbh) shape.

        Returns:
            torch.Tensor: layer output of (N_b x N_at x N_nbh x N_g) shape.

        """
        d_scaled = distances * self.inv_cutoff
        d_scaled = d_scaled.unsqueeze(-1)
        out = torch.sin(self.frequencies * d_scaled)

        return out

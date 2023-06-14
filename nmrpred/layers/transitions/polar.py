import numpy as np
import torch
from torch import nn


def cartesian_to_polar(atoms):
    """
    convert 3d cartesian coordinates to 3d polar coordinates.

    Parameters
    ----------
    atoms: torch.tensor
        cartesian system with shape: (...,3)

    Returns
    -------
    torch.Tensor: layer output with the same shape as input.
        the order of coordinates in last dimension: r, theta, phi

    """
    x = atoms.select(-1, 0)
    y = atoms.select(-1, 1)
    z = atoms.select(-1, 2)

    # Sphere
    S = torch.zeros_like(atoms)

    r = torch.sqrt(torch.pow(x,2) + torch.pow(y,2) + torch.pow(z,2))
    S[..., 0] = r
    S[..., 1] = torch.acos(z/r)
    S[..., 2] = torch.atan(y/x)

    return S


def cartesian_to_polar_planar(atoms, plane='xy'):
    """
    convert 3d cartesian coordinates to 2d polar coordinates specified by axis of 2d plane.

    Parameters
    ----------
    atoms: torch.tensor
        cartesian system with shape: (...,3)

    plane: str, default: 'xy'
        Any combination of two letters from this list: ('x', 'y', 'z')

    Returns
    -------
    torch.Tensor: layer output with the same shape as input atoms.
        the order of coordinates in last dimension: r, theta

    """
    assert len(plane.strip()) == 2

    x = atoms.select(-1, 0)
    y = atoms.select(-1, 1)
    z = atoms.select(-1, 2)

    coords = {'x':x, 'y': y, 'z': z}

    a = coords[plane[0]]
    b = coords[plane[1]]

    # circle
    Cr = torch.zeros_like(atoms[...,:1])
    Ct = torch.zeros_like(atoms[...,:1])

    r = torch.sqrt(torch.pow(a,2) + torch.pow(b,2))
    Cr[..., 0] = r
    Ct[..., 0] = torch.atan(b/a)

    return Cr, Ct
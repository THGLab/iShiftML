import os
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt


def fermi_dirac(x, mu, sigma):
    """
    The Fermi-Dirac distribution with simplified hyperparameters.

    Parameters
    ----------
    x: ndarray
        In this case the array of distances (e.g., OH distances)

    mu: flaot
        equivalent to the total chemical potential

    sigma: float
        equivalent to the 1/kT
        CJS: the equilibrium bond length between two atoms

    Returns
    -------
    ndarray: The F-D distribution
    CJS: I multiplied it by 2 to stretch the distribution and have the inflection point at 1
    """
    return 2.0 / ( 1 + np.exp(sigma * (x-mu)))


def coord_num(arrays, mu, sigma):
    """
    This function compute the Fermi Dirac distributions of
    each input array and returns sum of all values.

    Parameters
    ----------
    arrays: list
        A list of 1D arrays with same shape.

    mu: float
        The Fermi-Dirac parameter equivalent to the total chemical potential

    sigma: float
        The Fermi-Dirac parameterequivalent to the 1/kT

    Returns
    -------
    ndarray: The coordination number in same shape as each of the input arrays.

    Examples
    --------
    # example based on reaction #4
    >>> import numpy as np
    >>> import pandas as pd
    >>> from scipy.spatial import distance_matrix
    >>> from nmrpred.utils import parse_reax4
    >>> from nmrpred.utils import coord_num, cn_visualize2D

    >>> data_path = "local_path_to/AIMD/04/combined/"
    >>> carts, atomic_numbers, energy, forces = parse_reax4(data_path)
    >>> carts = carts.reshape(carts.shape[0], 12)

    >>> def get_distances(atoms, n_atoms):
    >>>     atoms = atoms.reshape(n_atoms, 3)
    >>>     dist = distance_matrix(atoms, atoms, p=2)
    >>>     return dist[np.triu_indices(n_atoms, k=1)]

    >>> dist = np.apply_along_axis(get_distances, 1, carts, n_atoms=4)
    >>> dist = pd.DataFrame(dist, columns=['H1_O1', 'H1_H2', 'H1_O2', 'O1_H2', 'O1_O2', 'H2_O2'])
    >>> cn1 = coord_num([dist.H1_O1.values, dist.O1_H2.values], mu=1.0, sigma=3.0)
    >>> cn2 = coord_num([dist.H1_O2.values, dist.H2_O2.values], mu=1.0, sigma=3.0)

    >>> cn_visualize2D(cn1, cn2, "cn_figs")

    """
    cn = 0
    for array in arrays:
        cn += fermi_dirac(array, mu, sigma)

    return cn


def cn_visualize2D(cn1, cn2, output_path):
    """
    A quick 2D scatter plot with point density, mostly
    hard-coded for coordination numbers (cn).

    Parameters
    ----------
    cn1: ndarray
        1D array of the first coordination numbers.

    cn2: ndarray
        1D array of the second coordination numbers.

    output_path: str
        The full path to the output directory.
        will be created if it doesn't exist.

    """
    # Calculate the point density
    combined = np.vstack([cn1, cn2])
    z = gaussian_kde(combined)(combined)

    # visualizatoin
    fig, ax = plt.subplots()

    ax.scatter(cn1, cn2, s=5, c=z, marker='.')

    plt.xlabel('CN1 [O1-(H1,H2)]')
    plt.ylabel('CN2 [O2-(H1,H2)]')

    # grid line
    ax.set_axisbelow(True)
    ax.grid(color='gray', linestyle='dashed')

    # plt.show()

    plt.tight_layout()

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    fig.savefig(os.path.join(output_path, "cn.eps"), close=True, verbose=True)
    fig.savefig(os.path.join(output_path, "cn.png"), close=True, verbose=True)
    plt.close(fig)


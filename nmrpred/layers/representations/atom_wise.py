import torch
from torch import nn


class ShellProvider(nn.Module):
    """
    This layer calculates distance of each atom in a molecule to its
    closest neighbouring atoms.

    Parameters
    ----------

    """
    def __init__(self,
                 return_vecs=False,
                 normalize_vecs=False,
                 pbc=False,
                 cutoff=None):
        super(ShellProvider, self).__init__()
        self.return_vecs = return_vecs
        self.normalize_vecs = normalize_vecs
        self.epsilon = 1e-8
        self.pbc = pbc
        self.cutoff = cutoff

    def forward(self,
                atoms,
                neighbors,
                neighbor_mask=None,
                lattice=None):
        """
        The main driver to calculate distances of atoms in a shell from center atom.
        Based on the SchnetPack AtomDistances

        Parameters
        ----------
        atoms: torch.Tensor
            XYZ coordinates of atoms in molecules.
            shape: (B, A, 3)

        neighbors: torch.Tensor
            indices of adjacent atoms.
            shape: (B, A, N), for small systems N=A-1

        neighbor_mask: torch.tensor
            boolean mask for neighbor positions.

        lattice: torch.Tensor or None
            the pbc lattice array for current batch of data
            shape: (9,)


        Returns
        -------
        torch.Tensor: distances with shape: (B, A, N)
        torch.Tensor: distance vector with shape: (B, A, N, 3)

        Notes
        -----
        shape of tensors are specified with following symbols throughout the documentation:
            - B: batch size
            - A: max number of atoms
            - N: max number of neighbors (upper limit is A-1)

        """
        # Construct auxiliary index vector
        B, A, _ = atoms.size()
        idx_m = torch.arange(B, device=atoms.device, dtype=torch.long)[
                :, None, None
                ]

        # Get atomic positions of all neighboring indices
        ngh_atoms_xyz = atoms[idx_m, neighbors[:, :, :], :]

        # Subtract positions of central atoms to get distance vectors
        distance_vector = ngh_atoms_xyz - atoms[:, :, None, :]


        # pbc: for distance in a direction (d) and boxlength (L), d = (d + L/2) % L - L/2
        if self.pbc:
            lattice_shift_arr = torch.tensor([[i, j, k] for i in [-1, 0, 1] for j in [-1, 0, 1] for k in [-1, 0, 1]],
                                             dtype=lattice.dtype,
                                             device=lattice.device)  # 27 x 3
            lattice_batchlast = lattice.view((-1, 3, 3)).moveaxis(0, 2) # 3 x 3 x B
            distance_shift_arr = torch.tensordot(lattice_shift_arr, lattice_batchlast, 1).moveaxis(2, 1) # 27 x B x 3
            distance_vector_pbc = distance_vector[None] + distance_shift_arr[:, :, None, None] # 27 x B x A x N x 3
            distance_vector = distance_vector_pbc.moveaxis(0, -2).flatten(start_dim=2, end_dim=3) # B x A x N*27 x 3
            distances = torch.linalg.norm(distance_vector, dim=-1)

            # expand neighbor (and neighbor mask)
            neighbors = neighbors[:, :, :, None].tile((1, 1, 1, 27)).flatten(start_dim=2)  # B x A x Nx27
            if neighbor_mask is not None:
                neighbor_mask = neighbor_mask[:, :, :, None].tile((1, 1, 1, 27)).flatten(start_dim=2)
            # distance_min_idx = torch.argmin(distances_pbc, dim=0)
            # distance_min_idx_tiled = distance_min_idx[None, ..., None].tile((1, 1, 1, 1, 3))
            # distance_vector = torch.gather(distance_vector_pbc, 0, distance_min_idx_tiled).squeeze(0)
            # distances = torch.gather(distances_pbc, 0, distance_min_idx[None]).squeeze(0)


        #     x_vec = distance_vector[:, :, :, 0]
        #     x_vec = (x_vec + 0.5 * self.box[0]) % self.box[0] - 0.5 * self.box[0]
        #     y_vec = distance_vector[:, :, :, 1]
        #     y_vec = (y_vec + 0.5 * self.box[1]) % self.box[1] - 0.5 * self.box[1]
        #     z_vec = distance_vector[:, :, :, 2]
        #     z_vec = (z_vec + 0.5 * self.box[2]) % self.box[2] - 0.5 * self.box[2]
        #     distance_vector[:, :, :, 0] = x_vec
        #     distance_vector[:, :, :, 1] = y_vec
        #     distance_vector[:, :, :, 2] = z_vec
        # # print(distance_vector)
        else:
            distances = torch.norm(distance_vector, 2, 3)   # B, A, N

        if neighbor_mask is not None:
            # Avoid problems with zero distances in forces (instability of square
            # root derivative at 0) This way is neccessary, as gradients do not
            # work with inplace operations, such as e.g.
            # -> distances[mask==0] = 0.0
            tmp_dist = torch.zeros_like(distances)
            tmp_dist[neighbor_mask != 0] = distances[neighbor_mask != 0]
            distances = tmp_dist
        # print(distances)

        if self.cutoff is not None:
            # remove all neighbors beyond cutoff to save computation
            within_cutoff = distances < self.cutoff
            if neighbor_mask is not None:
                within_cutoff[neighbor_mask == 0] = False
            neighbor_counts = torch.zeros((B, A), dtype=int)
            temporal_distances = [[[] for _ in range(A)] for _ in range(B)]
            temporal_distance_vec = [[[] for _ in range(A)] for _ in range(B)]
            temporal_neighbor = [[[] for _ in range(A)] for _ in range(B)]
            temporal_neighbor_mask = [[[] for _ in range(A)] for _ in range(B)]
            for i in range(B):
                for j in range(A):
                    neighbor_count = within_cutoff[i, j].sum()
                    neighbor_counts[i, j] = neighbor_count
                    temporal_distances[i][j] = distances[i, j, within_cutoff[i, j]]
                    temporal_distance_vec[i][j] = distance_vector[i, j, within_cutoff[i, j]]
                    temporal_neighbor[i][j] = neighbors[i, j, within_cutoff[i, j]]
                    temporal_neighbor_mask[i][j] = torch.tensor([1] * neighbor_count)
            N = neighbor_counts.max()
            distances = torch.zeros((B, A, N), device=atoms.device)
            distance_vector = torch.zeros((B, A, N, 3), device=atoms.device)
            neighbors = torch.zeros((B, A, N), device=atoms.device, dtype=torch.int64)
            neighbor_mask = torch.zeros((B, A, N), device=atoms.device)
            for i in range(B):
                for j in range(A):
                    distances[i, j, :neighbor_counts[i, j]] = temporal_distances[i][j]
                    distance_vector[i, j, :neighbor_counts[i, j]] = temporal_distance_vec[i][j]
                    neighbors[i, j, :neighbor_counts[i, j]] = temporal_neighbor[i][j]
                    neighbor_mask[i, j, :neighbor_counts[i, j]] = temporal_neighbor_mask[i][j]

        if self.return_vecs:
            tmp_distances = torch.ones_like(distances)
            tmp_distances[neighbor_mask != 0] = distances[neighbor_mask != 0] + self.epsilon

            if self.normalize_vecs:
                distance_vector = distance_vector / tmp_distances[:, :, :, None]
            return distances, distance_vector, neighbors, neighbor_mask

        return distances, neighbors, neighbor_mask


def atom_distances(
    positions,
    neighbors,
    cell=None,
    cell_offsets=None,
    return_vecs=False,
    normalize_vecs=False,
    neighbor_mask=None,
):
    r"""Compute distance of every atom to its neighbors.

    This function uses advanced torch indexing to compute differentiable distances
    of every central atom to its relevant neighbors.

    Args:
        positions (torch.Tensor):
            atomic Cartesian coordinates with (N_b x N_at x 3) shape
        neighbors (torch.Tensor):
            indices of neighboring atoms to consider with (N_b x N_at x N_nbh) shape
        cell (torch.tensor, optional):
            periodic cell of (N_b x 3 x 3) shape
        cell_offsets (torch.Tensor, optional) :
            offset of atom in cell coordinates with (N_b x N_at x N_nbh x 3) shape
        return_vecs (bool, optional): if True, also returns direction vectors.
        normalize_vecs (bool, optional): if True, normalize direction vectors.
        neighbor_mask (torch.Tensor, optional): boolean mask for neighbor positions.

    Returns:
        (torch.Tensor, torch.Tensor):
            distances:
                distance of every atom to its neighbors with
                (N_b x N_at x N_nbh) shape.

            dist_vec:
                direction cosines of every atom to its
                neighbors with (N_b x N_at x N_nbh x 3) shape (optional).

    """

    # Construct auxiliary index vector
    n_batch = positions.size()[0]
    idx_m = torch.arange(n_batch, device=positions.device,
                         dtype=torch.long)[:, None, None]
    # Get atomic positions of all neighboring indices
    pos_xyz = positions[idx_m, neighbors[:, :, :], :]

    # Subtract positions of central atoms to get distance vectors
    dist_vec = pos_xyz - positions[:, :, None, :]

    # add cell offset
    if cell is not None:
        B, A, N, D = cell_offsets.size()
        cell_offsets = cell_offsets.view(B, A * N, D)
        offsets = cell_offsets.bmm(cell)
        offsets = offsets.view(B, A, N, D)
        dist_vec += offsets

    # Compute vector lengths
    distances = torch.norm(dist_vec, 2, 3)

    if neighbor_mask is not None:
        # Avoid problems with zero distances in forces (instability of square
        # root derivative at 0) This way is neccessary, as gradients do not
        # work with inplace operations, such as e.g.
        # -> distances[mask==0] = 0.0
        tmp_distances = torch.zeros_like(distances)
        tmp_distances[neighbor_mask != 0] = distances[neighbor_mask != 0]
        distances = tmp_distances

    if return_vecs:
        tmp_distances = torch.ones_like(distances)
        tmp_distances[neighbor_mask != 0] = distances[neighbor_mask != 0]

        if normalize_vecs:
            dist_vec = dist_vec / tmp_distances[:, :, :, None]
        return distances, dist_vec
    return distances


class AtomDistances(nn.Module):
    r"""Layer for computing distance of every atom to its neighbors.

    Args:
        return_directions (bool, optional): if True, the `forward` method also returns
            normalized direction vectors.

    """
    def __init__(self, return_directions=False):
        super(AtomDistances, self).__init__()
        self.return_directions = return_directions

    def forward(self,
                positions,
                neighbors,
                cell=None,
                cell_offsets=None,
                neighbor_mask=None):
        r"""Compute distance of every atom to its neighbors.

        Args:
            positions (torch.Tensor): atomic Cartesian coordinates with
                (N_b x N_at x 3) shape.
            neighbors (torch.Tensor): indices of neighboring atoms to consider
                with (N_b x N_at x N_nbh) shape.
            cell (torch.tensor, optional): periodic cell of (N_b x 3 x 3) shape.
            cell_offsets (torch.Tensor, optional): offset of atom in cell coordinates
                with (N_b x N_at x N_nbh x 3) shape.
            neighbor_mask (torch.Tensor, optional): boolean mask for neighbor
                positions. Required for the stable computation of forces in
                molecules with different sizes.

        Returns:
            torch.Tensor: layer output of (N_b x N_at x N_nbh) shape.

        """
        return atom_distances(
            positions,
            neighbors,
            cell,
            cell_offsets,
            return_vecs=self.return_directions,
            normalize_vecs=True,
            neighbor_mask=neighbor_mask,
        )


def triple_distances(
    positions,
    neighbors_j,
    neighbors_k,
    offset_idx_j=None,
    offset_idx_k=None,
    cell=None,
    cell_offsets=None,
):
    """
    Get all distances between atoms forming a triangle with the central atoms.
    Required e.g. for angular symmetry functions.

    Args:
        positions (torch.Tensor): Atomic positions
        neighbors_j (torch.Tensor): Indices of first neighbor in triangle
        neighbors_k (torch.Tensor): Indices of second neighbor in triangle
        offset_idx_j (torch.Tensor): Indices for offets of neighbors j (for PBC)
        offset_idx_k (torch.Tensor): Indices for offets of neighbors k (for PBC)
        cell (torch.tensor, optional): periodic cell of (N_b x 3 x 3) shape.
        cell_offsets (torch.Tensor, optional): offset of atom in cell coordinates
            with (N_b x N_at x N_nbh x 3) shape.

    Returns:
        torch.Tensor: Distance between central atom and neighbor j
        torch.Tensor: Distance between central atom and neighbor k
        torch.Tensor: Distance between neighbors

    """
    nbatch, _, _ = neighbors_k.size()
    idx_m = torch.arange(nbatch, device=positions.device,
                         dtype=torch.long)[:, None, None]

    pos_j = positions[idx_m, neighbors_j[:], :]
    pos_k = positions[idx_m, neighbors_k[:], :]

    if cell is not None:
        # Get the offsets into true cartesian values
        B, A, N, D = cell_offsets.size()

        cell_offsets = cell_offsets.view(B, A * N, D)
        offsets = cell_offsets.bmm(cell)
        offsets = offsets.view(B, A, N, D)

        # Get the offset values for j and k atoms
        B, A, T = offset_idx_j.size()

        # Collapse batch and atoms position for easier indexing
        offset_idx_j = offset_idx_j.view(B * A, T)
        offset_idx_k = offset_idx_k.view(B * A, T)
        offsets = offsets.view(B * A, -1, D)

        # Construct auxiliary aray for advanced indexing
        idx_offset_m = torch.arange(B * A,
                                    device=positions.device,
                                    dtype=torch.long)[:, None]

        # Restore proper dmensions
        offset_j = offsets[idx_offset_m, offset_idx_j[:]].view(B, A, T, D)
        offset_k = offsets[idx_offset_m, offset_idx_k[:]].view(B, A, T, D)

        # Add offsets
        pos_j = pos_j + offset_j
        pos_k = pos_k + offset_k

    # if positions.is_cuda:
    #    idx_m = idx_m.pin_memory().cuda(async=True)

    # Get the real positions of j and k
    R_ij = pos_j - positions[:, :, None, :]
    R_ik = pos_k - positions[:, :, None, :]
    R_jk = pos_j - pos_k

    # + 1e-9 to avoid division by zero
    r_ij = torch.norm(R_ij, 2, 3) + 1e-9
    r_ik = torch.norm(R_ik, 2, 3) + 1e-9
    r_jk = torch.norm(R_jk, 2, 3) + 1e-9

    return r_ij, r_ik, r_jk


class TriplesDistances(nn.Module):
    """
    Layer that gets all distances between atoms forming a triangle with the
    central atoms. Required e.g. for angular symmetry functions.
    """
    def __init__(self):
        super(TriplesDistances, self).__init__()

    def forward(self, positions, neighbors_j, neighbors_k):
        """
        Args:
            positions (torch.Tensor): Atomic positions
            neighbors_j (torch.Tensor): Indices of first neighbor in triangle
            neighbors_k (torch.Tensor): Indices of second neighbor in triangle

        Returns:
            torch.Tensor: Distance between central atom and neighbor j
            torch.Tensor: Distance between central atom and neighbor k
            torch.Tensor: Distance between neighbors

        """
        return triple_distances(positions, neighbors_j, neighbors_k)


def neighbor_elements(atomic_numbers, neighbors):
    """
    Return the atomic numbers associated with the neighboring atoms. Can also
    be used to gather other properties by neighbors if different atom-wise
    Tensor is passed instead of atomic_numbers.

    Args:
        atomic_numbers (torch.Tensor): Atomic numbers (Nbatch x Nat x 1)
        neighbors (torch.Tensor): Neighbor indices (Nbatch x Nat x Nneigh)

    Returns:
        torch.Tensor: Atomic numbers of neighbors (Nbatch x Nat x Nneigh)

    """
    # Get molecules in batch
    n_batch = atomic_numbers.size()[0]
    # Construct auxiliary index
    idx_m = torch.arange(n_batch,
                         device=atomic_numbers.device,
                         dtype=torch.long)[:, None, None]
    # Get neighbors via advanced indexing
    neighbor_numbers = atomic_numbers[idx_m, neighbors[:, :, :]]
    return neighbor_numbers


class NeighborElements(nn.Module):
    """
    Layer to obtain the atomic numbers associated with the neighboring atoms.
    """
    def __init__(self):
        super(NeighborElements, self).__init__()

    def forward(self, atomic_numbers, neighbors):
        """
        Args:
            atomic_numbers (torch.Tensor): Atomic numbers (Nbatch x Nat x 1)
            neighbors (torch.Tensor): Neighbor indices (Nbatch x Nat x Nneigh)

        Returns:
            torch.Tensor: Atomic numbers of neighbors (Nbatch x Nat x Nneigh)
        """
        return neighbor_elements(atomic_numbers, neighbors)

import numpy as np
from numpy import linalg as LA
import pandas as pd

default_data_ranges = {1: {'DIA': (5, 65), 'PARA': (-50, 25), 'ISO': (11, 42) }, 
                       6: {'DIA': (185, 305), 'PARA': (-560, 10), 'ISO': (-120, 221) }, 
                       7: {'DIA': (268, 358), 'PARA': (-1020, 30), 'ISO': (-390, 323) }, 
                       8: {'DIA': (335, 440), 'PARA': (-2000, 100), 'ISO': (-841, 430) }}

def calculate_eta_t_n(min_value, max_value, num_values):
    '''
    Calculate the eta and t_n values
    '''
    gap = (max_value - min_value) / (num_values - 1)
    eta = 1 / ( gap**2)
    return eta, np.array([min_value + gap * i for i in range(num_values)])

dimension_mag = {'DIA': 8, 'PARA': 8, 'ISO': 32 }
default_etas = {}
default_t_n_values = {}
for atomtype in [1, 6, 7, 8]:
    default_etas[atomtype] = {}
    default_t_n_values[atomtype] = {}
    for tensortype in ['DIA', 'PARA', 'ISO']:
        default_etas[atomtype][tensortype], default_t_n_values[atomtype][tensortype] = calculate_eta_t_n(default_data_ranges[atomtype][tensortype][0], default_data_ranges[atomtype][tensortype][1], dimension_mag[tensortype])
# print(etas)
# print(t_n_values)

class TEVCalculator:
    '''
    Calculate the Tensor Environment Variables of a molecule, just like AEVs
    Need coordinates, atom_list, tensor_vecs 
    We embed the each eigen vectors into a 4 (number of nuclei) * 4 (number of nuclei) dimensional vector
    We utilize the following facts:
    r' = R r, R is a rotation matrix, r is the original coordinates
    T' = R T R^T, R is a rotation matrix, T is the NMR shielding tensor
    Then r'^T T' r' = r^T R^T R T R^T R r = r^T T r
    '''
    def __init__(self, atom_types = [1, 6, 7, 8] , R_C = 5.2, etas=None, t_n_values = None, xi=64, theta_m_values = np.array([(2*m+1)/16*np.pi for m in range(4)])):
        self.R_C =R_C
        if etas is None:
            self.etas = default_etas
        else:
            self.etas = etas
        if t_n_values is None:
            self.t_n_values = default_t_n_values
        else:
            self.t_n_values = t_n_values
        self.xi = xi
        self.theta_m_values = theta_m_values
        self.atom_types = atom_types # H, C, N, O
        # print(self.t_n_values)
        self.dim_magnitude_1tensor = len(self.t_n_values[atom_types[0]]['DIA'])
        self.dim_direction_m = len(self.theta_m_values) 
        self.dim_direction_uv = self.dim_direction_m * len(atom_types) 
        self.dim_direction_1tensor = self.dim_direction_uv * 2
        self.dim_1tensor = self.dim_direction_1tensor  + self.dim_magnitude_1tensor
        
        self.dim_magnitude_iso = len(self.t_n_values[atom_types[0]]['ISO'])
        self.dim_total = self.dim_magnitude_iso + 54
        self.dim = 6 + self.dim_1tensor * 6  + self.dim_total
        self.atom_dict = {value: index for index, value in enumerate(atom_types)}

    def cutoff_function(self, R):
        return np.where(R <= self.R_C, (1 + np.cos(np.pi * R / self.R_C)) / 2, 0)

    def svd_keep_sign(self, array_1d):
        array = array_1d.reshape((3,3))
        u, s, vh = LA.svd(array)
        tc = np.trace(array)
        sum_s = np.sum(s)
        # print(tc, sum_s)
        # If the trace of the matrix is negative, we need to change the sign of the s and vh of SVD
        if tc * sum_s / np.abs(sum_s) < 0.0001:
            s = -s
            vh = -vh
        return u.T, s, vh 

    def calculate_tensor_svd_one_nuclei(self, tensor_i):
        u_diamag, s_diamag, vh_diamag = self.svd_keep_sign(tensor_i[:9])
        u_paramag, s_paramag, vh_paramag = self.svd_keep_sign(tensor_i[9:])
        # Here we transpose u_diamag to let the first index be the DIA1, DIA2, DIA3, PARA1, PARA2, PARA3
        return np.concatenate((s_diamag, s_paramag), axis=None), np.concatenate((u_diamag.T, u_paramag.T), axis=0), np.concatenate((vh_diamag, vh_paramag), axis=0)

    def calculate_TEVs(self, coordinates, atom_list, tensors):
        num_atoms = len(atom_list)
        TEVs = np.zeros((num_atoms, self.dim))
        atom_type_list = [self.atom_dict[atom] for atom in atom_list]
        # print(atom_type_list)

        # Calculate TEVs
        for i in range(num_atoms):
            # if i > 0:
            #     break
            # Do SVD for each tensor
            s, uh, vh= self.calculate_tensor_svd_one_nuclei(tensors[i])
            # print("s", s)

            #first to get all r_ji and R_ji to save time
            r_ji_list = coordinates - coordinates[i]

            # Normalize r_ji
            R_ji_list  = LA.norm(r_ji_list , axis=1)
            R_ji_list[i] = np.inf # set diagonal elements to inf
            r_ji_list = r_ji_list / R_ji_list[:, None]

            # use R_ji_list to store cutoff list
            R_ji_list[i] = 0 # set diagonal elements to 0
            R_ji_list = self.cutoff_function(R_ji_list)

            # Store the singular values
            TEVs[i, :6] = s

            # Embed the singular values and vectors for each of DIA1, DIA2, DIA3, PARA1, PARA2, PARA3
            etas = self.etas[atom_list[i]]
            T_n_values= self.t_n_values[atom_list[i]]
            for idx_tensor in range(6):
                off = 6 + idx_tensor * self.dim_1tensor
                # magnitude part
                if idx_tensor < 3:
                    type = 'DIA'
                else:
                    type = 'PARA'
                for n in range(self.dim_magnitude_1tensor):
                    TEVs[i, off + n] = np.exp(-etas[type] * (s[idx_tensor] - T_n_values[type][n])**2) 

                # direction part
                off += self.dim_magnitude_1tensor
                for j in range(num_atoms):
                    if j == i:
                        continue
                    cutoff_j = R_ji_list[j]

                    index_start = atom_type_list[j] * self.dim_direction_m + off
                    index_end = index_start + self.dim_direction_m

                    # embedding uh
                    abscos_theta = np.abs(np.dot(r_ji_list[j], uh[idx_tensor]))
                    abscos_theta = np.clip(abscos_theta, 0, 1)
                    theta = np.arccos(abscos_theta)
                    TEVs[i, index_start:index_end] += (1 + np.cos(theta - self.theta_m_values))**self.xi * cutoff_j
                    # print("theta", theta, "uh", uh[idx_tensor], "r_ji", r_ji_list[j], "abscos_theta", abscos_theta, "cutoff_j", cutoff_j, "TEVs", TEVs[i, index_start:index_end])

                    # embedding vh
                    abscos_theta = np.abs(np.dot(r_ji_list[j], vh[idx_tensor]))
                    abscos_theta = np.clip(abscos_theta, 0, 1)
                    theta = np.arccos(abscos_theta)
                    TEVs[i, index_start + self.dim_direction_uv : index_end + self.dim_direction_uv ] += (1 + np.cos(theta - self.theta_m_values))**self.xi * cutoff_j

                TEVs[i, off:off + self.dim_direction_1tensor] *=  2**(1-self.xi) 

            # Embed the info for entire tensor
            off = 6 + 6 * self.dim_1tensor
            iso = np.sum(s) / 3
            for n in range(self.dim_magnitude_iso):
                TEVs[i, off + n] = np.exp(-etas['ISO'] * (iso - T_n_values['ISO'][n])**2) 
            off += self.dim_magnitude_iso
            uhvh = uh @ vh.T
            uhuh = uh[:3] @ uh[3:].T
            vhvh = vh[:3] @ vh[3:].T
            TEVs[i, off:] = np.abs(np.concatenate((uhvh, uhuh, vhvh), axis=None))

        return TEVs

orca_tensor_columns = ['DIA00', 'DIA01', 'DIA02', 'DIA10', 'DIA11', 'DIA12', 'DIA20', 'DIA21', 'DIA22',
                        'PARA00', 'PARA01', 'PARA02', 'PARA10', 'PARA11', 'PARA12', 'PARA20', 'PARA21', 'PARA22']
atom_types = {'H': 1, 'C': 6, 'N': 7, 'O': 8}


class TEV_generator:
    '''
    Calculate the Tensor Environment Variables of a molecule, just like AEVs, from a low level calculation result
    '''
    def __init__(self, atom_types = [1, 6, 7, 8] , R_C = 5.2):
        self.TEV_calculator = TEVCalculator(atom_types, R_C)

    def generate_TEVs(self, low_level_df_orca):
        coordinates = np.array(low_level_df_orca[['x', 'y', 'z']].values)
        tensors = np.array(low_level_df_orca[orca_tensor_columns].values)
        atom_list = np.array(low_level_df_orca['atom_symbol'].values)
        # convert atom_list to atom type
        atom_list = [atom_types[atom] for atom in atom_list]
        TEVs = self.TEV_calculator.calculate_TEVs(coordinates, atom_list, tensors)
        return TEVs
    

if __name__ == '__main__':
    import pickle
    # with open('../local/wB97X-V_pcSseg-1_ns372.pkl', 'rb') as f:
    #     wB97XV = pickle.load(f)
    # one_mol = wB97XV['ns372_1']
    # one_mol = pd.read_csv('../../rotation/0_0_45.csv', index_col = 0)
#     print(one_mol)
#     TEV_generator = TEV_generator()
#     result = TEV_generator.generate_TEVs(one_mol)[1]

#     print('shape', result.shape)
    # print('singular value', result[:6])
    # # Print the tensor  DIA1, DIA2, DIA3, PARA1, PARA2, PARA3
    # names = ['DIA1', 'DIA2', 'DIA3', 'PARA1', 'PARA2', 'PARA3']
    # for i in range(6):
    #     print(names[i])
    #     print(result[6 + i * 80:6 + (i+1) * 80].reshape((5, 16)))
    # print('ISO')
    # print(result[486:486+32])
    # print('interation')
    # print(result[518:].reshape((6, 9)))


    # aev_h5_handle = h5py.File("../local/ns372/tev.hdf5", "w")
    # for mol_name in wB97XV: 
    #     tev = TEV_generator.generate_TEVs(wB97XV[mol_name])
    #     aev_h5_handle.create_dataset(mol_name, data=tev)
    # aev_h5_handle.close()
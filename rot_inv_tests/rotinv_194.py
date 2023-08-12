'''
Script for generating Tensor Environment Variables (TEV) of each atom from its NMR shielding diamagnetic (DIA) and paramagnetic (PARA) tensors.
194 means the vector's dimension is 98, arranged as:
[a, b, 16 interval embedding of a,  16 interval embedding of b, 32 interval embedding of a + b, 
4 x 4 (atom types) x 4 (angles between two chemical bonds) embedding of normalized DIA (direction part),
4 x 4  x 4 embedding of normalized PARA (direction part)]
, where a = trace of DIA / 3, b = trace of PARA / 3.
'''


import numpy as np
from numpy import linalg as LA
import pandas as pd
import itertools

default_data_ranges = {1: {'DIA': (13, 58), 'PARA': (-30, 15), 'ISO': (11, 42) }, 
                       6: {'DIA': (205, 280), 'PARA': (-380, 10), 'ISO': (-120, 221) }, 
                       7: {'DIA': (280, 355), 'PARA': (-690, 30), 'ISO': (-390, 323) }, 
                       8: {'DIA': (367, 421), 'PARA': (-1300, 80), 'ISO': (-841, 430) }}

def calculate_eta_t_n(min_value, max_value, num_values):
    '''
    Calculate the eta and t_n values
    '''
    gap = (max_value - min_value) / (num_values - 1)
    eta = 1 / ( gap**2)
    return eta, np.array([min_value + gap * i for i in range(num_values)])

dimension_mag = {'DIA': 16, 'PARA': 16, 'ISO': 32 }
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
    For magnitude part, we embed each tensor into different intervals, like AEV's radial part
    For direction part, we embed each tensor into a 4 (number of nuclei) * 4 (number of nuclei) * 4 (number of angles) dimensional vector
    We utilize the following facts:
    r' = R r, R is a rotation matrix, r is the original vector coordinates
    T' = R T R^T, R is a rotation matrix, T is the NMR shielding tensor
    Then r'^T T' r' = r^T R^T R T R^T R r = r^T T r
    '''
    def __init__(self, atom_types = [1, 6, 7, 8] , R_C = 5.2, etas=None, t_n_values = None, xi=8, theta_m_values = np.array([m/2*np.pi for m in range(4)])):
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
        self.dim_magnitude_iso = len(self.t_n_values[atom_types[0]]['ISO'])
        self.dim_magnitude_1tensor = len(self.t_n_values[atom_types[0]]['DIA'])
        self.dim_magnitude = self.dim_magnitude_1tensor * 2 + self.dim_magnitude_iso
        self.dim_direction_m = len(self.theta_m_values) 
        self.dim_direction_tensor = self.dim_direction_m * len(atom_types) * len(atom_types)
        self.dim_direction =self.dim_direction_tensor  * 2
        self.dim = 2 + self.dim_magnitude  + self.dim_direction
        self.atom_dict = {value: index for index, value in enumerate(atom_types)}

    def cutoff_function(self, R):
        return np.where(R <= self.R_C, (1 + np.cos(np.pi * R / self.R_C)) / 2, 0)

    def calculate_TEVs(self, coordinates, atom_list, tensors):
        num_atoms = len(atom_list)
        TEVs = np.zeros((num_atoms, self.dim))
        atom_type_list = [self.atom_dict[atom] for atom in atom_list]
        # print(atom_type_list)

        # Calculate TEVs
        for i in range(num_atoms):

            # TEV[i, a, b] = sum_{j in a} sum_{k in b} r_ji^T * PARA (or DIA) * r_ki_function(R_ji)_function(R_ki)
            #first to get all r_ji and R_ji to save time
            r_ji_list = coordinates - coordinates[i]
            # print(r_ji_list)

            # Normalize r_ji
            R_ji_list  = LA.norm(r_ji_list , axis=1)
            R_ji_list[i] = np.inf # set diagonal elements to inf
            # Normalize r_ji_list
            r_ji_list = r_ji_list / R_ji_list[:, None]

            # use R_ji_list to store cutoff list
            R_ji_list[i] = 0 # set diagonal elements to 0
            R_ji_list = self.cutoff_function(R_ji_list)
            # print(r_ji_list)
            # print(R_ji_list)

            DIA = tensors[i, :9].reshape((3,3))
            PARA = tensors[i, 9:].reshape((3,3))
            # extract PARA and DIA trace
            iso_DIA = np.trace(DIA) / 3.0
            iso_PARA = np.trace(PARA) / 3.0
            isotropic = iso_DIA + iso_PARA

            #  Calculate the magnitude tensor
            TEVs[i, 0] = iso_DIA
            TEVs[i, 1] = iso_PARA
            etas = self.etas[atom_list[i]]
            T_n_values= self.t_n_values[atom_list[i]]
            for n in range(self.dim_magnitude_iso):
                TEVs[i, 2 + n] = np.exp(-etas['ISO'] * (isotropic - T_n_values['ISO'][n])**2) 
            for n in range(self.dim_magnitude_1tensor):
                TEVs[i, 2 + self.dim_magnitude_iso + n] = np.exp(-etas['DIA'] * (iso_DIA - T_n_values['DIA'][n])**2) 
                TEVs[i, 2 + self.dim_magnitude_iso + self.dim_magnitude_1tensor + n] = np.exp(-etas['PARA'] * (iso_PARA - T_n_values['PARA'][n])**2) 

            # normalize PARA and DIA
            DIA = DIA / LA.norm(DIA, 'fro')
            PARA = PARA / LA.norm(PARA, 'fro')

            #generate j list that belongs to allowed atom types and not equal to i
            j_list = [j for j in range(num_atoms) if atom_list[j] in self.atom_types and j != i]
            jk_combinations = itertools.combinations_with_replacement(j_list, 2)
            for j, k in jk_combinations:
                # extract r_ji and r_ki, R_ji_list now stores the cutoff function
                r_ji = r_ji_list[j]
                r_ki = r_ji_list[k]

                # Calculate the angel theta between r_ji and r_ki
                cos_theta = np.dot(r_ji, r_ki)
                cos_theta = np.clip(cos_theta, -1, 1)
                theta = np.arccos(cos_theta)

                # calculate index of TEVs, offset by 2 due to the trace
                index_jk_start = (atom_type_list[j] * len(self.atom_types) + atom_type_list[k]) * self.dim_direction_m + self.dim_magnitude + 2
                index_jk_end = index_jk_start + self.dim_direction_m
                index_kj_start = (atom_type_list[k] * len(self.atom_types) + atom_type_list[j]) * self.dim_direction_m + self.dim_magnitude + 2
                index_kj_end = index_kj_start + self.dim_direction_m
                
                r_ji *= R_ji_list[j]
                r_ki *= R_ji_list[k]

                # calculate r_ji^T * DIA * r_ki
                # TEVs[i, index_jk_start:index_jk_end] += r_ki^T * DIA * r_ji * (1 + cos(2 * theta - theta_m_values))^xi, theta_m_values is an array
                TEVs[i, index_jk_start:index_jk_end] += np.dot(np.dot(r_ji, DIA), r_ki) * (1 + np.cos(2 * theta - self.theta_m_values))**self.xi
                TEVs[i, index_kj_start:index_kj_end] += np.dot(np.dot(r_ki, DIA), r_ji) * (1 + np.cos(2 * theta - self.theta_m_values))**self.xi
                # calculate r_ji^T * PARA * r_ki
                TEVs[i, self.dim_direction_tensor + index_jk_start: self.dim_direction_tensor + index_jk_end] += np.dot(np.dot(r_ji, PARA), r_ki) * (1 + np.cos(2 * theta - self.theta_m_values))**self.xi
                TEVs[i, self.dim_direction_tensor + index_kj_start: self.dim_direction_tensor + index_kj_end] += np.dot(np.dot(r_ki, PARA), r_ji) * (1 + np.cos(2 * theta - self.theta_m_values))**self.xi

                # print(j ,k, index_jk, index_kj,TEVs[i, index_jk], TEVs[i, index_kj], TEVs[i, self.dim_direction_tensor + index_jk], TEVs[i, self.dim_direction_tensor + index_kj])
        TEVs[:, 2 + self.dim_magnitude:] *=  2**(1-self.xi) 
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
    import pickle, h5py
    with open('../local/wB97X-V_pcSseg-1_ns372.pkl', 'rb') as f:
        wB97XV = pickle.load(f)
    one_mol = wB97XV['ns372_1']
    # one_mol = pd.read_csv('../../rotation/0_0_0.csv', index_col = 0)
    # print(one_mol)
    TEV_generator = TEV_generator()
    print(TEV_generator.generate_TEVs(one_mol)[1][66:])

    # aev_h5_handle = h5py.File("../local/ns372/tev.hdf5", "w")
    # for mol_name in wB97XV: 
    #     tev = TEV_generator.generate_TEVs(wB97XV[mol_name])
    #     aev_h5_handle.create_dataset(mol_name, data=tev)
    # aev_h5_handle.close()


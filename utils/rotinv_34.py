'''
Script for generating Tensor Environment Variables (TEV) of each atom from its NMR shielding diamagnetic (DIA) and paramagnetic (PARA) tensors.
34 means the vector's dimension is 34, arranged as:
[trace of DIA, trace of PARA, 4 x 4 (atom types) embedding of normalized DIA (direction part), 4 x 4 embedding of normalized PARA (direction part)]
'''


import numpy as np
from numpy import linalg as LA
import pandas as pd
import itertools

pi_over_4 = np.pi / 4
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
    def __init__(self, atom_types = [1, 6, 7, 8] , R_C = 5.2):
        self.R_C =R_C
        # self.eta = eta
        # self.xi = xi
        # self.theta_m_values = theta_m_values
        # self.R_n_values = R_n_values
        self.atom_types = atom_types # H, C, N, O
        self.dim_1tensor = len(atom_types) * len(atom_types) + 1
        self.dim = self.dim_1tensor * 2
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
            r_ji_list = r_ji_list / R_ji_list[:, None]

            # Multiply r_ji_list by cutoff function
            R_ji_list[i] = 0 # set diagonal elements to 0
            R_ji_list = self.cutoff_function(R_ji_list)
            r_ji_list = r_ji_list * R_ji_list[:, None]
            # print(r_ji_list)
            # print(R_ji_list)

            # extract PARA and DIA
            DIA = tensors[i, :9].reshape((3,3))
            PARA = tensors[i, 9:].reshape((3,3))
            # extract trace of DIA and PARA, put into TEVs 
            trace_DIA = np.trace(DIA)
            trace_PARA = np.trace(PARA)
            TEVs[i, 0] = trace_DIA
            TEVs[i, self.dim_1tensor] = trace_PARA
            # normalize
            if trace_DIA < 1e-2:
                trace_DIA = 0.0
            else:
                DIA = DIA / trace_DIA
            if trace_PARA < 1e-2:
                trace_PARA = 0.0
            else:
                PARA = PARA / trace_PARA

            #generate j list that belongs to allowed atom types and not equal to i
            j_list = [j for j in range(num_atoms) if atom_list[j] in self.atom_types and j != i]
            jk_combinations = itertools.combinations_with_replacement(j_list, 2)
            for j, k in jk_combinations:
                # extract r_ji and r_ki
                r_ji = r_ji_list[j]
                r_ki = r_ji_list[k]

                # calculate index of TEVs, offset by 1 due to the trace
                index_jk = atom_type_list[j] * len(self.atom_types) + atom_type_list[k] + 1
                index_kj = atom_type_list[k] * len(self.atom_types) + atom_type_list[j] + 1
                # calculate r_ji^T * DIA * r_ki
                TEVs[i, index_jk] += np.dot(np.dot(r_ji, DIA), r_ki)
                TEVs[i, index_kj] += np.dot(np.dot(r_ki, DIA), r_ji)
                # calculate r_ji^T * PARA * r_ki
                TEVs[i, self.dim_1tensor + index_jk] += np.dot(np.dot(r_ji, PARA), r_ki)
                TEVs[i, self.dim_1tensor + index_kj] += np.dot(np.dot(r_ki, PARA), r_ji)

                # print(j ,k, index_jk, index_kj,TEVs[i, index_jk], TEVs[i, index_kj], TEVs[i, self.dim_1tensor + index_jk], TEVs[i, self.dim_1tensor + index_kj])

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
    with open('../local/ns372/wB97X-V_pcSseg-1.pkl', 'rb') as f:
        wB97XV = pickle.load(f)
    # one_mol = wB97XV['ns372_1']
    # one_mol = pd.read_csv('../../rotation/0_0_0.csv', index_col = 0)
    # print(one_mol)
    TEV_generator = TEV_generator()
    # print(TEV_generator.generate_TEVs(one_mol)[4])

    aev_h5_handle = h5py.File("../local/ns372/tev.hdf5", "w")
    for mol_name in wB97XV: 
        tev = TEV_generator.generate_TEVs(wB97XV[mol_name])
        aev_h5_handle.create_dataset(mol_name, data=tev)
    aev_h5_handle.close()


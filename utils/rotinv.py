import numpy as np
from numpy import linalg as LA
import math
import itertools

pi_over_4 = np.pi / 4
class TEVCalculator:
    '''
    Calculate the Tensor Environment Variables of a molecule, just like AEVs
    Need coordinates, atom_list, tensor_vecs 
    Each nuclear has 6 NMR eigen vectors, 3 for paramagnetic and 3 for diamagnetic
    We embed the each eigen vectors into a 4 (number of nuclei) * 8 (number of theta) *4 (number of R_n) dimensional vector
    '''
    def __init__(self, atom_types = [1, 6, 7, 8] , R_C =3.5, eta=16, xi=32, theta_m_values = np.array([(2*m+1)/16*np.pi for m in range(8)]), R_n_values = np.array([0.90, 1.55, 2.20, 2.85])):
        self.R_C =R_C
        self.eta = eta
        self.xi = xi
        self.theta_m_values = theta_m_values
        self.R_n_values =R_n_values
        self.atom_types = atom_types # H, C, N, O
        self.dim = len(atom_types) * len(theta_m_values) * len(R_n_values) 

    def cutoff_function(self, R):
        if R <= self.R_C:
            return (1 + np.cos(np.pi * R / self.R_C)) / 2
        else:
            return 0

    def calculate_angular_AEVs(self, coordinates, atom_list, tensor_vecs):
        num_atoms = len(atom_list)
        TEVs = np.zeros((num_atoms, 6, self.dim))

        # Calculate TEVs
        for i in range(num_atoms):
            for idxa, a in enumerate(self.atom_types):
                atom_indices = [j for j in range(num_atoms) if atom_list[j] == a and j != i]
                # print(atom_indices)
                for j in atom_indices:
                    R_ij = LA.norm(coordinates[i] - coordinates[j])
                    # print("R_ij", R_ij)
                    for k in range(6):
                        dot = np.dot(coordinates[j] - coordinates[i], tensor_vecs[i,k])
                        theta_jik = np.arccos(np.abs(dot) / R_ij )
                        for n in range(4):
                            factor_from_R_ij = np.exp(-self.eta * (R_ij - self.R_n_values[n])**2) * self.cutoff_function(R_ij)
                            for m in range(8):
                                    # print(i, k, a, idxa*32 + m*4 + n, theta_jik, (1 + np.cos(theta_jik - self.theta_m_values[m]))**self.xi * factor_from_R_ij)
                                    TEVs[i, k, idxa*32 + m*4 + n] +=  (1 + np.cos(theta_jik - self.theta_m_values[m]))**self.xi * factor_from_R_ij
                                    TEVs[i, k, idxa*32 + m*4 + n] +=  (1 + np.cos(theta_jik + pi_over_4 - self.theta_m_values[m]))**self.xi * factor_from_R_ij

        # 2**(1-self.xi) is a normalization factor. multiply it to all elements in TEVs
        TEVs *= 2**(1-self.xi)
        return TEVs


class TensorEigen:
    '''
    Calculate the 6 NMR eigen vectors of each nuclear
    '''
    def __init__(self, atom_types = [1, 6, 7, 8]):
        self.atom_types = atom_types  # H, C, N, O
        self.dim = 6
    
    def eigens(self, array_1d):
        array = array_1d.reshape((3,3))
        w, v = LA.eig(array)
        return w, v.T 

    def calculate_tensor_vecs_one_nuclei(self, tensor):
        w_diamag, v_diamag = self.eigens(tensor[:9])
        w_paramag, v_paramag = self.eigens(tensor[9:])
        return np.concatenate((w_diamag, w_paramag), axis=None), np.concatenate((v_diamag, v_paramag), axis=0)

    def calculate_tensor_vecs(self, tensors):
        num_atoms = len(tensors)
        tensor_eigen_vals = np.zeros((num_atoms, self.dim))
        tensor_eigen_vecs = np.zeros((num_atoms, self.dim, 3))
        for i in range(num_atoms):
            tensor_eigen_vals[i], tensor_eigen_vecs[i] = self.calculate_tensor_vecs_one_nuclei(tensors[i])
        return tensor_eigen_vals, tensor_eigen_vecs

import pickle


orca_tensor_columns = ['DIA00', 'DIA01', 'DIA02', 'DIA10', 'DIA11', 'DIA12', 'DIA20', 'DIA21', 'DIA22',
                        'PARA00', 'PARA01', 'PARA02', 'PARA10', 'PARA11', 'PARA12', 'PARA20', 'PARA21', 'PARA22']
atom_types = {'H': 1, 'C': 6, 'N': 7, 'O': 8}

class TEV_generator:
    '''
    Calculate the Tensor Environment Variables of a molecule, just like AEVs, from a low level calculation result
    '''
    def __init__(self, atom_types = [1, 6, 7, 8] , R_C =3.5, eta=16, xi=32, theta_m_values = np.array([(2*m+1)/16*np.pi for m in range(8)]), R_n_values = np.array([0.90, 1.55, 2.20, 2.85])):
        self.TEV_calculator = TEVCalculator(atom_types, R_C, eta, xi, theta_m_values, R_n_values)
        self.tensor_eigen = TensorEigen(atom_types=atom_types)

    def generate_TEVs(self, low_level_df_orca):
        coordinates = np.array(one_mol[['x', 'y', 'z']].values)
        tensors = np.array(one_mol[orca_tensor_columns].values)
        atom_list = np.array(one_mol['atom_symbol'].values)
        # convert atom_list to atom type
        atom_list = [atom_types[atom] for atom in atom_list]
        tensor_eigen_vals, tensor_eigen_vecs = self.tensor_eigen.calculate_tensor_vecs(tensors)
        TEVs = self.TEV_calculator.calculate_angular_AEVs(coordinates, atom_list, tensor_eigen_vecs)
        # Dimension of tensor_eigen_vals is (num_atoms, 6)
        # Dimension of TEVs is (num_atoms, 6, 128)
        # Concatenate them to (num_atoms, 6, 129)
        concated = np.concatenate((tensor_eigen_vals.reshape((len(tensor_eigen_vals), 6, 1)), TEVs), axis=2)
        return concated

# with open('../../local/ns372/wB97X-V_pcSseg-1.pkl', 'rb') as f:
#     wB97XV = pickle.load(f)

# one_mol = wB97XV['ns372_0']
# TEV_generator = TEV_generator()
# print(TEV_generator.generate_TEVs(one_mol)[0])

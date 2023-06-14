'''
Script for generating the pickle files and hdf5 files for storing the geometric data for the molecules,
including all the aev vectors, all coordinate vectors and atom types
'''

import subprocess
import numpy as np
from tqdm import tqdm
import os
import pickle
from ase import io
import h5py
import sys
import multiprocessing

atom_type_mapping={"H":"1","C":"6","N":"7","O":"8"}

#save_addr = "processed_data"
save_addr = "/global/cscratch1/sd/jerryli/" 
max_atoms = 8
sample_xyz_folder = "./"

def get_aev(xyz_file, temp_name, aev_binary="./xyz_to_aev"):
    with open(xyz_file) as f1:
        with open(temp_name + ".xyz","w") as f2:
            f2.write(f1.readline())
            f1.readline()
            for line in f1:
                atom_type=line.split()[0]
                f2.write(line.replace(atom_type,atom_type_mapping[atom_type]))
                
    command = aev_binary + " " + temp_name + ".xyz" + " >" + temp_name + ".aev" 
    subprocess.call(command, shell=True)
    aev=np.genfromtxt(temp_name + ".aev" ,delimiter=",")
    return aev

def get_atomic_properties(xyz_file):
    mol = io.read(xyz_file)
    at_prop = {
        "R": mol.positions,
        "Z": mol.numbers
    }
    return at_prop

def process_single_file(inputs):
    mol_name, xyz_file = inputs
    hashed_name = str(hash(xyz_file) % ((sys.maxsize + 1) * 2)) 
    try:
        aev = get_aev(xyz_file, "/tmp/" + hashed_name)
        at_prop = get_atomic_properties(xyz_file)
        return mol_name, aev, at_prop
    except:
        print("Error in processing file: " + mol_name)
        return None

if __name__ == '__main__':
    aev_h5_handle = h5py.File(save_addr + "/aev.hdf5", "w")

    current_keys = aev_h5_handle.keys()
    all_files = []
    atomics = {}
    for n_atom in tqdm(range(1, max_atoms + 1)):
        n_atom = str(n_atom)
        for mol_id in tqdm([item.split("_")[-1] for item in os.listdir(sample_xyz_folder + f"{n_atom}_atoms")], leave=False):
            mol_dir = sample_xyz_folder + f"{n_atom}_atoms/mol_{mol_id}"
            for filename in os.listdir(mol_dir):
                if filename.endswith(".xyz"):
                    conf_id = filename.replace(".xyz", "")
                else:
                    continue

                mol_name = "_".join([n_atom, mol_id, conf_id])
                # if mol_name not in current_keys:
                all_files.append([mol_name, mol_dir + "/" + filename])
                # aev = get_aev(mol_dir + "/" + filename)
                # data_aev[mol_name] = aev

                # at_prop = get_atomic_properties(mol_dir + "/" + filename)
                # data_at_prop[mol_name] = at_prop

    pool = multiprocessing.Pool(processes=int(multiprocessing.cpu_count() * 0.9))
    pbar = tqdm(total=len(all_files))
    for result in pool.imap_unordered(process_single_file, all_files):
        if result is not None:
            mol_name, aev, at_prop = result
            aev_h5_handle.create_dataset(mol_name, data=aev)
            atomics[mol_name] = at_prop
        pbar.update(1)




    aev_h5_handle.close()

    # with open(save_addr + "/aev.pkl", "wb") as f:
    #     pickle.dump(data_aev, f)

    with open(save_addr + "/atomic.pkl", "wb") as f:
        pickle.dump(atomics, f)
            

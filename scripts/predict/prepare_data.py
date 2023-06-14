'''
Script for generating the pickle files and hdf5 files for storing the geometric data for the molecules,
including all the aev vectors, all coordinate vectors and atom types
'''

import subprocess
import numpy as np
import pandas as pd
import os
import pickle
from ase import io
import h5py
import sys
import argparse

atom_type_mapping={"H":"1","C":"6","N":"7","O":"8"}

max_atoms = 8
sample_xyz_folder = "./"

def get_aev(xyz_file, temp_name, aev_binary="../../utils/xyz_to_aev"):
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

def process_single_file(xyz_file):
    hashed_name = str(hash(xyz_file) % ((sys.maxsize + 1) * 2)) 
    try:
        aev = get_aev(xyz_file, "/tmp/" + hashed_name)
        at_prop = get_atomic_properties(xyz_file)
        return aev, at_prop
    except:
        print("Error in processing file: " + mol_name)
        sys.exit()
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("xyz_file")
    parser.add_argument("low_level_QM_calculation")
    parser.add_argument("--low_level_theory", default="wB97X-V_pcSseg-1")
    parser.add_argument("--high_level_QM_calculation", default=None, help="When provided, high level data will also be prepared")
    parser.add_argument("--high_level_theory", default="composite_high", help="Level of theory for the high level method")
    parser.add_argument("--name", default=None, help="Name of data. When not provided, infer from necessary input file names")
    parser.add_argument("--prediction_index", default=None, help="In the format of i.e. 0-8, where 8 is inclusive")
    parser.add_argument("--save_folder", default="processed_data", help="A folder to save the processed data")
    args = parser.parse_args()
    if args.name is None:
        args.name = os.path.basename(args.xyz_file).replace(".xyz", "")
    return args
    
def convert_index(args):
    df = pd.read_csv(args.low_level_QM_calculation)
    n_atoms = len(df)
    if args.prediction_index is None:
        prediction_index = np.arange(n_atoms)
    else:
        idx_start, idx_end = args.prediction_index.split("-")
        idx_start = int(idx_start)
        idx_end = int(idx_end)
        assert idx_start >= 0 and idx_end < n_atoms, "Invalid atom index range!"
        prediction_index = np.arange(idx_start, idx_end + 1)
    return prediction_index

args = parse_args()
os.makedirs(args.save_folder, exist_ok=True)
needed_indices = convert_index(args)

# process xyz files to obtain atomic environment vectors (AEVs) and atomic properties
aev, atomic_props = process_single_file(args.xyz_file)
aev_h5_handle = h5py.File(os.path.join(args.save_folder, "aev.hdf5"), "w")
aev_h5_handle.create_dataset(args.name, data=aev[needed_indices])
aev_h5_handle.close()

atomics = {}
filtered_atomic_props = {}
for key in atomic_props:
    filtered_atomic_props[key] = atomic_props[key][needed_indices]
atomics[args.name] = filtered_atomic_props
with open(os.path.join(args.save_folder, "atomic.pkl"), "wb") as f:
    pickle.dump(atomics, f)

    
# process QM calculation dataframe
qm_data = {}
df = pd.read_csv(args.low_level_QM_calculation)
df = df.iloc[needed_indices]
qm_data[args.name] = df
with open(os.path.join(args.save_folder, "{}.pkl".format(args.low_level_theory)), "wb") as f:
    pickle.dump(qm_data, f)

if args.high_level_QM_calculation is not None:
    qm_data = {}
    df = pd.read_csv(args.high_level_QM_calculation)
    df = df.iloc[needed_indices]
    qm_data[args.name] = df
    with open(os.path.join(args.save_folder, "{}.pkl".format(args.high_level_theory)), "wb") as f:
        pickle.dump(qm_data, f)

    
# write data input file
with open("predict_data.txt", "w") as f:
    f.write("test\n")
    f.write("  {}\n".format(args.name))
    
print("Finished processing", args.name)

'''
Script for generating the pickle files and hdf5 files for storing the geometric data for the molecules,
including all the aev vectors, all coordinate vectors and atom types
'''

import subprocess
import numpy as np
import pandas as pd
import os, sys
import pickle
from ase import io
import h5py
import argparse
import nmrpred
from nmrpred.utils import rotinv

aev_binary_path = os.path.join( os.path.abspath(os.path.dirname(nmrpred.__file__)), 'utils', 'xyz_to_aev')


atom_type_mapping={"H":"1","C":"6","N":"7","O":"8"}

save_addr = "./temp"
max_atoms = 8
sample_xyz_folder = "./"

def get_aev(xyz_file, temp_name, aev_binary=aev_binary_path):
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


def convert_index(args):
    df = pd.read_csv(args.low_level_QM_file)
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

def convert_index(low_level_QM_file, prediction_index =None):
    df = pd.read_csv(low_level_QM_file)
    n_atoms = len(df)
    if prediction_index is None:
        prediction_index = np.arange(n_atoms)
    else:
        idx_start, idx_end = prediction_index.split("-")
        idx_start = int(idx_start)
        idx_end = int(idx_end)
        assert idx_start >= 0 and idx_end < n_atoms, "Invalid atom index range!"
        prediction_index = np.arange(idx_start, idx_end + 1)
    return prediction_index

def xyzfile_from_low_level_QM(low_level_QM_file, save_folder, name):
    df = pd.read_csv(low_level_QM_file)
    xyzfile = os.path.join(save_folder, name + ".xyz")
    with open(xyzfile, 'w') as f:
        f.write(str(len(df)) + "\n")
        f.write(name +"\n")
        f.write(df[['atom_symbol', 'x', 'y', 'z']].to_string(header=False, index=False))
    return xyzfile

def prepare_data(low_level_QM_file, low_level_theory, without_tev = False, xyz_file=None, high_level_QM_calculation=None, high_level_theory =None, name=None, prediction_index =None, save_folder='./temp'):
    if name is None:
        name = os.path.basename(low_level_QM_file).split(".")[0]
    os.makedirs(save_folder, exist_ok=True)
    needed_indices = convert_index(low_level_QM_file, prediction_index)
    if xyz_file is None:
        xyz_file = xyzfile_from_low_level_QM(low_level_QM_file, save_folder, name)
        
    # Write Atomic environment variables
    aev, atomic_props = process_single_file(xyz_file)
    aev_h5_handle = h5py.File(os.path.join(save_folder, "aev.hdf5"), "w")
    aev_h5_handle.create_dataset(name, data=aev[needed_indices])
    aev_h5_handle.close()
    
    atomics = {}
    filtered_atomic_props = {}
    for key in atomic_props:
        filtered_atomic_props[key] = atomic_props[key][needed_indices]
    atomics[name] = filtered_atomic_props
    with open(os.path.join(save_folder, "atomic.pkl"), "wb") as f:
        pickle.dump(atomics, f)
        
    df = pd.read_csv(low_level_QM_file)
    # Write Tensor environment variables
    if not without_tev:
        TEV_generator = rotinv.TEV_generator()
        tev = TEV_generator.generate_TEVs(df)
        tev_h5_handle = h5py.File(os.path.join(save_folder, "tev.hdf5"), "w")
        tev_h5_handle.create_dataset(name, data=tev[needed_indices])
        tev_h5_handle.close()
    
    # Write low level data
    df = df.iloc[needed_indices]
    qm_data = {}
    qm_data[name] = df
    with open(os.path.join(save_folder, "{}.pkl".format(low_level_theory)), "wb") as f:
        pickle.dump(qm_data, f)
        
    # high level
    if high_level_QM_calculation is not None:
        qm_data = {}
        df = pd.read_csv(args.high_level_QM_calculation)
        df = df.iloc[needed_indices]
        qm_data[args.name] = df
        with open(os.path.join(args.save_folder, "{}.pkl".format(high_level_theory)), "wb") as f:
            pickle.dump(qm_data, f)

    # write data input file
    with open(os.path.join(save_folder,"predict_data.txt"), "w") as f:
        f.write("test\n")
        f.write("  {}\n".format(name))
    print("Finished processing", name)




if __name__ == "__main__":
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("low_level_QM_file")
        parser.add_argument("--xyz_file", default=None, help="The xyz file for the molecule")
        parser.add_argument("--without_tev", action="store_true", help="whether to calculate Tensor environment variables")
        parser.add_argument("--low_level_theory", default="wB97X-V_pcSseg-1")
        parser.add_argument("--high_level_QM_calculation", default=None, help="When provided, high level data will also be prepared")
        parser.add_argument("--high_level_theory", default="composite_high", help="Level of theory for the high level method")
        parser.add_argument("--name", default=None, help="Name of data. When not provided, infer from necessary input file names")
        parser.add_argument("--prediction_index", default=None, help="In the format of i.e. 0-8, where 8 is inclusive")
        parser.add_argument("--save_folder", default="processed_data", help="A folder to save the processed data")
        args = parser.parse_args()
        if args.name is None:
            args.name = os.path.basename(args.low_level_QM_file).split('.')[0]
        return args

    args = parse_args()
    prepare_data(args.low_level_QM_file, args.low_level_theory, args.without_tev, args.xyz_file, args.high_level_QM_calculation, args.high_level_theory, args.name, args.prediction_index, args.save_folder)
    # os.makedirs(args.save_folder, exist_ok=True)
    # needed_indices = convert_index(args)

    # # process xyz files to obtain atomic environment vectors (AEVs) and atomic properties
    # aev, atomic_props = process_single_file(args.xyz_file)
    # aev_h5_handle = h5py.File(os.path.join(args.save_folder, "aev.hdf5"), "w")
    # aev_h5_handle.create_dataset(args.name, data=aev[needed_indices])
    # aev_h5_handle.close()

    # atomics = {}
    # filtered_atomic_props = {}
    # for key in atomic_props:
    #     filtered_atomic_props[key] = atomic_props[key][needed_indices]
    # atomics[args.name] = filtered_atomic_props
    # with open(os.path.join(args.save_folder, "atomic.pkl"), "wb") as f:
    #     pickle.dump(atomics, f)

        
    # # process QM calculation dataframe
    # qm_data = {}
    # df = pd.read_csv(args.low_level_QM_file)
    # df = df.iloc[needed_indices]
    # qm_data[args.name] = df
    # with open(os.path.join(args.save_folder, "{}.pkl".format(args.low_level_theory)), "wb") as f:
    #     pickle.dump(qm_data, f)

        
    # # write data input file
    # with open("predict_data.txt", "w") as f:
    #     f.write("test\n")
    #     f.write("  {}\n".format(args.name))
        
    # print("Finished processing", args.name)

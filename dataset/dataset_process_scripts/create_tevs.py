'''
Script for generating the hdf5 file for storing the embedded NMR tensors, which are converted from the wB97X-V_pcSseg-1.pkl file
'''

import subprocess
import numpy as np
from tqdm import tqdm
import pickle
import h5py
import multiprocessing

import sys
sys.path.append('/global/cfs/cdirs/m2963/nmr_Composite/NMR_QM_jiashu/utils')
from rotinv_98 import TEV_generator

TEV_generator = TEV_generator()
atom_type_mapping={"H":"1","C":"6","N":"7","O":"8"}

data_addr = "/global/cscratch1/sd/jerryli/" 

def process_single_file(inputs):
    try:
        mol_name, df = inputs
        tev = TEV_generator.generate_TEVs(df)
        return mol_name, tev
    except:
        print("Error in processing file: " + mol_name)
        return None

if __name__ == '__main__':
    with open(data_addr + '/wB97X-V_pcSseg-1.pkl', 'rb') as f:
        wB97XV = pickle.load(f)

    print("wB97X-V_pcSseg-1.pkl read successfully!")

    tev_h5_handle = h5py.File(data_addr + "tev.hdf5", "w")

    current_keys = tev_h5_handle.keys()
    all_files = list(wB97XV.keys())

    pool = multiprocessing.Pool(processes=int(multiprocessing.cpu_count() * 0.9))
    pbar = tqdm(total=len(all_files))
    for result in pool.imap_unordered(process_single_file, list(wB97XV.items())):
        if result is not None:
            mol_name, tev = result
            tev_h5_handle.create_dataset(mol_name, data=tev)
        pbar.update(1)

    tev_h5_handle.close()

            

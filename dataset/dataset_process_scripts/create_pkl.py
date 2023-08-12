'''
Create a pickle file for all molecules up to a specified number of heavy atoms for a specific level of theory
'''
import os
import pandas as pd
import pickle
from tqdm import tqdm

##### SHOULD CHECK THESE BEFORE RUNNING #####

save_addr = "../local" 
sample_xyz_folder = "../sampled_xyzs"
n_atoms = [7]
level_of_theory = "composite_high"


############################################
    
data = {}
n_atom_str = "".join([str(n) for n in n_atoms])
for n_atom in tqdm(n_atoms):
    n_atom = str(n_atom)
    for mol_id in tqdm([item.split("_")[-1] for item in os.listdir(f"{n_atom}_atoms")], leave=False):
        nmr_dir = f"{n_atom}_atoms/mol_{mol_id}/nmr_{level_of_theory}"
        if not os.path.exists(nmr_dir):
            print("Folder does not exist!", nmr_dir)
            continue
        for filename in os.listdir(nmr_dir):
            if filename.endswith(".csv"):
                conf_id = filename.replace(".csv", "")
            else:
                continue
            mol_name = "_".join([n_atom, mol_id, conf_id])
#             name_in_selector=f"mol_{mol_id}/{conf_id}.xyz"

#             if name_in_selector not in selector and conf_id!="equilibrium":
#                 continue
            
            df=pd.read_csv(nmr_dir + f"/{conf_id}.csv")
            data[mol_name] = df
            
# with open(save_addr + "/" + level_of_theory + ".pkl", "wb") as f:
#     pickle.dump(data, f)

with open(save_addr + "/" + level_of_theory + f"_{n_atom_str}.pkl", "wb") as f:
    pickle.dump(data, f)
            
"""
Defines dataset class for all data used for NMR QM prediction
"""

import math
import pickle
from os.path import join
from sklearn.model_selection import train_test_split
import numpy as np
import h5py
from tqdm import tqdm

EFS_columns = ['EFS00', 'EFS01', 'EFS02',
            'EFS10', 'EFS11', 'EFS12', 'EFS20', 'EFS21', 'EFS22']
SOLIP_columns = ['SOLIP00', 'SOLIP01', 
            'SOLIP02', 'SOLIP10', 'SOLIP11', 'SOLIP12', 'SOLIP20', 'SOLIP21', 'SOLIP22']
SOI_columns = ['SOI00', 'SOI01', 'SOI02', 'SOI10', 'SOI11', 'SOI12', 'SOI20', 'SOI21', 'SOI22']
qchem_tensor_columns = EFS_columns + SOLIP_columns + SOI_columns
orca_tensor_columns = ['DIA00', 'DIA01', 'DIA02', 'DIA10', 'DIA11', 'DIA12', 'DIA20', 'DIA21', 'DIA22',
                        'PARA00', 'PARA01', 'PARA02', 'PARA10', 'PARA11', 'PARA12', 'PARA20', 'PARA21', 'PARA22']

default_data_ranges = {
    1: (0, 50),
    6: (-100, 250),
    7: (-300, 350),
    8: (-700, 500)
}     

def default_data_filter(input_arr, atom_type):
    # Default data filter filters data according to the expected min and max of atom type, 
    # filters all data that are exactly zero, and all nan data
    expected_min, expected_max = default_data_ranges[atom_type]
    non_nan_filter = ~np.isnan(input_arr)
    upper_filter = input_arr > expected_min
    lower_filter = input_arr < expected_max
    non_zero_filter = input_arr != 0
    final_filter = non_nan_filter & upper_filter & lower_filter & non_zero_filter
    return final_filter

def tensor_rotation(tensor,d_x,d_y,d_z, degrees=True):
    # rotate shielding tensors by d_x, d_y, d_z in degrees
    # tensor should be [:,18]

    #build the rotation matrix for xyz
    if degrees:
        d_x = d_x*np.pi/180
        d_y = d_y*np.pi/180
        d_z = d_z*np.pi/180
    sin_x = np.sin(d_x)
    cos_x = np.cos(d_x)
    sin_y = np.sin(d_y)
    cos_y = np.cos(d_y)
    sin_z = np.sin(d_z)
    cos_z = np.cos(d_z)

    rotate_x = np.matrix([[1,0,0],[0,cos_x,sin_x],[0,-sin_x,cos_x]])
    rotate_y = np.matrix([[cos_y,0,-sin_y],[0,1,0],[sin_y,0,cos_y]])
    rotate_z = np.matrix([[cos_z,sin_z,0],[-sin_z,cos_z,0],[0,0,1]])
    rotate_total = rotate_x * rotate_y * rotate_z
    
    #build the rotation matrix for shielding
    hess_tensor_rotate = np.kron(rotate_total,rotate_total).T
    # create an 18x18 matrix by putting two hess_tensor_rotate matrices together
    B = np.block([[hess_tensor_rotate, np.zeros((9, 9))], [np.zeros((9, 9)), hess_tensor_rotate]])

    # rotate in place
    np.matmul(tensor, B, out=tensor)
    
    return

class NMRData:
    def __init__(self, theory_levels, with_aev=True, with_tev=False, data_path="/home/jerry/data/NMR_QM/processed_data", quiet=True) -> None:
        with open(join(data_path, "atomic.pkl"), "rb") as f:
            self.atomic = pickle.load(f)
        file_ids = set(self.atomic)
        if with_aev:
            self.with_aev = True
            self.aev = h5py.File(join(data_path, "aev.hdf5"), "r")
            # with open(join(data_path, "aev.pkl"), "rb") as f:
            #     self.aev = pickle.load(f)
        else:
            self.with_aev = False
        if with_tev:
            self.with_tev = True
            self.tev = h5py.File(join(data_path, "tev.hdf5"), "r")
        else:
            self.with_tev = False
        self.qm_values = {}
        for theory_level in theory_levels:
            with open(join(data_path, theory_level + ".pkl"), "rb") as f:
                self.qm_values[theory_level] = pickle.load(f)
            file_ids = file_ids & set(self.qm_values[theory_level])
        self.file_ids = list(file_ids)
        self.prepared_data = {}
        self.quiet = quiet

    def assign_train_val_test(self, mode="simple", proportions=None):
        '''
        options for mode:
            simple - mix all together, then do train/val/test split. Requiring proportions to have keys of train, val, test
            id/ood - train and val in distribution, test out of distribution. id/ood based on molecules. 
                Requiring proportions to have keys of train, val, test
            resample - train, ood val, id test and ood test. 
                Requiring proportions to have keys of train, ood_val, id_test and ood_test
        '''
        if mode == "simple":
            train_val_ids, test_ids = train_test_split(self.file_ids, test_size=proportions["test"])
            train_ids, val_ids = train_test_split(train_val_ids, test_size=proportions["val"] / (proportions["train"] + proportions["val"]))
            self.splits = {"train": train_ids,
                            "val": val_ids,
                            "test": test_ids}
        else:
            if mode not in ["id/ood", "resample"]:
                raise RuntimeError("Mode has to be one of [simple], [id/ood] or [resample]")
            all_mols = set(["_".join(item.split("_")[:-1]) for item in self.file_ids])
            if mode == "id/ood":
                train_val_mols, test_mols = train_test_split(list(all_mols), test_size=proportions["test"])
                test_ids = [item for item in self.file_ids if "_".join(item.split("_")[:-1]) in test_mols]
                train_val_ids = [item for item in self.file_ids if item not in test_ids]
                train_ids, val_ids = train_test_split(train_val_ids, test_size=proportions["val"] / (proportions["train"] + proportions["val"]))
                self.splits = {"train": train_ids,
                            "val": val_ids,
                            "test": test_ids}
            else:
                ood_proportion = (proportions["ood_val"] + proportions["ood_test"]) / sum(proportions.values())
                id_mols, ood_mols = train_test_split(all_mols, ood_proportion)
                ood_val_mols, ood_test_mols = train_test_split(ood_mols, 
                    test_size=proportions["ood_test"] / (proportions["ood_val"] + proportions["ood_test"]))
                ood_val_ids = [item for item in self.file_ids if "_".join(item.split("_")[:-1]) in ood_val_mols]
                ood_test_ids = [item for item in self.file_ids if "_".join(item.split("_")[:-1]) in ood_test_mols]
                id_ids = [item for item in self.file_ids if "_".join(item.split("_")[:-1]) in id_mols]
                train_ids, id_test_ids = train_test_split(id_ids, 
                    test_size=proportions["id_test"] / (proportions["train"] + proportions["id_test"]))
                self.splits = {"train": train_ids,
                            "ood_val": ood_val_ids,
                            "id_test": id_test_ids,
                            "ood_test": ood_test_ids}

    def save_data_splitting(self, save_addr):
        "save the current data splitting into a file"
        contents = []
        for cat in sorted(self.splits):
            contents.append(cat + "\n")
            for ids in sorted(self.splits[cat]):
                contents.append("  " + ids + "\n")
        with open(save_addr, "w") as f:
            f.writelines(contents)

    def read_data_splitting(self, data_splitting_file):
        "read data splitting from a file"
        splits = {}
        current_category = ""
        with open(data_splitting_file) as f:
            for line in f:
                if len(line.strip()) > 0:
                    if line[0] != " ":
                        current_category = line.strip()
                        splits[current_category] = []
                    else:
                        splits[current_category].append(line.strip())
        self.splits = splits

    def get_all_data(self, atom=None, input_level=None, tensor_level=None, target_level=None, format="orca", combine_efs_solip=True, splitting=None, data_filter="default"):
        """
        get all data in the needed splitting corresponding to the specified atom from the dataset
        atom: which atom data to get
        tensor_level: which theory of level for the tensor features
        target_level: which theory of level for the targets
        format: whether the data is in orca or qchem format
        combine_efs_solip: whether combine EFS tensors and SOLIP tensors into a single one (only for format=qchem)
        splitting: which splitting of the data to use, when None, use all data
        """
        if (atom, input_level, tensor_level, target_level, combine_efs_solip, splitting) in self.prepared_data:
            return self.prepared_data[(atom, input_level, tensor_level, target_level, combine_efs_solip, splitting)]
        else:
            if splitting is None:
                needed_indices = self.file_ids
            else:
                needed_indices = self.splits[splitting]
            return_dict = {"R": [], "Z":[], "labels":[]}
            if not self.quiet:
                needed_indices = tqdm(needed_indices)
            for idx in needed_indices:
                atomic_filter = self.atomic[idx]['Z'] == atom
                if data_filter is None:
                    filter = atomic_filter
                else:
                    if data_filter == "default":
                        data_filter = default_data_filter
                    target_level_filter = data_filter(self.qm_values[target_level][idx][target_level][atomic_filter].values, atom)
                    if tensor_level is None:
                        tensor_level_filter = np.ones_like(target_level_filter, dtype=bool)
                    else:
                        tensor_level_filter = data_filter(self.qm_values[tensor_level][idx][tensor_level][atomic_filter].values, atom)
                    
                    filter = np.zeros_like(atomic_filter, dtype=bool)
                    atomic_indices = np.where(atomic_filter)[0]
                    filter[atomic_indices[tensor_level_filter & target_level_filter]] = True


                atomic_indices = np.where(filter)[0]
                return_dict["labels"].extend([idx + "_" + str(atom_idx) for atom_idx in atomic_indices])
                return_dict["R"].extend(self.atomic[idx]["R"][filter])
                return_dict["Z"].extend(self.atomic[idx]["Z"][filter])
                if self.with_aev:
                    if "aev" not in return_dict:
                        return_dict["aev"] = list(self.aev[idx][filter, 1:])
                    else:
                        return_dict["aev"].extend(self.aev[idx][filter, 1:])
                if self.with_tev:
                    if "tev" not in return_dict:
                        return_dict["tev"] = list(self.tev[idx][filter, :])
                    else:
                        return_dict["tev"].extend(self.tev[idx][filter, :])
                if input_level is not None:
                    input_cs_values = self.qm_values[input_level][idx][input_level].values
                    if "low_level_inputs" not in return_dict:
                        return_dict["low_level_inputs"] = list(input_cs_values[filter])
                    else:
                        return_dict["low_level_inputs"].extend(input_cs_values[filter])
                if tensor_level is not None:
                    if format == "orca":
                        tensor_features = self.qm_values[tensor_level][idx][orca_tensor_columns]
                        atomic_tensor_features = tensor_features.values[filter]
                    elif format == "qchem":
                        tensor_features = self.qm_values[tensor_level][idx][qchem_tensor_columns]
                        if combine_efs_solip:
                            efs_solip = tensor_features[EFS_columns].values + tensor_features[SOLIP_columns].values
                            soi = tensor_features[SOI_columns].values
                            atomic_tensor_features = np.hstack([efs_solip, soi])[filter]
                        else:
                            atomic_tensor_features = tensor_features.values[filter]
                    if "tensor_features" not in return_dict:
                        return_dict["tensor_features"] = list(atomic_tensor_features)
                    else:
                        return_dict["tensor_features"].extend(atomic_tensor_features)
                # for theory_level in self.qm_values:
                # targets
                if target_level is not None:
                    cs_values = self.qm_values[target_level][idx][target_level].values
                    if "targets" not in return_dict:
                        return_dict["targets"] = list(cs_values[filter])
                    else:
                        return_dict["targets"].extend(cs_values[filter])
            
            # convert all items in return_dict into numpy array
            for key in return_dict:
                if key != "labels":
                    return_dict[key] = np.array(return_dict[key])
            n_total_data = len(return_dict["R"])
            self.prepared_data[(atom, tensor_level, target_level, combine_efs_solip, splitting)] = return_dict, n_total_data
        return return_dict, n_total_data

    def get_data_generator(self, atom=None, input_level=None, tensor_level=None, target_level=None, combine_efs_solip=True, splitting=None, batch_size=128, collate_fn=None, random_rotation=True):
        "returns the data generator and the number of steps for each epoch"
        # when splitting is provided as a list, prepare all data generators
        if type(splitting) is list:
            generator_collection = {}
            for splitting_level in splitting:
                generator, n_steps = self.get_data_generator(atom, input_level, tensor_level, target_level, combine_efs_solip,
                                            splitting_level, batch_size, collate_fn)
                generator_collection[splitting_level + "_gen"] = generator
                generator_collection[splitting_level + "_steps"] = n_steps 
            return generator_collection

        # otherwise prepare just a single data generator
        if target_level is None:
            data_filter = None
        else:
            data_filter = 'default'
        data_dict, n_total_data = self.get_all_data(atom, input_level, tensor_level, target_level, 
                            combine_efs_solip=combine_efs_solip, 
                            splitting=splitting,
                            data_filter=data_filter)
        n_steps = math.ceil(n_total_data / batch_size)

        def generator():
            while 1:
                indices = np.arange(n_total_data)
                np.random.shuffle(indices)
                for step in range(n_steps):
                    batch_idx = indices[step * batch_size: (step+1) * batch_size]
                    batch = {}

                    for key in data_dict:
                        if key == "labels":
                            batch["labels"] = [data_dict[key][i] for i in batch_idx]
                        else:
                            batch[key] = data_dict[key][batch_idx]
                            
                    # generate random degrees for rotation
                    d_x = np.random.uniform(0, 2 * np.pi)
                    d_y = np.random.uniform(0, 2 * np.pi)
                    d_z = np.random.uniform(0, 2 * np.pi)
                    # apply tensor_rotation to each element in batch["tensor_features"]
                    tensor_rotation(batch["tensor_features"], d_x, d_y, d_z, degrees=False)

                    if collate_fn is not None:
                        batch = collate_fn(batch)
                    yield batch

        return generator(), n_steps


    def get_normalizer(self, atom, target_level, splitting="train"):
        """
        get the mean and standard deviations for a specific target level of theory and splitting
        """
        found_in_prepared = False
        for data_record in self.prepared_data: #[(atom, tensor_level, target_level, combine_efs_solip, splitting)]
            if data_record[0] == atom and data_record[-1] == splitting:
                found_in_prepared = True
                needed_data = self.prepared_data[data_record][0]["targets"]
        if not found_in_prepared:
            all_data = self.get_all_data(atom, None, target_level, False, splitting)[0]
            needed_data = all_data["targets"]
        mean = np.mean(needed_data)
        std = np.std(needed_data)
        data_count = len(needed_data)
        return mean, std, data_count


if __name__ == '__main__':
    data = NMRData(["B97-D_pcS-1"])
    data.assign_train_val_test(mode="id/ood", proportions={"train":0.6, "val": 0.2, "test": 0.2})
    pass
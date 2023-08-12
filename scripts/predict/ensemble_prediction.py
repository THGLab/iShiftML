import argparse
import os,sys

import numpy as np
import torch
from torch.optim import Adam
from functools import partial
from glob import glob
import pandas as pd

from nmrpred.layers import swish, shifted_softplus
from nmrpred.data.nmr_data import NMRData, default_data_filter
from nmrpred.data.loader import batch_dataset_converter
from nmrpred.train import Trainer

from functools import partial
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--low_level_QM_file", default=None, help="the low level QM calculation organized in csv format. This is to predict single molecule")
    parser.add_argument("--xyz_file", default=None, help="The xyz file for the molecule. Not needed if low_level_QM_file contains xyz info")
    parser.add_argument('-e', "--element", default="C", help="The element to predict")
    parser.add_argument("--model_path", help="The path to the models folder", type=str, default="../../models/TEV")
    parser.add_argument("--low_level_theory", default="wB97X-V_pcSseg-1")
    parser.add_argument("--target_level_theory", default="composite_high")
    parser.add_argument("--name", default=None, help="Name of data. When not provided, infer from necessary input file names")
    parser.add_argument("--scratch_folder", default="processed_data", help="A folder to save the scratch data generated in data preparation")
    parser.add_argument("--output_folder", default="local", help="A folder to save the output")
    parser.add_argument("--has_target", action="store_true", help="When the high level target data has been prepared, \
                        setting this argument to True will add the high level target data in the prediction files.")
    parser.add_argument("--include_low_level", action="store_true", help="setting this argument to True \
                        will add the low level calculations to the prediction files.")
    parser.add_argument("--batch_size", default=128, help="The batch size for prediction")
    parser.add_argument("--device", default="cpu", help="The device to use for prediction")
    parser.add_argument("--without_tev", action="store_true", help="whether the model is trained without tev. Setting this argument to True will ignore TEVs, usually used when you are using original model or data_aug model.")
    parser.add_argument("--self_trained_model", action="store_true", help="whether the model is trained by yourself. Setting this argument to True will change the model paths from model_path/element/*.pt to model_path/element/training_*/models/best_model.pt")
    parser.add_argument("--input_folder", default=None, help="The folder to store all input data. This is to get the ensemble prediction result after preparing your data. Need to be used with --split_file. Need to be used with --self_trained_model if you are using your model. Could not be used together with --low_level_QM_file.")
    parser.add_argument("--split_file", default=None, help="The file tell which molecules to predict when predicting multiple molecules")
    
    args = parser.parse_args()
    if args.name is None and args.low_level_QM_file is not None:
        args.name = os.path.basename(args.low_level_QM_file).split('.')[0]
    return args

args = parse_args()  

### Check and process inputs ###

# Check input is from input_folder or single molecule.
if args.input_folder is None and args.low_level_QM_file is None:
    raise ValueError("Please provide either input_folder or low_level_QM_file")
if args.input_folder is not None and args.low_level_QM_file is not None:
    raise ValueError("Please provide either input_folder or low_level_QM_file, not both") 

if args.input_folder is not None:
    #  If from input_folder, check if input_folder is valid
    data_path = args.input_folder
    if not os.path.exists(data_path):
        raise ValueError("Please provide a valid input_folder")
    #  Check if aev.hdf5, atomic.pkl, args.low_level_theory.pkl are in input_folder
    if not os.path.exists(os.path.join(data_path, "aev.hdf5")):
        raise ValueError("There is no aev.hdf5 in the input_folder.")
    if not os.path.exists(os.path.join(data_path, "atomic.pkl")):
        raise ValueError("There is no atomic.pkl in the input_folder.")
    if not os.path.exists(os.path.join(data_path, args.low_level_theory+".pkl")):
        raise ValueError("There is no "+args.low_level_theory+".pkl in the input_folder.")
    #If without_tev is False, check if tev.hdf5 is in input_folder
    if not args.without_tev:
        if not os.path.exists(os.path.join(data_path, "tev.hdf5")):
            raise ValueError("There is no tev.hdf5 in the input_folder. Please set --without_tev to True if you are using original model or data_aug model.")
    #  Check if split_file is provided
    if args.split_file is None:
        data_split_file = os.path.join(args.input_folder, "predict_data.txt")
    else:
        data_split_file = args.split_file
    if not os.path.exists(data_split_file):
        raise ValueError("Please provide a valid split_file. The default path is input_folder/predict_data.txt")
    
else:
    #If from single molecule, prepare data
    if not os.path.exists(args.low_level_QM_file):
        raise ValueError("Please provide a valid low_level_QM_file")
    from prepare_data import prepare_data
    prepare_data(args.low_level_QM_file, args.low_level_theory, xyz_file=args.xyz_file, without_tev= args.without_tev, save_folder=args.scratch_folder, name=args.name, prediction_index=None)
    data_split_file = os.path.join(args.scratch_folder, "predict_data.txt")
    data_path=args.scratch_folder
    #Finish preparing data

#If has_target is True, check if args.target_level_theory.pkl is provided in data_path
if args.has_target:
    if not os.path.exists(os.path.join(data_path, args.target_level_theory+".pkl")):
        raise ValueError("There is no "+args.target_level_theory+".pkl in the input_folder.")

# Check the model path is valid
atom = args.element
model_path = os.path.join(args.model_path, atom)
if not os.path.exists(args.model_path):
    raise ValueError("Please provide a valid model_path. The default path is ../../models/TEV")
if not os.path.exists(model_path):
    raise ValueError("Please provide a valid model_path of specified element. The default path is ../../models/TEV/element")

#if its a self trained model, change the model path
model_star_path = os.path.join(model_path, "*.pt")
if args.self_trained_model:
    model_star_path = os.path.join(model_path, "training_*/models/best_model.pt")


### Check and process inputs ###

torch.set_default_tensor_type(torch.FloatTensor)
ATOM_MAP = {"H": 1, "C": 6, "N": 7, "O": 8}

output_path = args.output_folder
if not os.path.exists(output_path):
    os.makedirs(output_path, exist_ok=True)

print(" Now using the model in ", model_path)
print(" Will save the output in ", output_path)

# data
if args.has_target:
    required_data = [args.low_level_theory, args.target_level_theory]
else:
    required_data = [args.low_level_theory]

data_collection = NMRData(required_data, data_path=data_path, quiet=False, with_tev=(not args.without_tev))
data_collection.read_data_splitting(data_split_file)

test_data_categories = [item for item in data_collection.splits if "test" in item]
test_data_generators = []
test_data_steps = []

if args.include_low_level:
    input_level= args.low_level_theory
else:
    input_level=None

for category in test_data_categories:   
    gen, st = data_collection.get_data_generator(atom=ATOM_MAP[atom],
                                input_level=input_level,
                                tensor_level=args.low_level_theory,
                                target_level=args.target_level_theory if args.has_target else None,
                                combine_efs_solip=False,
                                splitting=category,
                                batch_size=args.batch_size,
                                collate_fn=partial(batch_dataset_converter, device=args.device))
    test_data_generators.append(gen)
    test_data_steps.append(st)
                                
ensemble_models = []
# normalizers = []
    
for model_file in glob(model_star_path):
    model = torch.load(model_file, map_location=torch.device(args.device))
    ensemble_models.append(model.to(args.device))
    # normalizers.append(normalizer)

# define function for calculating statistics ignoring outliers
from sklearn.neighbors import LocalOutlierFactor
outlier_detector=LocalOutlierFactor(n_neighbors=3)
def non_outlier_stats(inputs):
    outlier_filter=outlier_detector.fit_predict(inputs.reshape(-1,1))
    return inputs[outlier_filter==1].mean(),inputs[outlier_filter==1].std()

# predicting
for category, data_gen, steps in zip(test_data_categories, test_data_generators, test_data_steps):
    preds = []
    low_level_cs = []
    targets = []
    labels = []
    for step in tqdm(range(steps)):
        batch = next(data_gen)
        if args.include_low_level:
            low_level_cs.append(batch["low_level_inputs"].detach().cpu().numpy())
        if args.has_target:
            targets.append(batch["targets"].detach().cpu().numpy())
        labels.extend(batch["labels"])
        batch_ensemble_preds = []
        for model in ensemble_models:
            with torch.no_grad():
                pred = model(batch).detach().cpu().numpy()
            batch_ensemble_preds.append(pred)
        preds.append(np.array(batch_ensemble_preds).T)
    preds = np.concatenate(preds, axis=0)
    if args.has_target:
        targets = np.concatenate(targets, axis=0)
    if args.include_low_level:
        low_level_cs = np.concatenate(low_level_cs, axis=0)
    preds_means, preds_stds = [], []
    for item in preds:
        mean, std = non_outlier_stats(item)
        preds_means.append(mean)
        preds_stds.append(std)
    df = pd.DataFrame(preds, columns=["preds_{}".format(i) for i in range(preds.shape[1])], index=labels)
    if args.include_low_level:
        df["low_level_inputs"] = low_level_cs
    if args.has_target:
        df["target"] = targets
    df["preds_mean"] = preds_means
    df["preds_std"] = preds_stds
    if args.input_folder is not None:
        df.to_csv(os.path.join(output_path, "ensemble_prediction_{}_{}.csv".format(atom, category)))
    else:
        df.to_csv(os.path.join(output_path, "{}_{}_{}.csv".format(args.name, atom, category)))

print("All finished!")

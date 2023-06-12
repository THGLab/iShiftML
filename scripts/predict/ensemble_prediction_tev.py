import os
import sys
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

from nmrpred.models.MLP import AEVMLP
from nmrpred.models.metamodels import Attention
from nmrpred.models.decoders import AttentionMask
from functools import partial
from tqdm import tqdm

import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("low_level_QM_file")
    parser.add_argument("--xyz_file", default=None, help="The xyz file for the molecule")
    parser.add_argument('-e', "--element", default="C", help="The element to predict")
    parser.add_argument("--model_path", help="The path to the models folder", type=str, default="/global/cfs/cdirs/m2963/nmr_Composite/NMR_QM_jiashu/scripts/active_learning/7")
    parser.add_argument("--low_level_theory", default="wB97X-V_pcSseg-1")
    parser.add_argument("--target_level_theory", default="composite_high")
    parser.add_argument("--name", default=None, help="Name of data. When not provided, infer from necessary input file names")
    parser.add_argument("--prediction_index", default=None, help="In the format of i.e. 0-8, where 8 is inclusive")
    parser.add_argument("--scratch_folder", default="temp", help="A folder to save the processed data")
    parser.add_argument("--output_folder", default="local", help="A folder to save the output")
    parser.add_argument("--has_target", action="store_true", help="When the high level target data has been prepared, \
                        setting this argument to True will add the high level target data in the prediction files.")
    parser.add_argument("--include_low_level", action="store_true", help="setting this argument to True \
                        will add the low level calculations to the prediction files.")
    parser.add_argument("--batch_size", default=128, help="The batch size for prediction")
    parser.add_argument("--device", default="cpu", help="The device to use for prediction")
    parser.add_argument("--with_tev", action="store_true", help="whether the model is trained with tev. Setting this argument to True will calculate TEVs")
    
    args = parser.parse_args()
    if args.name is None:
        args.name = os.path.basename(args.low_level_QM_file).split('.')[0]
    return args

# model_path = f"/global/cfs/cdirs/m2963/nmr_Composite/NMR_QM_jiashu/scripts/active_learning/7/{atom}"

args = parse_args()   

### Now preparing the data
from prepare_data import prepare_data
prepare_data(args.low_level_QM_file, args.low_level_theory, xyz_file=args.xyz_file, need_tev= args.with_tev, save_folder=args.scratch_folder, name=args.name, prediction_index=args.prediction_index)
data_split_file = os.path.join(args.scratch_folder, "predict_data.txt")
data_path=args.scratch_folder
#Finish preparing data


torch.set_default_tensor_type(torch.FloatTensor)
ATOM_MAP = {"H": 1, "C": 6, "N": 7, "O": 8}

atom = args.element
model_path = os.path.join(args.model_path, atom)
output_path = os.path.join(args.output_folder, atom)
if not os.path.exists(output_path):
    os.makedirs(output_path, exist_ok=True)

print(" Now using the model in ", model_path)
print(" Will save the output in ", output_path)

# data
if args.has_target:
    required_data = [args.low_level_theory, args.target_level_theory]
else:
    required_data = [args.low_level_theory]

data_collection = NMRData(required_data, data_path=data_path, quiet=False, with_tev=args.with_tev)
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
for model_file in glob(os.path.join(model_path, "training_*/models/best_model.pt")):
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
    df.to_csv(os.path.join(output_path, "{}_{}_{}.csv".format(args.name, atom, category)))

print("All finished!")

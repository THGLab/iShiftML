import os
import numpy as np
import torch
from functools import partial
from glob import glob
import pandas as pd

from nmrpred.data.nmr_data import NMRData
from nmrpred.data.loader import batch_dataset_converter
from functools import partial
from tqdm import tqdm
import argparse


torch.set_default_tensor_type(torch.FloatTensor)
ATOM_MAP = {"H": 1, "C": 6, "N": 7, "O": 8}
input_lot = "wB97X-V_pcSseg-1" # input level of theory
target_lot = "composite_high" # target level of theory

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("atom", help="The element to be predicted. One of H/C/N/O.")
    parser.add_argument("--output_path", default="./local/", help="Path to save the output files. \
                         The prediction files will be named as 'ensemble_prediction_{atom}_{category}.csv'. ")
    parser.add_argument("--model_path", default="../../models/", help="Path to the trained models.")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size for prediction.")
    parser.add_argument("--prediction_data", default="predict_data.txt", help="Path to the file specifying data to be predicted.")
    parser.add_argument("--data_root", default="./processed_data", help="Folder containing necessary data for prediction.")
    parser.add_argument("--has_target", action="store_true", help="When the high level target data has been prepared, \
                        setting this argument to True will add the high level target data in the prediction files.")
    parser.add_argument("--include_low_level", action="store_true", help="setting this argument to True \
                        will add the low level calculations to the prediction files.")
    return parser.parse_args()

args = parse_args()

atom = args.atom
has_target = args.has_target
include_low_level = args.include_low_level
data_root = args.data_root
data_split_file = args.prediction_data
model_path = os.path.join(args.model_path, atom)
output_path = args.output_path
batch_size = args.batch_size

if not os.path.exists(output_path):
    os.makedirs(output_path, exist_ok=True)


# device
device = ["cpu"]

# data

if has_target:
    required_data = [input_lot, target_lot]
else:
    required_data = [input_lot]
data_collection = NMRData(required_data, data_path=data_root, quiet=False)
data_collection.read_data_splitting(data_split_file)

test_data_categories = [item for item in data_collection.splits if "test" in item]
test_data_generators = []
test_data_steps = []

if include_low_level:
    input_level=input_lot
else:
    input_level=None

for category in test_data_categories:   
    gen, st = data_collection.get_data_generator(atom=ATOM_MAP[atom],
                                input_level=input_level,
                                tensor_level=input_lot,
                                target_level=target_lot if has_target else None,
                                splitting=category,
                                batch_size=batch_size,
                                collate_fn=partial(batch_dataset_converter, device=device[0]))
    test_data_generators.append(gen)
    test_data_steps.append(st)
                                
ensemble_models = []
# normalizers = []
for model_file in glob(os.path.join(model_path, "model*.pt")):
    model = torch.load(model_file, map_location=torch.device('cpu'))
    ensemble_models.append(model.to(device[0]))
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
        if include_low_level:
            low_level_cs.append(batch["low_level_inputs"].detach().cpu().numpy())
        if has_target:
            targets.append(batch["targets"].detach().cpu().numpy())
        labels.extend(batch["labels"])
        batch_ensemble_preds = []
        for model in ensemble_models:
            with torch.no_grad():
                pred = model(batch).detach().cpu().numpy()
            batch_ensemble_preds.append(pred)
        preds.append(np.array(batch_ensemble_preds).T)
    preds = np.concatenate(preds, axis=0)
    if has_target:
        targets = np.concatenate(targets, axis=0)
    if include_low_level:
        low_level_cs = np.concatenate(low_level_cs, axis=0)
    preds_means, preds_stds = [], []
    for item in preds:
        mean, std = non_outlier_stats(item)
        preds_means.append(mean)
        preds_stds.append(std)
    df = pd.DataFrame(preds, columns=["preds_{}".format(i) for i in range(preds.shape[1])], index=labels)
    if include_low_level:
        df["low_level_inputs"] = low_level_cs
    if has_target:
        df["target"] = targets
    df["preds_mean"] = preds_means
    df["preds_std"] = preds_stds
    df.to_csv(os.path.join(output_path, "ensemble_prediction_{}_{}.csv".format(atom, category)))

print("All finished!")

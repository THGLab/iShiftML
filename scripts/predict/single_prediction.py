import os
import sys
import numpy as np
import torch
from functools import partial
import pandas as pd

from nmrpred.data.nmr_data import NMRData
from nmrpred.data.loader import batch_dataset_converter

from functools import partial
from tqdm import tqdm


torch.set_default_tensor_type(torch.FloatTensor)
ATOM_MAP = {"H": 1, "C": 6, "N": 7, "O": 8}

atom = "H"

if len(sys.argv) > 1:
    atom = sys.argv[1]

### settings ###
has_target = True
data_split_file = "./predict_data.txt" 
model_path = f"../../models/H/model0.pt"
data_root = "./processed_data"
batch_size = 128
output_path = "./local/"
output_name = "single_prediction_{}.csv".format(atom)
input_lot = "wB97X-V_pcSseg-1" # input level of theory
target_lot = "composite_high" # target level of theory
################

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
for category in test_data_categories:   
    gen, st = data_collection.get_data_generator(atom=ATOM_MAP[atom],
                                tensor_level=input_lot,
                                target_level=target_lot if has_target else None,
                                splitting=category,
                                batch_size=batch_size,
                                collate_fn=partial(batch_dataset_converter, device=device[0]))
    test_data_generators.append(gen)
    test_data_steps.append(st)
    
model = torch.load(model_path, map_location=torch.device('cpu'))
     
# predicting
for category, data_gen, steps in zip(test_data_categories, test_data_generators, test_data_steps):
    preds = []
    targets = []
    labels = []
    for step in tqdm(range(steps)):
        batch = next(data_gen)
        if has_target:
            targets.append(batch["targets"].detach().cpu().numpy())
        labels.extend(batch["labels"])
        with torch.no_grad():
            batch_pred = model(batch).detach().cpu().numpy()
        preds.append(batch_pred)
    preds = np.concatenate(preds, axis=0)

    df = pd.DataFrame(preds, columns=["pred"], index=labels)
    if has_target:
        targets=np.concatenate(targets, axis=0)
        df["target"] = targets
    df.to_csv(os.path.join(output_path, output_name))

print("All finished!")

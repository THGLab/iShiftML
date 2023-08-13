import os, sys
import numpy as np
import torch
from torch.optim import Adam, SGD
import yaml
from functools import partial


from nmrpred.data.loader import batch_dataset_converter
from nmrpred.utils.get_gpu import handle_gpu
from nmrpred.train import Trainer

from nmrpred.data.nmr_data import NMRData
from nmrpred.models.MLP import AEVMLP
from nmrpred.models.metamodels import Attention
from nmrpred.models.decoders import AttentionMask
from functools import partial


torch.set_default_tensor_type(torch.FloatTensor)

# first read in settings and fix random number seeds 
settings_path = 'config.yml'
if len(sys.argv) > 1:
    settings_path = sys.argv[-1]
settings = yaml.safe_load(open(settings_path, "r"))
seed=settings["general"]["seed"]
np.random.seed(seed)
torch.manual_seed(seed)
print("Using random seed", seed)

# device
if type(settings['general']['device']) is list:
    device = [torch.device(item) for item in settings['general']['device']]
elif settings['general']['device'] == "auto":
    device = [handle_gpu()]
else:
    device = [torch.device(settings['general']['device'])]

# data
data_collection = NMRData([settings['data']['input_lot'], settings['data']['target_lot']], data_path=settings['data']['root'])
# Check if the data is already split by checking if the splitting is a string
if type(settings['data']['splitting']) is str:
    data_collection.read_data_splitting(settings['data']['splitting'])
else:
    # If the splitting is not a string, then it is a list of proportions
    split_list = settings['data']['splitting']
    split_list =np.array(split_list)
    split_list /= np.sum(split_list)
    data_collection.assign_train_val_test(mode="simple", proportions={"train":split_list[0], "val":split_list[1], "test":split_list[2]})

generators = data_collection.get_data_generator(atom=settings['data']['shift_types'],
                                input_level=settings['data']['input_lot'],
                                tensor_level=settings['data']['input_lot'],
                                target_level=settings['data']['target_lot'],
                                splitting=list(data_collection.splits),
                                batch_size=settings['training']['batch_size'],
                                collate_fn=partial(batch_dataset_converter, device=device[0]),
                                random_rotation=settings['training']['random_rotation'])
                                
# data_collection = parse_nmr_data_aev(settings,device[0])
print('normalizer: ', data_collection.get_normalizer(atom=settings['data']['shift_types'],
                                    target_level=settings['data']['target_lot']))
# print('target hash:', data_collection.hash)

# model

dropout = settings['training']['dropout']
AEV_outdim = settings['model']['AEV_outdim']
feature_extractor = AEVMLP([384, 128, AEV_outdim], dropout)
if settings['data'].get("combine_efs_solip", True):
    attention_input_dim = AEV_outdim + 18
    attention_output_dim = 19
else:
    attention_input_dim = AEV_outdim + 27
    attention_output_dim = 28
attention_mask_network = AttentionMask([attention_input_dim, AEV_outdim, attention_output_dim], dropout)
model = Attention(feature_extractor, attention_mask_network)



# optimizer
trainable_params = filter(lambda p: p.requires_grad, model.parameters())
optimizer = Adam(trainable_params,
                 lr=settings['training']['lr'],
                 weight_decay=settings['training']['weight_decay'])


# scaled loss fn
def mse_loss(preds, batch, sample_weights=None):
    targets = batch["targets"]
    mse = torch.mean((preds - targets) ** 2)
    return mse

# training

test_names = [item for item in data_collection.splits if "test" in item]
trainer = Trainer(model=model,
                  loss_fn=mse_loss,
                  optimizer=optimizer,
                  requires_dr=False,
                  device=device,
                  normalizer=None,
                  yml_path=settings['general']['me'],
                  output_path=settings['general']['output'],
                  script_name=os.path.basename(__file__),
                  lr_scheduler=settings['training']['lr_scheduler'],
                  checkpoint_log=settings['checkpoint']['log'],
                  checkpoint_val=settings['checkpoint']['val'],
                  checkpoint_test=settings['checkpoint']['test'],
                  verbose=settings['checkpoint']['verbose'],
                  preempt=settings['training']['allow_preempt'],
                  test_names=test_names)

trainer.print_layers()
# trainer.log_statistics(data_collection)

# tr_steps=10; val_steps=10; test_steps=10



trainer.train(train_generator=generators["train_gen"],
              epochs=settings['training']['epochs'],
              steps=generators["train_steps"],
              val_generator=generators["val_gen"],
              val_steps=generators["val_steps"],
              test_generators=[generators[test_name + "_gen"] for test_name in test_names],
              test_steps=[generators[test_name + "_steps"] for test_name in test_names],
              err_inclusion_fn=lambda x: x < 30)   # maximum considered error: N-30

print('done!')
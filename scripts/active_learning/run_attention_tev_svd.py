import os
import sys
import yaml
import numpy as np
import torch
# first read in settings and fix random number seeds 
settings_path = 'config.yml'
if len(sys.argv) > 1:
    settings_path = sys.argv[-1]
settings = yaml.safe_load(open(settings_path, "r"))
seed=settings["general"]["seed"]
np.random.seed(seed)
torch.manual_seed(seed)
print("Using random seed", seed)



from torch.optim import Adam, SGD

from functools import partial

from nmrpred.layers import swish, shifted_softplus
from nmrpred.data.nmr_data import NMRData, default_data_filter
from nmrpred.data.loader import batch_dataset_converter
from nmrpred.utils.get_gpu import handle_gpu
from nmrpred.train import Trainer

from nmrpred.models.MLP import AEVMLP
from nmrpred.models.metamodels import Attention_TEV_SVD
from nmrpred.models.decoders import AttentionMask
from functools import partial



# device
if type(settings['general']['device']) is list:
    device = [torch.device(item) for item in settings['general']['device']]
else:
    device = [torch.device(settings['general']['device'])]

# data
data_collection = NMRData([settings['data']['input_lot'], settings['data']['target_lot']], data_path=settings['data']['root'], with_tev=True)
data_collection.read_data_splitting(settings['data']['splitting'])
# data_collection.assign_train_val_test(mode="simple", proportions={"train":0.8, "val":0.1, "test":0.1})
generators = data_collection.get_data_generator(atom=settings['data']['shift_types'],
                                input_level=settings['data']['input_lot'],
                                tensor_level=settings['data']['input_lot'],
                                target_level=settings['data']['target_lot'],
                                splitting=list(data_collection.splits),
                                batch_size=settings['training']['batch_size'],
                                collate_fn=partial(batch_dataset_converter, device=device[0]))

# data_collection = parse_nmr_data_aev(settings,device[0])
print('normalizer: ', data_collection.get_normalizer(atom=settings['data']['shift_types'],
                                    target_level=settings['data']['target_lot']))
# print('target hash:', data_collection.hash)

# model
AEV_output_dim = settings['model']['AEV_outdim']
dropout = settings['training']['dropout']
feature_extractor = AEVMLP([384, 128, AEV_output_dim], dropout)
with_low_level_inputs = settings['model'].get('with_low_level_inputs', False)

dim_1tensor = 80
dim_overall = 86
weight_input_dim = AEV_output_dim + dim_1tensor*3 + dim_overall
bias_input_dim = AEV_output_dim + dim_1tensor*2 + dim_overall
attention_output_dim = 1

DIA_attention_net = AttentionMask([weight_input_dim, 128, 64, 16, attention_output_dim], dropout)
PARA_attention_net = AttentionMask([weight_input_dim, 128, 64, 16, attention_output_dim], dropout)
Bias_attention_net = AttentionMask([bias_input_dim, 128, 64, 16, attention_output_dim], dropout)
model = Attention_TEV_SVD(feature_extractor, DIA_attention_net, PARA_attention_net, Bias_attention_net, dim_1tensor = dim_1tensor, with_low_level_input=with_low_level_inputs)


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

import os
import sys
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

import nni


torch.set_default_tensor_type(torch.FloatTensor)

# settings
settings_path = 'config.yml'
if len(sys.argv) > 1:
    settings_path = sys.argv[-1]
settings = yaml.safe_load(open(settings_path, "r"))

# NNI hyperparameter tuning
tuned_params = nni.get_next_parameter()
settings["model"]["network_dim"] = tuned_params["network_dim"]
settings["training"]["batch_size"] = tuned_params["bs"]
settings["training"]["lr"] = tuned_params["lr"]
settings["training"]["lr_scheduler"][-2] = tuned_params["lr_decay"]
settings["training"]["weight_decay"] = tuned_params["weight_decay"]
settings["training"]["optimizer"] = tuned_params["optimizer"]
settings["training"]["dropout"] = tuned_params["dropout"]
settings["training"]["momentum"] = tuned_params["momentum"]

# device
if type(settings['general']['device']) is list:
    device = [torch.device(item) for item in settings['general']['device']]
elif settings['general']['device'] == "auto":
    device = [handle_gpu()]
else:
    device = [torch.device(settings['general']['device'])]

# data
data_collection = NMRData([settings['data']['input_lot'], settings['data']['target_lot']], data_path=settings['data']['root'])
data_collection.read_data_splitting(settings['data']['splitting'])
generators = data_collection.get_data_generator(atom=settings['data']['shift_types'],
                                tensor_level=settings['data']['input_lot'],
                                target_level=settings['data']['target_lot'],
                                combine_efs_solip=settings['data']['combine_efs_solip'],
                                splitting=list(data_collection.splits),
                                batch_size=settings['training']['batch_size'],
                                collate_fn=partial(batch_dataset_converter, device=device[0]))
                                
# data_collection = parse_nmr_data_aev(settings,device[0])
print('normalizer: ', data_collection.get_normalizer(atom=settings['data']['shift_types'],
                                    target_level=settings['data']['target_lot']))
# print('target hash:', data_collection.hash)

# model

dropout = settings['training']['dropout']
network_dim = settings['model']['network_dim']
feature_extractor = AEVMLP([384, 128, network_dim], dropout)
if settings['data'].get("combine_efs_solip", True):
    attention_input_dim = network_dim + 18
    attention_output_dim = 19
else:
    attention_input_dim = network_dim + 27
    attention_output_dim = 28
attention_mask_network = AttentionMask([attention_input_dim, network_dim, attention_output_dim], dropout)
model = Attention(feature_extractor, attention_mask_network)



# optimizer
trainable_params = filter(lambda p: p.requires_grad, model.parameters())
if settings['training']['optimizer'] == 'Adam':
    optimizer = Adam(trainable_params,
                    lr=settings['training']['lr'],
                    weight_decay=settings['training']['weight_decay'])
elif settings['training']['optimizer'] == 'SGD':
    optimizer = SGD(trainable_params,
                    lr=settings['training']['lr'],
                    weight_decay=settings['training']['weight_decay'],
                    momentum=settings['training']['momentum'])


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
                  checkpoint_model=settings['checkpoint']['model'],
                  verbose=settings['checkpoint']['verbose'],
                  preempt=settings['training']['allow_preempt'],
                  test_names=test_names,
                  save_validation_data=True,
                  nni_module=nni)

trainer.print_layers()
# trainer.log_statistics(data_collection)

# tr_steps=10; val_steps=10; test_steps=10



best_error = trainer.train(train_generator=generators["train_gen"],
              epochs=settings['training']['epochs'],
              steps=generators["train_steps"],
              val_generator=generators["val_gen"],
              val_steps=generators["val_steps"],
              test_generators=[generators[test_name + "_gen"] for test_name in test_names],
              test_steps=[generators[test_name + "_steps"] for test_name in test_names],
              err_inclusion_fn=lambda x: x < 30)   # maximum considered error: N-30

nni.report_final_result(best_error)
print('done!')

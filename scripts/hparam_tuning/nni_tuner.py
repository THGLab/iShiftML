"""
NNI hyperparameter optimization example.

Check the online tutorial for details:
https://nni.readthedocs.io/en/stable/tutorials/hpo_quickstart_pytorch/main.html
"""

from pathlib import Path
import signal

from nni.experiment import Experiment

# Define search space
search_space = {
    'network_dim': {'_type': 'choice', '_value': [32, 64, 128, 256]},
    'bs': {'_type': 'choice', '_value': [32, 64, 128, 256, 512]},
    'lr': {'_type': 'loguniform', '_value': [5e-5, 0.05]},
    'lr_decay': {'_type': 'uniform', '_value': [0.1, 0.9]},
    'weight_decay': {'_type': 'loguniform', '_value': [1e-6, 1e-2]},
    'dropout': {'_type': 'uniform', '_value': [0, 0.75]},
    'optimizer': {'_type': 'choice', '_value': ['Adam', 'SGD']},
    'momentum': {'_type': 'uniform', '_value': [0, 1]},
}

# Configure experiment
experiment = Experiment('local')
experiment.config.trial_command = 'python run_attention_aev.py'
experiment.config.trial_code_directory = Path(__file__).parent
experiment.config.search_space = search_space
experiment.config.tuner.name = 'Anneal'
experiment.config.max_trial_number = 128
experiment.config.trial_concurrency = 8

# Run it!
experiment.run(port=8080, wait_completion=False)

print('Experiment is running. Press Ctrl-C to quit.')
signal.pause()

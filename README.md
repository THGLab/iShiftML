# iShiftML: Highly Accurate Prediction of NMR Chemical Shifts from Low-Level Quantum Mechanics Calculations Using Machine Learning

## Authors 
* Jie Li `<jerry-li1996@hotmail.com>`
* Jiashu Liang `<jsliang@berkeley.edu>`
* Zhe Wang `<wang_j@berkeley.edu>`
* Aleksandra Ptaszek `<aleksandra.ptaszek@berkeley.edu>`
* Xiao Liu `<xiao_liu@berkeley.edu>`
* Brad Ganoe `<ganoe@berkeley.edu>`
* Martin Head-Gordon `<mhg@cchem.berkeley.edu>`
* Teresa Head-Gordon `<thg@berkeley.edu>`

## Installation and Dependencies
We recommend using conda environment to install dependencies of this library.
Please install (or load) conda and then proceed with the following commands:

    conda create --name iShiftML python=3.8
    conda activate iShiftML
    conda install -c conda-forge numpy scipy pandas pyyaml scikit-learn ase
    conda install -c pytorch pytorch cudatoolkit=10.1 
    pip install tqdm tensorboard h5py==2.10
    
The developer installation is available and for that you need to first clone iShiftML from this repository:

    git clone https://github.com/THGLab/iShiftML.git

and then run the following command inside the repository:

    pip install -e .

Now, you have iShiftML code installed as `nmrpred` and you can run related code anywhere on your computer as long as you are in the `iShiftML` environment.
Please note that this is a developer version, and thus you should reinstall the library whenever you pull new changes. 
Otherwise, you always use the previously installed version of this library.

## Usage
### To predict the shifts of a single molecule
Run `iShiftML/scripts/predict/ensemble_prediction.py` to make predictions using ensemble model, or run `iShiftML/scripts/predict/single_prediction.py` to predict using a single model (not recommended).

```python
usage: ensemble_prediction.py [-h] [--input_folder INPUT_FOLDER] [--split_file SPLIT_FILE] [--low_level_QM_file LOW_LEVEL_QM_FILE] [--xyz_file XYZ_FILE] [-e ELEMENT] [--model_path MODEL_PATH]
                              [--low_level_theory LOW_LEVEL_THEORY] [--target_level_theory TARGET_LEVEL_THEORY] [--name NAME] [--scratch_folder SCRATCH_FOLDER] [--output_folder OUTPUT_FOLDER] [--has_target]
                              [--include_low_level] [--batch_size BATCH_SIZE] [--device DEVICE] [--without_tev]

optional arguments:
  -h, --help            show this help message and exit
  --low_level_QM_file LOW_LEVEL_QM_FILE
                        the low level QM calculation organized in csv format. This is to predict single molecule
  --xyz_file XYZ_FILE   The xyz file for the molecule. Not needed if low_level_QM_file contains xyz info
  -e ELEMENT, --element ELEMENT
                        The element to predict
  --model_path MODEL_PATH
                        The path to the models folder
  --low_level_theory LOW_LEVEL_THEORY
  --target_level_theory TARGET_LEVEL_THEORY
  --name NAME           Name of data. When not provided, infer from necessary input file names
  --scratch_folder SCRATCH_FOLDER
                        A folder to save the scratch data generated in data preparation
  --output_folder OUTPUT_FOLDER
                        A folder to save the output
  --has_target          When the high level target data has been prepared, setting this argument to True will add the high level target data in the prediction files.
  --include_low_level   setting this argument to True will add the low level calculations to the prediction files.
  --batch_size BATCH_SIZE
                        The batch size for prediction
  --device DEVICE       The device to use for prediction
  --without_tev         whether the model is trained without tev. Setting this argument to True will ignore TEVs, usually used when you are using original model or data_aug model.
  --self_trained_model  whether the model is trained by yourself. Setting this argument to True will change the model paths from model_path/element/*.pt to
                        model_path/element/training_*/models/best_model.pt
  --input_folder INPUT_FOLDER
                        The folder to store all input data. This is to get the ensemble prediction result after preparing your data. Need to be used with --split_file. Need to be used with
                        --self_trained_model if you are using your model. Could not be used together with --low_level_QM_file.
  --split_file SPLIT_FILE
                        The file tell which molecules to predict when predicting multiple molecules
```

`low_level_QM_file` is the low level QM calculation organized in csv format. It is required to predict a single molecule. The csv file should contain following columns:

[atom_idx, atom_symbol, x, y, z, wB97X-V_pcSseg-1, DIA00, DIA01, DIA02, DIA10, DIA11, DIA12, DIA20, DIA21, DIA22, PARA00, PARA01, PARA02, PARA10, PARA11, PARA12, PARA20, PARA21, PARA22] 

This prediction script will call `prepare_data.py` first to prepare data so keep the two scripts in the same folder. Its full usage can be seen by `python prepare_data.py -h`. When the code runs successfully, it will prepare all necessary files, including `predict_data.txt`, together with `aev.hdf5`, `atomic.pkl`, `wB97X-V_pcSseg-1.pkl` and `tev.hdf5` under default `processed_data` folder or specified folder. 

Once the prediction is made, a prediction `.csv` file will be generated under the specified `output_folder`, or by default the `./local` folder. The prediction file will contain the predicted chemical shieldings from each model in the ensemble, the mean and standard deviations with outliers excluded, together with the low level calculations and high level target data if specified.

Usage examples:
The following command predicts the chemical shielding of carbon for a molecule whose `low_level_QM_file` is `./temp/mol.csv`. It uses our trained TEV models and output to the folder `./local` 
```python
python ensemble_prediction.py --low_level_QM_file ./temp/mol.csv --model_path iShiftML/models/TEV --output_folder ./local --include_low_level -e C 
```

`iShiftML/scripts/predict/single_prediction.py` is used to make predictions using a single model. You should check the settings in line 24-32 to make sure the settings are correct. You can then run the script without arguments to make predictions. 


### Optional: Batch processing and predicting
1. Recommended: `iShiftML/scripts/predict/predict.sh` is an example bash script to run predictions for multiple molecules in batch. You can change this code to fit your own needs.

2. If you want to get the ensemble prediction result after preparing your data and training your own models (see below), you can use the argument `--input_folder` of `iShiftML/scripts/predict/ensemble_prediction.py` to predict multiple molecules. Refer to `iShiftML/dataset/README.md` for an explanation of dataset preparation.


## Train models
Please first make sure data has been prepared. Refer to `iShiftML/dataset/README.md` for an explanation of dataset preparation.

### Hyperparameter tuning using NNI
`iShiftML/scripts/hparam_tuning` has necessary code for doing hyperparameter tuning using the NNI package. If you want to do your own hyperparameter tuning, first install NNI according to https://github.com/microsoft/nni, and then run `nni_tuner.py` to start the tuning process. Make sure you have checked the `config.yml` file and changed all data paths.

### Training models
The entrance script for training models is `iShiftML/scripts/active_learning/run_attention_aev.py` or `iShiftML/scripts/active_learning/run_attention_tev.py` (requiring tev.hdf5 prepared), which will train a model using parameters and dataset specified in `config.yml`. You can change the settings in `config.yml` to fit your own needs. Check comments in the `config.yml` to understand its usage.


## Development guidelines

- Please push your changes to a new branch and avoid merging with the master branch unless
your work is reviewed by at least one other contributor.

- The documentation of the modules are available at most cases. Please look up local classes or functions and consult with the docstrings in the code.



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
The developer installation is available and for that you need to first clone iShiftML from this repository:

    git clone https://github.com/THGLab/iShiftML.git

and then run the following command inside the repository:

    pip install -e .


We recommend using conda environment to install dependencies of this library.
Please install (or load) conda and then proceed with the following commands:

    conda create --name iShiftML python=3.8
    conda activate iShiftML
    conda install -c conda-forge numpy scipy pandas pyyaml scikit-learn ase
    conda install -c pytorch pytorch cudatoolkit=10.1 
    pip install tqdm, h5py==2.10

Now, you have iShiftML code installed as `nmrpred` and you can run related code anywhere on your computer as long as you are in the `iShiftML` environment.
Please note that this is a developer version, and thus you should reinstall the library whenever you pull new changes. 
Otherwise, you always use the previously installed version of this library.

## Usage
### Step 1: Prepare data
Run `iShiftML/scripts/predict/prepare_data.py` to prepare data for making predictions. The full usage note is given below:
```python
usage: prepare_data.py [-h] [--low_level_theory LOW_LEVEL_THEORY] [--high_level_QM_calculation HIGH_LEVEL_QM_CALCULATION] [--high_level_theory HIGH_LEVEL_THEORY] [--name NAME]
                       [--prediction_index PREDICTION_INDEX] [--save_folder SAVE_FOLDER]
                       xyz_file low_level_QM_calculation

positional arguments:
  xyz_file
  low_level_QM_calculation

optional arguments:
  -h, --help            show this help message and exit
  --low_level_theory LOW_LEVEL_THEORY
  --high_level_QM_calculation HIGH_LEVEL_QM_CALCULATION
                        When provided, high level data will also be prepared
  --high_level_theory HIGH_LEVEL_THEORY
                        Level of theory for the high level method
  --name NAME           Name of data. When not provided, infer from necessary input file names
  --prediction_index PREDICTION_INDEX
                        In the format of i.e. 0-8, where 8 is inclusive
  --save_folder SAVE_FOLDER
                        A folder to save the processed data
```
`xyz_file` is the molecule geometry file in xyz format. 

`low_level_QM_calculation` is the low level QM calculation organized in csv format. The csv file should contain following columns:

[atom_idx, atom_symbol, x, y, z, wB97X-V_pcSseg-1, DIA00, DIA01, DIA02, DIA10, DIA11, DIA12, DIA20, DIA21, DIA22, PARA00, PARA01, PARA02, PARA10, PARA11, PARA12, PARA20, PARA21, PARA22] 

If only part of the atoms need to be calculated, please specify the indices of the atoms required in `--prediction_index` argument.

When the code runs successfully, it will prepare all necessary files for running model inference under current folder or the folder specified by `--save_folder` argument. The files include `predict_data.txt`, together with `aev.hdf5`, `atomic.pkl` and `wB97X-V_pcSseg-1.pkl` under `processed_data` folder. 

### Step 2: Make predictions
Run `iShiftML/scripts/predict/ensemble_prediction.py` to make predictions using ensemble model, or run `iShiftML/scripts/predict/single_prediction.py` to predict using a single model.

```python
usage: ensemble_prediction.py [-h] [--output_path OUTPUT_PATH] [--model_path MODEL_PATH] [--batch_size BATCH_SIZE] [--prediction_data PREDICTION_DATA] [--data_root DATA_ROOT]
                              [--has_target] [--include_low_level]
                              atom

positional arguments:
  atom                  The element to be predicted. One of H/C/N/O.

optional arguments:
  -h, --help            show this help message and exit
  --output_path OUTPUT_PATH
                        Path to save the output files. The prediction files will be named as 'ensemble_prediction_{atom}_{category}.csv'.
  --model_path MODEL_PATH
                        Path to the trained models.
  --batch_size BATCH_SIZE
                        Batch size for prediction.
  --prediction_data PREDICTION_DATA
                        Path to the file specifying data to be predicted.
  --data_root DATA_ROOT
                        Folder containing necessary data for prediction.
  --has_target          When the high level target data has been prepared, setting this argument to True will add the high level target data in the prediction files.
  --include_low_level   setting this argument to True will add the low level calculations to the prediction files.
```

Once the prediction is made, a prediction `.csv` file will be generated under the specified `output_path`, or by default the `./local` folder. The prediction file will contain the predicted chemical shieldings from each model in the ensemble, the mean and standard deviations with outliers excluded, together with the low level calculations and high level target data if specified.

`iShiftML/scripts/predict/single_prediction.py` is used to make predictions using a single model. You should check the settings in line 24-32 to make sure the settings are correct. You can then run the script without arguments to make predictions. 

### Optional: Batch processing and predicting
`iShiftML/scripts/predict/process_and_predict.sh` is an example bash script to run predictions for multiple molecules in batch. The script will first prepare the data for prediction, and then make predictions using the ensemble model. You can change this code to fit your own needs.


## Train models
Please first make sure data has been prepared. Refer to `iShiftML/dataset/README.md` for an explanation of dataset preparation.

### Hyperparameter tuning using NNI
`iShiftML/scripts/hparam_tuning` has necessary code for doing hyperparameter tuning using the NNI package. If you want to do your own hyperparameter tuning, first install NNI according to https://github.com/microsoft/nni, and then run `nni_tuner.py` to start the tuning process. Make sure you have checked the `config.yml` file and changed all data paths.

### Training models
The entrance script for training models is `iShiftML/scripts/active_learning/run_attention_aev.py`, which will train a model using parameters and dataset specified in `config.yml`. You can change the settings in `config.yml` to fit your own needs. Check comments in the `config.yml` to understand its usage.


## Development guidelines

- Please push your changes to a new branch and avoid merging with the master branch unless
your work is reviewed by at least one other contributor.

- The documentation of the modules are available at most cases. Please look up local classes or functions and consult with the docstrings in the code.



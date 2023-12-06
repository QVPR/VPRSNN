# Modular SNN: Ensembles of Compact, Region-specific & Regularized Spiking Neural Networks for Scalable Place Recognition

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![stars](https://img.shields.io/github/stars/QVPR/VPRSNN.svg?style=flat-square)](https://github.com/QVPR/VPRSNN/stargazers)
![GitHub repo size](https://img.shields.io/github/repo-size/QVPR/VPRSNN.svg?style=flat-square)
[![QUT Centre for Robotics](https://img.shields.io/badge/collection-QUT%20Robotics-%23043d71?style=flat-square)](https://qcr.ai)

## License and Citation

This code is licensed under [MIT License](./LICENSE). If you use our modular SNN code, please cite our [paper](https://arxiv.org/abs/2209.08723):

```
@inproceedings{hussaini2023ensembles,
  title={Ensembles of compact, region-specific \& regularized spiking neural networks for scalable place recognition},
  author={Hussaini, Somayeh and Milford, Michael and Fischer, Tobias},
  booktitle={2023 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={4200--4207},
  year={2023},
  organization={IEEE}
}
```


## Setup

Our recommended tool to install all dependencies is conda (or better: mamba). Please download and install [conda](https://docs.conda.io/en/latest/) or [mamba](https://mamba.readthedocs.io/en/latest/), if you have not installed it yet. 


You can create the conda environment using one of the following options. 

Option 1: 

```bash
conda create -n vprsnn -c conda-forge python numpy matplotlib pathlib opencv tqdm pickle5 brian2 scikit-learn ipykernel numba cudatoolkit autopep8 pandas seaborn wandb
```

Option 2 (using the provided environment.yml file): 

```bash 
conda env create -f environment.yml
```

Option 3 (using the provided requirements.txt file):
```bash 
conda create --name vprsnn --file requirements.txt -c conda-forge
```

Activate the conda environment: 

```bash
conda activate vprsnn
```


## Run 
### Prerequisites
1. Please ensure you have created and activated the conda environment.  
2. Nordland dataset

### Data Processing Instructions

#### Reference Dataset: 
* We use the entire Spring and Fall traverses to train our modular SNN network. These traverses are our reference dataset which we use to train our network. 

#### Query Dataset:
* The entire Summer traverse is our query dataset, with less than 20% for calibration (600 images) and the rest for testing.

#### Filtering Criteria for Nordland dataset:
* Exclude images from segments where the train's speed is below 15 km/h. Refer to the filtered list in [`dataset_imagenames/nordland_imageNames.txt`](https://github.com/QVPR/VPRSNN/blob/modularSNN/dataset_imagenames/nordland_imageNames.txt) relevant to the Nordland dataset variation found [here](https://cloudstor.aarnet.edu.au/plus/s/2LtwUtLUFpUiUC8).

#### Sampling Method:
* Both datasets are sampled to extract images approximately every 100 meters (every 8th image). The data processing code is in [`tools/data_utils.py`](https://github.com/QVPR/VPRSNN/blob/main/tools/data_utils.py). 

The scripts [`modular_snn/modular_snn_processing.py`](https://github.com/QVPR/VPRSNN/blob/main/modular_snn/modular_snn_processing.py) and [`modular_snn/modular_snn_evaluation.py`](https://github.com/QVPR/VPRSNN/blob/main/modular_snn/modular_snn_evaluation.py) provide the configurations to train, test, calibrate, and evaluate our Modular SNN on the Nordland dataset.


## Modular SNN

### Training a modular spiking neural network:

1. Run `modular_snn/modular_snn_processing.py` with `args.process_mode="train"` to individually:

    * Process training all modules of the network on geographically non-overlapping regions of the reference dataset. 
    * Record the responses of all trained modules to the entire reference images (used by all modules) after training (for hyperactive neuron detection). Process mode is set to record once the training is finished, `args.process_mode="record"`.
    * Calibrate the calibration modules of the network using a geographically separate set of query images from the test set. Process mode is set to calibrate after the record process is finished, `args.process_mode="calibrate"`.   
  
2. Run `modular_snn/modular_snn_evaluation.py` with `args.process_mode = "calibrate"` to evaluate the performance of the calibration modules using a range of threshold values to detect and then remove the hyperactive neurons. Select the best-performing threshold for testing. 


Train the modular SNN with the default configs locally: 
```bash
python modular_snn/modular_snn_processing.py --run_mode="local" --process_mode="train"
```


Notes:

* Setting the `args.process_mode="train"` will also automatically perform the record and calibrate processes after the training is finished. 
* The record process record the activity of the neurons which is used in the calibration and test processes of Modular SNN to detect the hyperactive neurons, done in `modular_snn/modular_snn_evaluation.py`.
* The calibration process evaluates the performance of Modular SNN on the calibration data using a range of values for the hyperparameters. The hyperparameter, epochs, related to train process is defined in `modular_snn/one_modular_snn_processing.py`. The hyperparameter, threshold, that is used for hyperactive neuron detection, is defined in `modular_snn/modular_snn_evaluation.py`. 
* Set `args.run_mode` to one of the available choices `["local", "wandb_local", "wandb_hpc"]` to either process the modules locally (sequential), locally using wandb tool, or on a high performance computing (hpc) platform using wandb tool. 



**[UPDATE]** We have released the trained weights of our Modular SNN on the Nordland dataset using reference traverses spring and fall, as requested in [QVPR/VPRSNN#4](https://github.com/QVPR/VPRSNN/issues/4). You can access them [here](https://drive.google.com/drive/u/1/folders/1Qwp3h6D1s2CMLXisAUDVGN1Z9EOAQbwA). 



### Testing with learned weights

1. Run `modular_snn/modular_snn_processing.py` with `args.process_mode="test"` to test the modular spiking neural network on the test region of the query dataset. 
2. Run `modular_snn/modular_snn_evaluation.py` with `args.process_mode="test"` to evaluate the performance of the modular spiking neural network on the test region of the query dataset. 

Test the modular SNN with the default configs locally: 
```bash
python modular_snn/modular_snn_processing.py --run_mode="local" --process_mode="test"
```


## Acknowledgements
This work is supported by the Australian Government, Intel Labs, and the Queensland University of Technology (QUT) through the Centre for Robotics.
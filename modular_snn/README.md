# Modular SNN: Ensembles of Compact, Region-specific & Regularized Spiking Neural Networks for Scalable Place Recognition

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![stars](https://img.shields.io/github/stars/QVPR/VPRSNN.svg?style=flat-square)](https://github.com/QVPR/VPRSNN/stargazers)
![GitHub repo size](https://img.shields.io/github/repo-size/QVPR/VPRSNN.svg?style=flat-square)
[![QUT Centre for Robotics](https://img.shields.io/badge/collection-QUT%20Robotics-%23043d71?style=flat-square)](https://qcr.ai)

## License and Citation

This code is licensed under [MIT License](./LICENSE). If you use our modular SNN code, please cite our [paper](https://arxiv.org/abs/2209.08723):

```
@article{hussaini2022ensembles,
  title={Ensembles of Compact, Region-specific \& Regularized Spiking Neural Networks for Scalable Place Recognition},
  author={Hussaini, Somayeh and Milford, Michael and Fischer, Tobias},
  journal={arXiv preprint arXiv:2209.08723},
  year={2022}
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
2. Nordland datasets, which can be downloaded from: https://cloudstor.aarnet.edu.au/plus/s/2LtwUtLUFpUiUC8 (If not already available)

Notes: 
* We use the entire Spring and Fall traverses to train our modular SNN network. We consider these traverses as our reference dataset, and use it to train our network. 
* We use the entire Summer traverse as our query dataset, using less than 20% (600 images) for calibration, and using the remaining 80% for testing. 
* We remove sections were the train is moving at speeds less than 15 km/h, the filtered list of images are provided in [`dataset_imagenames/nordland_imageNames.txt`](https://github.com/QVPR/VPRSNN/blob/modularSNN/dataset_imagenames/nordland_imageNames.txt). Please note that this filetered image list file is for the variation of the Nordland dataset provided in the [link](https://cloudstor.aarnet.edu.au/plus/s/2LtwUtLUFpUiUC8) above.
* We sample both our reference and query datasets to extract places at approximately every 100 m (every 8th image). Our code is provided in [`tools/data_utils.py`](https://github.com/QVPR/VPRSNN/blob/main/tools/data_utils.py) 


## Modular SNN

### Training a modular spiking neural network:

1. Run `modular_snn/modular_snn_processing.py` with `args.process_mode="train"` to individually:

    * Process training all modules of the network on geographically non-overlapping regions of the reference dataset. 
    * Record the responses of all trained modules to the entire reference images (used by all modules) after training (for hyperactive neuron detection)
    * Calibrate the calibration modules of the network using a geographically separate set of query images from the test set.  
        * Run `modular_snn/modular_snn_evaluation.py` with `args.process_mode = "calibrate"` to evaluate the performance of the calibration modules using a range of threshold values to detect and then remove the hyperactive neurons. Select the best-performing threshold for testing. 


Train the modular SNN with the default configs locally: 
```bash
python modular_snn/modular_snn_processing.py --run_mode="local" --process_mode="train"
```


Notes:

* Setting the `args.process_mode="train"` will also automatically performs the record and calibrate processes after the training is finished. 
* Set `args.run_mode` to one of the available choices `["local", "wandb_local", "wandb_hpc"]` to either process the modules locally (sequential), locally using wandb tool, or on a high performance computing (hpc) platform using wandb tool. 


### Testing with learned weights

1. Run `modular_snn/modular_snn_processing.py` with `args.process_mode="test"` to test the modular spiking neural network on the test region of the query dataset. 
2. Run `modular_snn/modular_snn_evaluation.py` with `args.process_mode="test"` to evaluate the performance of the modular spiking neural network on the test region of the query dataset. 

Test the modular SNN with the default configs locally: 
```bash
python modular_snn/modular_snn_processing.py --run_mode="local" --process_mode="test"
```


## Acknowledgements
This work is supported by the Australian Government, Intel Labs, and the Queensland University of Technology (QUT) through the Centre for Robotics.
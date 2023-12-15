# Ensembles of Modular SNNs with/without sequence matching: Applications of Spiking Neural Networks in Visual Place Recognition

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![stars](https://img.shields.io/github/stars/QVPR/VPRSNN.svg?style=flat-square)](https://github.com/QVPR/VPRSNN/stargazers)
![GitHub repo size](https://img.shields.io/github/repo-size/QVPR/VPRSNN.svg?style=flat-square)
[![QUT Centre for Robotics](https://img.shields.io/badge/collection-QUT%20Robotics-%23043d71?style=flat-square)](https://qcr.ai)

## License and Citation

This code is licensed under [MIT License](./LICENSE). If you use our Ensemble of Modular SNNs with/without sequence matching code, please cite the following [paper](https://arxiv.org/abs/2311.13186):

```
@article{hussaini2023applications,
  title={Applications of Spiking Neural Networks in Visual Place Recognition},
  author={Hussaini, Somayeh and Milford, Michael and Fischer, Tobias},
  journal={arXiv preprint arXiv:2311.13186},
  year={2023}
}
```


## Setup

Our recommended tool to install all dependencies is conda (or better: mamba). Please download and install [conda](https://docs.conda.io/en/latest/) or [mamba](https://mamba.readthedocs.io/en/latest/), if you have not installed it yet. 


You can create the conda environment using one of the following options. 

Option 1: 

```bash
conda create -n vprsnn -c conda-forge python torchvision pytorch torchaudio numpy matplotlib pathlib opencv tqdm pickle5 brian2 scikit-learn ipykernel numba cudatoolkit autopep8 pandas seaborn wandb
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

Repeating the same process of training, testing, calibrating, and evaluating our Modular SNN with different seed values, produces Modular SNNs that have the same architecture, and differ in the random order of input images and initialisation of learned weights. The Ensemble of Modular SNNs is formed by the predictions of these multiple Modular SNNs. 

Note that both `ens_seq/process_seqmatch.py` and `ens_seq/process_ensembles.py` operate on the predictions of one and/or multiple Modular SNNs.  


## Ensemble of Modular SNNs

### Forming the Ensemble of Modular SNNs:

1. Run `ens_seq/process_ensembles.py` with configs about the data and the hyperparameters of ensemble members, Modular SNNs:

Compute the recall at 1 (R@1) performance of Ensemble of Modular SNNs on Nordland dataset with spring and fall as reference traverses, that are used for training the Modular SNNs, and summer as the query dataset, that is used for calibration and testing. The epochs and thresholds hyperparameter values of the ensemble members, Modular SNNs, are selected based on the best performing values on the calibration set. 

```bash
python process_ensembles.py --SNN_data_path outputs_ne43200_L2700_offset3275_tcgi0.5_S{}_M2/standard/epoch{}_T3300_T{} --mainfolder_path ./outputs/outputs_models_Nordland_SFS --seeds 0 10 20 30 40 --epochs 70 60 60 60 70 --thresholds 80 180 140 220 60 --num_query_imgs 3300 --num_cal_labels 600
```



**[Note]** We have released the trained weights of one instance of our Modular SNN on the Nordland dataset using reference traverses spring and fall, as requested in [QVPR/VPRSNN#4](https://github.com/QVPR/VPRSNN/issues/4). You can access them [here](https://drive.google.com/drive/u/1/folders/1Qwp3h6D1s2CMLXisAUDVGN1Z9EOAQbwA). 



## Ensemble of Modular SNNs with sequence matching 

### Applying sequence matching to Modular SNNs and Ensemble of Modular SNNs:

1. Run `ens_seq/process_seqmatch.py` with configs about the data and the hyperparameters of ensemble members, Modular SNNs:

    * Sequence matching can be applied to the predictions of both our Modular SNN and Ensemble of Modular SNNs.

Compute the recall at 1 (R@1) performance of our Ensemble of Modular SNNs with sequence matching on the Nordland dataset using reference traverses spring and fall, and query traverse summer:
```bash
python process_seqmatch.py --SNN_data_path merged_results_ens_SNN_S5/ --mainfolder_path ./outputs/outputs_models_Nordland_SFS --seed 5 --num_query_imgs 3300 --num_cal_labels 600 --use_ensemble True
```

Compute the recall at 1 (R@1) performance of our Modular SNN with sequence matching on the Nordland dataset using reference traverses spring and fall, and query traverse summer:

```bash
python3 process_seqmatch.py --SNN_data_path outputs_ne43200_L2700_offset3275_tcgi0.5_S{}_M2/standard/epoch{}_T3300_T{}/ --mainfolder_path ./outputs/outputs_models_Nordland_SFS --seed 0 --epochs 70 --thresholds 80 --num_query_imgs 3300 --num_cal_labels 600
```


## Acknowledgements
This work is supported by the Australian Government, Intel Labs, and the Queensland University of Technology (QUT) through the Centre for Robotics.
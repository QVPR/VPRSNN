# Non-modular SNN: Spiking Neural Networks for Visual Place Recognition via Weighted Neuronal Assignments

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![stars](https://img.shields.io/github/stars/QVPR/VPRSNN.svg?style=flat-square)](https://github.com/QVPR/VPRSNN/stargazers)
![GitHub repo size](https://img.shields.io/github/repo-size/QVPR/VPRSNN.svg?style=flat-square)
[![QUT Centre for Robotics](https://img.shields.io/badge/collection-QUT%20Robotics-%23043d71?style=flat-square)](https://qcr.ai)

## License and Citation

This code is licensed under [MIT License](./LICENSE). If you use our non-modular SNN code, please cite our [paper](https://arxiv.org/abs/2109.06452):


```
@article{hussaini2022spiking,
  title={Spiking Neural Networks for Visual Place Recognition via Weighted Neuronal Assignments},
  author={Hussaini, Somayeh and Milford, Michael J and Fischer, Tobias},
  journal={IEEE Robotics and Automation Letters},
  year={2022},
  publisher={IEEE}
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
* We use the Spring and Fall traverses to train our modular SNN network. We consider these traverses as our reference dataset, and use it to train our network. 
* We use the Summer traverse as our query dataset. 
* We remove sections were the train is moving at speeds less than 15 km/h, the filtered list of images are provided in [`dataset_imagenames/nordland_imageNames.txt`](https://github.com/QVPR/VPRSNN/blob/modularSNN/dataset_imagenames/nordland_imageNames.txt). Please note that this filetered image list file is for the variation of the Nordland dataset provided in the [link](https://cloudstor.aarnet.edu.au/plus/s/2LtwUtLUFpUiUC8) above.
* We sample both our reference and query datasets to extract places at approximately every 100 m (every 8th image). Our code is provided in [`tools/data_utils.py`](https://github.com/QVPR/VPRSNN/blob/main/tools/data_utils.py) 


## Non-modular SNN 
### Training a new network:

1. run `non_modular_snn/single_snn_model_processing.py` with `args.process_mode="train"` to: 

    * Generate the initial synaptic weights of the SNN model using `tools/random_connection_generator.py`.
    * Train the snn model using `non_modular_snn/snn_model.py` on the reference set. 
    * The trained weights will be stored in a subfolder in the folder "weights", which can be used to test the performance.
    * The output will be stored in a subfolder in the folder "outputs", which also contains log files. 

Train the non-modular SNN with the default configs locally: 
```bash
python non_modular_snn/single_snn_model_processing.py --process_mode="train"
```

The trained weights of our Non-modular SNN on the Nordland dataset, using reference traverses spring and fall, are available [here](https://drive.google.com/drive/u/1/folders/1Qwp3h6D1s2CMLXisAUDVGN1Z9EOAQbwA). 


### Testing with learned weights

1. run `non_modular_snn/single_snn_model_processing.py` with `args.process_mode="test"` to: 

    * Test your trained model on the query set. The trained weights for a model with 100 places (current configuration across all files) is provided in a subfolder in weights folder.  
    * Evaluate the performance of the model on the query set using `non_modular_snn/snn_model_evaluation.py`. 
2. Run `tools/weight_visualisations.py` to visualise the learnt weights.
3. The output will be stored in the same subfolder as in the training folder "outputs", which also contains log files. 

Test the non-modular SNN with the default configs locally: 
```bash
python non_modular_snn/single_snn_model_processing.py --process_mode="test"
```


## Acknowledgements
This work is supported by the Australian Government, Intel Labs, and the Queensland University of Technology (QUT) through the Centre for Robotics.

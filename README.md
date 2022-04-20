# Spiking Neural Networks for Visual Place Recognition via Weighted Neuronal Assignments
[![GitHub repo size](https://img.shields.io/github/repo-size/QVPR/Patch-NetVLAD.svg?style=flat-square)](./README.md)
[![QUT Centre for Robotics](https://img.shields.io/badge/collection-QUT%20Robotics-%23043d71?style=flat-square)](https://qcr.ai)


This repository contains code for our RA-L paper "Spiking Neural Networks for Visual Place Recognition via Weighted Neuronal Assignments" which will also be presented at ICRA 2022. DOI: [10.1109/LRA.2022.3149030](https://doi.org/10.1109/LRA.2022.3149030)

The pre-print version of the paper is available on [arxiv](https://arxiv.org/abs/2109.06452). 

Video: https://www.youtube.com/watch?v=VGfv4ZVOMkw

<p style="width: 50%; display: block; margin-left: auto; margin-right: auto">
  <img src="./resources/cover_photo.png" alt="VPRSNN method diagram"/>
</p>

## Citation

If using code within this repository, please refer to our [paper](https://doi.org/10.1109/LRA.2022.3149030) in your publications:
```
@article{hussaini2022spiking,
  title={Spiking Neural Networks for Visual Place Recognition via Weighted Neuronal Assignments},
  author={Hussaini, Somayeh and Milford, Michael J and Fischer, Tobias},
  journal={IEEE Robotics and Automation Letters},
  year={2022},
  publisher={IEEE}
}
```

This work is an adaptation of the spiking neural network model from "Unsupervised Learning of Digit Recognition Using Spike-Timing-Dependent Plasticity", Diehl and Cook, (2015) for Visual Place Recognition (VPR). DOI: [10.3389/fncom.2015.00099](https://doi.org/10.3389/fncom.2015.00099).
Visual Place Recognition is the problem of how a robot can identify whether it has previously visited a place given an image of the place despite challenges including changes in appearance and perceptual aliasing (where two different places look similar). 

The code is based on the following reposities, that include the original code and the modified versions of the original code. 

Original code (Peter U. Diehl): https://github.com/peter-u-diehl/stdp-mnist

Updated for Brian2: zxzhijia: https://github.com/zxzhijia/Brian2STDPMNIST

Updated for Python3: sdpenguin: https://github.com/sdpenguin/Brian2STDPMNIST


This code is licensed under [MIT License](./LICENSE).


## Setup

Our recommended tool to install all dependencies is conda (or better: mamba). Please download and install [conda](https://docs.conda.io/en/latest/) or [mamba](https://mamba.readthedocs.io/en/latest/), if you have not installed it yet. 


You can create the conda environment using one of the following options. 

Option 1: 

```bash
conda create -n vprsnn -c conda-forge python==3.7.9 numpy matplotlib pathlib opencv tqdm pickle5 brian2 scikit-learn ipykernel numba cudatoolkit autopep8 pandas seaborn
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
2. Nordland datasets, which can be downloaded from: https://webdiis.unizar.es/~jmfacil/pr-nordland/#download-dataset (If not already available)

### Training a new network:
1. Generate the initial synaptic weights using DC_MNIST_random_conn_generator.py, modify the number of output neurons if required. 
2. Modify the training configurations on main file "snn_model.py" including changing "args.test_mode" to "False" (if needed) and run the code. 
3. The trained weights will be stored in a subfolder in the folder "weights", which can be used to test the performance.
4. The output will be stored in a subfolder in the folder "outputs", which also contains log files. 

### Testing with pretrained weights
1. To test your trained model, set "args.test_mode" to "True", and run snn_model.py file. The trained weights for a model with 100 places (current configuration across all files) is provided in a subfolder in weights folder.  
2. Run the "WeightReadout.py" to visualise the learnt weights. 
3. Run "DC_MNIST_evaluation.py" to do the neuronal assignments and perform the predictions. 
4. The output will be stored in the same subfolder as in the training folder "outputs", which also contains log files. 


## Acknowledgements
This work was supported by the Australian Government via grant AUSMURIB000001 associated with ONR MURI grant N00014-19-1-2571, Intel Research via grant RV3.248.Fischer, and the Queensland University of Technology (QUT) through the Centre for Robotics.



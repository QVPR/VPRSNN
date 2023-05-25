# Spiking Neural Networks for Visual Place Recognition via Weighted Neuronal Assignments
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![stars](https://img.shields.io/github/stars/QVPR/VPRSNN.svg?style=flat-square)](https://github.com/QVPR/VPRSNN/stargazers)
![GitHub repo size](https://img.shields.io/github/repo-size/QVPR/VPRSNN.svg?style=flat-square)
[![QUT Centre for Robotics](https://img.shields.io/badge/collection-QUT%20Robotics-%23043d71?style=flat-square)](https://qcr.ai)


This repository contains code for two papers: 
* Modular SNN: [Ensembles of Compact, Region-specific & Regularized Spiking Neural Networks for Scalable Place Recognition (ICRA 2023)](https://arxiv.org/abs/2209.08723)

* Non-modular SNN: [Spiking Neural Networks for Visual Place Recognition via Weighted Neuronal Assignments (RAL + ICRA2022)](https://arxiv.org/abs/2109.06452) DOI: [10.1109/LRA.2022.3149030](https://doi.org/10.1109/LRA.2022.3149030)


## Overview
Please refer to the readme files of the `modular_snn` and `non_modular_snn` folders for instructions to run the code for modular SNN and non-modular SNN works. 


## Modular SNNs for scalable place recognition (Modular SNN)

Video: https://www.youtube.com/watch?v=TNDdfmPSe1U&t=137s

<p style="width: 50%; display: block; margin-left: auto; margin-right: auto">
  <img src="./resources/ICRA2023.png" alt="ModularSNN for scalable place recognition"/>
</p>

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

## SNNs for VPR (non-modular SNN)

Video: https://www.youtube.com/watch?v=VGfv4ZVOMkw

<p style="width: 50%; display: block; margin-left: auto; margin-right: auto">
  <img src="./resources/cover_photo.png" alt="VPRSNN method diagram"/>
</p>

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

This work is an adaptation of the spiking neural network model from "Unsupervised Learning of Digit Recognition Using Spike-Timing-Dependent Plasticity", Diehl and Cook, (2015) for Visual Place Recognition (VPR). DOI: [10.3389/fncom.2015.00099](https://doi.org/10.3389/fncom.2015.00099).
Visual Place Recognition is the problem of how a robot can identify whether it has previously visited a place given an image of the place despite challenges including changes in appearance and perceptual aliasing (where two different places look similar). 

The code is based on the following reposities, that include the original code and the modified versions of the original code. 

Original code (Peter U. Diehl): https://github.com/peter-u-diehl/stdp-mnist

Updated for Brian2: zxzhijia: https://github.com/zxzhijia/Brian2STDPMNIST

Updated for Python3: sdpenguin: https://github.com/sdpenguin/Brian2STDPMNIST




Please refer to the [wiki tab](https://github.com/QVPR/VPRSNN/wiki) for additional ablation studies. 



## Acknowledgements
These works were supported by the Australian Government, Intel Labs, and the Queensland University of Technology (QUT) through the Centre for Robotics.



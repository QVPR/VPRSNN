#! /usr/bin/env python

'''
MIT License

Copyright (c) 2023 Somayeh Hussaini, Michael Milford and Tobias Fischer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import argparse
import os
import sys
import time
import numpy as np
import wandb


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


from tools.wandb_utils import create_hpc_bashscript_wandb, get_wandb_sweep_id, setup_wandb_config



'''
To submit the jobs on hpc, simply:
- conda activate vprsnn 
- python modular_snn/modular_snn_processing.py --project_name="VPRSNN_0" --username="username" --sweep_name="sweep_1" --run_mode="hpc" --process_mode="train" --dataset='nordland' --num_labels=25 --num_cal_labels=600 --num_test_labels=2700 --num_query_imgs=3300 --skip=8 --offset_after_skip=0 --update_interval=250 --folder_id='NRD_SFS' --n_e=400 --epochs=60 --seed=0
- python modular_snn/modular_snn_processing.py --project_name="VPRSNN_0" --username="username" --sweep_name="sweep_1" --run_mode="hpc" --process_mode="test" --dataset='nordland' --num_labels=25 --num_cal_labels=600 --num_test_labels=2700 --num_query_imgs=3300 --skip=8 --offset_after_skip=0 --update_interval=250 --folder_id='NRD_SFS' --n_e=400 --epochs=60 --seed=0

_1: python3 modular_snn/modular_snn_processing.py --run_mode="wandb_hpc" --process_mode="train" --dataset='nordland' --num_labels=25 --num_cal_labels=600 --num_test_labels=2700 --num_query_imgs=3300 --skip=8 --offset_after_skip=0 --update_interval=250 --folder_id='NRD_SFS' --n_e=400 --epochs=80 --seed=0 
_1: python3 modular_snn/modular_snn_processing.py --run_mode="wandb_hpc" --process_mode="test" --dataset='nordland' --num_labels=25 --num_cal_labels=600 --num_test_labels=2700 --num_query_imgs=3300 --skip=8 --offset_after_skip=600 --update_interval=250 --folder_id='NRD_SFS' --n_e=400 --epochs=60 --seed=0 

_2: python3 modular_snn/modular_snn_processing.py --run_mode="wandb_hpc" --process_mode="train" --dataset='nordland' --num_labels=25 --num_cal_labels=600 --num_test_labels=2700 --num_query_imgs=3300 --skip=8 --offset_after_skip=0 --update_interval=250 --folder_id='NRD_SFW' --n_e=400 --epochs=200 --seed=0 
_2: python3 modular_snn/modular_snn_processing.py --run_mode="wandb_hpc" --process_mode="test" --dataset='nordland' --num_labels=25 --num_cal_labels=600 --num_test_labels=2700 --num_query_imgs=3300 --skip=8 --offset_after_skip=600 --update_interval=250 --folder_id='NRD_SFW' --n_e=400 --epochs=60 --seed=0 

_3: python3 modular_snn/modular_snn_processing.py --run_mode="wandb_hpc" --process_mode="train" --dataset='ORC' --num_labels=25 --num_cal_labels=75 --num_test_labels=375 --num_query_imgs=450 --skip=8 --offset_after_skip=0 --update_interval=250 --folder_id='ORC' --n_e=400 --epochs=200 --seed=0 
_3: python3 modular_snn/modular_snn_processing.py --run_mode="wandb_hpc" --process_mode="test" --dataset='ORC' --num_labels=25 --num_cal_labels=75 --num_test_labels=375 --num_query_imgs=450 --skip=8 --offset_after_skip=75 --update_interval=250 --folder_id='ORC' --n_e=400 --epochs=60 --seed=0 

_4: python3 modular_snn/modular_snn_processing.py --run_mode="wandb_hpc" --process_mode="train" --dataset='SFU-Mountain' --num_labels=25 --num_cal_labels=75 --num_test_labels=300 --num_query_imgs=375 --skip=1 --offset_after_skip=0 --update_interval=250 --folder_id='SFU-Mountain' --n_e=400 --epochs=200 --seed=0 
_4: python3 modular_snn/modular_snn_processing.py --run_mode="wandb_hpc" --process_mode="test" --dataset='SFU-Mountain' --num_labels=25 --num_cal_labels=75 --num_test_labels=300 --num_query_imgs=375 --skip=1 --offset_after_skip=75 --update_interval=250 --folder_id='SFU-Mountain' --n_e=400 --epochs=10 --seed=0 

_5: python3 modular_snn/modular_snn_processing.py --run_mode="wandb_hpc" --process_mode="train" --dataset='Synthia-NightToFall' --num_labels=25 --num_cal_labels=50 --num_test_labels=225 --num_query_imgs=275 --skip=1 --offset_after_skip=0 --update_interval=250 --folder_id='Synthia-NightToFall' --n_e=400 --epochs=200 --seed=0
_5: python3 modular_snn/modular_snn_processing.py --run_mode="wandb_hpc" --process_mode="test" --dataset='Synthia-NightToFall' --num_labels=25 --num_cal_labels=50 --num_test_labels=225 --num_query_imgs=275 --skip=1 --offset_after_skip=50 --update_interval=250 --folder_id='Synthia-NightToFall' --n_e=400 --epochs=20 --seed=0

_6: python3 modular_snn/modular_snn_processing.py --run_mode="wandb_hpc" --process_mode="train" --dataset='St-Lucia' --num_labels=25 --num_cal_labels=50 --num_test_labels=300 --num_query_imgs=350 --skip=1 --offset_after_skip=0 --update_interval=250 --folder_id='St-Lucia' --n_e=400 --epochs=200 --seed=0
_6: python3 modular_snn/modular_snn_processing.py --run_mode="wandb_hpc" --process_mode="test" --dataset='St-Lucia' --num_labels=25 --num_cal_labels=50 --num_test_labels=300 --num_query_imgs=350 --skip=1 --offset_after_skip=50 --update_interval=250 --folder_id='St-Lucia' --n_e=400 --epochs=60 --seed=0



'''




def main(args):
    
    if args.process_mode == "calibrate":
        
        offset_after_skip_list = np.arange(args.offset_after_skip, args.num_cal_labels+args.offset_after_skip, args.num_labels).tolist()
    else:
        offset_after_skip_list = np.arange(args.offset_after_skip, args.num_cal_labels+args.num_test_labels, args.num_labels).tolist()
    
    ad_path_base = args.ad_path
    ad_path_test_base = args.ad_path_test
    args_multi_path_base = args.multi_path
    num_test_labels_base = args.num_test_labels
    mode_base = args.process_mode

    if args.run_mode == "wandb_hpc" or args.run_mode == "wandb_local":
        sweep_config = setup_wandb_config(offset_after_skip_list, args)
        sweep_id = get_wandb_sweep_id(args, sweep_config)
        print("sweep id: ", sweep_id)

    for offset_after_skip in offset_after_skip_list:
        
        args.offset_after_skip = offset_after_skip
        args.num_test_labels = num_test_labels_base
        args.ad_path = ad_path_base.format(args.offset_after_skip)
        
        if args.run_mode == "local":
            
            from modular_snn.one_snn_module_processing import main as process_one_snn_module
            args.ad_path = ad_path_base
            args.ad_path_test = ad_path_test_base
            args.multi_path = args_multi_path_base
            args.process_mode = mode_base
            process_one_snn_module(args) 
        
    
        elif args.run_mode == "wandb_hpc":
            
            create_hpc_bashscript_wandb(args, sweep_id)


        elif args.run_mode == "wandb_local":

            from modular_snn.one_snn_module_processing import main as process_one_snn_module
            args.ad_path = ad_path_base
            args.ad_path_test = ad_path_test_base
            args.multi_path = args_multi_path_base
            args.process_mode = mode_base
            wandb.agent(sweep_id=sweep_id, project=args.project_name, function=process_one_snn_module, count=len(offset_after_skip_list))



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', type=str, default="nordland", 
                        help='Dataset folder name that is relative to this repo. The folder must exist in this directory: ./../data/')
    parser.add_argument('--num_labels', type=int, default=5, 
                        help='Number of training place labels for a single module.')
    parser.add_argument('--num_cal_labels', type=int, default=5, 
                        help="Number of calibration place labels.")
    parser.add_argument('--num_test_labels', type=int, default=15, 
                        help='Number of testing place labels.')
    parser.add_argument('--num_query_imgs', type=int, default=5, 
                        help='Number of query images used for testing and calibration.')
    parser.add_argument('--tc_ge', type=float, default=1.0, 
                        help='Time constant of conductance of excitatory synapses AeAi')
    parser.add_argument('--tc_gi', type=float, default=0.5, 
                        help='Time constant of conductance of inhibitory synapses AiAe')
    parser.add_argument('--intensity', type=int, default=4, 
                        help="Intensity scaling factor to change the range of input pixel values")
    parser.add_argument('--use_weighted_assignments', type=bool, default=False, 
                        help='Value to define the type of neuronal assignment to use: standard=False, weighted=True')

    parser.add_argument('--skip', type=int, default=8, 
                        help='The number of images to skip between each place label.')
    parser.add_argument('--offset_after_skip', type=int, default=0, 
                        help='The offset to apply for selecting places after skipping every n images.')
    parser.add_argument('--folder_id', type=str, default="NRD_SFS", 
                        help='Id to distinguish the traverses used from the dataset.')
    parser.add_argument('--update_interval', type=int, default=50, 
                        help='The number of iterations to save at one time in output matrix.')
    parser.add_argument('--epochs', type=int, default=20, 
                        help='Number of passes through the dataset.')
    parser.add_argument('--n_e', type=int, default=100, 
                        help='Number of excitatory output neurons. The number of inhibitory neurons are defined the same.')
    parser.add_argument('--threshold_i', type=int, default=0, 
                        help='Threshold value used to ignore the hyperactive neurons.')
    parser.add_argument('--seed', type=int, default=0, 
                        help='Set seed for random generator.')

    parser.add_argument('--ad_path_test', type=str, default="_test_E{}", 
                        help='Additional string arguments to use for saving test outputs in testing')
    parser.add_argument('--ad_path', type=str, default="_offset{}_S{}")             
    parser.add_argument('--multi_path', type=str, default="epoch{}_T{}_T{}")   
    
    parser.add_argument('--process_mode', type=str, choices=["train", "record", "calibrate", "test"], default="train", 
                        help='String indicator to define the mode (train, record, calibrate, test).')
    
    parser.add_argument('--run_mode', type=str, choices=["local", "wandb_local", "wandb_hpc"], default="local", 
                        help='Mode to run the modular network.')
    parser.add_argument('--sweep_name', type=str, default="sweep_1", 
                        help='Wandb sweep name.')
    parser.add_argument('--project_name', type=str, default="modularSNN", 
                        help='Wandb project name.')
    parser.add_argument('--username', type=str, default="my_username", 
                        help='Wandb user name.')
    
    parser.set_defaults()
    args = parser.parse_args()

    main(args)
    

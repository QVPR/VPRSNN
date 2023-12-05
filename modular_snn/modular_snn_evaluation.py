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


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


from modular_snn.modular_snn_config_evaluation import main as evaluate_modular_snn_config


'''

python3 modular_snn/modular_snn_evaluation.py --process_mode="calibrate" --dataset="nordland" --num_labels=25 --num_cal_labels=600 --num_test_labels=2700 --num_query_imgs=3300 --skip=8 --offset_after_skip=0 --folder_id='NRD_SFS' --n_e=400 --epochs=60 --seed=0 
python3 modular_snn/modular_snn_evaluation.py --process_mode="test" --dataset="nordland" --num_labels=25 --num_cal_labels=600 --num_test_labels=2700 --num_query_imgs=3300 --skip=8 --offset_after_skip=600 --folder_id='NRD_SFS' --n_e=400 --epochs=60 --seed=0 


_1: python3 modular_snn/modular_snn_evaluation.py --process_mode="calibrate" --dataset='nordland' --num_labels=25 --num_cal_labels=600 --num_test_labels=2700 --num_query_imgs=3300 --skip=8 --offset_after_skip=0 --folder_id='NRD_SFS' --n_e=400 --epochs=80 --seed=0 
_1: python3 modular_snn/modular_snn_evaluation.py --process_mode="test" --dataset='nordland' --num_labels=25 --num_cal_labels=600 --num_test_labels=2700 --num_query_imgs=3300 --skip=8 --offset_after_skip=600 --folder_id='NRD_SFS' --n_e=400 --epochs=60 --seed=0 

_2: python3 modular_snn/modular_snn_evaluation.py --process_mode="calibrate" --dataset='nordland' --num_labels=25 --num_cal_labels=600 --num_test_labels=2700 --num_query_imgs=3300 --skip=8 --offset_after_skip=0 --folder_id='NRD_SFW' --n_e=400 --epochs=200 --seed=0 
_2: python3 modular_snn/modular_snn_evaluation.py --process_mode="test" --dataset='nordland' --num_labels=25 --num_cal_labels=600 --num_test_labels=2700 --num_query_imgs=3300 --skip=8 --offset_after_skip=600 --folder_id='NRD_SFW' --n_e=400 --epochs=60 --seed=0 

_3: python3 modular_snn/modular_snn_evaluation.py --process_mode="calibrate" --dataset='ORC' --num_labels=25 --num_cal_labels=75 --num_test_labels=375 --num_query_imgs=450 --skip=8 --offset_after_skip=0 --folder_id='ORC' --n_e=400 --epochs=200 --seed=0
_3: python3 modular_snn/modular_snn_evaluation.py --process_mode="test" --dataset='ORC' --num_labels=25 --num_cal_labels=75 --num_test_labels=375 --num_query_imgs=450 --skip=8 --offset_after_skip=75 --folder_id='ORC' --n_e=400 --epochs=60 --seed=0

_4: python3 modular_snn/modular_snn_evaluation.py --process_mode="calibrate" --dataset='SFU-Mountain' --num_labels=25 --num_cal_labels=75 --num_test_labels=300 --num_query_imgs=375 --skip=1 --offset_after_skip=0 --folder_id='SFU-Mountain' --n_e=400 --epochs=200 --seed=0 
_4: python3 modular_snn/modular_snn_evaluation.py --process_mode="test" --dataset='SFU-Mountain' --num_labels=25 --num_cal_labels=75 --num_test_labels=300 --num_query_imgs=375 --skip=1 --offset_after_skip=75 --folder_id='SFU-Mountain' --n_e=400 --epochs=10 --seed=0 

_5: python3 modular_snn/modular_snn_evaluation.py --process_mode="calibrate" --dataset='Synthia-NightToFall' --num_labels=25 --num_cal_labels=50 --num_test_labels=225 --num_query_imgs=275 --skip=1 --offset_after_skip=0 --folder_id='Synthia-NightToFall' --n_e=400 --epochs=200 --seed=0 
_5: python3 modular_snn/modular_snn_evaluation.py --process_mode="test" --dataset='Synthia-NightToFall' --num_labels=25 --num_cal_labels=50 --num_test_labels=225 --num_query_imgs=275 --skip=1 --offset_after_skip=50 --folder_id='Synthia-NightToFall' --n_e=400 --epochs=20 --seed=0 

_6: python3 modular_snn/modular_snn_evaluation.py --process_mode="calibrate" --dataset='St-Lucia' --num_labels=25 --num_cal_labels=50 --num_test_labels=300 --num_query_imgs=350 --skip=1 --offset_after_skip=0 --folder_id='St-Lucia' --n_e=400 --epochs=200 --seed=0
_6: python3 modular_snn/modular_snn_evaluation.py --process_mode="test" --dataset='St-Lucia' --num_labels=25 --num_cal_labels=50 --num_test_labels=300 --num_query_imgs=350 --skip=1 --offset_after_skip=50 --folder_id='St-Lucia' --n_e=400 --epochs=60 --seed=0


'''


def main(args):
    
    args.num_query_imgs = args.num_test_labels + args.num_cal_labels

    threshold_i_list = [0, 5, 10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]   

    args_multi_path_base = args.multi_path
    args_num_test_labels_base = args.num_test_labels
    args_offset_after_skip_base = args.offset_after_skip


    for threshold_i in threshold_i_list:

        args.threshold_i = threshold_i
        args.multi_path = args_multi_path_base.format(args.epochs, args.num_query_imgs, args.threshold_i)

        if args.process_mode == "calibrate": 
            
            args.num_test_labels = args.num_cal_labels 
            args.offset_after_skip = 0
            evaluate_modular_snn_config(args)
        
        
        elif args.process_mode == "test":
            
            args.num_test_labels = args_num_test_labels_base  
            args.offset_after_skip = args_offset_after_skip_base
            evaluate_modular_snn_config(args)
            



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
    parser.add_argument('--use_weighted_assignments', type=bool, default=False, 
                        help='Value to define the type of neuronal assignment to use: standard=False, weighted=True')
    parser.add_argument('--shuffled', type=bool, default=True, 
                        help='Value to define whether the order of input images should be shuffled: shuffled order of images=True, consecutive image order=False') 

    parser.add_argument('--skip', type=int, default=8, 
                        help='The number of images to skip between each place label.')
    parser.add_argument('--offset_after_skip', type=int, default=5, 
                        help='The offset to apply for selecting places after skipping every n images.')
    parser.add_argument('--folder_id', type=str, default="NRD_SFS", 
                        help='Id to distinguish the traverses used from the dataset.')
    parser.add_argument('--num_query_imgs', type=int, default=20, 
                        help='Number of query images used for testing and calibration.')
    
    parser.add_argument('--epochs', type=int, default=20, 
                        help='Number of passes through the dataset.')
    parser.add_argument('--n_e', type=int, default=100, 
                        help='Number of excitatory output neurons. The number of inhibitory neurons are defined the same.')
    parser.add_argument('--threshold_i', type=int, default=0, 
                        help='Threshold value used to ignore the hyperactive neurons.')
    parser.add_argument('--seed', type=int, default=0, 
                        help='Set seed for random generator.')
    
    parser.add_argument('--process_mode', type=str, choices=["calibrate", "test"], default="calibrate", 
                        help='String indicator to define the mode (calibrate, test).')
    
    parser.add_argument('--ad_path', type=str, default="_offset{}_S{}")             
    parser.add_argument('--multi_path', type=str, default="epoch{}_T{}_T{}")   
        

    parser.set_defaults()
    args = parser.parse_args()
    
    main(args)


            
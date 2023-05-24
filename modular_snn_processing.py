#! /usr/bin/env python

'''
MIT License

Copyright (c) 2022 Somayeh Hussaini, Michael Milford and Tobias Fischer

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
import wandb

from tools.wandb_utils import create_hpc_bashscript_wandb, get_wandb_sweep_id, setup_wandb_config



'''
To submit the jobs on hpc, simply:
- conda activate snn
- python modular_snn_processing.py or
- python modular_snn_processing.py --project_name="EnsSNN" --num_test_labels=100 --num_labels=25 --epochs=60 --dataset='nordland' --folder_id='NRD_SFS' --sweep_name='sweep_1'
- python modular_snn_processing.py --project_name="EnsSNN" --num_test_labels=2700 --num_labels=25 --epochs=60 --dataset='nordland' --folder_id='NRD_SFS' --sweep_name='sweep_2'
'''



print(wandb.__path__)


def main(args):
    

    offset_after_skip_list = list(range(0, args.num_cal_labels+args.num_test_labels, args.num_labels))
    
    ad_path_base = args.ad_path
    ad_path_test_base = args.ad_path_test
    args_multi_path_base = args.multi_path
    num_test_labels_base = args.num_test_labels

    if args.run_mode == 'wandb_hpc' or args.run_mode == 'wandb_local':
        sweep_config = setup_wandb_config(offset_after_skip_list, args)
        sweep_id = get_wandb_sweep_id(args, sweep_config)
        print("sweep id: ", sweep_id)

    for offset_after_skip in offset_after_skip_list:
        
        args.offset_after_skip = offset_after_skip
        args.num_test_labels = num_test_labels_base
        
        if args.run_mode == 'local':
            
            from single_snn_module_processing import main as process_single_module
            args.ad_path = ad_path_base
            args.ad_path_test = ad_path_test_base
            args.multi_path = args_multi_path_base
            process_single_module(args) 
        
    
        elif args.run_mode == 'wandb_hpc':
            
            create_hpc_bashscript_wandb(args, offset_after_skip_list, sweep_id)


        elif args.run_mode == 'wandb_local':

            from single_snn_module_processing import main as process_single_module

            # only pass sweep_id to function form but full path for terminal command 
            wandb.agent(sweep_id=sweep_id, project=args.project_name, function=process_single_module, count=len(offset_after_skip_list))







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
    parser.add_argument('--num_train_imgs', type=int, default=10, 
                        help='Number of entire training images.')
    parser.add_argument('--num_test_imgs', type=int, default=15, 
                        help='Number of entire testing images.')
    parser.add_argument('--first_epoch', type=int, default=200, 
                        help='For use of neuronal assignments, the first training iteration number in saved items.')
    parser.add_argument('--last_epoch', type=int, default=201, 
                        help='For use of neuronal assignments, the last training iteration number in saved items.')
    parser.add_argument('--update_interval', type=int, default=50, 
                        help='The number of iterations to save at one time in output matrix.')
    parser.add_argument('--epochs', type=int, default=20, 
                        help='Number of passes through the dataset.')
    parser.add_argument('--n_e', type=int, default=100, 
                        help='Number of excitatory output neurons. The number of inhibitory neurons are defined the same.')
    parser.add_argument('--threshold_i', type=int, default=0, 
                        help='Threshold value used to ignore the hyperactive neurons.')

    parser.add_argument('--ad_path_test', type=str, default="_test_E{}", 
                        help='Additional string arguments to use for saving test outputs in testing')
    parser.add_argument('--ad_path', type=str, default="_offset{}")             
    parser.add_argument('--multi_path', type=str, default="epoch{}_T{}_T{}")   
    
    parser.add_argument('--mode', type=str, default="test",  # "test", #"train", 
                        help='String indicator to define the mode (train, record, calibrate, test).')
    parser.add_argument('--run_mode', type=str, default="local", 
                        help='Mode to run the modular network.')
    parser.add_argument('--sweep_name', type=str, default="sweep_1", 
                        help='Wandb sweep name.')
    parser.add_argument('--project_name', type=str, default="modularSNN", 
                        help='Wandb project name.')
    
    parser.set_defaults()
    args = parser.parse_args()

    main(args)
    

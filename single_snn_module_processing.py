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
import numpy as np

from snn_model_tools.random_connection_generator import main as generate_random_connections
from snn_model import main as snn_model_main
from snn_model_tools.snn_model_evaluation import main as evaluate_snn_module





def main(args):    
    
    args.ad_path = args.ad_path.format(args.offset_after_skip, args.tc_gi) 
    
    num_test_labels_base = args.num_test_labels
    offset_after_skip_base = args.offset_after_skip
    
    if args.mode == "test":
        
        if offset_after_skip_base >= args.num_cal_labels: 
            
            args.ad_path_test = "_test_E{}".format(args.epochs)
            args.num_test_labels = num_test_labels_base
            args.offset_after_skip = args.num_cal_labels
            snn_model_main(args)
            
            args.first_epoch = (args.num_train_imgs*args.epochs)
            args.last_epoch = (args.num_train_imgs*args.epochs) + 1       
            args.multi_path = args.multi_path.format(args.epochs, args.num_test_labels, args.threshold_i)  
            args.offset_after_skip = offset_after_skip_base
            evaluate_snn_module(args)
        
    
    else: 
        # update the initial random values of connections 
        generate_random_connections(args)
        
        
        # Run the python script - train        
        args.mode = "train"
        args.ad_path_test = ""
        args.offset_after_skip = offset_after_skip_base
        snn_model_main(args)


        # Run the python script - record 
        args.mode = "record"
        args.ad_path_test = ""
        args.num_test_labels = args.num_cal_labels + num_test_labels_base
        args.offset_after_skip = 0
        snn_model_main(args)
        
                
        if offset_after_skip_base < args.num_cal_labels: 

            # Run the python script - Calibrate
            args.mode = "calibrate"
            args.ad_path_test = "_test_E{}".format(args.epochs)
            args.num_test_labels = args.num_cal_labels
            args.offset_after_skip = 0
            snn_model_main(args)
            
            args.first_epoch = (args.num_train_imgs*args.epochs)
            args.last_epoch = (args.num_train_imgs*args.epochs) + 1       
            args.multi_path = args.multi_path.format(args.epochs, args.num_test_labels, args.threshold_i)  
            evaluate_snn_module(args)




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

    parser.add_argument('--ad_path_test', type=str, default="_test_E{}", 
                        help='Additional string arguments to use for saving test outputs in testing')
    parser.add_argument('--ad_path', type=str, default="_offset{}")             
    parser.add_argument('--multi_path', type=str, default="epoch{}_T{}_T{}") 

    parser.add_argument('--mode', type=str, default="test", #"train", 
                        help='String indicator to define the mode (train, record, calibrate, test).')

    parser.set_defaults()
    args = parser.parse_args()
    
    
    main(args)



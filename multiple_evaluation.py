#! /usr/bin/env python
import argparse
import numpy as np

from DC_MNIST_evaluation import main as module_evaluation_main
from DC_MNIST_random_conn_generator import main as conn_generator_main
from snn_model import main as snn_model_main





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
            module_evaluation_main(args)
        
    
    else: 
        # update the initial random values of connections 
        conn_generator_main(args)
        
        
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
            module_evaluation_main(args)




if __name__ == "__main__":
    
    dataset = "nordland"
    num_test_labels = 10
    skip = 8 
    num_labels = 5

    folder_id = "NRD_SFS"
    epochs = 20
    n_e = 100

    tc_gi = 0.5
    offset_after_skip = 5
    
    intensity = 4
    num_cal_labels = 5
    update_interval = num_labels * 10

    num_train_imgs = num_labels * 2 if folder_id == 'NRD_SFS' or folder_id == 'ORC' else num_labels
    num_test_imgs = num_test_labels

    tc_ge = 1.0
    first_epoch = num_train_imgs * epochs
    last_epoch = first_epoch + 1 


    ad_path_test = "_test_E{}"
    ad_path = "_offset{}"               
    multi_path = "epoch{}_T{}_T{}"


    mode = "train"
    use_weighted_assignments = False
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=dataset, help='Folder name of dataset to be used. Relative to this repo, the folder must exist in this directory: ./../data/')
    parser.add_argument('--num_labels', type=int, default=num_labels, help='Number of distinct places to use from the dataset.')
    parser.add_argument('--num_test_labels', type=int, default=num_test_labels, help='Number of distinct places to use from the dataset for testing.')
    parser.add_argument('--tc_ge', type=float, default=tc_ge, help='Time constant of conductance of excitatory synapses AeAi')
    parser.add_argument('--tc_gi', type=float, default=tc_gi, help='Time constant of conductance of inhibitory synapses AiAe')
    parser.add_argument('--intensity', type=int, default=intensity, help="Intensity scaling to change the range of input pixel values")
    parser.add_argument('--num_cal_labels', type=int, default=num_cal_labels, help="Number of images needed for calibration. Needed for shuffling input images.")
    
    parser.add_argument('--mode', type=str, default=mode, help='String indicator to define the mode (train, record, calibrate, test).')
    parser.add_argument('--use_weighted_assignments', type=bool, default=use_weighted_assignments, help='Value to define the type of neuronal assignment to use: standard=0, weighted=1')

    parser.add_argument('--skip', type=int, default=skip, help='The number of images to skip between each place label.')
    parser.add_argument('--offset_after_skip', type=int, default=offset_after_skip, help='The offset to apply for selecting places after skipping every n images.')
    parser.add_argument('--folder_id', type=str, default=folder_id, help='Folder name of dataset to be used.')
    parser.add_argument('--num_train_imgs', type=int, default=num_train_imgs, help='Number of entire training images.')
    parser.add_argument('--num_test_imgs', type=int, default=num_test_imgs, help='Number of entire testing images.')
    parser.add_argument('--first_epoch', type=int, default=first_epoch, help='For use of neuronal assignments, the first training iteration number in saved items.')
    parser.add_argument('--last_epoch', type=int, default=last_epoch, help='For use of neuronal assignments, the last training iteration number in saved items.')
    parser.add_argument('--update_interval', type=int, default=update_interval, help='The number of iterations to save at one time in output matrix.')
    parser.add_argument('--epochs', type=int, default=epochs, help='Number of passes through the dataset.')
    parser.add_argument('--n_e', type=int, default=n_e, help='Number of excitatory output neurons. The number of inhibitory neurons are the same.')

    parser.add_argument('--ad_path_test', type=str, default=ad_path_test, help='Additional string arguments to use for saving test outputs in testing')
    parser.add_argument('--ad_path', type=str, default=ad_path)             
    parser.add_argument('--multi_path', type=str, default=multi_path)   


    parser.set_defaults()
    args = parser.parse_args()
    
    
    main(args)



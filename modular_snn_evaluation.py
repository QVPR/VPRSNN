#! /usr/bin/env python
import argparse
import os
import time


from modular_snn_config_evaluation import main as evaluate_modular_snn_config


'''

python3 modular_snn_evaluation.py --num_test_imgs=100 --num_labels=25 --folder_id='NRD_SFS' --offset_after_skip=600 --skip=8 --multi_path='epoch{}_T{}_T{}' 
python3 modular_snn_evaluation.py --num_test_imgs=2700 --num_labels=25 --folder_id='NRD_SFS' --offset_after_skip=600 --skip=8 --multi_path='epoch{}_T{}_T{}' 


'''




def main(args, use_all_data=False):
    

    args.num_train_imgs = args.num_labels * 2 if args.folder_id == 'NRD_SFS' or args.folder_id == 'NRD_SFW' or args.folder_id == 'ORC' else args.num_labels
    args.org_num_test_imgs = args.num_test_imgs + args.num_cal_labels

    threshold_i_list = [0, 5, 10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]   

    args_multi_path_base = args.multi_path
    args_num_test_imgs_base = args.num_test_imgs
    args_offset_after_skip_base = args.offset_after_skip


    for threshold_i in threshold_i_list:

        args.threshold_i = threshold_i
        
        if use_all_data: 
            args.num_test_imgs = args_num_test_imgs_base
            args.offset_after_skip = 0
            args.multi_path = args_multi_path_base.format(args.epochs, args.num_test_imgs, args.threshold_i)
            
            evaluate_modular_snn_config(args)
            
            
        else: 
            args.num_test_imgs = args.num_cal_labels 
            args.offset_after_skip = 0
            args.multi_path = args_multi_path_base.format(args.epochs, args.org_num_test_imgs, args.threshold_i)
            
            evaluate_modular_snn_config(args)


            args.num_test_imgs = args_num_test_imgs_base  
            args.offset_after_skip = args_offset_after_skip_base
            args.multi_path = args_multi_path_base.format(args.epochs, args.org_num_test_imgs, args.threshold_i)
            
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
    parser.add_argument('--use_weighted_assignments', type=bool, default=False, 
                        help='Value to define the type of neuronal assignment to use: standard=False, weighted=True') 

    parser.add_argument('--skip', type=int, default=8, 
                        help='The number of images to skip between each place label.')
    parser.add_argument('--offset_after_skip', type=int, default=5, 
                        help='The offset to apply for selecting places after skipping every n images.')
    parser.add_argument('--folder_id', type=str, default="NRD_SFS", 
                        help='Id to distinguish the traverses used from the dataset.')
    parser.add_argument('--num_train_imgs', type=int, default=10, 
                        help='Number of entire training images.')
    parser.add_argument('--num_test_imgs', type=int, default=15, 
                        help='Number of entire testing images.')
    parser.add_argument('--org_num_test_imgs', type=int, default=20, 
                        help='Number of entire testing images and calibration images.')
    
    parser.add_argument('--epochs', type=int, default=20, 
                        help='Number of passes through the dataset.')
    parser.add_argument('--n_e', type=int, default=100, 
                        help='Number of excitatory output neurons. The number of inhibitory neurons are defined the same.')
    parser.add_argument('--threshold_i', type=int, default=0, 
                        help='Threshold value used to ignore the hyperactive neurons.')
    
    parser.add_argument('--ad_path', type=str, default="_offset{}")             
    parser.add_argument('--multi_path', type=str, default="epoch{}_T{}_T{}")   
        

    parser.set_defaults()
    args = parser.parse_args()
    
    main(args)


            
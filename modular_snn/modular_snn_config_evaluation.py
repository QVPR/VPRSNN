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
import os.path
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


from non_modular_snn.snn_model_evaluation import (get_accuracy,
                                    get_recognized_number_ranking, 
                                    get_training_neuronal_spikes,
                                    compute_recall, invert_dMat,
                                    compute_binary_distance_matrix, 
                                    compute_distance_matrix, 
                                    plot_precision_recall)

from tools.logger import Logger
matplotlib.rcParams['ps.fonttype'] = 42




def main(args): 
    
    use_precomputed = False 
    
    merged_path = './outputs/outputs_ne{}_L{}'
    assignment_types = ["standard", "weighted"]   
    NA_name = assignment_types[args.use_weighted_assignments] 
    
    print(args)

    offset_after_skip_list = np.arange(args.offset_after_skip, args.num_test_labels+args.offset_after_skip, args.num_labels)
    print("Offset after skip: ", offset_after_skip_list)
    
    test_labels = []
    validation_result_monitor = []
    testing_result_monitor = []
    assignments = []
    all_assignments = []


    for offset_after_skip in offset_after_skip_list:
        
        results_path = './outputs/outputs_ne{}_L{}'.format(args.n_e, args.num_labels) + args.ad_path.format(offset_after_skip) + '/'        
        data_path =  results_path + NA_name + "/" + args.multi_path + "/" 
        
        validation_result_filename = results_path + "resultPopVecs{}.npy".format(args.num_query_imgs)
        testing_result_filename = results_path + "resultPopVecs{}_test_E{}.npy".format(args.num_test_labels, args.epochs)
        print("Validation result monitor (available = {}): {}".format(os.path.isfile(validation_result_filename), validation_result_filename) )
        print("Testing result monitor (available = {}): {}".format(os.path.isfile(testing_result_filename), testing_result_filename) )
        validation_result_monitor.append(np.load(validation_result_filename))
        testing_result_monitor.append(np.load(testing_result_filename))
        
        test_labels_i = np.arange(offset_after_skip, offset_after_skip+args.num_labels)
        test_labels.append(test_labels_i)
        
        path_id = 'L{}_S{}_O{}'.format(args.num_labels, args.skip, int(offset_after_skip))
        assignments_filepath = results_path + "assignments_{}.npy".format(path_id)
        all_assignments_filepath = results_path + "all_assignments_{}.npy".format(path_id)

        print("Assignments (available = {}): {}".format(os.path.isfile(assignments_filepath), assignments_filepath) )
        print("All assignments (available = {}): {}".format(os.path.isfile(all_assignments_filepath), all_assignments_filepath) )
        
        assignments.append(np.load(assignments_filepath))
        all_assignments.append(np.load(all_assignments_filepath, allow_pickle=True))


    if np.any(np.array(test_labels)):
        test_labels = np.concatenate(tuple(test_labels))
 

    num_offsets = len(offset_after_skip_list)
    num_neurons = args.n_e*num_offsets
    num_labels_all = args.num_labels*num_offsets 
    
    
    data_path = merged_path.format(num_neurons, num_labels_all) + args.ad_path.format(offset_after_skip_list[-1]) + "_M2/"
    print(data_path)
    Path(data_path).mkdir(parents=True, exist_ok=True)  
    
    data_path += "{}/".format(NA_name)
    Path(data_path).mkdir(parents=True, exist_ok=True)

    data_path += args.multi_path + "/"
    Path(data_path).mkdir(parents=True, exist_ok=True)
    
    sys.stdout = Logger(data_path, logfile_name="logfile_evaluation_merged")
    print("\nArgument values:\n{}".format(args))
    

    if not use_precomputed:         
        validation_result_monitor = np.concatenate(tuple(validation_result_monitor), axis=1)            
        testing_result_monitor_all = np.concatenate(tuple(testing_result_monitor), axis=1)           
        assignments = np.concatenate(np.array(assignments))
        all_assignments = np.concatenate(np.array(all_assignments)) 

        np.save(data_path + "validation_result_monitor.npy", validation_result_monitor)
        np.save(data_path + "testing_result_monitor_all.npy", testing_result_monitor_all)
        np.save(data_path + "assignments.npy", assignments)
        np.save(data_path + "all_assignments.npy", all_assignments)
    
    else:
        validation_result_monitor = np.load(data_path + "validation_result_monitor.npy")
        testing_result_monitor_all = np.load(data_path + "testing_result_monitor_all.npy")    
        assignments = np.load(data_path + "assignments.npy")
        all_assignments = np.load(data_path + "all_assignments.npy", allow_pickle=True)


    unique_assignments = np.unique(assignments)
    neuron_spikes = np.count_nonzero(validation_result_monitor, axis=0)        
    testing_result_monitor_all = ignore_hyperactive_neurons(neuron_spikes, testing_result_monitor_all, args.threshold_i)

    if np.all(testing_result_monitor_all == 0):
        print("All neurons are categorised as hyperactive with a threshold of {} spikes (based on their responses to the reference data). Exiting...".format(args.threshold_i))
        return
        
    if not use_precomputed:        
            
        test_results = np.zeros((len(unique_assignments), args.num_test_labels))
        summed_rates = np.zeros((len(unique_assignments), args.num_test_labels))
        
        sum_train_spikes_list, train_spikes_list, learnt_neurons_list, len_learnt_labels_list = get_training_neuronal_spikes(unique_assignments, args.use_weighted_assignments, all_assignments)

        for i in range(args.num_test_labels):
            test_results[:,i], summed_rates[:,i] = get_recognized_number_ranking(assignments, testing_result_monitor_all[i,:], unique_assignments, sum_train_spikes_list, train_spikes_list, learnt_neurons_list, len_learnt_labels_list, args.use_weighted_assignments)

        np.save(data_path + "summed_rates.npy", summed_rates)
        
    else:
        summed_rates = np.load(data_path + "summed_rates.npy")
        sorted_indices = np.argsort(summed_rates, axis=0)[::-1]
        test_results = unique_assignments[sorted_indices]

    print("summed_rates shape: {}".format(summed_rates.shape))
    print("test_labels shape: {}".format(np.array(test_labels).shape))
    
    # Invert the scale of the distance matrix 
    rates_matrix = invert_dMat(summed_rates)
    sorted_pred_idx = np.argsort(rates_matrix, axis=0)

    # compute recall at N
    n_values = [1, 5, 10, 15, 20, 25]
    numQ = test_labels.shape[0]
    gt_labels = np.arange(test_labels.shape[0])
    recallAtN = compute_recall(gt_labels, sorted_pred_idx, numQ, n_values, data_path, name="recallAtN_SNN.npy")
    plot_recallAtN(data_path, n_values, recallAtN, "recallAtN_plot")
        
    difference = test_results[0,:] - test_labels
    tolerance = 0
    correct, incorrect, accurracy = get_accuracy(abs(difference), tolerance=tolerance)
    print( "\nTolerance: {}, accuracy: {}, num correct: {}, num incorrect: {}".format(tolerance, np.mean(accurracy), len(correct), len(incorrect)) )
    
    
    # Get Distance matrices 
    dMat = compute_binary_distance_matrix(summed_rates)
    plot_name = "DM_{}_ne{}_L{}".format(args.folder_id, num_neurons, args.num_labels)
    compute_distance_matrix(dMat, data_path, name="binary_distMatrix")
    compute_distance_matrix(summed_rates, data_path, "distMatrix_spike_rates")
    compute_distance_matrix(rates_matrix, data_path, plot_name)


    if summed_rates.shape[0] != summed_rates.shape[1]:
        return

    
    # Plot PR Curves 
    sn.set_context("paper", font_scale=2, rc={"lines.linewidth": 2})
    fig_name = "PR_{}_{}".format(args.folder_id, path_id)
    plot_precision_recall(rates_matrix, data_path, fig_name=fig_name, label='SNN_T{}'.format(args.threshold_i), png_ending=".png")

    return 




def plot_recallAtN(data_path, n_values, recallAtN, filename, label=''):
    
    sn.set_context("paper", font_scale=2, rc={"lines.linewidth": 2})
    
    fig, ax = plt.subplots()            
    ax.plot(n_values, list(recallAtN.values()), label=label)
    
    ax.set_xlabel("N values")
    ax.set_ylabel("Recall at N")
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(data_path, filename))
    plt.close()



def ignore_hyperactive_neurons(neuron_spikes, testing_result_monitor, threshold_val):
    
    testing_result_monitor = np.copy(testing_result_monitor)
    
    if threshold_val != 0: 

        for n in range(testing_result_monitor.shape[1]):
            if neuron_spikes[n] < threshold_val: 
                continue 
            
            # set the response of the neuron to 0 for all query images 
            testing_result_monitor[:, n] = np.zeros(testing_result_monitor.shape[0])

    return testing_result_monitor

    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', type=str, default="nordland", 
                        help='Dataset folder name that is relative to this repo. The folder must exist in this directory: ./../data/')
    parser.add_argument('--num_labels', type=int, default=5, 
                        help='Number of training place labels for a single module.')
    parser.add_argument('--num_test_labels', type=int, default=5, 
                        help='Number of testing place labels.')
    parser.add_argument('--use_weighted_assignments', type=bool, default=False, 
                        help='Value to define the type of neuronal assignment to use: standard=False, weighted=True') 
    
    parser.add_argument('--skip', type=int, default=8, 
                        help='The number of images to skip between each place label.')
    parser.add_argument('--offset_after_skip', type=int, default=0, 
                        help='The offset to apply for selecting places after skipping every n images.')
    parser.add_argument('--folder_id', type=str, default="NRD_SFS", 
                        help='Id to distinguish the traverses used from the dataset.')
    parser.add_argument('--num_query_imgs', type=int, default=15, 
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

    

    
    
    
    
    
    
    
    
    
    

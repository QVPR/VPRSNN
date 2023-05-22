#! /usr/bin/env python
import argparse
import os
import os.path
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn


from DC_MNIST_evaluation import (Logger, get_accuracy,
                                    get_recognized_number_ranking, 
                                    get_training_neuronal_spikes,
                                    compute_recall, invert_dMat,
                                    compute_binary_distance_matrix, 
                                    compute_distance_matrix, 
                                    plot_precision_recall)


matplotlib.rcParams['ps.fonttype'] = 42

use_latex = False

if use_latex:
    plt.rcParams.update({"text.usetex": True, "font.family": "serif", "font.sans-serif": ["Helvetica"]})
    custom_preamble = {
        "text.latex.preamble": r"\usepackage{amsmath}",  # for the align, center,... environment
    }
    plt.rcParams.update(custom_preamble)





def main(args): 
    
    use_precomputed = False 
    
    merged_path = './outputs/outputs_ne{}_L{}'
    assignment_types = ["standard", "weighted"]   
    NA_name = assignment_types[args.use_weighted_assignments] 
    
    print(args)

    offset_after_skip_list = np.arange(args.offset_after_skip, args.num_test_imgs+args.offset_after_skip, args.num_labels)
    print("Offset after skip: ", offset_after_skip_list)
    
    
    test_labels = []
    validation_result_monitor = []
    testing_result_monitor = []
    assignments = []
    all_assignments = []


    for offset_after_skip in offset_after_skip_list:
        
        results_path = './outputs/outputs_ne{}_L{}'.format(args.n_e, args.num_labels) + args.ad_path.format(offset_after_skip, args.tc_gi, args.seed) + '/'        
        data_path =  results_path + NA_name + "/" + args.multi_path + "/" 
        
        validation_result_filename = results_path + "resultPopVecs{}.npy".format(args.org_num_test_imgs)
        testing_result_filename = results_path + "resultPopVecs{}_test_E{}.npy".format(args.num_test_imgs, args.epochs)
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
    
    
    data_path = merged_path.format(num_neurons, num_labels_all) + args.ad_path.format(offset_after_skip_list[-1], args.tc_gi, args.seed) + "_M2/"
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

        
    if not use_precomputed:        
            
        test_results = np.zeros((len(unique_assignments), args.num_test_imgs))
        summed_rates = np.zeros((len(unique_assignments), args.num_test_imgs))
        
        sum_train_spikes_list, train_spikes_list, learnt_neurons_list, len_learnt_labels_list = get_training_neuronal_spikes(unique_assignments, args.use_weighted_assignments, all_assignments)

        for i in range(args.num_test_imgs):
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
    numQ = summed_rates.shape[0]
    gt_labels = np.arange(summed_rates.shape[0])
    recallAtN = compute_recall(gt_labels, sorted_pred_idx, numQ, n_values, data_path, name="recallAtN_SNN.npy")
    plot_recallAtN(data_path, n_values, recallAtN, "recallAtN_plot")
        
    difference = test_results[0,:] - gt_labels[:len(test_results[0,:])]
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
    
    
    val_mode = False
    test_mode = False 

    skip = 8
    offset_after_skip = 600 
    folder_id = 'NRD_SFS' 
    dataset = "nordland"
    epochs = 60
    num_labels = 25 
    num_train_imgs = num_labels * 2 if folder_id == 'NRD_SFS' or folder_id == 'ORC' else num_labels
    num_test_imgs = 3300 - offset_after_skip
    org_num_test_imgs = 3300
    use_weighted_assignments = 0
    n_e = 400
    threshold_i = 40
    tc_gi = 0.5
    seed = 0

    ad_path = '_offset{}'
    multi_path = 'epoch{}_T{}_T{}'.format(epochs, org_num_test_imgs, threshold_i) 


    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=dataset, help='Folder name of dataset to be used. Relative to this repo, the folder must exist in this directory: ./../data/')
    parser.add_argument('--skip', type=int, default=skip, help='The number of images to skip between each place label.')
    parser.add_argument('--offset_after_skip', type=int, default=offset_after_skip, help='The offset to apply for selecting places after skipping every n images.')
    parser.add_argument('--folder_id', type=str, default=folder_id, help='Folder name of dataset to be used.')
    parser.add_argument('--num_train_imgs', type=int, default=num_train_imgs, help='Number of entire training images.')
    parser.add_argument('--num_test_imgs', type=int, default=num_test_imgs, help='Number of entire testing images.')
    parser.add_argument('--org_num_test_imgs', type=int, default=org_num_test_imgs, help='Number of entire testing images for both cal and testing.')
    parser.add_argument('--num_labels', type=int, default=num_labels, help='Number of distinct places to use from the dataset.')
    parser.add_argument('--epochs', type=int, default=epochs, help='Number of sweeps through the dataset in training.')
    parser.add_argument('--use_weighted_assignments', type=bool, default=use_weighted_assignments, help='Value to define the type of neuronal assignment to use: standard=0, weighted=1')
    parser.add_argument('--n_e', type=int, default=n_e, help='Number of excitatory output neurons. The number of inhibitory neurons are the same.')
    parser.add_argument('--threshold_i', type=int, default=threshold_i, help='Threshold value to ignore hyperactive neurons.')
    parser.add_argument('--tc_gi', type=float, default=tc_gi, help='Time constant of conductance of inhibitory synapses AiAe')
    parser.add_argument('--seed', type=int, default=seed, help='Set seed for random generator.')

    parser.add_argument('--val_mode', dest="val_mode", action="store_true", help='Boolean indicator to switch to validation mode.')
    parser.add_argument('--test_mode', dest="test_mode", action="store_true", help='Boolean indicator to switch between training and testing mode.')

    parser.add_argument('--ad_path', type=str, default=ad_path, help='Additional string arguments for folder names to save evaluation outputs')         # _tcgi{:3.1f}
    parser.add_argument('--multi_path', type=str, default=multi_path, help='Additional string arguments for subfolder names to save evaluation outputs.')

    parser.set_defaults()
    args = parser.parse_args()

    main(args)

    

    
    
    
    
    
    
    
    
    
    

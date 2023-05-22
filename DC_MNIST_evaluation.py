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
import os
import os.path
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
from sklearn.metrics import auc


class Logger(object):
    '''
    Class to save the terminal output into a file
    '''
    def __init__(self, outputsPath, logfile_name):
        self.terminal = sys.stdout
        self.log = open(outputsPath + "{}.log".format(logfile_name), "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass 
    


def get_training_neuronal_spikes(unique_assignments, use_weighted_na, all_assignments=[]):
    '''
    Get the characteristic information of output spikes in training 
    '''
    
    len_all_assignments = len(all_assignments) 
    
    sum_train_spikes_list = []
    train_spikes_list = []
    learnt_neurons_list = []
    len_learnt_labels_list = []  
    
    if use_weighted_na != 1:
        return sum_train_spikes_list, train_spikes_list, learnt_neurons_list, len_learnt_labels_list

    for v in unique_assignments:
        
        all_train_spikes_all = np.array([ [int(x), all_assignments[x]] for x in range(len_all_assignments) if v in all_assignments[x]])
        all_train_spikes = np.array([ x for x in all_train_spikes_all if (x[1].get(v) / sum(list(x[1].values()))) > 0.1 ])  
        
        sum_train_spikes = np.array([ sum(list(x[1].values()))  for x in all_train_spikes ])
        sum_train_spikes_list.append(sum_train_spikes)
        
        train_spikes = np.array([ x[1].get(v) for x in all_train_spikes ])
        train_spikes_list.append(train_spikes)

        learnt_neurons = all_train_spikes[:, 0].astype(int) if len(all_train_spikes.shape) == 2 else all_train_spikes[:] 
        learnt_neurons_list.append(learnt_neurons)
        
        len_learnt_labels = np.array([ len(list(x[1].values()))  for x in all_train_spikes ])
        len_learnt_labels_list.append(len_learnt_labels)


    return sum_train_spikes_list, train_spikes_list, learnt_neurons_list, len_learnt_labels_list    


def get_recognized_number_ranking(assignments, spike_rates, unique_assignments, sum_train_spikes_list=[], train_spikes_list=[], learnt_neurons_list=[], len_learnt_labels_list=[], use_weighted_na=0):
    '''
    Given the neuronal assignments, and the number of spikes fired for the image with the given label in testing, 
    predict the label of the corresponding image  
    '''
    summed_rates = np.zeros(len(unique_assignments))
    
    if np.all(spike_rates == 0):
        sorted_summed_rates = np.arange(-1, len(unique_assignments)-1)
        return sorted_summed_rates, summed_rates

    for i, v in enumerate(unique_assignments):
        assigned_neurons_indices = np.where(assignments == v)[0]  
        num_assignments = len(assigned_neurons_indices)

        if num_assignments == 0:            
            continue 
        
        test_spikes = spike_rates[assignments == v]
        
        # 0: standard assignments 
        if use_weighted_na == 0:            
            summed_rates[i] = ( np.sum(test_spikes) / num_assignments )

        # 1: weighted assignments 
        elif use_weighted_na == 1:
            
            sum_train_spikes = sum_train_spikes_list[i]
            train_spikes = train_spikes_list[i]
            learnt_neurons = learnt_neurons_list[i]
            
            len_learnt_labels = len_learnt_labels_list[i]
            len_learnt_labels_reciprocal = (1/len_learnt_labels)
            learnt_label_ratio = train_spikes / sum_train_spikes
            
            test_spikes = np.array(spike_rates[learnt_neurons])
            norm_factor = ( np.sum(train_spikes[test_spikes != 0])  /  np.sum(train_spikes) ) if np.any(train_spikes) else 0 

            # a) Regularization by involvement 
            regularised_indices = len_learnt_labels >= 0.02*len(unique_assignments)
            test_spikes[regularised_indices] = test_spikes[regularised_indices] * len_learnt_labels_reciprocal[regularised_indices]
            
            # b) Normalization by response strength             
            test_spikes = test_spikes * learnt_label_ratio
            
            # c) Penalize relevant neurons that did not fire 
            summed_rates[i] =  ( np.sum(test_spikes) ) * norm_factor if len(learnt_neurons) > 0 else 0   
        
        # 2: spikes assignments 
        else:
            summed_rates[i] = np.sum(test_spikes)                       
    
    if np.all(summed_rates == 0):
        sorted_summed_rates = np.arange(-1, len(unique_assignments)-1)
    else: 
        sorted_indices = np.argsort(summed_rates)[::-1]
        sorted_summed_rates = unique_assignments[sorted_indices]
        
    return sorted_summed_rates, summed_rates


def get_new_assignments(result_monitor, input_numbers, n_e):
    '''
    Update the assignments of classes to neurons based on the spiking rate 
    '''

    assignments = np.ones(n_e) * -1 
    maximum_rate = np.zeros(n_e)  

    for j in np.unique(input_numbers):
        num_assignments = len(np.where(input_numbers == j)[0])

        if num_assignments > 0:
            rate = np.sum(result_monitor[input_numbers == j], axis = 0) / num_assignments
        else:
            rate = np.zeros((n_e, 1))

        for i in range(n_e):
            if rate[i] > maximum_rate[i]:
                maximum_rate[i] = rate[i]
                assignments[i] = j

    return assignments



def get_new_assignments_weighted(result_monitor, input_numbers, n_e):
    '''
    Update the assignments of all classes associated with neurons based on the spiking rate 
    '''
    
    assignments = [{} for _ in range(n_e)] 

    for j in np.unique(input_numbers):
        num_assignments = len(np.where(input_numbers == j)[0])
        maximum_rate = np.zeros(n_e) 

        if num_assignments > 0:
            num_spikes = np.sum(result_monitor[input_numbers == j], axis = 0)
            rate = num_spikes / num_assignments
        else:
            rate = np.zeros((n_e, 1))
            
        for i in range(n_e):
            if rate[i] > maximum_rate[i]:
                maximum_rate[i] = rate[i]                
                if j in assignments[i]:
                    assignments[i][j] += num_spikes[i]
                else:  
                    assignments[i][j] = num_spikes[i] 

    top_assignments = np.array([-1 if not assignments[i] else max(assignments[i], key=assignments[i].get) for i in range(len(assignments))])

    return top_assignments, assignments


def compute_distance_matrix(dMat, data_path, name, png_ending=".png"):
    
    fig, ax = plt.subplots()
    sn.set_context("paper", font_scale=2)

    im = ax.matshow(dMat, cmap=plt.cm.Blues)

    cbar = ax.figure.colorbar(im, ax=ax)
    cbarlabel = "Distance"
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    ax.xaxis.set_ticks_position('bottom')

    plt.xlabel("Output labels")
    plt.ylabel("True labels")
    plt.tight_layout()
    plt.savefig(data_path + name + png_ending)
    plt.close()


def compute_binary_distance_matrix(summed_rates):
    
    binary_summed_rates = np.ones(summed_rates.shape)
    
    for col in range(summed_rates.shape[1]):
        idx_max = np.argmax(summed_rates[:, col])
        binary_summed_rates[idx_max, col] = 0 

    return binary_summed_rates


def get_accuracy(difference, tolerance):

    correct = np.where(difference <= tolerance)[0]
    incorrect = np.where(difference > tolerance)[0]
    accurracy = len(correct)/(len(correct)+len(incorrect)) * 100

    return correct, incorrect, accurracy


def get_precision_recall_f1_score(dMat):

    all_labels = np.arange(len(dMat))
    gt = np.abs(all_labels - np.argmin(dMat, axis=0)) 
    
    mInds = np.argmin(dMat, axis=0)
    mDists = np.min(dMat, axis=0)

    min_val = np.min(dMat)
    max_val = np.max(dMat)

    thresholds = np.linspace(min_val, max_val, 99)
    precision = []
    recall = []
    f1_scores = [] 

    precision.append(1)
    recall.append(0)

    for threshold in thresholds:

        # get boolean of all indices of dists whose value is <= threshold 
        matchFlags = mDists <= threshold

        # set all unmatched items to -1 in a fresh copy of mInds 
        mInds_filtered = np.copy(mInds)
        mInds_filtered[~matchFlags] = -1 

        # get positives: matched mInds whose distance <= threshold 
        positives = np.argwhere(mInds_filtered!=-1)[:,0]
        tps = np.sum( gt[positives] == 0 )
        fps = len(positives) - tps 

        if tps == 0:
            precision.append(0)
            recall.append(0)

        # get negatives: matched mInds whose distance > threshold 
        negatives = np.argwhere(mInds_filtered==-1)[:,0]
        tns = np.sum( gt[negatives] < 0 )
        fns = len(negatives) - tns 

        assert(tps+tns+fps+fns==len(gt))

        precision_i = tps / float(tps+fps)
        recall_i = tps / float(tps+fns)
        f1_score = (2*precision_i*recall_i) / (precision_i+recall_i)

        precision.append( precision_i )
        recall.append(recall_i)
        f1_scores.append(f1_score)
        print( 'tps: {}, fps:{}, fns:{}, tns:{}, tot:{}, T:{:.2f}, P:{:.2f}, R:{:.2f}'.format(tps, fps, fns, tns, tps+fps+fns+tns, threshold, precision_i, recall_i) ) 
    
    precision = np.array(precision)
    recall = np.array(recall)
    f1_scores = np.array(f1_scores)
    
    return precision, recall, f1_scores


def getAUCPR(precision, recall):
    AUC_PRs = auc(recall, precision)

    return AUC_PRs


def getRat100P(precision, recall):

    if not np.any(precision >= 1.0): 
        return 0
    Rat100P = np.max(recall[precision >= 1.0])

    return Rat100P


def plot_precision_recall(dMat, data_path, fig_name='', label='', png_ending=".png"):
    
    precision, recall, f1_scores = get_precision_recall_f1_score(dMat)

    AUC_PRs = getAUCPR(precision, recall)
    print("\nArea under precision recall curve: \nsklearn AUC: {:.4f}\n".format(AUC_PRs))

    Rat100P = getRat100P(precision, recall)
    print("\nR @ 100% P is: {:.4f}\n".format(Rat100P))

    Pat100R = f1_scores[-1]
    print("\nP a@ 100% R (last item of f1-score) is: {:.4f}\n".format(Pat100R))

    sn.set_context("paper", font_scale=2, rc={"lines.linewidth": 2})

    plt.figure()
    plt.plot(recall, precision, label=label)
    plt.xlabel("Recall")
    plt.ylabel("Precision")

    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    plt.legend() 
    plt.tight_layout()
    plt.savefig(data_path + "{}".format(fig_name) + png_ending)
    plt.close()

    plt.figure()
    plt.plot(recall, precision, label=label)
    plt.plot(recall, precision, '*r')

    plt.xlabel("Recall")
    plt.ylabel("Precision")

    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(data_path + "{}_show_thresholds".format(fig_name) + png_ending)
    plt.close()

    return Rat100P, AUC_PRs


def compute_recall(gt, sorted_pred, numQ, n_values, data_path, name, allow_save=True): 

    correct_at_n = np.zeros(len(n_values)) 
    sorted_pred_transposed = np.transpose(sorted_pred)
    for qIx, pred in enumerate(sorted_pred_transposed):
        for i, n in enumerate(n_values):
            # if in top N then also in top NN, where NN > N
            if np.any(np.in1d(pred[:n], gt[qIx])):
                correct_at_n[i:] += 1
                break
    recall_at_n = correct_at_n / numQ
    all_recalls = {}  # make dict for output
    for i, n in enumerate(n_values):
        all_recalls[n] = recall_at_n[i]
        print("====> Recall {}@{}: {:.4f}".format("Recall", n, recall_at_n[i]))

    if allow_save:
        np.save(data_path + name, all_recalls)
    return all_recalls


def invert_dMat(dMat):
    '''
    Inverts the scale of the given distance matrix
    '''

    max_dMat = np.max(dMat)
    inverted_dMat = np.zeros_like(dMat)
    
    inverted_dMat = max_dMat - dMat
    
    return inverted_dMat 

                

def main(args):
    
    
    data_path = './outputs/outputs_ne{}_L{}'.format(args.n_e, args.num_labels) + args.ad_path + '/' 
    path_id = 'L{}_S{}_O{}'.format(args.num_labels, args.skip, args.offset_after_skip)

    main_folder_path = data_path 
    Path(main_folder_path).mkdir(parents=True, exist_ok=True)

    folder_id = args.folder_id 
    num_training_imgs = args.num_train_imgs 
    num_testing_imgs = args.num_test_imgs    
    first_epoch = args.first_epoch          
    last_epoch = args.last_epoch           
    update_interval = args.update_interval  
    num_training_sweeps = args.epochs 
    n_e = args.n_e
    use_weighted_assignments = args.use_weighted_assignments    
    

    result_type = "weighted/" if use_weighted_assignments else "standard/"  
    data_path += result_type
    Path(data_path).mkdir(parents=True, exist_ok=True)

    multiple_UI_assignments = use_weighted_assignments
 
    if args.multi_path != "": 
        data_path += args.multi_path + "/"

    Path(data_path).mkdir(parents=True, exist_ok=True)
    sys.stdout = Logger(data_path, logfile_name="logfile_evaluation")
    print(args)
    ending = '.npy'

    training_ending = str( int(num_training_imgs * num_training_sweeps )) 
    testing_ending = str(num_testing_imgs )  

    training_result_monitor = np.load(main_folder_path + 'resultPopVecs' + training_ending + ending)
    epoch_list = np.arange(first_epoch, last_epoch, update_interval) 

    if multiple_UI_assignments:
        for x in epoch_list:
        
            if not os.path.isfile(data_path + 'resultPopVecs' + str(int(x)) + ending):
                if x == epoch_list[-1]:
                    training_result_monitors = training_result_monitor
                continue
            training_result_monitors_i = np.load(data_path + 'resultPopVecs' + str(int(x)) + ending)

            if x == epoch_list[0]:
                training_result_monitors = training_result_monitors_i
            else:
                training_result_monitors = np.append(training_result_monitors, training_result_monitors_i, axis=0)
                print("training result shape: ", training_result_monitors.shape )
    else:
        training_result_monitors = training_result_monitor
        
    training_input_numbers = np.load(main_folder_path + 'inputNumbers' + training_ending + ending)


    testing_result_monitor = np.load(main_folder_path + 'resultPopVecs' + testing_ending + args.ad_path_test + ending)     
    testing_input_numbers = np.load(main_folder_path + 'inputNumbers' + testing_ending + args.ad_path_test + ending)



    print('Assignments')
    assignments = get_new_assignments(training_result_monitor, training_input_numbers[0:training_result_monitor.shape[0]], n_e) 
    all_assignments = [{} for _ in range(n_e)]

    if use_weighted_assignments: 
        assignments, all_assignments = get_new_assignments_weighted(training_result_monitor, training_input_numbers[0:training_result_monitor.shape[0]], n_e) 
    
    if use_weighted_assignments and multiple_UI_assignments: 
        assignments, all_assignments = get_new_assignments_weighted(training_result_monitors, training_input_numbers[0:training_result_monitors.shape[0]], n_e) 

    np.save(main_folder_path + "assignments_" + path_id, assignments)
    np.save(main_folder_path + "all_assignments_" + path_id, all_assignments)
    
    unique_assignments = np.unique(assignments)
    num_unique_assignments = len(unique_assignments) 

    print("Neuron Assignments ( shape = {} ): \n{}".format( assignments.shape, assignments) )
    print("Unique labels learnt ( count: {} ): \n{}".format( len(unique_assignments), unique_assignments ) ) 

    norm_summed_rates = np.zeros((num_unique_assignments, num_testing_imgs)) 
    P_i = np.zeros((num_unique_assignments, num_testing_imgs))    

    test_results = np.zeros((num_unique_assignments, num_testing_imgs))
    summed_rates = np.zeros((num_unique_assignments, num_testing_imgs))
    
    sum_train_spikes_list, train_spikes_list, learnt_neurons_list, len_learnt_labels_list = get_training_neuronal_spikes(unique_assignments, use_weighted_assignments, all_assignments)
    
    for i in range(num_testing_imgs):
        test_results[:,i], summed_rates[:,i] = get_recognized_number_ranking(assignments, testing_result_monitor[i,:], unique_assignments, sum_train_spikes_list, train_spikes_list, learnt_neurons_list, len_learnt_labels_list, use_weighted_assignments)

        # Probability-based neuronal assignment 
        norm_summed_rates[:,i] = (summed_rates[:,i] - np.min(summed_rates[:,i]) ) / ( np.max(summed_rates[:,i]) - np.min(summed_rates[:,i]) )
        P_i[:, i] = norm_summed_rates[:,i] / np.sum(norm_summed_rates[:,i])

    np.save(data_path + "summed_rates_" + path_id, summed_rates)
    
    difference = test_results[0, args.offset_after_skip : args.offset_after_skip+args.num_labels] - testing_input_numbers[args.offset_after_skip : args.offset_after_skip+args.num_labels]
    
    print("Testing input numbers: \n", testing_input_numbers)
    print("Testing result: \n", test_results[0,:])
    print( "\nDifferences: \n{}".format(difference) )

    difference = abs(difference)
    correct, incorrect, accurracy = get_accuracy(difference, tolerance=0)
    print( "\nAccuracy: {}, num correct: {}, num incorrect: {}".format(accurracy, len(correct), len(incorrect)) )
    print("Correctly predicted label indices: \n{}\nIncorrectly predicted label indices: \n{}\n".format(correct, incorrect))
    

    # Binary distance matrix 
    dMat = compute_binary_distance_matrix(summed_rates) 
    compute_distance_matrix(dMat, data_path, "binary_distMatrix")

    # Distance matrix where 0 represents furthest 
    compute_distance_matrix(summed_rates, data_path, "spike_rates_distMatrix")

    # Distance matrix where 0 represents closest 
    rates_matrix = invert_dMat(summed_rates)
    rates_matrix_P_i = invert_dMat(P_i)
    
    sorted_pred_idx = np.argsort(rates_matrix, axis=0)
    
    # compute recall at N - use num_labels to only compute the R@N at module level 
    n_values = [1, 5, 10, 15, 20, 25]
    numQ = args.num_labels
    gt = np.arange(len(summed_rates))
    compute_recall(gt, sorted_pred_idx[:, args.offset_after_skip : args.offset_after_skip+args.num_labels], numQ, n_values, data_path, name="recallAtN_SNN")

    plot_name = "DM_{}_{}".format(folder_id, path_id)
    compute_distance_matrix(rates_matrix, data_path, plot_name)
    compute_distance_matrix(rates_matrix_P_i, data_path, plot_name + "_Pi")
    
    if summed_rates.shape[0] != summed_rates.shape[1]:
        return

    fig_name = "PR_{}_{}".format(folder_id, path_id)
    plot_name = "Weighted" if use_weighted_assignments else "Standard" 
    print("Results based on {} assignments: ".format(plot_name))
    plot_precision_recall(rates_matrix, data_path, fig_name, "{}".format(plot_name))
            
    print("Results based on {} probability-based assignments: ".format(plot_name))
    plot_precision_recall(rates_matrix_P_i, data_path, fig_name + "_Pi", "{} Probability-based".format(plot_name))

    print('done')



if __name__ == "__main__":
    
    
    skip = 8 
    offset_after_skip = 0
    update_interval = 50 
    epochs = 20
    n_e = 100
    folder_id = 'NRD'
    num_train_imgs = 2 * 5
    num_test_imgs = 5

    first_epoch = (num_train_imgs * epochs)
    last_epoch = (num_train_imgs * epochs) + 1 
    use_weighted_assignments = False
    
    ad_path_test = "_test"
    ad_path = "_offset{}".format(offset_after_skip)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip', type=int, default=skip, help='The number of images to skip between each place label.')
    parser.add_argument('--offset_after_skip', type=int, default=offset_after_skip, help='The offset to apply for selecting places after skipping every n images.')
    parser.add_argument('--folder_id', type=str, default=folder_id, help='Folder name of dataset to be used.')
    parser.add_argument('--num_train_imgs', type=int, default=num_train_imgs, help='Number of entire training images.')
    parser.add_argument('--num_test_imgs', type=int, default=num_test_imgs, help='Number of entire testing images.')
    parser.add_argument('--first_epoch', type=int, default=first_epoch, help='For use of neuronal assignments, the first training iteration number in saved outputs.')
    parser.add_argument('--last_epoch', type=int, default=last_epoch, help='For use of neuronal assignments, the last training iteration number in saved outputs.')
    parser.add_argument('--update_interval', type=int, default=update_interval, help='The number of iterations to save at one time in training output matrix.')
    parser.add_argument('--epochs', type=int, default=epochs, help='Number of sweeps through the dataset in training.')
    parser.add_argument('--use_weighted_assignments', type=bool, default=use_weighted_assignments, help='Value to define the type of neuronal assignment to use: standard=0, weighted=1')
    parser.add_argument('--n_e', type=int, default=n_e, help='Number of excitatory output neurons. The number of inhibitory neurons are the same.')

    parser.add_argument("--ad_path_test", type=str, default=ad_path_test, help='Additional string arguments for subfolder names to save testing outputs.')
    parser.add_argument('--ad_path', type=str, default=ad_path, help='Additional string arguments for subfolder names to save outputs.')
    parser.add_argument('--multi_path', type=str, default='epoch{}'.format(epochs), help='Additional string arguments for subfolder names to save evaluation outputs.')
    parser.set_defaults()

    args = parser.parse_args()
    

    main(args)
            
    
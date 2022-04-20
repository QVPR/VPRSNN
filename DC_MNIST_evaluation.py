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
    
    


def get_recognized_number_ranking(assignments, spike_rates, num_labels, use_weighted_assignments=False, all_assignments=[]):
    '''
    Given the neuronal assignments, and the number of spikes fired for the image with the given label in testing, 
    predict the label of the corresponding image  
    '''

    summed_rates = np.zeros(num_labels)

    for i in range(num_labels):
        assigned_neurons_indices = np.where(assignments == i)[0]  
        num_assignments = len(assigned_neurons_indices)

        if num_assignments > 0:            
            test_spikes = spike_rates[assignments == i]

            if use_weighted_assignments and len(all_assignments) > 0: 

                all_train_spikes_all = np.array([ [x, all_assignments[x]] for x in range(len(all_assignments)) if i in all_assignments[x]])

                all_train_spikes = np.array([ x for x in all_train_spikes_all if (x[1].get(i) / sum(list(x[1].values()))) > 0.1 ])    

                sum_train_spikes = np.array([ sum(list(x[1].values()))  for x in all_train_spikes ])
                train_spikes = np.array([ x[1].get(i) for x in all_train_spikes ])
                learnt_neurons = np.array([ x[0] for x in all_train_spikes ])

                len_learnt_labels = [ len(list(x[1].values()))  for x in all_train_spikes ]

                test_spikes = np.array([ spike_rates[x] for x in range(len(spike_rates)) if x in learnt_neurons ])  

                norm_factor = ( np.sum(train_spikes[test_spikes != 0])  /  np.sum(train_spikes) ) if np.any(train_spikes) else 0 

                # a) Regularization by involvement 
                test_spikes = np.array([ test_spikes[x] if len_learnt_labels[x] <= 0.02*num_labels else test_spikes[x]*(1/len_learnt_labels[x]) for x in range(len(test_spikes))  ]) 

                # b) Normalization by response strength 
                test_spikes = np.array([test_spikes[x] * ( train_spikes[x] / (sum_train_spikes[x]) ) for x in range(len(test_spikes)) ]) 

                # c) Penalize relevant neurons that did not fire 
                summed_rates[i] =  ( np.sum(test_spikes) ) * norm_factor if len(learnt_neurons) > 0 else 0   
                
            else:
                summed_rates[i] = ( np.sum(test_spikes) / num_assignments )            

    if np.all(summed_rates == 0):
        sorted_summed_rates = np.arange(-1, num_labels-1)
    else: 
        sorted_summed_rates = np.argsort(summed_rates)[::-1]

    return sorted_summed_rates, summed_rates


def get_new_assignments(result_monitor, input_numbers, num_labels, n_e):
    '''
    Update the assignments of classes to neurons based on the spiking rate 
    '''

    assignments = np.ones(n_e) * -1 
    input_nums = np.asarray(input_numbers)
    maximum_rate = np.zeros(n_e)  

    for j in range(num_labels):
        num_assignments = len(np.where(input_nums == j)[0])

        if num_assignments > 0:
            rate = np.sum(result_monitor[input_nums == j], axis = 0) / num_assignments
        else:
            rate = np.zeros((n_e, 1))

        for i in range(n_e):
            if rate[i] > maximum_rate[i]:
                maximum_rate[i] = rate[i]
                assignments[i] = j

    return assignments


def get_new_assignments_weighted(result_monitor, input_numbers, num_labels, n_e):
    '''
    Update the assignments of all classes associated with neurons based on the spiking rate 
    '''
    
    assignments = [{} for _ in range(n_e)] 
    input_nums = np.asarray(input_numbers)

    for j in range(num_labels):
        num_assignments = len(np.where(input_nums == j)[0])
        maximum_rate = np.zeros(n_e) 

        if num_assignments > 0:
            num_spikes = np.sum(result_monitor[input_nums == j], axis = 0)
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

    print("Min value: {} Max value: {}\n".format(min_val, max_val))

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
    AUC_PR_trap = np.trapz(precision, recall)

    return AUC_PRs, AUC_PR_trap


def getRat100P(precision, recall):

    if not np.any(precision >= 1.0): 
        return 0
    Rat100P = np.max(recall[precision >= 1.0])

    return Rat100P


def plot_precision_recall(dMat, data_path, fig_name='', label='', png_ending=".png"):
    
    precision, recall, f1_scores = get_precision_recall_f1_score(dMat)

    AUC_PRs, AUC_PR_trap = getAUCPR(precision, recall)
    print("\nArea under precision recall curve: \nsklearn AUC: {}\nnumpy trapz: {}\n".format(AUC_PRs, AUC_PR_trap))

    Rat100P = getRat100P(precision, recall)
    print("\nR @ 100% P is: {}\n".format(Rat100P))

    Pat100R = f1_scores[-1]
    print("\nP a@ 100% R (last item of f1-score) is: {}\n".format(Pat100R))
    print("\n\nf1 scores with wa: {}\n\n".format(f1_scores))

    sn.set_context("paper", font_scale=2, rc={"lines.linewidth": 2})

    plt.figure()
    plt.plot(recall, precision, label=label)
    plt.xlabel("Recall")
    plt.ylabel("Precision")

    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    plt.legend(fontsize=10, loc='lower center') 
    plt.tight_layout()
    plt.savefig(data_path + "{}".format(fig_name) + png_ending)
    plt.close()

    plt.figure()
    plt.plot(recall, precision, label=label)
    plt.plot(recall, precision, '*r')

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision and Recall")  

    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(data_path + "{}_show_thresholds".format(fig_name) + png_ending)
    plt.close()

    return Rat100P, AUC_PRs

                

def main():
    
    skip = 8 
    offset_after_skip = 0
    update_interval = 600 
    epochs = 60
    n_e = 400
    folder_id = 'NRD'
    num_train_imgs = 2 * 100
    num_test_imgs = 100

    first_epoch = (num_train_imgs * epochs)
    last_epoch = (num_train_imgs * epochs) + 1 
    use_weighted_assignments = True
    
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

    data_path = './outputs/outputs_ne{}_L{}'.format(args.n_e, args.num_test_imgs) + args.ad_path + '/' 
    path_id = 'L{}_S{}_O{}'.format(args.num_test_imgs, args.skip, args.offset_after_skip)

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
        
    if os.path.isfile(main_folder_path + 'inputNumbers' + training_ending + ending): 
        training_input_numbers = np.load(main_folder_path + 'inputNumbers' + training_ending + ending)
    else:
        training_input_numbers = np.array( 2 * num_training_sweeps * list(range(num_testing_imgs)) ) 


    testing_result_monitor = np.load(main_folder_path + 'resultPopVecs' + testing_ending + args.ad_path_test + ending)     
    testing_input_numbers = np.load(main_folder_path + 'inputNumbers' + testing_ending + args.ad_path_test + ending)

    unique_test_labels = np.unique(testing_input_numbers)
    num_unique_test_labels = len(unique_test_labels)

    test_results = np.zeros((num_unique_test_labels, num_testing_imgs))
    summed_rates = np.zeros((num_unique_test_labels, num_testing_imgs))
    rates_matrix = np.zeros((num_unique_test_labels, num_testing_imgs))

    print('Assignments')
    assignments = get_new_assignments(training_result_monitor, training_input_numbers[0:training_result_monitor.shape[0]], num_unique_test_labels, n_e) 
    all_assignments = [{} for _ in range(n_e)]

    if use_weighted_assignments: 
        assignments, all_assignments = get_new_assignments_weighted(training_result_monitor, training_input_numbers[0:training_result_monitor.shape[0]], num_unique_test_labels, n_e) 
    
    if use_weighted_assignments and multiple_UI_assignments: 
        assignments, all_assignments = get_new_assignments_weighted(training_result_monitors, training_input_numbers[0:training_result_monitors.shape[0]], num_unique_test_labels, n_e) 

    unique_assignments = np.unique(assignments)

    print("Neuron Assignments ( shape = {} ): \n{}".format( assignments.shape, assignments) )
    print("Unique labels learnt ( count: {} ): \n{}".format( len(unique_assignments), unique_assignments ) ) 

    norm_summed_rates = np.zeros((num_unique_test_labels, num_testing_imgs)) 
    P_i = np.zeros((num_unique_test_labels, num_testing_imgs))    

    for i in range(num_testing_imgs):
        test_results[:,i], summed_rates[:,i] = get_recognized_number_ranking(assignments, testing_result_monitor[i,:], num_unique_test_labels, use_weighted_assignments, all_assignments)

        # Probability-based neuronal assignment 
        norm_summed_rates[:,i] = (summed_rates[:,i] - np.min(summed_rates[:,i]) ) / ( np.max(summed_rates[:,i]) - np.min(summed_rates[:,i]) )
        P_i[:, i] = norm_summed_rates[:,i] / np.sum(norm_summed_rates[:,i])

    difference = test_results[0,:] - testing_input_numbers[0:test_results.shape[0]]
    
    print("Testing input numbers: \n", testing_input_numbers)
    print("Testing result: \n", test_results[0,:])
    print( "\nDifferences: \n{}\nSummed rates (shape = {} ): \n{}".format(difference, summed_rates.shape, summed_rates) )

    difference = abs(difference)
    correct, incorrect, accurracy = get_accuracy(difference, tolerance=0)
    print( "\nAccuracy: {}, num correct: {}, num incorrect: {}".format(accurracy, len(correct), len(incorrect)) )
    print("Correctly predicted labels: \n{}\nIncorrectly predicted labels: \n{}\n".format(correct, incorrect))
    

    # Binary distance matrix 
    dMat = compute_binary_distance_matrix(summed_rates) 
    compute_distance_matrix(dMat, data_path, "binary_distMatrix")

    # Distance matrix where 0 represents furthest 
    compute_distance_matrix(summed_rates, data_path, "spike_rates_distMatrix")

    # Distance matrix where 0 represents closest 
    max_rate = (max(map(max, summed_rates)))
    max_P_i = (max(map(max, P_i)))

    rates_matrix_P_i = np.zeros((num_unique_test_labels, num_testing_imgs))

    plot_name = "DM_{}_{}".format(folder_id, path_id)
    for i in range(num_testing_imgs):
        rates_matrix[:, i] = [max_rate - l for l in summed_rates[:, i] ]
        rates_matrix_P_i[:, i] = [max_P_i - l for l in P_i[:, i] ]

    compute_distance_matrix(rates_matrix, data_path, plot_name)
    compute_distance_matrix(rates_matrix_P_i, data_path, plot_name + "_Pi")

    fig_name = "PR_{}_{}".format(folder_id, path_id)
    plot_name = "Weighted" if use_weighted_assignments else "Standard" 
    print("Results based on {} assignments: ".format(plot_name))
    plot_precision_recall(rates_matrix, data_path, fig_name, "{}".format(plot_name))
            
    print("Results based on {} probability-based assignments: ".format(plot_name))
    plot_precision_recall(rates_matrix_P_i, data_path, fig_name + "_Pi", "{} Probability-based".format(plot_name))

    print('done')



if __name__ == "__main__":

    main()
            
    
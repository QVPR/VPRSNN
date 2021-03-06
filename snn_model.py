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
import math
import os
import sys
from pathlib import Path

import brian2 as b2
import matplotlib.cm as cmap
import matplotlib.pyplot as plt
import numpy as np

import plotting as pl
import snn_model_utils as model
from DC_MNIST_evaluation import (Logger, get_new_assignments,
                                 get_recognized_number_ranking)
from snn_util import get_train_test_datapath, processImageDataset


def save_connections(snn_model, weightsPath, interval = '', save_all=True, use_initial_name=False):
    '''
    Saves a zip of presynaptic and postsynaptic neuron connection indices and their corresponding weight value 
    '''

    print('save connections')

    for conn in snn_model.connections: 
        if conn != "XeAe" and not save_all:
            continue 
        elif conn == "AiAe" or conn == "AeAi":
            updated_interval = ""
        else:
            updated_interval = interval 
        
        if use_initial_name:
            updated_interval = ""
            
        connListSparse = zip(snn_model.connections[conn].i, snn_model.connections[conn].j, snn_model.connections[conn].w)

        np.save(weightsPath + conn + updated_interval, list(connListSparse)) 


def save_theta(theta, weightsPath, population_name, interval='', use_initial_name=False):
    '''
    Saves theta parameter of the Ae population group 
    '''

    print('save theta')
    
    if use_initial_name:
        interval = ""
    np.save(weightsPath + 'theta_' + population_name + interval, theta) 


def normalize_weights(connections, n_e):
    '''
    For excitatory to excitatory connections, the weight is normalised by a factor of weight / col sum
    '''
    weight_ee = 78

    for connName in connections:
        
        if connName[1] == 'e' and connName[3] == 'e':
            len_source = len(connections[connName].source)
            len_target = len(connections[connName].target)
            connection = np.zeros((len_source, len_target))
            connection[connections[connName].i, connections[connName].j] = connections[connName].w
            temp_conn = np.copy(connection)
            colSums = np.sum(temp_conn, axis = 0)
            colFactors = weight_ee/colSums
            
            for j in range(n_e):
                temp_conn[:,j] *= colFactors[j]

            connections[connName].w = temp_conn[connections[connName].i, connections[connName].j]


def get_2d_input_weights(weight_matrix, n_input, n_e):
    '''
    Reshapes the weight matrix to a square matrix for plotting 
    '''

    n_e_sqrt = int(np.sqrt(n_e))
    n_in_sqrt = int(np.sqrt(n_input))

    num_values_col = n_e_sqrt*n_in_sqrt
    num_values_row = num_values_col
    rearranged_weights = np.zeros((num_values_col, num_values_row))

    for i in range(n_e_sqrt):
        for j in range(n_e_sqrt):
                rearranged_weights[i*n_in_sqrt : (i+1)*n_in_sqrt, j*n_in_sqrt : (j+1)*n_in_sqrt] = \
                    weight_matrix[:, i + j*n_e_sqrt].reshape((n_in_sqrt, n_in_sqrt))
    return rearranged_weights


def plot_2d_input_weights(connection, n_input, n_e, outputsPath, fig_id, figsize=(8, 8), tag=""):
    '''
    Plots the 2d input weights
    '''

    name = 'XeAe'
    connMatrix = np.zeros((n_input, n_e))
    connMatrix[connection.i, connection.j] = connection.w
    weight_matrix = np.copy(connMatrix)
    weights = get_2d_input_weights(weight_matrix, n_input, n_e)
    wmax_eee = np.amax(weights)

    plt.figure(fig_id, figsize=figsize)
    plt.cla()
    
    img = plt.imshow(weights, interpolation = "nearest", vmin = 0, vmax = wmax_eee, cmap = cmap.get_cmap('gray_r'))
    cb = plt.colorbar(img)

    plot_name = "Connection weights of " + name + tag
    plt.title(plot_name)

    plt.savefig(outputsPath + plot_name)  
    cb.remove()


def get_current_performance(performance, current_example_num, update_interval, outputNumbers, input_numbers):
    '''
    Calculates the performance for current example, and appends the result to the performance array
    '''

    current_evaluation = int(current_example_num/update_interval)

    start_num = current_example_num - update_interval 
    end_num = current_example_num

    difference = outputNumbers[start_num:end_num, 0] - input_numbers[start_num:end_num]
    correct = len(np.where(difference == 0)[0])

    performance[current_evaluation] = correct / float(update_interval) * 100

    return performance


def plot_initial_performance(num_examples, update_interval, outputsPath, fig_id, figsize=(8, 8)):
    '''
    Plots the initial figure for performance 
    '''

    num_evaluations = math.ceil(num_examples/update_interval)
    time_steps = range(0, num_evaluations)
    performance = np.zeros(num_evaluations)

    plt.figure(fig_id, figsize=figsize)
    plt.cla()
    
    plt.plot(time_steps, performance) 
    plt.ylim(ymin = 0, ymax = 100)
    plt.xlabel("Number of evaluations")
    plt.ylabel("Performance")
    plot_name = "Initial performance"
    plt.title(plot_name)

    plt.savefig(outputsPath + plot_name)  

    return performance


def plot_performance(performance, current_example_num, update_interval, outputNumbers, input_numbers, outputsPath, fig_id, figsize=(8, 8)):
    '''
    Updates the performance plot 
    '''
    
    performance = get_current_performance(performance, current_example_num, update_interval, outputNumbers, input_numbers)
    time_steps = range(0, len(performance))
    
    plt.figure(fig_id, figsize=figsize)
    plt.cla() 
    
    plt.plot(time_steps, performance) 

    plt.ylim(ymin = 0, ymax = 100)
    plt.xlabel("Number of evaluations")
    plt.ylabel("Performance")
    plot_name = "Updated performance"
    plt.title(plot_name)
    
    plt.savefig(outputsPath + plot_name) 

    return performance 


def create_snn_model(weight_path, num_training_examples, n_e, n_i, n_input, population_name, Xe, Ae, test_mode=False, use_monitors=False, ending='.npy'):

    Ai = 'Ai'
    input_connName = 'XeAe'
    recurrent_connNames = ['ei', 'ie']

    snn_model = model.SNNModel()
    
    # excitatory and inhibitory neuronal populations 
    snn_model.create_excitatory_neurons(weight_path, n_e, num_training_examples, test_mode)
    snn_model.create_inhibitory_neurons(n_i)

    # monitors 
    snn_model.create_spike_monitor(snn_model.neuron_groups[Ae], name=Ae)

    if use_monitors: 
        snn_model.create_spike_monitor(snn_model.neuron_groups[Ai], name=Ai)

        snn_model.create_population_rate_monitor(snn_model.neuron_groups[Ae], name=Ae)
        snn_model.create_population_rate_monitor(snn_model.neuron_groups[Ai], name=Ai)

    # connections between the excitatory and inhibitory neurons
    for conn_type in recurrent_connNames:

        connName = population_name + conn_type[0] + population_name + conn_type[1]
        weight_filename = weight_path + connName + ending

        snn_model.create_population_connections(connName, weight_filename, n_input=n_input, n_e=n_e, n_i=n_i, is_input=False, test_mode=test_mode)

    # input population 
    snn_model.create_input_neurons(n_input, name="Xe")

    # monitors 
    if use_monitors: 
        snn_model.create_spike_monitor(snn_model.input_groups[Xe], name=Xe)
        snn_model.create_population_rate_monitor(snn_model.input_groups[Xe], name=Xe)

    # connection between input and excitatory neurons 
    connName = input_connName 
    weight_filename = weight_path + connName + str(num_training_examples) + ending
    if not os.path.isfile(weight_filename): 
        weight_filename = weight_path + connName + ending

    snn_model.create_population_connections(connName, weight_filename, n_input=n_input, n_e=n_e, n_i=n_i, is_input=True, test_mode=test_mode)

    snn_model.create_network()
    
    return snn_model



def main():

    dataset = 'nordland'
    num_labels = 100
    tc_ge = 1.0
    tc_gi = 2.0 
    intensity = 4 

    test_mode = True

    skip = 8
    offset_after_skip = 0
    update_interval = 600  
    epochs = 60
    n_e = 400

    ad_path_test = "_test"
    ad_path = "_offset{}".format(offset_after_skip)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=dataset, help='Folder name of dataset to be used. Relative to this repo, the folder must exist in this directory: ./../data/')
    parser.add_argument('--num_labels', type=int, default=num_labels, help='Number of distinct places to use from the dataset.')
    parser.add_argument('--skip', type=int, default=skip, help='The number of images to skip between each place label.')
    parser.add_argument('--offset_after_skip', type=int, default=offset_after_skip, help='The offset to apply for selecting places after skipping every n images.')
    parser.add_argument('--update_interval', type=int, default=update_interval, help='The number of iterations to save at one time in training output matrix.')
    parser.add_argument('--epochs', type=int, default=epochs, help='Number of sweeps through the dataset in training.')
    parser.add_argument('--n_e', type=int, default=n_e, help='Number of excitatory output neurons in model. The number of inhibitory neurons are the same.')
    parser.add_argument('--tc_ge', type=float, default=tc_ge, help='Time constant of conductance of excitatory synapses AeAi.')
    parser.add_argument('--tc_gi', type=float, default=tc_gi, help='Time constant of conductance of inhibitory synapses AiAe.')
    parser.add_argument('--intensity', type=int, default=intensity, help="Intensity scaling to change the range of input pixel values.")

    parser.add_argument('--test_mode', dest="test_mode", action="store_true", help='Boolean indicator to switch between training and testing mode.')

    parser.add_argument("--ad_path_test", type=str, default=ad_path_test, help='Additional string arguments for subfolder names to save testing outputs.')
    parser.add_argument('--ad_path', type=str, default=ad_path, help='Additional string arguments for subfolder names to save outputs.')

    parser.set_defaults(test_mode=test_mode)
    args = parser.parse_args()

    weightsPath = './weights/weights_ne{}_L{}'.format(args.n_e, args.num_labels) + args.ad_path + '/'
    outputsPath = './outputs/outputs_ne{}_L{}'.format(args.n_e, args.num_labels) + args.ad_path + '/'
    random_path = './random/random_ne{}'.format(args.n_e) + args.ad_path + '/'

    Path(weightsPath).mkdir(parents=True, exist_ok=True)
    Path(outputsPath).mkdir(parents=True, exist_ok=True)

    logfile_name = "logfile" if not args.test_mode else "logfile_test" 
    sys.stdout = Logger(outputsPath, logfile_name)
    print(args)

    org_data_path = ['./../data/{}/'.format(args.dataset)]  

    train_data_path, test_data_path = get_train_test_datapath(org_data_path)

    ad_path_test = args.ad_path_test if args.test_mode else ""
    skip = args.skip
    num_labels = args.num_labels 
    train_update_interval = args.update_interval 
    test_mode = args.test_mode
    n_e = args.n_e
    n_i = args.n_e
    offset_after_skip = args.offset_after_skip  
    num_training_epochs = args.epochs       

    use_monitors = True 
    repeat_no_spikes = False 

    # Set img width and height, also regenerate random initialised weights when img shape changes 
    imWidth = 28   
    imHeight = 28   
    num_patches = 7
    min_num_spikes = 1

    np.random.seed(0)

    training_data = processImageDataset(train_data_path, "train", imWidth, imHeight, num_patches, num_labels, skip, offset_after_skip)

    testing_data = processImageDataset(test_data_path, "test", imWidth, imHeight, num_patches, num_labels, skip, offset_after_skip) 

    print("\nTraining labels:\n{}\n\nTesting labels:\n{}\n".format(training_data['y'].flatten(), testing_data['y'].flatten() ))

    train_labels = np.unique(training_data['y'])
    num_training_imgs = training_data['x'].shape[0]

    test_labels = np.unique(testing_data['y'])
    num_testing_imgs = testing_data['x'].shape[0]
    np.save(outputsPath + 'test_labels.npy', test_labels)


    weight_path = weightsPath if test_mode else random_path
    num_examples = num_testing_imgs if test_mode else num_training_imgs * num_training_epochs            
    update_interval = num_examples if test_mode else train_update_interval
    num_training_examples =  num_training_imgs * num_training_epochs
    do_plot_performance = True    

    initial_resting_time = 0.5 * b2.second
    single_example_time = 0.35 * b2.second
    resting_time = 0.15 * b2.second
    n_input = imWidth * imHeight             

    Xe = 'Xe'
    population_name = 'A'
    Ae = 'Ae'

    # create snn model 
    print("Create model")
    snn_model = create_snn_model(weight_path, num_training_examples, n_e, n_i, n_input, population_name, Xe, Ae, test_mode=test_mode, use_monitors=use_monitors)

    # run the simulation 
    labels = train_labels if not test_mode else test_labels
    num_labels = len(labels)
    
    figures = [1,2]

    if not test_mode:
        plot_2d_input_weights(snn_model.connections[Xe+Ae], n_input, n_e, outputsPath, figures[1], tag="1")
        
    if do_plot_performance:
        performance = plot_initial_performance(num_examples, update_interval, outputsPath, figures[0])

    snn_model.input_groups[Xe].rates = 0 * b2.Hz
    snn_model.run_network(initial_resting_time)

    input_intensity = args.intensity 
    start_input_intensity = input_intensity

    previous_spike_count = np.zeros(n_e)
    assignments = np.zeros(n_e)
    input_numbers = np.zeros(num_examples)
    outputNumbers = np.zeros((num_examples, num_labels))
    result_monitor = np.zeros((update_interval,n_e))
    previous_spike_count = np.copy(snn_model.spike_monitors[Ae].count[:])

    j = 0
    kkk = 0 

    for j in range(int(num_examples)):

        if test_mode:
            spike_rates = testing_data['x'][j%num_testing_imgs,:,:].reshape((n_input)) / input_intensity                  
        else:
            normalize_weights(snn_model.connections, n_e)
            spike_rates = training_data['x'][j%num_training_imgs,:,:].reshape((n_input)) / input_intensity              
        
        snn_model.input_groups[Xe].rates = spike_rates * b2.Hz

        print('run number:', j+1, 'of', int(num_examples))
        snn_model.run_network(single_example_time)

        if j % update_interval == 0 and j > 0:
            assignments = get_new_assignments(result_monitor[:], input_numbers[j-update_interval : j], num_labels, n_e)

            print("Unique labels learnt: \n", np.unique(assignments))
            np.save(outputsPath + "resultPopVecs" + str(j), result_monitor) 

            if not test_mode:                
                plot_2d_input_weights(snn_model.connections[Xe+Ae], n_input, n_e, outputsPath, figures[1])                
                save_connections(snn_model, weightsPath, str(j), save_all=False)
                save_theta(snn_model.neuron_groups[Ae].theta, weightsPath, population_name, str(j))

        current_spike_count = np.asarray(snn_model.spike_monitors[Ae].count[:]) - previous_spike_count
        previous_spike_count = np.copy(snn_model.spike_monitors[Ae].count[:])

        if repeat_no_spikes and np.sum(current_spike_count) < min_num_spikes and kkk == 0:
            input_intensity += 1
            snn_model.input_groups[Xe].rates = 0 * b2.Hz        
            snn_model.run_network(resting_time)
            kkk += 1 

        else:
            kkk = 0
            result_monitor[j%update_interval,:] = current_spike_count
            input_numbers[j] = testing_data['y'][j%num_testing_imgs][0] if test_mode else training_data['y'][j%num_training_imgs][0]

            outputNumbers[j,:], summed_rates = get_recognized_number_ranking(assignments, result_monitor[j%update_interval,:], num_labels)

            if j % update_interval == 0 and j > 0 and do_plot_performance: 
                performance = plot_performance(performance, j, update_interval, outputNumbers, input_numbers, outputsPath, figures[0])
                
                print("Classification performance at {}: \n{}".format(j, performance[:int((j/float(update_interval))+1)] ) )      

            snn_model.input_groups[Xe].rates = 0 * b2.Hz
            snn_model.run_network(resting_time)
            input_intensity = start_input_intensity
            j += 1
                
        b2.device.delete(code=False)

    print('output numbers: \n', outputNumbers, '\nSummed rates: \n', summed_rates)


    # save results
    print('save results')
    if not test_mode:
        save_theta(snn_model.neuron_groups[Ae].theta, weightsPath, population_name, str(num_examples), use_initial_name=True)
        save_connections(snn_model, weightsPath, str(num_examples), use_initial_name=True)

    np.save(outputsPath + "resultPopVecs" + str(num_examples) + ad_path_test, result_monitor)
    np.save(outputsPath + "inputNumbers" + str(num_examples) + ad_path_test, input_numbers)


    # plot results
    plot_2d_input_weights(snn_model.connections[Xe+Ae], n_input, n_e, outputsPath, figures[1], tag="3")

    if snn_model.rate_monitors:
        pl.plot_rateMonitors(snn_model.rate_monitors, outputsPath, num_training_epochs, test_mode)

    if snn_model.spike_monitors:
        pl.plot_spikeMonitors(snn_model.spike_monitors, outputsPath, num_training_epochs, test_mode)
        pl.plot_spikeMonitorsCount(snn_model.spike_monitors, outputsPath)

    print('done')
    


if __name__ == "__main__":
    
    main()

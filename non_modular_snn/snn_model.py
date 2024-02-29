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
import math
import os
import sys
from pathlib import Path

import brian2 as b2
import matplotlib.cm as cmap
import matplotlib.pyplot as plt
import numpy as np


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


from tools.logger import Logger
from tools.data_utils import get_train_test_imagenames_path, processImageDataset
import tools.snn_model_utils as snn_model_class
from non_modular_snn.snn_model_evaluation import get_new_assignments, get_recognized_number_ranking
from tools.snn_model_plot import plot_rateMonitors, plot_spikeMonitors, plot_spikeMonitorsCount


# set the default target for code generation: 'auto', 'cython', 'numpy'
target = 'cython'                 
b2.prefs.codegen.target = target
b2.prefs.codegen.cpp.extra_compile_args_gcc = []

# on hpc with PBS scheduling system, set cache directory to temp directory of the job  
cache_dir = os.environ['TMPDIR']
print("Cache dir: ", cache_dir)
b2.prefs.codegen.runtime.cython.cache_dir = cache_dir
b2.prefs.codegen.runtime.cython.multiprocess_safe = False



def main(args):

    weightsPath = './weights/weights_ne{}_L{}'.format(args.n_e, args.num_labels) + args.ad_path + '/'
    outputsPath = './outputs/outputs_ne{}_L{}'.format(args.n_e, args.num_labels) + args.ad_path + '/'
    random_path = './random/random_ne{}'.format(args.n_e) + args.ad_path + '/'

    Path(weightsPath).mkdir(parents=True, exist_ok=True)
    Path(outputsPath).mkdir(parents=True, exist_ok=True)
    
    modes = {
        "train": "logfile_train",
        "record": "logfile_record",
        "calibrate": "logfile_calibrate",
        "test": "logfile_test",
    }
    
    test_mode, logfile_name = set_mode(args.process_mode, modes)
    sys.stdout = Logger(outputsPath, logfile_name)
    print(args)

    train_data_path, test_data_path = get_train_test_imagenames_path(args.dataset, args.folder_id)

    ad_path_test = args.ad_path_test if test_mode else ""
    skip = args.skip
    num_labels = args.num_labels
    n_i = args.n_e

    use_monitors = False 
    repeat_no_spikes = False 


    # Set img width and height, also regenerate random initialised weights when img shape changes 
    imWidth = 28   
    imHeight = 28   
    num_patches = 7
    min_num_spikes = 1

    np.random.seed(args.seed)
    b2.seed(args.seed)
    
    shuffled_indices = snn_model_class.get_shuffled_indices(args.num_query_imgs, args.num_cal_labels, shuffled=args.shuffled, seed=args.seed)   

    if not test_mode:
        training_data = processImageDataset(train_data_path, "train", imWidth, imHeight, num_patches, num_labels=args.num_labels, 
                                            num_test_labels=args.num_test_labels, num_query_imgs=args.num_query_imgs, skip=args.skip, 
                                            offset_after_skip=args.offset_after_skip, shuffled_indices=shuffled_indices)
        print("\nTraining labels:\n{}\n".format(training_data['y'].flatten()))
    
    else:
        testing_data = processImageDataset(test_data_path, "test", imWidth, imHeight, num_patches, num_labels=args.num_labels, 
                                           num_test_labels=args.num_test_labels, num_query_imgs=args.num_query_imgs, skip=skip, 
                                           offset_after_skip=args.offset_after_skip, shuffled_indices=shuffled_indices) 
        print("\nTesting labels:\n{}\n".format(testing_data['y'].flatten() ))

    num_training_imgs = len(train_data_path)*args.num_labels
    num_testing_imgs = args.num_test_labels


    weight_path = weightsPath if test_mode else random_path
    num_examples = num_testing_imgs if test_mode else num_training_imgs * args.epochs            
    update_interval = num_examples if test_mode else args.update_interval
    num_training_examples =  num_training_imgs * args.epochs
    do_plot_performance = True    

    initial_resting_time = 0.5 * b2.second
    single_example_time = 0.35 * b2.second
    resting_time = 0.15 * b2.second
    n_input = imWidth * imHeight             

    Xe = 'Xe'
    population_name = 'A'
    Ae = 'Ae'
    Ai = 'Ai'
    ending = '.npy'

    # create snn model 
    print("Create model")
    # snn_model = create_snn_model(weight_path, num_training_examples, args.n_e, n_i, n_input, population_name, Xe, Ae, test_mode=test_mode, use_monitors=use_monitors)
        
    v_rest_e = -65 * b2.mV
    v_rest_i = -60 * b2.mV
    v_reset_e = -65 * b2.mV
    v_reset_i = -45 * b2.mV
    v_thresh_e = -52 * b2.mV
    v_thresh_i = -40 * b2.mV
    refrac_e = 5 * b2.ms
    refrac_i = 2 * b2.ms


    input_connection_name = 'XA'
    input_conn_name = 'ee_input'
    recurrent_conn_names = ['ei', 'ie']

    weight = {}
    delay = {}
    delay['ee_input'] = (0*b2.ms,10*b2.ms)
    input_intensity = args.intensity 
    start_input_intensity = input_intensity

    tc_pre_ee = 20*b2.ms
    tc_post_1_ee = 20*b2.ms
    tc_post_2_ee = 40*b2.ms
    wmax_ee = 1.0

    tc_e = 100 * b2.ms 
    tc_i = 10 * b2.ms 

    tc_ge = args.tc_ge * b2.ms
    tc_gi = args.tc_gi * b2.ms 

    # learning rates 
    nu_ee_pre =  0.0001     
    nu_ee_post = 0.01       

    if test_mode:
        scr_e = 'v = v_reset_e; timer = 0*ms'
    else:
        tc_theta = 1e7 * b2.ms
        theta_plus_e = 0.05 * b2.mV
        scr_e = 'v = v_reset_e; theta += theta_plus_e; timer = 0*ms'

    offset = 20.0*b2.mV
    v_thresh_e_str = '(v>(theta - offset + v_thresh_e)) and (timer>refrac_e)'
    v_thresh_i_str = 'v>v_thresh_i'
    v_reset_i_str = 'v=v_reset_i'

    neuron_eqs_e = '''
            dv/dt = ((v_rest_e - v) + (I_synE+I_synI) / nS) / (tc_e)  : volt (unless refractory)
            I_synE = ge * nS * -v               : amp
            I_synI = gi * nS * (-100.*mV-v)     : amp
            dge/dt = -ge/(tc_ge)               : 1
            dgi/dt = -gi/(tc_gi)               : 1
            '''

    if test_mode:
        neuron_eqs_e += '\n  theta :volt'
    else:
        neuron_eqs_e += '\n  dtheta/dt = -theta / (tc_theta)  : volt'

    neuron_eqs_e += '\n  dtimer/dt = 0.1  : second'

    neuron_eqs_i = '''
            dv/dt = ((v_rest_i - v) + (I_synE+I_synI) / nS) / (tc_i)  : volt (unless refractory)
            I_synE = ge * nS * -v               : amp
            I_synI = gi * nS * (-85.*mV-v)      : amp
            dge/dt = -ge/(tc_ge)               : 1
            dgi/dt = -gi/(tc_gi)               : 1
            '''

    eqs_stdp_ee = '''
                    post2before                            : 1
                    dprees/dt  = -prees/(tc_pre_ee)        : 1 (event-driven)
                    dpost1/dt  = -post1/(tc_post_1_ee)     : 1 (event-driven)
                    dpost2/dt  = -post2/(tc_post_2_ee)     : 1 (event-driven)
                '''

    eqs_stdp_pre_ee = '''   
                        prees = 1.0
                        w = clip(w + nu_ee_pre * post1, 0, wmax_ee)
                        '''

    eqs_stdp_post_ee = '''
                        post2before = post2 
                        w = clip(w + nu_ee_post * prees * post2before, 0, wmax_ee)
                        post1 = 1.0
                        post2 = 1.0
                        '''

    neuron_groups = {}
    input_groups = {}
    connections = {}
    spike_monitors = {}
    result_monitor = np.zeros((update_interval,args.n_e))
    
    neuron_groups['e'] = b2.NeuronGroup(args.n_e, neuron_eqs_e, threshold= v_thresh_e_str, refractory= refrac_e, reset= scr_e, method='euler', name='e')
    neuron_groups['i'] = b2.NeuronGroup(n_i, neuron_eqs_i, threshold= v_thresh_i_str, refractory= refrac_i, reset= v_reset_i_str, method='euler', name='i')


    # create network population and recurrent connections
    neuron_groups[Ae] = neuron_groups['e'][0 : args.n_e]
    neuron_groups[Ai] = neuron_groups['i'][0 : n_i]

    neuron_groups[Ae].v = v_rest_e - 40. * b2.mV
    neuron_groups[Ai].v = v_rest_i - 40. * b2.mV
    
    if test_mode: 
        theta_filename = weight_path + 'theta_' + population_name + str(num_training_examples) + ending 
        if not os.path.isfile(theta_filename): 
            theta_filename = weight_path + 'theta_' + population_name + ending 
            
        print(theta_filename)
        neuron_groups['e'].theta = np.load(theta_filename) * b2.volt 

    else:
        neuron_groups['e'].theta = np.ones((args.n_e)) * 20.0*b2.mV

    weight['ei'] = 10.4
    weight['ie'] = 17.0

    for conn_type in recurrent_conn_names:
        connName = population_name + conn_type[0] + population_name + conn_type[1]
        
        if connName == 'AeAi':
            weightList = np.array([(i, i, weight['ei']) for i in range(args.n_e)])
            weightMatrix = snn_model_class.get_matrix_from_file(weightList=weightList, fileName=None, n_input=n_input, n_e=args.n_e, n_i=n_i) 
        elif connName == 'AiAe':
            weightList = np.full((n_i, args.n_e), weight['ie'])
            np.fill_diagonal(weightList, 0)
            weightList = np.array([(i, j, weightList[i,j]) for i in range(n_i) for j in range(args.n_e)])
            weightMatrix = snn_model_class.get_matrix_from_file(weightList=weightList, fileName=None, n_input=n_input, n_e=args.n_e, n_i=n_i)  

        model = 'w : 1'
        pre = 'g%s_post += w' % conn_type[0]
        post = ''

        connections[connName] = b2.Synapses(neuron_groups[connName[0:2]], neuron_groups[connName[2:4]], model=model, on_pre=pre, on_post=post, name='S_'+connName[0:2])
        connections[connName].connect() # all-to-all connection
        connections[connName].w = weightMatrix[connections[connName].i, connections[connName].j]
    
    spike_monitors[Ae] = b2.SpikeMonitor(neuron_groups[Ae], name='Ae_SM')


    # create input population and connections from input populations
    input_groups[Xe] = b2.PoissonGroup(n_input, 0*b2.Hz, name='Xe')

    weight['ee_input'] = 0.3
    connName = input_connection_name[0] + input_conn_name[0] + input_connection_name[1] + input_conn_name[1]
    
    weight_filename = weight_path + connName + str(num_training_examples) + ending
    if not os.path.isfile(weight_filename): 
        weight_filename = weight_path + connName + ending

    if test_mode:
        print(weight_filename)
        weightMatrix = snn_model_class.get_matrix_from_file(weightList=None, fileName=weight_filename, n_input=n_input, n_e=args.n_e, n_i=n_i)
    else:
        weightList = np.random.random((n_input, args.n_e)) + 0.01
        weightList *= weight['ee_input']
        weightList = np.array([(i, j, weightList[i,j]) for j in range(args.n_e) for i in range(n_input)])
        weightMatrix = snn_model_class.get_matrix_from_file(weightList=weightList, fileName=None, n_input=n_input, n_e=args.n_e, n_i=n_i)
            
    
    model = 'w : 1'
    pre = 'g%s_post += w' % input_conn_name[0]
    post = ''
    if not test_mode:
        model += eqs_stdp_ee
        pre += '\n ' + eqs_stdp_pre_ee 
        post = eqs_stdp_post_ee

    connections[connName] = b2.Synapses(input_groups[connName[0:2]], neuron_groups[connName[2:4]], model=model, on_pre=pre, on_post=post, name='S_'+connName)

    minDelay = delay[input_conn_name][0]
    maxDelay = delay[input_conn_name][1]
    deltaDelay = maxDelay - minDelay

    connections[connName].connect(True) # all-to-all connection
    connections[connName].delay = 'minDelay + rand() * deltaDelay'
    connections[connName].w = weightMatrix[connections[connName].i, connections[connName].j]


    # run the simulation and set inputs
    net = b2.Network()
    for obj_list in [neuron_groups, input_groups, connections, spike_monitors]: 
        for key in obj_list:
            net.add(obj_list[key])        
        
    # run the simulation 
    num_labels = args.num_labels if not test_mode else args.num_test_labels
    
    figures = [1,2]

    if not test_mode:
        plot_2d_input_weights(connections[Xe+Ae], n_input, args.n_e, outputsPath, figures[1], tag="1")
        
    if do_plot_performance:
        performance = plot_initial_performance(num_examples, update_interval, outputsPath, figures[0])

    input_groups[Xe].rates = 0 * b2.Hz
    net.run(initial_resting_time)

    input_intensity = args.intensity 
    start_input_intensity = input_intensity

    previous_spike_count = np.zeros(args.n_e)
    assignments = np.zeros(args.n_e)
    input_numbers = np.zeros(num_examples)
    outputNumbers = np.zeros((num_examples, num_labels))
    result_monitor = np.zeros((update_interval,args.n_e))
    previous_spike_count = np.copy(spike_monitors[Ae].count[:])

    j = 0
    kkk = 0 

    for j in range(int(num_examples)):

        if test_mode:
            spike_rates = testing_data['x'][j%num_testing_imgs,:,:].reshape((n_input)) / input_intensity                  
        else:
            normalize_weights(connections, args.n_e)
            spike_rates = training_data['x'][j%num_training_imgs,:,:].reshape((n_input)) / input_intensity              
        
        input_groups[Xe].rates = spike_rates * b2.Hz

        print('run number:', j+1, 'of', int(num_examples))
        net.run(single_example_time, report='text')

        if j % update_interval == 0 and j > 0:
            assignments = get_new_assignments(result_monitor[:], input_numbers[j-update_interval : j], args.n_e)

            print("Unique labels learnt: \n", np.unique(assignments))
            np.save(outputsPath + "resultPopVecs" + str(j), result_monitor) 

            if not test_mode:                
                plot_2d_input_weights(connections[Xe+Ae], n_input, args.n_e, outputsPath, figures[1])                
                save_connections(connections, weightsPath, str(j), save_all=False)
                save_theta(neuron_groups[Ae].theta, weightsPath, population_name, str(j))

        current_spike_count = np.asarray(spike_monitors[Ae].count[:]) - previous_spike_count
        previous_spike_count = np.copy(spike_monitors[Ae].count[:])

        if repeat_no_spikes and np.sum(current_spike_count) < min_num_spikes and kkk == 0:
            input_intensity += 1
            input_groups[Xe].rates = 0 * b2.Hz        
            net.run(resting_time)
            kkk += 1 

        else:
            kkk = 0
            result_monitor[j%update_interval,:] = current_spike_count
            input_numbers[j] = testing_data['y'][j%num_testing_imgs][0] if test_mode else training_data['y'][j%num_training_imgs][0]

            outputNumbers[j,:], summed_rates = get_recognized_number_ranking(assignments, result_monitor[j%update_interval,:], np.arange(args.offset_after_skip, args.offset_after_skip+num_labels))

            if j % update_interval == 0 and j > 0 and do_plot_performance: 
                performance = plot_performance(performance, j, update_interval, outputNumbers, input_numbers, outputsPath, figures[0])
                
                print("Classification performance at {}: \n{}".format(j, performance[:int((j/float(update_interval))+1)] ) )      

            input_groups[Xe].rates = 0 * b2.Hz
            net.run(resting_time)
            input_intensity = start_input_intensity
            j += 1
                
    
    b2.device.delete(code=False)

    print('output numbers: \n', outputNumbers, '\nSummed rates: \n', summed_rates)

    print('save results')
    if not test_mode:
        save_theta(neuron_groups[Ae].theta, weightsPath, population_name, str(num_examples), use_initial_name=True)
        save_connections(connections, weightsPath, str(num_examples), use_initial_name=True)

    np.save(outputsPath + "resultPopVecs" + str(num_examples) + ad_path_test, result_monitor)
    np.save(outputsPath + "inputNumbers" + str(num_examples) + ad_path_test, input_numbers)


    # plot results
    plot_2d_input_weights(connections[Xe+Ae], n_input, args.n_e, outputsPath, figures[1], tag="3")

    if spike_monitors:
        plot_spikeMonitors(spike_monitors, outputsPath, args.epochs, test_mode)
        plot_spikeMonitorsCount(spike_monitors, outputsPath)

    print('done')
    


def save_connections(connections, weightsPath, interval = '', save_all=True, use_initial_name=False):
    '''
    Saves a zip of presynaptic and postsynaptic neuron connection indices and their corresponding weight value 
    '''

    print('save connections')

    for conn in connections: 
        if conn != "XeAe" and not save_all:
            continue 
        elif conn == "AiAe" or conn == "AeAi":
            updated_interval = ""
        else:
            updated_interval = interval 
        
        if use_initial_name:
            updated_interval = ""
            
        connListSparse = zip(connections[conn].i, connections[conn].j, connections[conn].w)

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






def set_mode(mode, modes):
    
    if mode in modes:
        test_mode = True if mode != "train" else False
        logfile_name = modes[mode]
    else:
        raise ValueError("Invalid mode.")
    
    return test_mode, logfile_name
    



    


if __name__ == "__main__":
    

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', type=str, default="nordland", 
                        help='Dataset folder name that is relative to this repo. The folder must exist in this directory: ./../data/')
    parser.add_argument('--num_labels', type=int, default=5, 
                        help='Number of training place labels for a single module.')
    parser.add_argument('--num_cal_labels', type=int, default=0, 
                        help="Number of calibration place labels.")
    parser.add_argument('--num_test_labels', type=int, default=5, 
                        help='Number of testing place labels.')
    parser.add_argument('--num_query_imgs', type=int, default=5, 
                        help='Number of query images used for testing and calibration.')
    parser.add_argument('--tc_ge', type=float, default=1.0, 
                        help='Time constant of conductance of excitatory synapses AeAi')
    parser.add_argument('--tc_gi', type=float, default=0.5, 
                        help='Time constant of conductance of inhibitory synapses AiAe')
    parser.add_argument('--intensity', type=int, default=4, 
                        help="Intensity scaling factor to change the range of input pixel values")
    parser.add_argument('--use_weighted_assignments', type=bool, default=False, 
                        help='Value to define the type of neuronal assignment to use: standard=False, weighted=True')
    parser.add_argument('--shuffled', type=bool, default=True, 
                        help='Value to define whether the order of input images should be shuffled: shuffled order of images=True, consecutive image order=False')
    
    parser.add_argument('--skip', type=int, default=8, 
                        help='The number of images to skip between each place label.')
    parser.add_argument('--offset_after_skip', type=int, default=0, 
                        help='The offset to apply for selecting places after skipping every n images.')
    parser.add_argument('--folder_id', type=str, default="NRD_SFS", 
                        help='Id to distinguish the traverses used from the dataset.')
    parser.add_argument('--update_interval', type=int, default=50, 
                        help='The number of iterations to save at one time in output matrix.')
    parser.add_argument('--epochs', type=int, default=20, 
                        help='Number of passes through the dataset.')
    parser.add_argument('--n_e', type=int, default=100, 
                        help='Number of excitatory output neurons. The number of inhibitory neurons are defined the same.')
    parser.add_argument('--threshold_i', type=int, default=0, 
                        help='Threshold value used to ignore the hyperactive neurons.')
    parser.add_argument('--seed', type=int, default=0, 
                        help='Set seed for random generator to define the shuffled order of input images, and random initialisation of learned weights.')

    parser.add_argument('--ad_path_test', type=str, default="_test_E{}", 
                        help='Additional string arguments to use for saving test outputs in testing')
    parser.add_argument('--ad_path', type=str, default="_offset{}")             
    parser.add_argument('--multi_path', type=str, default="epoch{}_T{}_T{}") 

    parser.add_argument('--process_mode', type=str, choices=["train", "record", "calibrate", "test"], default="test", 
                        help='String indicator to define the mode (train, record, calibrate, test).')

    parser.set_defaults()
    args = parser.parse_args()
    
    
    main(args)

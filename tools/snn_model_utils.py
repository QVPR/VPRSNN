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



import os
import os.path

import brian2 as b2
import numpy as np


# set the default target for code generation: 'auto', 'cython', 'numpy'
target = 'cython'                 
b2.prefs.codegen.target = target
b2.prefs.codegen.cpp.extra_compile_args_gcc = []

# on hpc with PBS scheduling system, set cache directory to temp directory of the job  
cache_dir = os.environ['TMPDIR']
print("Cache dir: ", cache_dir)
b2.prefs.codegen.runtime.cython.cache_dir = cache_dir
b2.prefs.codegen.runtime.cython.multiprocess_safe = False



def get_matrix_from_file(weightList, fileName=None, n_input=784, n_e=400, n_i=400):
    '''
    Loads a weight matrix from given file name, shaped based on number of input neurons, 
    and number of connections 
    '''

    if fileName != None: 
        assert os.path.isfile(fileName), "Weight matrix file name does not exist: " + fileName
        readout = np.load(fileName, allow_pickle=True)
    elif np.any(weightList):
        readout = np.array(weightList)
    else:
        assert False, "No weight matrix file name or weight list provided"

    if (fileName != None and "XeAe" in fileName) or (np.any(weightList) and weightList.shape[0]==n_e*n_input):
        n_tgt = n_e
        n_src = n_input  
    else:
        n_tgt = n_e
        n_src = n_i

    print(readout.shape, fileName)
    value_arr = np.zeros((n_src, n_tgt))

    if not readout.shape == (0,):
        value_arr[np.int32(readout[:, 0]), np.int32(readout[:, 1])] = readout[:,2]
    return value_arr


class SNNModel(object):

    initial_resting_time = 0.5 * b2.second
    single_example_time = 0.35 * b2.second
    resting_time = 0.15 * b2.second

    v_rest_e = -65 * b2.mV
    v_rest_i = -60 * b2.mV
    v_reset_e = -65 * b2.mV
    v_reset_i = -45 * b2.mV
    v_thresh_e = -52 * b2.mV
    v_thresh_i = -40 * b2.mV
    refrac_e = 5 * b2.ms
    refrac_i = 2 * b2.ms

    # input population name 
    Xe = 'Xe'

    # population A 
    population_name = 'A'
    Ae = 'Ae'
    Ai = 'Ai'

    e = 'e'
    i = 'i'

    tc_pre_ee = 20*b2.ms
    tc_post_1_ee = 20*b2.ms
    tc_post_2_ee = 40*b2.ms
    wmax_ee = 1.0

    tc_e = 100 * b2.ms 
    tc_i = 10 * b2.ms 

    tc_ge = 1.0 * b2.ms
    tc_gi = 2.0 * b2.ms 

    # learning rates 
    nu_ee_pre =  0.0001     
    nu_ee_post = 0.01       

    tc_theta = 1e7 * b2.ms
    theta_plus_e = 0.05 * b2.mV

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


    def __init__(self):
        
        self.input_groups = {}
        self.neuron_groups = {}
        self.connections = {} 
        self.rate_monitors = {}
        self.spike_monitors = {}
        self.net = b2.Network()


    def create_input_neurons(self, n_input, name="Xe"):

        self.input_groups[self.Xe] = b2.PoissonGroup(n_input, 0*b2.Hz, name=name)


    def create_excitatory_neurons(self, weight_path, n_e, num_training_examples, test_mode=False, ending=".npy"):

        if test_mode: 
            scr_e = 'v = v_reset_e; timer = 0*ms'
            self.neuron_eqs_e += '\n  theta :volt'
        else: 
            scr_e = 'v = v_reset_e; theta += theta_plus_e; timer = 0*ms'
            self.neuron_eqs_e += '\n  dtheta/dt = -theta / (tc_theta)  : volt'
        
        self.neuron_eqs_e += '\n  dtimer/dt = 0.1  : second'

        self.neuron_groups['e'] = b2.NeuronGroup(n_e, self.neuron_eqs_e, threshold= self.v_thresh_e_str, refractory= self.refrac_e, reset= scr_e, method='euler', name='e')

        self.neuron_groups[self.Ae] = self.neuron_groups['e'][0 : n_e]
        self.neuron_groups[self.Ae].v = self.v_rest_e - 40. * b2.mV 

        if test_mode: 
            theta_filename = weight_path + 'theta_' + self.population_name + str(num_training_examples) + ending 
            if not os.path.isfile(theta_filename): 
                theta_filename = weight_path + 'theta_' + self.population_name + ending 
            self.neuron_groups['e'].theta = np.load(theta_filename) * b2.volt 
        else:
            self.neuron_groups['e'].theta = np.ones((n_e)) * 20.0*b2.mV
        
        return self.neuron_groups


    def create_inhibitory_neurons(self, n_i):

        self.neuron_groups['i'] = b2.NeuronGroup(n_i, self.neuron_eqs_i, threshold= self.v_thresh_i_str, refractory= self.refrac_i, reset= self.v_reset_i_str, method='euler', name='i')

        self.neuron_groups[self.Ai] = self.neuron_groups['i'][0 : n_i]
        self.neuron_groups[self.Ai].v = self.v_rest_i - 40. * b2.mV
        
        return self.neuron_groups


    def create_population_connections(self, connName, weight_filename, n_input=784, n_e=400, n_i=400, is_input=False, test_mode=False):
        
        weightMatrix = get_matrix_from_file(weight_filename, n_input=n_input, n_e=n_e, n_i=n_i)  
        
        model = 'w : 1'
        pre = 'g%s_post += w' % connName[1]
        post = ''

        if is_input and not test_mode:
            model += self.eqs_stdp_ee
            pre += '\n ' + self.eqs_stdp_pre_ee 
            post = self.eqs_stdp_post_ee
        
        source_neurons = self.neuron_groups[connName[0:2]] if not is_input else self.input_groups[connName[0:2]]
        destination_neurons = self.neuron_groups[connName[2:4]]

        self.connections[connName] = b2.Synapses(source_neurons, destination_neurons, model=model, on_pre=pre, on_post=post, name='S_'+connName)
        self.connections[connName].connect(True)
        self.connections[connName].w = weightMatrix[self.connections[connName].i, self.connections[connName].j]

        if is_input:
            minDelay = 0*b2.ms
            maxDelay = 10*b2.ms
            deltaDelay = maxDelay - minDelay
            delay_eq = 'minDelay + rand() * deltaDelay'
            self.connections[connName].delay = delay_eq

        return self.connections 
    

    def create_network(self):

        for obj_list in [self.neuron_groups, self.input_groups, self.connections, self.rate_monitors, self.spike_monitors]: 
            for key in obj_list:
                self.net.add(obj_list[key])
    

    def run_network(self, time, report='text', profile=False):

        v_rest_e = self.v_rest_e
        v_rest_i = self.v_rest_i
        v_reset_e = self.v_reset_e
        v_reset_i = self.v_reset_i
        v_thresh_e = self.v_thresh_e
        v_thresh_i = self.v_thresh_i
        refrac_e = self.refrac_e
        refrac_i = self.refrac_i

        tc_pre_ee = self.tc_pre_ee
        tc_post_1_ee = self.tc_post_1_ee
        tc_post_2_ee = self.tc_post_2_ee
        wmax_ee = self.wmax_ee

        tc_e = self.tc_e
        tc_i = self.tc_i
        tc_ge = self.tc_ge
        tc_gi = self.tc_gi

        nu_ee_pre =  self.nu_ee_pre   
        nu_ee_post = self.nu_ee_post      

        offset = self.offset
        tc_theta = self.tc_theta
        theta_plus_e = self.theta_plus_e

        self.net.run(time, report=report, profile=profile) 


    def create_spike_monitor(self, neuron_group, name):

        self.spike_monitors[name] = b2.SpikeMonitor(neuron_group, name="SM_"+name)


    def create_population_rate_monitor(self, neuron_group, name):
        
        self.rate_monitors[name] = b2.PopulationRateMonitor(neuron_group, name="RM_"+name)

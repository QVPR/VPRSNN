
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
import numpy as np
import wandb




def create_hpc_bashscript_wandb(args, offset_after_skip, sweep_id):

    run_filename = 'run_offset{}.sh'.format(offset_after_skip)
    with open (run_filename, 'w') as rsh:
        rsh.write('''\
#!/bin/bash -l
#PBS -N SVN_2L
#PBS -l walltime=10:00:00
#PBS -l mem=40GB
#PBS -l ncpus=1
#PBS -j oe

cd $PBS_O_WORKDIR

# Load conda environment
conda activate vprsnn


wandb agent --count 1 ${username}/${project_name}/${sweep_id} 

''')

    os.system('qsub -v username={},project_name={},sweep_id={} run_offset{}.sh'.format(args.username, args.project_name, sweep_id, offset_after_skip))



def get_wandb_sweep_id(args, sweep_config):
    if os.path.isfile("sweep_id_{}.npy".format(args.sweep_name)):
        sweep_id = np.load("sweep_id_{}.npy".format(args.sweep_name), allow_pickle=True)

    else:
        sweep_id = wandb.sweep(sweep_config, project=args.project_name)
        np.save("sweep_id_{}.npy".format(args.sweep_name), sweep_id)
    return sweep_id



def setup_wandb_config(offset_after_skip_list, args):
    
    sweep_config = {
    'program': 'modular_snn/one_snn_module_processing.py',
    'method': 'grid',
    'name': args.sweep_name,
    'parameters': {
        'offset_after_skip': {'values': offset_after_skip_list},
        'epochs': {'value': args.epochs},
        'dataset': {'value': args.dataset},
        'folder_id': {'value': args.folder_id},
        'num_test_labels' : {'value': args.num_test_labels}
    },
    }
    
    return sweep_config

#! /usr/bin/env python
import argparse
import os 
import numpy as np
import wandb

'''
To submit the jobs on hpc, simply:
- conda activate snn
- python main.py or
- python main.py --project_name="EnsSNN" --num_test_labels=100 --num_labels=25 --epochs=60 --dataset='nordland' --folder_id='NRD_SFS' --sweep_name='sweep_1'
- python main.py --project_name="EnsSNN" --num_test_labels=2700 --num_labels=25 --epochs=60 --dataset='nordland' --folder_id='NRD_SFS' --sweep_name='sweep_2'
'''

print(wandb.__path__)


def main(args):
    

    offset_after_skip_list = list(range(0, args.num_cal_labels+args.num_test_labels, args.num_labels))
    
    ad_path_base = args.ad_path
    ad_path_test_base = args.ad_path_test
    args_multi_path_base = args.multi_path

    if args.run_mode == 'wandb_hpc' or args.run_mode == 'wandb_local':
        sweep_config = setup_wandb_config(offset_after_skip_list, args)
        sweep_id = get_wandb_sweep_id(args, sweep_config)
        print("sweep id: ", sweep_id)

    for offset_after_skip in offset_after_skip_list:
        args.offset_after_skip = offset_after_skip
        
        if args.run_mode == 'local':
            
            from multiple_evaluation import main as multiple_evaluation_main
            args.ad_path = ad_path_base
            args.ad_path_test = ad_path_test_base
            args.multi_path = args_multi_path_base
            multiple_evaluation_main(args) 
        
    
        elif args.run_mode == 'wandb_hpc':
            
            create_hpc_bashscript_wandb(args, offset_after_skip_list, sweep_id)


        elif args.run_mode == 'wandb_local':

            from multiple_evaluation import main as multiple_evaluation_main

            # only pass sweep_id to function form but full path for terminal command 
            wandb.agent(sweep_id=sweep_id, project=project_name, function=multiple_evaluation_main, count=len(offset_after_skip_list))





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
conda activate snn_v2


wandb agent --count 1 somayeh-h/${project_name}/${sweep_id} 

''')

    os.system('qsub -v project_name={},sweep_id={} run_offset{}.sh'.format(args.project_name, sweep_id, offset_after_skip))




def get_wandb_sweep_id(args, sweep_config):
    if os.path.isfile("sweep_id_{}.npy".format(args.sweep_name)):
        sweep_id = np.load("sweep_id_{}.npy".format(args.sweep_name), allow_pickle=True)

    else:
        sweep_id = wandb.sweep(sweep_config, project=args.project_name)
        np.save("sweep_id_{}.npy".format(args.sweep_name), sweep_id)
    return sweep_id



def setup_wandb_config(offset_after_skip_list, args):
    
    sweep_config = {
    'program': 'multiple_evaluation.py',
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



if __name__ == "__main__":
    

    dataset = "nordland"
    num_test_labels = 15 # 15
    skip = 8 
    num_labels = 5
    folder_id = "NRD_SFS"
    epochs = 20
    n_e = 100
    threshold_i = 0

    tc_gi = 0.5
    offset_after_skip = 0
    
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

    mode = "test" #  "train"
    use_weighted_assignments = False

    run_mode = "local"
    sweep_name = "sweep_T{}".format(num_test_labels)
    project_name = "EnsSNN"

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
    parser.add_argument('--threshold_i', type=int, default=threshold_i, help='Threshold value to ignore hyperactive neurons.')

    parser.add_argument('--ad_path_test', type=str, default=ad_path_test, help='Additional string arguments to use for saving test outputs in testing')
    parser.add_argument('--ad_path', type=str, default=ad_path)             
    parser.add_argument('--multi_path', type=str, default=multi_path)   
    
    parser.add_argument('--run_mode', type=str, default=run_mode, help='Mode to run the modular network.')
    parser.add_argument('--sweep_name', type=str, default=sweep_name, help='Sweep name to be used.')
    parser.add_argument('--project_name', type=str, default=project_name, help='Project name to be used.')
    
    parser.set_defaults()
    args = parser.parse_args()

    main(args)
    

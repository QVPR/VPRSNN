#! /usr/bin/env python
import argparse
import os
import time



'''

python3 merged_bashscript.py --num_test_imgs=100 --num_labels=25 --folder_id='NRD_SFS' --offset_after_skip=600 --skip=8 --multi_path='epoch{}_T{}_T{}' 
python3 merged_bashscript.py --num_test_imgs=2700 --num_labels=25 --folder_id='NRD_SFS' --offset_after_skip=600 --skip=8 --multi_path='epoch{}_T{}_T{}' 


'''




def main(use_all_data=False):
    
    skip = 8 
    n_e = 100        

    epochs = 20
    folder_id = 'NRD_SFS'
    dataset = "nordland"

    num_labels = 5
    num_train_imgs = num_labels * 2 if folder_id == 'NRD_SFS' or folder_id == 'ORC' else num_labels


    offset_after_skip = 5
    num_test_imgs = 15          
    org_num_test_imgs = num_test_imgs
    base_res = 1

    num_cal_labels = 5
    num_test_labels = 15

    ad_path = '_offset{}'     
    multi_path = 'epoch{}_T{}_T{}'    

    use_weighted_assignments = False
    tc_gi = 0.5
    threshold_i = 0
    seed = 0


    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default=dataset, help='Folder name of dataset to be used. Relative to this repo, the folder must exist in this directory: ./../data/')
    parser.add_argument('--skip', type=int, default=skip, help='The number of images to skip between each place label.')
    parser.add_argument('--offset_after_skip', type=int, default=offset_after_skip, help='The offset to apply for selecting places after skipping every n images.')
    parser.add_argument('--epochs', type=int, default=epochs, help='Number of passes through the dataset.')
    parser.add_argument('--n_e', type=int, default=n_e, help='Number of excitatory output neurons. The number of inhibitory neurons are the same.')

    parser.add_argument('--folder_id', type=str, default=folder_id, help='Folder name of dataset to be used.')
    parser.add_argument('--num_train_imgs', type=int, default=num_train_imgs, help='Number of entire training images.')
    parser.add_argument('--num_test_imgs', type=int, default=num_test_imgs, help='Number of entire testing images.')
    parser.add_argument('--org_num_test_imgs', type=int, default=org_num_test_imgs, help='Number of entire testing images for both cal and testing.')
    parser.add_argument('--num_labels', type=int, default=num_labels, help='Number of distinct places to use from the dataset.')
    parser.add_argument('--num_cal_labels', type=int, default=num_cal_labels, help="Number of images needed for calibration. Needed for shuffling input images.")
    parser.add_argument('--num_test_labels', type=int, default=num_test_labels, help='Number of distinct places to use from the dataset for testing.')
    parser.add_argument('--use_weighted_assignments', type=bool, default=use_weighted_assignments, help='Value to define the type of neuronal assignment to use: standard=0, weighted=1')
    parser.add_argument('--base_res', type=int, default=base_res, help='The type of base results to use for hyperactive neuron detection.')
    parser.add_argument('--threshold_i', type=int, default=threshold_i, help='Threshold value to ignore hyperactive neurons.')
    parser.add_argument('--tc_gi', type=float, default=tc_gi, help='Time constant of conductance of inhibitory synapses AiAe')
    parser.add_argument('--seed', type=int, default=seed, help='Set seed for random generator.')
            
    parser.add_argument('--val_mode', dest="val_mode", action="store_true", help='Boolean indicator to switch to validation mode.')
    parser.add_argument('--test_mode', dest="test_mode", action="store_true", help='Boolean indicator to switch between training and testing mode.')

    parser.add_argument('--ad_path', type=str, default=ad_path)
    parser.add_argument('--multi_path', type=str, default=multi_path)

    parser.set_defaults()
    args = parser.parse_args()
    
    

    args.num_train_imgs = args.num_labels * 2 if args.folder_id == 'NRD_SFS' or args.folder_id == 'NRD_SFW' or args.folder_id == 'ORC' else args.num_labels
    args.org_num_test_imgs = args.num_test_imgs + args.num_cal_labels

    threshold_i_list = [0, 5, 10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 400, 500, 600]   


    args_multi_path_base = args.multi_path
    args_num_test_imgs_base = args.num_test_imgs
    args_offset_after_skip_base = args.offset_after_skip


    for threshold_i in threshold_i_list:

        args.threshold_i = threshold_i
        
        if use_all_data: 
            args.num_test_imgs = args_num_test_imgs_base
            args.offset_after_skip = 0
            args.multi_path = args_multi_path_base.format(args.epochs, args.num_test_imgs, args.threshold_i)
            args.test_mode = True
            
            create_bashscript_merged(args)
        
            args.test_mode = False 
            
            
        else: 
            args.num_test_imgs = args.num_cal_labels 
            args.offset_after_skip = 0
            args.multi_path = args_multi_path_base.format(args.epochs, args.org_num_test_imgs, args.threshold_i)
            
            create_bashscript_merged(args)
            time.sleep(1)


            args.num_test_imgs = args_num_test_imgs_base  
            args.offset_after_skip = args_offset_after_skip_base
            args.multi_path = args_multi_path_base.format(args.epochs, args.org_num_test_imgs, args.threshold_i)
            args.test_mode = True
            
            create_bashscript_merged(args)
            time.sleep(1)
            
            
            args.test_mode = False 

            





def create_bashscript_merged(args): 


    if args.test_mode:

        filename = 'run{}_T{}.sh'.format(args.multi_path, args.threshold_i)
        with open (filename, 'w') as rsh:
            rsh.write('''\
#!/bin/bash -l

#PBS -N SVN_2L
#PBS -l walltime=00:30:00
#PBS -l mem=40GB
#PBS -l ncpus=1
#PBS -j oe

cd $PBS_O_WORKDIR

# Load conda environment
conda activate snn_v2_clone 

# perform merged evaluations 
python3 merged_evaluation.py --dataset ${dataset} --skip ${skip} --offset_after_skip ${offset_after_skip} --folder_id ${folder_id} --num_train_imgs ${num_train_imgs} --num_test_imgs ${num_test_imgs} --org_num_test_imgs ${org_num_test_imgs} --num_labels ${num_labels} --epochs ${epochs} --use_weighted_na ${use_weighted_na} --threshold_i ${threshold_i} --tc_gi ${tc_gi} --seed ${seed} --base_res ${base_res} --n_e ${n_e} --ad_path ${ad_path} --multi_path ${multi_path} --test_mode

    ''')

            
    else:
        filename = 'run{}_t{}.sh'.format(args.multi_path, args.threshold_i)
        with open (filename, 'w') as rsh:
            rsh.write('''\
#!/bin/bash -l

#PBS -N SVN_2L
#PBS -l walltime=00:20:00
#PBS -l mem=40GB
#PBS -l ncpus=1
#PBS -j oe

cd $PBS_O_WORKDIR

# Load conda environment
conda activate snn_v2_clone 

# perform merged evaluations 
python3 merged_evaluation.py --dataset ${dataset} --skip ${skip} --offset_after_skip ${offset_after_skip} --folder_id ${folder_id} --num_train_imgs ${num_train_imgs} --num_test_imgs ${num_test_imgs} --org_num_test_imgs ${org_num_test_imgs} --num_labels ${num_labels} --epochs ${epochs} --use_weighted_na ${use_weighted_na} --threshold_i ${threshold_i} --tc_gi ${tc_gi} --seed ${seed} --base_res ${base_res} --n_e ${n_e} --ad_path ${ad_path} --multi_path ${multi_path}

    ''')
            

    # os.system('qsub -v dataset={},skip={},offset_after_skip={},folder_id={},num_train_imgs={},num_test_imgs={},org_num_test_imgs={},num_labels={},epochs={},n_e={},use_weighted_na={},threshold_i={},tc_gi={},seed={},base_res={},ad_path={},multi_path={} {}'.
    #         format(args.dataset, args.skip, args.offset_after_skip, args.folder_id, args.num_train_imgs, args.num_test_imgs, args.org_num_test_imgs, args.num_labels, args.epochs, args.n_e, args.use_weighted_na, args.threshold_i, args.tc_gi, args.seed, args.base_res, args.ad_path, args.multi_path, filename))  



    from merged_evaluation import main as merged_evaluation_main
    merged_evaluation_main(args)





if __name__ == "__main__":
    
    main()


            
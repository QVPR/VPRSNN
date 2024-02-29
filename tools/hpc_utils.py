'''
MIT License

Copyright (c) 2024 Somayeh Hussaini, Michael Milford and Tobias Fischer

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


def create_bashscript(args):


    if args.process_mode == "train":
        with open("run{}.sh".format(args.ad_path), 'w') as rsh:
            rsh.write('''\
#!/bin/bash -l

#PBS -N SVN_2L
#PBS -l walltime=28:00:00
#PBS -l mem=70GB
#PBS -l ncpus=1
#PBS -j oe

cd $PBS_O_WORKDIR

# Load conda environment
micromamba activate vprsnn


# Run evaluation script to get the result across all segmented data 
python3 modular_snn/one_snn_module_processing.py --dataset ${dataset} --num_labels ${num_labels} --num_test_labels ${num_test_labels} --tc_gi ${tc_gi} --num_cal_labels ${num_cal_labels} --num_query_imgs ${num_query_imgs} --skip ${skip} --offset_after_skip ${offset_after_skip} --update_interval ${update_interval} --folder_id ${folder_id} --epochs ${epochs} --n_e ${n_e} --seed ${seed} --process_mode ${process_mode} --ad_path ${ad_path} --ad_path_test ${ad_path_test} --multi_path ${multi_path}

    ''')

    else:

        with open("run{}.sh".format(args.ad_path), 'w') as rsh:
            rsh.write('''\
#!/bin/bash -l

#PBS -N SVN_2L
#PBS -l walltime=15:00:00
#PBS -l mem=70GB
#PBS -l ncpus=1
#PBS -j oe

cd $PBS_O_WORKDIR

# Load conda environment
micromamba activate vprsnn


# Run evaluation script to get the result across all segmented data 
python3 modular_snn/one_snn_module_processing.py --dataset ${dataset} --num_labels ${num_labels} --num_test_labels ${num_test_labels} --tc_gi ${tc_gi} --num_cal_labels ${num_cal_labels} --num_query_imgs ${num_query_imgs} --skip ${skip} --offset_after_skip ${offset_after_skip} --update_interval ${update_interval} --folder_id ${folder_id} --epochs ${epochs} --n_e ${n_e} --seed ${seed} --process_mode ${process_mode} --ad_path ${ad_path} --ad_path_test ${ad_path_test} --multi_path ${multi_path}

    ''')

    os.system('qsub -v dataset={},num_labels={},num_test_labels={},tc_gi={},num_cal_labels={},num_query_imgs={},skip={},offset_after_skip={},update_interval={},folder_id={},epochs={},n_e={},seed={},process_mode={},ad_path={},ad_path_test={},multi_path={} run{}.sh'.
            format(args.dataset, args.num_labels, args.num_test_labels, args.tc_gi, args.num_cal_labels, args.num_query_imgs, args.skip, args.offset_after_skip, args.update_interval, args.folder_id, args.epochs, args.n_e, args.seed, args.process_mode, args.ad_path, args.ad_path_test, args.multi_path, args.ad_path))  
    
    return 


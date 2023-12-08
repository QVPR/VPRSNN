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
import os
import os.path
import sys
from pathlib import Path
import numpy as np
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


from tools.logger import Logger
from non_modular_snn.snn_model_evaluation import (compute_recall, invert_dMat, get_unshuffled_results)
from ens_seq.process_ensembles import get_pred_results




def main(args):
        
    results_path = f"merged_results_seq_SNN_S{args.seed}/" if not args.use_ensemble else f"merged_results_ens_SNN_S{args.seed}/" 
    data_path = os.path.join(args.mainfolder_path, results_path)
    Path(data_path).mkdir(parents=True, exist_ok=True)
    
    sys.stdout = Logger(data_path, logfile_name=f"logfile_seqMatching")

    if args.use_ensemble:
        summed_rates_i = np.load(os.path.join(args.SNN_data_path, "ensemble_summed_rates.npy"))
    else:
        summed_rates_i = np.load(os.path.join(args.SNN_data_path, "summed_rates.npy"))
        
        shuffled_indices_path = os.path.join(args.mainfolder_path, f"shuffled_indices_L{args.num_query_imgs}_S{args.seed}.npy")
        if  args.shuffled:
            if not os.path.isfile(shuffled_indices_path):
                assert f"Shuffled indices file not found! {shuffled_indices_path}"
            shuffled_indices = np.load(shuffled_indices_path)    
            summed_rates_i = get_unshuffled_results(summed_rates_i, shuffled_indices, args.num_cal_labels)
            
    gt_labels = np.arange(summed_rates_i.shape[0])
    rates_matrix = invert_dMat(summed_rates_i)
    test_results = get_pred_results(rates_matrix)
    
    n_values = [1, 5, 10, 15, 20, 25]
    numQ = rates_matrix.shape[1]
    print("\nseq len: 1")
    compute_recall(gt_labels, test_results, numQ, n_values, data_path, name=f"recallAtN_SNN{args.seed}_SL1")
    
    print("\nSequence matching with varying sequence lengths:")
    for seq_len in args.seq_len_list:
        
        print("\nseq len: ", seq_len)
        
        dist_matrix_seqslam = compute_dist_matrix_seqslam(rates_matrix, seq_len=seq_len)        
        test_results = get_pred_results(dist_matrix_seqslam)
 
        numQ = dist_matrix_seqslam.shape[0]
        compute_recall(gt_labels, test_results, numQ, n_values, data_path, name=f"recallAtN_SNN{args.seed}_SL{seq_len}.npy")

    print("done")
    return 


def compute_dist_matrix_seqslam(rates_matrix, seq_len=5):

    dist_matrix_numpy = torch.from_numpy(rates_matrix)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    dist_matrix = dist_matrix_numpy.to(device).type(torch.float).unsqueeze(0).unsqueeze(0)
    precomputed_convWeight = torch.eye(seq_len, device=device).unsqueeze(0).unsqueeze(0)
    
    dist_matrix_seqslam = torch.nn.functional.conv2d(dist_matrix, precomputed_convWeight).squeeze()
    dist_matrix_seqslam = dist_matrix_seqslam.cpu().detach().numpy()

    return dist_matrix_seqslam




if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--SNN_data_path', type=str, default="outputs_ne43200_L2700_offset3275_tcgi0.5_S{}_M2/standard/epoch{}_T3300_T{}/", 
                        help='data path of the ensemble members, individual Modular SNN outputs')
    parser.add_argument('--mainfolder_path', type=str, default="./outputs/outputs_models_Nordland_SFW", 
                        help='data path of the individual Modular SNN outputs for a particular dataset')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed value of the ensemble members, individual Modular SNNs, that is used to define the shuffled order of input images, and random initialisation of learned weights.')
    parser.add_argument('--epochs', type=int, default=60, 
                        help='List of epoch values of the ensemble members, individual Modular SNNs.')
    parser.add_argument('--thresholds', type=int, default=80, 
                        help='List of threshold values of the ensemble members, individual Modular SNNs.')
    parser.add_argument('--num_query_imgs', type=int, default=3300, 
                        help='Number of query images used for testing and calibration.')
    parser.add_argument('--num_cal_labels', type=int, default=600, 
                        help="Number of calibration place labels.")
    parser.add_argument('--seq_len_list', nargs='+', type=int, default=[2, 4, 6, 10], 
                        help='List of sequence lengths to apply sequence matching to individually.')
    parser.add_argument('--use_ensemble', type=bool, default=False, 
                        help='Value to define the type of model to apply sequence matching to: Modular SNN=False, Ensemble of Modular SNN=True')
    parser.add_argument('--shuffled', type=bool, default=True, 
                        help='Value to define whether the order of input images should be shuffled: shuffled order of images=True, consecutive image order=False') 
    
    parser.set_defaults()
    args = parser.parse_args()
    
    args.SNN_data_path = os.path.join(args.mainfolder_path, args.SNN_data_path.format(args.seed, args.epochs, args.thresholds))

    main(args)
    
    
    
'''

_1: python3 ens_seq/process_seqmatch.py --SNN_data_path merged_results_ens_SNN_S5/ --mainfolder_path ./outputs/outputs_models_Nordland_SFS --seed 5 --num_query_imgs 3300 --num_cal_labels 600 --use_ensemble True

_1: python3 ens_seq/process_seqmatch.py --SNN_data_path outputs_ne43200_L2700_offset3275_tcgi0.5_S{}_M2/standard/epoch{}_T3300_T{}/ --mainfolder_path ./outputs/outputs_models_Nordland_SFS --seed 0 --epochs 70 --thresholds 80 --num_query_imgs 3300 --num_cal_labels 600

_2: python3 ens_seq/process_seqmatch.py --SNN_data_path merged_results_ens_SNN_S5/ --mainfolder_path ./outputs/outputs_models_Nordland_SFW --seed 5 --num_query_imgs 3300 --num_cal_labels 600 --use_ensemble True

_2: python3 ens_seq/process_seqmatch.py --SNN_data_path outputs_ne43200_L2700_offset3275_tcgi0.5_S{}_M2/standard/epoch{}_T3300_T{}/ --mainfolder_path ./outputs/outputs_models_Nordland_SFW --seed 0 --epochs 60 --thresholds 80 --num_query_imgs 3300 --num_cal_labels 600

_3: python3 ens_seq/process_seqmatch.py --SNN_data_path merged_results_ens_SNN_S5/ --mainfolder_path ./outputs/outputs_models_ORC --seed 5 --num_query_imgs 450 --num_cal_labels 75 --use_ensemble True

_3: python3 ens_seq/process_seqmatch.py --SNN_data_path outputs_ne6000_L375_offset425_tcgi0.5_S{}_M2/standard/epoch{}_T450_T{}/ --mainfolder_path ./outputs/outputs_models_ORC --seed 0 --epochs 20 --thresholds 120 --num_query_imgs 450 --num_cal_labels 75

_4: python3 ens_seq/process_seqmatch.py --SNN_data_path merged_results_ens_SNN_S5/ --mainfolder_path ./outputs/outputs_models_SFU_Mountain --seed 5 --num_query_imgs 375 --num_cal_labels 75 --use_ensemble True

_4: python3 ens_seq/process_seqmatch.py --SNN_data_path outputs_ne4800_L300_offset350_tcgi0.5_S{}_M2/standard/epoch{}_T375_T{}/ --mainfolder_path ./outputs/outputs_models_SFU_Mountain --seed 10 --epochs 30 --thresholds 40 --num_query_imgs 375 --num_cal_labels 75

_5: python3 ens_seq/process_seqmatch.py --SNN_data_path merged_results_ens_SNN_S5/ --mainfolder_path ./outputs/outputs_models_Synthia --seed 5 --num_query_imgs 275 --num_cal_labels 50 --use_ensemble True

_5: python3 ens_seq/process_seqmatch.py --SNN_data_path outputs_ne3600_L225_offset250_tcgi0.5_S{}_M2/standard/epoch{}_T275_T{}/ --mainfolder_path ./outputs/outputs_models_Synthia --seed 0 --epochs 20 --thresholds 40 --num_query_imgs 275 --num_cal_labels 50

_6: python3 ens_seq/process_seqmatch.py --SNN_data_path merged_results_ens_SNN_S5/ --mainfolder_path ./outputs/outputs_models_St_Lucia --seed 5 --num_query_imgs 350 --num_cal_labels 50 --use_ensemble True

_6: python3 ens_seq/process_seqmatch.py --SNN_data_path outputs_ne4800_L300_offset325_tcgi0.5_S{}_M2/standard/epoch{}_T350_T{}/ --mainfolder_path ./outputs/outputs_models_St_Lucia --seed 20 --epochs 30 --thresholds 20 --num_query_imgs 350 --num_cal_labels 50


'''


    
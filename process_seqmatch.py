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

from tools.logger import Logger
from non_modular_snn.snn_model_evaluation import (compute_recall, invert_dMat)
from process_ensembles import get_pred_results, get_unshuffled_results




def main(args):
        
    results_path = f"merged_results_seq_SNN_S{args.seeds}/" if not args.use_ensemble else f"merged_results_ens_SNN_S{args.seeds}/" 
    data_path = os.path.join(args.mainfolder_path, results_path)
    Path(data_path).mkdir(parents=True, exist_ok=True)
    
    sys.stdout = Logger(data_path, logfile_name=f"logfile_seqMatching")

    if args.use_ensemble:
        summed_rates_i = np.load(os.path.join(args.SNN_data_path, "ensemble_summed_rates.npy"))
    else:
        summed_rates_i = np.load(os.path.join(args.SNN_data_path, "summed_rates.npy"))
        shuffled_indices = np.load(os.path.join(args.mainfolder_path, "shuffled_indices_L{}_S{}.npy".format(args.num_test_labels+args.num_cal_labels, args.seeds)))    
        summed_rates_i = get_unshuffled_results(summed_rates_i, shuffled_indices, args.num_cal_labels)
        
    gt_labels = np.arange(summed_rates_i.shape[0])
    rates_matrix = invert_dMat(summed_rates_i)
    inv_test_results_idx = get_pred_results(rates_matrix)
    
    n_values = [1, 5, 10, 15, 20, 25]
    numQ = rates_matrix.shape[1]
    print("\nseq len: 1")
    compute_recall(gt_labels, inv_test_results_idx, numQ, n_values, data_path, name=f"recallAtN_SNN{args.seeds}_SL1")
    
    print("\nSequence matching with varying sequence lengths:")        
    seq_len_list = np.array([2, 4, 6, 10])

    for seq_len in seq_len_list:
        
        print("\nseq len: ", seq_len)
        
        dist_matrix_seqslam = compute_dist_matrix_seqslam(rates_matrix, seq_len=seq_len)        
        sorted_pred_idx_seq = get_pred_results(dist_matrix_seqslam)
 
        numQ = dist_matrix_seqslam.shape[0]
        compute_recall(gt_labels, sorted_pred_idx_seq, numQ, n_values, data_path, name=f"recallAtN_SNN{args.seeds}_SL{seq_len}.npy")

    print("done")
    return 


def compute_dist_matrix_seqslam(rates_matrix, seq_len=5):
    
    dist_matrix_numpy = torch.from_numpy(rates_matrix)
    
    dist_matrix = dist_matrix_numpy.to('cuda').type(torch.cuda.FloatTensor).unsqueeze(0).unsqueeze(0)
    precomputed_convWeight = torch.eye(seq_len, device='cuda').unsqueeze(0).unsqueeze(0)
    
    dist_matrix_seqslam = torch.nn.functional.conv2d(dist_matrix, precomputed_convWeight).squeeze()
    dist_matrix_seqslam = dist_matrix_seqslam.cpu().detach().numpy()

    return dist_matrix_seqslam




if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--SNN_data_path', type=str, default="outputs_ne43200_L2700_offset3275_tcgi0.5_S{}_M2/standard/epoch{}_T3300_T{}/", 
                        help='data path of the ensemble members, individual Modular SNN outputs')
    parser.add_argument('--mainfolder_path', type=str, default="./outputs/outputs_models_Nordland_SFW", 
                        help='data path of the individual Modular SNN outputs for a particular dataset')
    parser.add_argument('--seeds', type=int, default=0,
                        help='List of seed values of the ensemble members, individual Modular SNNs.')
    parser.add_argument('--epochs', type=int, default=60, 
                        help='List of epoch values of the ensemble members, individual Modular SNNs.')
    parser.add_argument('--thresholds', type=int, default=80, 
                        help='List of threshold values of the ensemble members, individual Modular SNNs.')
    parser.add_argument('--num_test_labels', type=int, default=2700, 
                        help='Number of testing place labels.')
    parser.add_argument('--num_cal_labels', type=int, default=600, 
                        help="Number of calibration place labels.")
    parser.add_argument('--use_ensemble', type=bool, default=False, 
                        help='Value to define the type of model to apply sequence matching to: Modular SNN=False, Ensemble of Modular SNN=True')
    
    parser.set_defaults()
    args = parser.parse_args()
    
    args.SNN_data_path = os.path.join(args.mainfolder_path, args.SNN_data_path.format(args.seeds, args.epochs, args.thresholds))

    main(args)
    
    
    
'''

python3 process_seqmatch_v2.py --SNN_data_path merged_results_ens_SNN_S5/ --mainfolder_path ./outputs/outputs_models_Nordland_SFS --seeds 5 --num_test_labels 2700 --num_cal_labels 600 --use_ensemble True

python3 process_seqmatch_v2.py --SNN_data_path outputs_ne43200_L2700_offset3275_tcgi0.5_S{}_M2/standard/epoch{}_T3300_T{}/ --mainfolder_path ./outputs/outputs_models_Nordland_SFS --seeds 0 --epochs 70 --thresholds 80 --num_test_labels 2700 --num_cal_labels 600

python3 process_seqmatch_v2.py --SNN_data_path merged_results_ens_SNN_S5/ --mainfolder_path ./outputs/outputs_models_Nordland_SFW --seeds 5 --num_test_labels 2700 --num_cal_labels 600 --use_ensemble True

python3 process_seqmatch_v2.py --SNN_data_path outputs_ne43200_L2700_offset3275_tcgi0.5_S{}_M2/standard/epoch{}_T3300_T{}/ --mainfolder_path ./outputs/outputs_models_Nordland_SFW --seeds 0 --epochs 60 --thresholds 80 --num_test_labels 2700 --num_cal_labels 600

python3 process_seqmatch_v2.py --SNN_data_path merged_results_ens_SNN_S5/ --mainfolder_path ./outputs/outputs_models_ORC --seeds 5 --num_test_labels 375 --num_cal_labels 75 --use_ensemble True

python3 process_seqmatch_v2.py --SNN_data_path outputs_ne6000_L375_offset425_tcgi0.5_S{}_M2/standard/epoch{}_T450_T{}/ --mainfolder_path ./outputs/outputs_models_ORC --seeds 0 --epochs 20 --thresholds 120 --num_test_labels 375 --num_cal_labels 75

python3 process_seqmatch_v2.py --SNN_data_path merged_results_ens_SNN_S5/ --mainfolder_path ./outputs/outputs_models_SFU_Mountain --seeds 5 --num_test_labels 300 --num_cal_labels 75 --use_ensemble True

python3 process_seqmatch_v2.py --SNN_data_path outputs_ne4800_L300_offset350_tcgi0.5_S{}_M2/standard/epoch{}_T375_T{}/ --mainfolder_path ./outputs/outputs_models_SFU_Mountain --seeds 10 --epochs 30 --thresholds 40 --num_test_labels 300 --num_cal_labels 75

python3 process_seqmatch_v2.py --SNN_data_path merged_results_ens_SNN_S5/ --mainfolder_path ./outputs/outputs_models_Synthia --seeds 5 --num_test_labels 225 --num_cal_labels 50 --use_ensemble True

python3 process_seqmatch_v2.py --SNN_data_path outputs_ne3600_L225_offset250_tcgi0.5_S{}_M2/standard/epoch{}_T275_T{}/ --mainfolder_path ./outputs/outputs_models_Synthia --seeds 0 --epochs 20 --thresholds 40 --num_test_labels 225 --num_cal_labels 50

python3 process_seqmatch_v2.py --SNN_data_path merged_results_ens_SNN_S5/ --mainfolder_path ./outputs/outputs_models_St_Lucia --seeds 5 --num_test_labels 300 --num_cal_labels 50 --use_ensemble True

python3 process_seqmatch_v2.py --SNN_data_path outputs_ne4800_L300_offset325_tcgi0.5_S{}_M2/standard/epoch{}_T350_T{}/ --mainfolder_path ./outputs/outputs_models_St_Lucia --seeds 20 --epochs 30 --thresholds 20 --num_test_labels 300 --num_cal_labels 50


'''


    
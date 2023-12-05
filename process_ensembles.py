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

from tools.logger import Logger
from non_modular_snn.snn_model_evaluation import compute_recall, invert_dMat 




def main(args):
    
    
    data_path = os.path.join(args.mainfolder_path, f"merged_results_ens_SNN_S{len(args.seeds)}/")
    Path(data_path).mkdir(parents=True, exist_ok=True)
    
    sys.stdout = Logger(data_path, logfile_name=f"logfile_ensembling")
    
    summed_rates_all = []
    n_values = [1, 5, 10, 15, 20, 25]
    
    for i in range(len(args.seeds)):
        
        print("Seed: ", args.seeds[i])
        module_path = args.SNN_data_path.format(args.seeds[i], args.epochs[i], args.thresholds[i])

        summed_rates_i_datapath = os.path.join(module_path, "summed_rates.npy")
        summed_rates_i = np.load(summed_rates_i_datapath)
        shuffled_indices_path = os.path.join(args.mainfolder_path, f"shuffled_indices_L{args.num_query_imgs}_S{args.seeds[i]}.npy")
        shuffled_indices = np.load(shuffled_indices_path)
        summed_rates_i = get_unshuffled_results(summed_rates_i, shuffled_indices, args.num_cal_labels)
        
        summed_rates_all.append(summed_rates_i)
        
        gt_labels = np.arange(summed_rates_i.shape[0])
        rates_matrix = invert_dMat(summed_rates_i)
        inv_test_results_idx = get_pred_results(rates_matrix)
        numQ = rates_matrix.shape[1]
        compute_recall(gt_labels, inv_test_results_idx, numQ, n_values, data_path, name=f"recallAtN_SNN{args.seeds[i]}_SL1")
        
    print("Ensembling...")
    summed_rates_all = np.array(summed_rates_all)
    summed_rates_i = np.sum(summed_rates_all, axis=0)
    np.save(os.path.join(data_path, "ensemble_summed_rates.npy"), summed_rates_i)
    
    rates_matrix = invert_dMat(summed_rates_i)
    inv_test_results_idx = get_pred_results(rates_matrix)

    numQ = rates_matrix.shape[1]
    compute_recall(gt_labels, inv_test_results_idx, numQ, n_values, data_path, name=f"recallAtN_ens_{len(args.seeds)}_SNNs_SL1")
    
    print("done")
    return 


def get_unshuffled_results(summed_rates_i, shuffled_indices, num_cal_labels, process_mode="test"):
        
    test_shuffled_indices = shuffled_indices[num_cal_labels:] if process_mode == "test" else shuffled_indices[:num_cal_labels]
    sorted_indices = np.argsort(test_shuffled_indices)
    
    summed_rates_i = summed_rates_i[sorted_indices, :]
    summed_rates_i = summed_rates_i[:, sorted_indices]
    
    return summed_rates_i


def get_pred_results(summed_rates_i):
    
    sorted_rates_indices = np.argsort(summed_rates_i, axis=0)
    
    return sorted_rates_indices


 
'''
_1: python3 process_ensembles.py --SNN_data_path outputs_ne43200_L2700_offset3275_tcgi0.5_S{}_M2/standard/epoch{}_T3300_T{} --mainfolder_path ./outputs/outputs_models_Nordland_SFS --seeds 0 10 20 30 40 --epochs 70 60 60 60 70 --thresholds 80 180 140 220 60 --num_query_imgs 3300 --num_cal_labels 600

_2: python3 process_ensembles.py --SNN_data_path outputs_ne43200_L2700_offset3275_tcgi0.5_S{}_M2/standard/epoch{}_T3300_T{} --mainfolder_path ./outputs/outputs_models_Nordland_SFW --seeds 0 10 20 30 40 --epochs 60 70 70 60 60 --thresholds 80 40 120 180 300 --num_query_imgs 3300 --num_cal_labels 600

_3: python3 process_ensembles.py --SNN_data_path outputs_ne6000_L375_offset425_tcgi0.5_S{}_M2/standard/epoch{}_T450_T{} --mainfolder_path ./outputs/outputs_models_ORC --seeds 0 10 20 30 40 --epochs 20 60 20 20 30 --thresholds 120 20 180 80 80 --num_query_imgs 450 --num_cal_labels 75

_4: python3 process_ensembles.py --SNN_data_path outputs_ne4800_L300_offset350_tcgi0.5_S{}_M2/standard/epoch{}_T375_T{} --mainfolder_path ./outputs/outputs_models_SFU_Mountain --seeds 10 20 30 40 50 --epochs 30 30 20 20 30 --thresholds 40 40 80 120 60 --num_query_imgs 375 --num_cal_labels 75

_5: python3 process_ensembles.py --SNN_data_path outputs_ne3600_L225_offset250_tcgi0.5_S{}_M2/standard/epoch{}_T275_T{} --mainfolder_path ./outputs/outputs_models_Synthia --seeds 0 10 20 30 40 --epochs 20 20 20 20 20 --thresholds 40 60 160 140 40 --num_query_imgs 275 --num_cal_labels 50

_6: python3 process_ensembles.py --SNN_data_path outputs_ne4800_L300_offset325_tcgi0.5_S{}_M2/standard/epoch{}_T350_T{} --mainfolder_path ./outputs/outputs_models_St_Lucia --seeds 20 30 40 50 70 --epochs 30 20 30 20 20 --thresholds 20 160 40 60 80 --num_query_imgs 350 --num_cal_labels 50

'''


if __name__ == "__main__":
    

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--SNN_data_path', type=str, default="outputs_ne43200_L2700_offset3275_tcgi0.5_S{}_M2/standard/epoch{}_T3300_T{}", 
                        help='data path of the ensemble members, individual Modular SNN outputs')
    parser.add_argument('--mainfolder_path', type=str, default="./outputs/outputs_models_Nordland_SFW", 
                        help='data path of the individual Modular SNN outputs for a particular dataset')
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 10, 20, 30, 40], 
                        help='List of seed values of the ensemble members, individual Modular SNNs.')
    parser.add_argument('--epochs', nargs='+', type=int, default=[60, 70, 70, 60, 60], 
                        help='List of epoch values of the ensemble members, individual Modular SNNs.')
    parser.add_argument('--thresholds', nargs='+', type=int, default=[80, 40, 120, 180, 300], 
                        help='List of threshold values of the ensemble members, individual Modular SNNs.')
    parser.add_argument('--num_query_imgs', type=int, default=3300, 
                        help='Number of query images used for testing and calibration.')
    parser.add_argument('--num_cal_labels', type=int, default=600, 
                        help="Number of calibration place labels.")
    
    parser.set_defaults()

    args = parser.parse_args()
    
    args.SNN_data_path = os.path.join(args.mainfolder_path, args.SNN_data_path)
    
    main(args)
    
    




import numpy as np
from save_utils import load_flr, load_nparray,save_prediction
from fair_eval import calculate_overall_accuracy, diff_calculate_impact, diff_calculate_misclassification, diff_calculate_mistreatment
import argparse
import os
from save_utils import *

def calculate_fairness_metrics(pred,y_te,z_te):
    #calculate fairness metrics
    acc = calculate_overall_accuracy(pred,y_te,True)
    di = diff_calculate_impact(pred,y_te,z_te)
    omr = diff_calculate_misclassification(pred,y_te,z_te)
    fpr = diff_calculate_mistreatment(pred,y_te,z_te,cond=-1)
    fnr = diff_calculate_mistreatment(pred,y_te,z_te,cond=1)
    
    #print results
    print("accuracy (Acc): ",acc)
    print("disparate impact (DI): ",di)
    print("overall misclassification rate (OMR): ",omr)
    print("false positive rate (FPR): ",fpr)
    print("false negative rate (FNR): ",fnr)

def certify_FLR(dataname,save_dir):
    # load test y and z
    save_path = os.path.join(save_dir,dataname+'_testset.te')
    _,y_te,z_te = load_testdata(save_path)
    
    # load prediction results
    save_path2 = os.path.join(save_dir,dataname+'_flr_pred.pr')
    pred = load_nparray(save_path2)
        
    calculate_fairness_metrics(pred,y_te.flatten(),z_te.flatten())

def main(**args):
    certify_FLR(**args)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description = 'Argument for certifying fair LR')
    parser.add_argument('--dataname',type=str,default='adult',help='select data among adult,bank,compas,german,lsac')
#     parser.add_argument('--saved_model',type=str,default='../results/adult_FLR_model.sm',help='saved model file')
    parser.add_argument('--result_dir', type=str, default='../results',help='result directory for the saved data and model')
    args = parser.parse_args()
      
    main(dataname = args.dataname, save_dir = args.result_dir)
import numpy as np
from save_utils import load_flr, load_nparray,save_prediction
import argparse
import os

import pickle

def load_lr(save_dir='',dataname='adult',filename='LR_model'):
    model_info = ['coef','intercept']
    res = {}
    for mi in model_info:
        save_path = os.path.join(save_dir,dataname+'_'+filename+'_'+mi+'.sm')
        with open(save_path,'rb') as f:
            aa = pickle.load(f)
        res[mi] = aa
    return res

def predict_lr(X_te, coef, intercept):
    dec_eval = np.dot(X_te,coef.T)+intercept
    return np.sign(dec_eval).flatten()

def test_lr(save_te, dataname='adult', save_dir = ''):
    print("load LR model")
    res = load_lr(save_dir,dataname,'LR_model')
    
    print("load test data from", save_te)
    X_te = load_nparray(save_te)
    
    print("predict LR")
    pred = predict_lr(X_te, res['coef'],res['intercept'])
    
    print("save prediction")
    save_prediction(pred,dataname,save_dir,'lr')

def main(**args):
    print(args)
    test_lr(**args)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Argument for testing LR')
    parser.add_argument('--dataname',type=str,default='adult',help='select data among adult,bank,compas,german,lsac')
    parser.add_argument('--result_dir', type=str, default='../results',help='result directory for model and data')
    args = parser.parse_args()
    
    test_data = os.path.join(args.result_dir,args.dataname+"_lr_testX.te")
    main(save_dir = args.result_dir, save_te = test_data,dataname = args.dataname)
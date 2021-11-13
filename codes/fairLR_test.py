import numpy as np
from save_utils import load_flr, load_nparray,save_prediction
from fair_eval import calculate_prule_clf, calculate_odds_clf, calculate_parity_reg, calculate_group_loss,l2_loss, calculate_overall_accuracy,bce_loss
import argparse
import os

def predict_FairLR(X_te, coef):
    res = np.dot(X_te, coef)
    return np.sign(res)
#     return res

def test_FairLR(save_md, save_te, dataname='adult', save_dir = ''):
    print("load FLR model")
    coef = load_flr(save_md)
    
    print("load test data from", save_te)
    X_te = load_nparray(save_te)
    
    print("predict FLR")
    pred = predict_FairLR(X_te, coef)
    
    print("save prediction")
    save_prediction(pred,dataname,save_dir,'flr')

def test_FLR(X_te, y_te,xs_te,load_path = '../results/compas_FLR_model.sm'):
    coef = load_flr(load_path)
    pred = predict_FairLR(X_te,coef)
    calculate_overall_accuracy(pred,y_te)
    calculate_prule_clf(pred,y_te,xs_te)
    calculate_odds_clf(pred,y_te,xs_te)
    
def main(**args):    
    print(args)
    test_FairLR(**args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Argument for testing fair LR')
    parser.add_argument('--dataname',type=str,default='adult',help='select data among adult,bank,compas,german,lsac')
#     parser.add_argument('--saved_model',type=str,default='../results/adult_FLR_model.sm',help='saved model file')
#     parser.add_argument('--test_data', type=str, default='../results/adult_testX.te',help='saved test data file')
    parser.add_argument('--result_dir', type=str, default='../results',help='result directory for model and data')
    args = parser.parse_args()
    
#     save_md = '../results/FLR_model.sm'
#     save_te = '../results/adult_testX.te'
#     save_dir = '../results'
    save_md = os.path.join(args.result_dir,args.dataname+"_FLR_model.sm")
    test_data = os.path.join(args.result_dir,args.dataname+"_flr_testX.te")
    main(save_md = save_md, save_te = test_data,save_dir = args.result_dir,dataname = args.dataname)

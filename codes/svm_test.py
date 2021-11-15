import numpy as np
from sklearn import svm
import pickle
import os
from kernel_fns import calculate_kernel
from save_utils import load_nparray,save_prediction

import argparse
import os


def load_svm(save_dir='',dataname='adult',filename='SVM_model'):
    model_info = ['sv','alpha','kernel','gamma','coef0','degree','intercept']
    res = {}
    for mi in model_info:
        save_path = os.path.join(save_dir,dataname+'_'+filename+'_'+mi+'.sm')
        with open(save_path,'rb') as f:
            aa = pickle.load(f)
        res[mi] = aa
    return res      

def predict_large_lin_svm(X_te,sv,alpha,intercept,kernel,gamma,**kwds):
    Ktests = []
    batch_num = 100
    if sv.shape[0]%batch_num ==0:
        sp_num = int(sv.shape[0]/batch_num)
    else:
        sp_num = int(sv.shape[0]/batch_num) +1
    dv = 0.0
    for i in range(sp_num):
        if i<sp_num-1:
            Kt = np.dot(sv[batch_num*i:batch_num*(i+1)],X_te.T)
#             print(Kt.shape,alpha.flatten()[batch_num*i:batch_num*(i+1)].shape)
            dv += Kt.T.dot(alpha.flatten()[batch_num*i:batch_num*(i+1)])
        else:
            Kt = np.dot(sv[batch_num*i:],X_te.T)
#             print(Kt.shape)
            dv += Kt.T.dot(alpha.flatten()[batch_num*i:])+intercept
    return np.sign(dv).flatten()

def predict_svm(X_te, sv, alpha, intercept, kernel,gamma, **kwds):
    if kernel == 'rbf':
        Ktest = calculate_kernel(sv, X_te, kernel, gamma=gamma)
    elif kernel =='linear':
        Ktest = calculate_kernel(sv, X_te, kernel)
    elif kernel =='poly':
        Ktest = calculate_kernel(sv, X_te, kernel, **kwds)
    dec_eval = Ktest.T.dot(alpha.flatten())+intercept
    return np.sign(dec_eval).flatten()

def test_svm(save_te, dataname='adult',save_dir=''):
    print("load FLR model")
    res = load_svm(save_dir,dataname, "SVM_model")
    
    print("load test data from", save_te)
    X_te = load_nparray(save_te)
    
    print("predict SVM")
    pred = predict_svm(X_te,**res)
#     pred = predict_large_lin_svm(X_te,**res)
    
    print("save prediction")
    save_prediction(pred,dataname,save_dir,'svm')

def main(**args):
    print(args)
    test_svm(**args)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Argument for testing SVM')
    parser.add_argument('--dataname',type=str,default='adult',help='select data among adult,bank,compas,german,lsac')
    parser.add_argument('--result_dir', type=str, default='../results',help='result directory for model and data')
    args = parser.parse_args()
    
    test_data = os.path.join(args.result_dir,args.dataname+"_svm_testX.te")
    main(save_dir = args.result_dir, save_te = test_data,dataname = args.dataname)
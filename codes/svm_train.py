import numpy as np
from sklearn import svm
import pickle
import os
import argparse
from load_data import *
from save_utils import *

def save_svm(clf, dataname='adult', save_dir = '', filename = 'SVM_model'):
    model_info = ['sv','alpha','kernel','gamma','coef0','degree','intercept']
    res = {}
    res['sv'] = clf.support_vectors_
    res['alpha'] = clf.dual_coef_
    res['kernel'] = clf.kernel
    res['gamma'] = clf._gamma
    res['coef0'] = clf.coef0
    res['degree'] = clf.degree
    res['intercept'] = clf.intercept_

    for mi in model_info:
        save_path = os.path.join(save_dir,dataname+'_'+filename+'_'+mi+'.sm')
        resi = res[mi]
        with open(save_path,'wb') as f:
            pickle.dump(resi,f)
      
            
def train_svm(kern = 'rbf', gamm = 0.1, data='adult',save_dir='',filename='SVM_model',rs=42):
    print("load data:", data)
    if data == 'adult':
        X_tr, X_te, y_tr, y_te, xs_tr, xs_te = load_adult_data_prev(svm=True,random_state=rs,intercept=False)
    elif data == 'bank':
        X_tr, X_te, y_tr, y_te, xs_tr, xs_te = load_bank_data(svm=True,random_state=rs,intercept=False)
    elif data == 'compas':
        X_tr, X_te, y_tr, y_te, xs_tr, xs_te = load_compas_data(svm=True,random_state=rs,intercept=False)
    elif data == 'german':
        X_tr, X_te, y_tr, y_te, xs_tr, xs_te = load_german_data(svm=True,random_state=rs,intercept=False)
    elif data == 'lsac':
        X_tr, X_te, y_tr, y_te, xs_tr, xs_te = load_lsac_data(svm=True,random_state=rs,intercept=False)
    
    print("train SVM model")
    clf = svm.SVC(kernel=kern,gamma = gamm)
    clf.fit(X_tr,y_tr)
    print(clf)
    print("test score",clf.score(X_te,y_te))
    
    print("save model")
    save_svm(clf,data,save_dir,filename)
    
    print("save testdata")
    save_testdata(X_te,y_te,xs_te,data+'_svm', save_dir)

def main(**args):
    train_svm(**args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Argument for training SVM')
    parser.add_argument('--dataname',type=str,default='adult',help='select data among adult,bank,compas,german,lsac')
    parser.add_argument('--result_dir', type=str, default='../results',help='save directory for data and model')
    parser.add_argument('--file_name', type=str, default='SVM_model',help='file name for dataset and model')
    parser.add_argument('--kernel', type=str, default='linear',help='kernel for SVM')
    parser.add_argument('--gamma', type=float, default=0.1,help='kernel parameter gamma for SVM')
    args = parser.parse_args()
    print(args)
    
    main(kern = args.kernel, gamm = args.gamma, data = args.dataname, save_dir = args.result_dir, filename = args.file_name)
    
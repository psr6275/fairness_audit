import numpy as np
from sklearn import linear_model as lm
import pickle
import os
import argparse
from load_data import *
from save_utils import *

def save_lr(clf,dataname, save_dir = '', filename = 'LR_model'):
    model_info = ['coef','intercept']
    res = {}
    res['coef'] = clf.coef_
    res['intercept'] = clf.intercept_
    for mi in model_info:
        save_path = os.path.join(save_dir,dataname+'_'+filename+'_'+mi+'.sm')
        resi = res[mi]
        with open(save_path,'wb') as f:
            pickle.dump(resi,f)



def train_LR(data='adult',save_dir='',filename='LR_model',rs=42):
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
    
    print("train LR model")
    clf = lm.LogisticRegression()
    clf.fit(X_tr,y_tr)
    print(clf)
    print("test score",clf.score(X_te,y_te))
    
    print("save model")
    save_lr(clf,data,save_dir,filename)
    
    print("save testdata")
    save_testdata(X_te,y_te,xs_te,data+'_lr', save_dir)

def main(**args):
	train_LR(**args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Argument for training LR')
    parser.add_argument('--dataname',type=str,default='adult',help='select data among adult,bank,compas,german,lsac')
    parser.add_argument('--result_dir', type=str, default='../results',help='save directory for data and model')
    parser.add_argument('--file_name', type=str, default='LR_model',help='file name for dataset and model')
    args = parser.parse_args()
    
#     data = 
# 	save_dir = '../results'
#     filename = 
    main(save_dir =args.result_dir,data=args.dataname, filename=args.file_name)
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from fair_eval import calculate_prule_clf, calculate_odds_clf, calculate_parity_reg, calculate_group_loss,l2_loss, calculate_overall_accuracy,bce_loss
import argparse
from load_data import *
from fairNN_train import Classifier
from save_utils import load_nparray,save_prediction

def test_NN(save_md, save_te, dataname = 'adult', save_dir='../results'):
    
    
    print("load test data from ", save_te)
    X_te = load_nparray(save_te)
    print(X_te.shape)
    print("load NN model from ", save_md)
    clf = Classifier(n_features=X_te.shape[1])
    clf.load_state_dict(torch.load(save_md))
    clf.eval()
    
    print("Prediction")
    clf_pred = clf(torch.Tensor(X_te))
    pred = (clf_pred>0.5).float().flatten().detach().cpu().numpy()
    
    print("save prediction")
    save_prediction(pred,dataname,save_dir,'fnn')
    

def main(**args):    
    print(args)
    test_NN(**args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Argument for testing NN')
    parser.add_argument('--dataname',type=str,default='adult',help='select data among adult,bank,compas,german,lsac')
    parser.add_argument('--result_dir', type=str, default='../results',help='result directory for model and data')
    args = parser.parse_args()
    
    save_md = os.path.join(args.result_dir,args.dataname+"_NN_model.sm")
    test_data = os.path.join(args.result_dir,args.dataname+"_nn_testX.te")
    main(save_md = save_md, save_te = test_data,save_dir = args.result_dir,dataname = args.dataname)
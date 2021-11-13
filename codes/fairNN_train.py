import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from fair_eval import calculate_prule_clf, calculate_odds_clf, calculate_parity_reg, calculate_group_loss,l2_loss, calculate_overall_accuracy,bce_loss
import argparse
from load_data import *
from save_utils import *

# Our code is revised from the following links
# https://github.com/equialgo/fairness-in-ml
# https://godatadriven.com/blog/towards-fairness-in-ml-with-adversarial-networks/
# https://github.com/equialgo/fairness-in-ml/blob/master/fairness-in-torch.ipynb 

class NPsDataSet(TensorDataset):

    def __init__(self, *dataarrays):
        tensors = (torch.tensor(da).float() for da in dataarrays)
        super(NPsDataSet, self).__init__(*tensors)

def make_dataloaders(X_tr, X_te, y_tr, y_te, xs_tr, xs_te):
    y_tr = y_tr.astype('float32')
    y_te = y_te.astype('float32')
    train_data = NPsDataSet(X_tr, y_tr, xs_tr)
    test_data = NPsDataSet(X_te, y_te, xs_te)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False, drop_last=False)
    return train_loader, test_loader

class Classifier(nn.Module):

    def __init__(self, n_features, n_hidden=32, p_dropout=0.2):
        super(Classifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, n_hidden),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(n_hidden, 1),
        )

    def forward(self, x):
        return F.sigmoid(self.network(x))

class Adversary(nn.Module):

    def __init__(self, n_sensitive, n_hidden=32):
        super(Adversary, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(1, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_sensitive),
        )

    def forward(self, x):
        return F.sigmoid(self.network(x))    
        
def pretrain_NN(clf,train_loader,clf_criterion,clf_optimizer, n_epoch = 5):
    for epoch in range(n_epoch):
        for x, y, _ in train_loader:
            clf.zero_grad()
            p_y = clf(x)
            loss = clf_criterion(p_y.flatten(), y)
            loss.backward()
            clf_optimizer.step()
    return clf
def pretrain_adv(clf,adv,train_loader,adv_criterion,adv_optimizer,lambdas, n_epoch = 5):
    for epoch in range(n_epoch):
        for x, _, z in train_loader:
            adv.zero_grad()
            p_y = clf(x).detach()
            p_z = adv(p_y)
            loss = (adv_criterion(p_z, z) * lambdas).mean()
            loss.backward()
            adv_optimizer.step()
    return adv
def train_FNN_iter(clf, adv, train_loader, clf_criterion, adv_criterion, clf_optimizer, adv_optimizer,lambdas):
    for x, y, z in train_loader:
        adv.zero_grad()
        p_y = clf(x)
        p_z = adv(p_y)
        loss_adv = (adv_criterion(p_z, z) * lambdas).mean()
        loss_adv.backward()
        adv_optimizer.step()

    # Train classifier on single batch
    for x, y, z in train_loader:
        pass  # Ugly way to get a single batch
    clf.zero_grad()
    p_y = clf(x)
    p_z = adv(p_y)
    loss_adv = (adv_criterion(p_z, z) * lambdas).mean()
    clf_loss = clf_criterion(p_y, y) - (adv_criterion(adv(p_y), z) * lambdas).mean()
    clf_loss.backward()
    clf_optimizer.step()
    return clf, adv
def train_FNN_torch(clf, adv, train_loader,test_loader, clf_criterion, adv_criterion,clf_optimizer,adv_optimizer, lambdas, n_epoch = 165):
    assert len(lambdas) == train_loader.dataset.tensors[2].shape[1]
    for epoch in range(1, n_epoch):
        # Train adversary
        train_FNN_iter(clf, adv, train_loader, clf_criterion, adv_criterion, clf_optimizer, adv_optimizer,lambdas)
        print("="*20)
        print("test for epoch:",epoch)
        print("="*20)
        test_FNN(clf,test_loader,'bce',thr=0.5)
    return clf, adv

def test_FNN(clf, test_loader,loss_fn = 'bce',thr = 0.5):
#     x_te = test_loader.dataset.tensors[0]
    x_te = test_loader.dataset.tensors[0]
    y_te = test_loader.dataset.tensors[1].detach().cpu().numpy()
    xs_te = test_loader.dataset.tensors[2].detach().cpu().numpy()
    with torch.no_grad():
        clf_pred = clf(x_te)
#         adv_pred = adv(clf_pred)
    pred = (clf_pred>thr).float().flatten().detach().cpu().numpy()
    clf_pred = clf_pred.flatten().detach().cpu().numpy()
    if loss_fn =='bce':
        loss_fn = bce_loss
    calculate_overall_accuracy(pred*2-1,y_te.flatten()*2-1)
    calculate_group_loss(loss_fn,clf_pred,y_te,xs_te)

def train_fnn(lmd = 30.0,data = 'adult',save_dir = '',filename='FNN_model',rs=42):
    print("load data:", data)
    if data == 'adult':
        X_tr, X_te, y_tr, y_te, xs_tr, xs_te = load_adult_data_prev(svm=False,random_state=rs,intercept=False)
    elif data == 'bank':
        X_tr, X_te, y_tr, y_te, xs_tr, xs_te = load_bank_data(svm=False,random_state=rs,intercept=False)
    elif data == 'compas':
        X_tr, X_te, y_tr, y_te, xs_tr, xs_te = load_compas_data(svm=False,random_state=rs,intercept=False)
    elif data == 'german':
        X_tr, X_te, y_tr, y_te, xs_tr, xs_te = load_german_data(svm=False,random_state=rs,intercept=False)
    elif data == 'lsac':
        X_tr, X_te, y_tr, y_te, xs_tr, xs_te = load_lsac_data(svm=False,random_state=rs,intercept=False)
    
    y_tr = y_tr.astype('float32')
    y_te = y_te.astype('float32')
    train_data = NPsDataSet(X_tr, y_tr, xs_tr)
    test_data = NPsDataSet(X_te, y_te, xs_te)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=True, drop_last=False)
    
    print('# training samples:', len(train_data))
    print('# batches:', len(train_loader))
    
    # construct classifier
    n_features = train_data.tensors[0].shape[1]
    clf = Classifier(n_features=n_features)
    clf_criterion = nn.BCELoss()
    clf_optimizer = optim.Adam(clf.parameters())
    
    # pretrain the clf
    pretrain_NN(clf,train_loader,clf_criterion,clf_optimizer, n_epoch = 5)
    
    # construct adversary
    lambdas = torch.Tensor([lmd])
    adv = Adversary(xs_tr.shape[1])
    adv_criterion = nn.BCELoss(reduce=False)
    adv_optimizer = optim.Adam(adv.parameters())
    
    # pretrain adv
    pretrain_adv(clf,adv,train_loader,adv_criterion,adv_optimizer,lambdas, n_epoch = 5)
    
    # Train the classfier with the adversary
    train_FNN_torch(clf, adv, train_loader,test_loader, clf_criterion, adv_criterion,clf_optimizer,adv_optimizer, lambdas, n_epoch = 50)
    
    save_path = os.path.join(save_dir,data+'_'+filename)
    torch.save(clf.state_dict(),save_path)
    print("save the model in: ", save_path)
    
    # save SVMfalse no-intercept test data
    save_testdata(X_te,y_te,xs_te,data+"_fnn",save_dir)
    
def main(**args):
    train_fnn(**args)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Argument for training fair NN')
    parser.add_argument('--dataname',type=str,default='adult',help='select data among adult,bank,compas,german,lsac')
    parser.add_argument('--result_dir', type=str, default='../results',help='save directory for data and model')
    parser.add_argument('--file_name', type=str, default='FNN_model.sm',help='file name for dataset and model')
    parser.add_argument('--lmbd', type=float, default=100.0,help='lambda parameter for fairNN')
    args = parser.parse_args()
    
#     data = 
# 	save_dir = '../results'
#     filename = 
    main(lmd = args.lmbd, save_dir =args.result_dir,data=args.dataname, filename=args.file_name)    
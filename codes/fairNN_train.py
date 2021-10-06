import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from fair_eval import calculate_prule_clf, calculate_odds_clf, calculate_parity_reg, calculate_group_loss,l2_loss, calculate_overall_accuracy,bce_loss

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
def train_FNN(clf, adv, train_loader,test_loader, clf_criterion, adv_criterion,clf_optimizer,adv_optimizer, lambdas, n_epoch = 165):
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
    calculate_overall_accuracy(pred,y_te)
    calculate_prule_clf(pred,y_te,xs_te)
    calculate_odds_clf(pred,y_te,xs_te)
    calculate_group_loss(loss_fn,clf_pred,y_te,xs_te)
    
    
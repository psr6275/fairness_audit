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
    
def train_NN(clf,train_loader,clf_criterion,clf_optimizer, n_epoch = 5):
    for epoch in range(n_epoch):
        print("epoch: ",epoch)
        for x, y, _ in train_loader:
            clf.zero_grad()
            p_y = clf(x)
            loss = clf_criterion(p_y.flatten(), y)
            loss.backward()
            clf_optimizer.step()
    return clf

def train_nn(data = 'adult',save_dir = '',filename='NN_model.sm'):
    print("load data:", data)
    if data == 'adult':
        X_tr, X_te, y_tr, y_te, xs_tr, xs_te = load_adult_data(svm=False,random_state=42,intercept=False)
    elif data == 'bank':
        X_tr, X_te, y_tr, y_te, xs_tr, xs_te = load_bank_data(svm=False,random_state=42,intercept=False)
    elif data == 'compas':
        X_tr, X_te, y_tr, y_te, xs_tr, xs_te = load_compas_data(svm=False,random_state=42,intercept=False)
    elif data == 'german':
        X_tr, X_te, y_tr, y_te, xs_tr, xs_te = load_german_data(svm=False,random_state=42,intercept=False)
    elif data == 'lsac':
        X_tr, X_te, y_tr, y_te, xs_tr, xs_te = load_lsac_data(svm=False,random_state=42,intercept=False)
    
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
    
    # train the clf
    train_NN(clf,train_loader,clf_criterion,clf_optimizer, n_epoch = 50)
    
    # save model
    save_path = os.path.join(save_dir,data+'_'+filename)
    torch.save(clf.state_dict(),save_path)
    print("save the model in: ", save_path)
    
    # save SVMfalse no-intercept test data
    save_testdata(X_te,y_te,xs_te,data+"_nn",save_dir)

def main(**args):
    train_nn(**args)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Argument for training NN')
    parser.add_argument('--dataname',type=str,default='adult',help='select data among adult,bank,compas,german,lsac')
    parser.add_argument('--result_dir', type=str, default='../results',help='save directory for data and model')
    parser.add_argument('--file_name', type=str, default='NN_model.sm',help='file name for dataset and model')
    
    args = parser.parse_args()
      
    main(data = args.dataname, save_dir = args.result_dir,filename = args.file_name)
import pickle
import os





def save_lr(clf, save_dir = '', filename = 'lr_model'):
    res = {}
    res['coef'] = clf.coef_
    res['intercept'] = clf.intercept_
    save_path = os.path.join(save_dir, filename+'.sm')
    with open(save_path,'wb') as f:
        pickle.dump(res,f)
    print("saved in ",save_path)

def load_lr(save_path):
    with open(save_path,'rb') as f:
        aa = pickle.load(f)
    return aa

def save_testdata(Xte, yte,zte,data = 'adult',save_dir = ''):
    res = {}
    res['Xte'] = Xte
    res['yte'] = yte
    res['zte'] = zte
    
#     save_path = os.path.join(save_dir,data+'_testset.te')
    save_path1 = os.path.join(save_dir,data+'_testX.te')
    save_path2 = os.path.join(save_dir,data+'_testY.te')
    save_path3 = os.path.join(save_dir,data+'_testZ.te')    
    with open(save_path1,'wb') as f:
        pickle.dump(Xte,f)
    with open(save_path2,'wb') as f:
        pickle.dump(yte,f)
    with open(save_path3,'wb') as f:
        pickle.dump(zte,f)
    print("saved in ", save_path1,save_path2,save_path3)

def load_testdata(save_dir,data):
    save_path1 = os.path.join(save_dir,data+'_testX.te')
    save_path2 = os.path.join(save_dir,data+'_testY.te')
    save_path3 = os.path.join(save_dir,data+'_testZ.te')    
    with open(save_path1,'rb') as f:
        Xte = pickle.load(f)
    with open(save_path2,'rb') as f:
        yte = pickle.load(f)
    with open(save_path3,'rb') as f:
        zte = pickle.load(f)

    return Xte,yte,zte

def load_nparray(save_path):
    with open(save_path,'rb') as f:
        aa = pickle.load(f)
    return aa

def save_prediction(pred,data,save_dir = '', model = 'flr'):
    save_path = os.path.join(save_dir, data +'_'+model+'_pred.pr')
    with open(save_path,'wb') as f:
        pickle.dump(pred,f)
        print("saved in ", save_path)
    

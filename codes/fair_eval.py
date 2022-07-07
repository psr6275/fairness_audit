import numpy as np

def calculate_misclassification(pred, y, xs):
    mis0 = sum((pred==-y)*(xs==0))
    mis1 = sum((pred==-y)*(xs==1))
    s0 = sum(xs==0)
    s1 = sum(xs==1)
    prule = min((mis0/s0)/(mis1/s1),(mis1/s1)/(mis0/s0))*100
    return prule
def diff_calculate_misclassification(pred, y, xs):
    mis0 = sum((pred==-y)*(xs==0))
    mis1 = sum((pred==-y)*(xs==1))
    s0 = sum(xs==0)
    s1 = sum(xs==1)
    diff = np.abs((mis0/s0)-(mis1/s1))*100
    return diff

def calculate_mistreatment(pred, y, xs,cond = 1):
    """
        cond = 1 : False negative rate
        cond = -1: False positive rate
    """
    xs = xs.flatten()

    if min(y)==0:
        y_ = (y*2)-1
        pred_ = pred*2-1
    else:
        y_ = y
        pred_ = pred
    assert cond in [-1,1]
    
    fr0 = sum((pred_==-cond)*(y_==cond)*(xs==0))
    fr1 = sum((pred_==-cond)*(y_==cond)*(xs==1))
    s0 = sum((y_==cond)*(xs==0))
    s1 = sum((y_==cond)*(xs==1))

    
    prule = min((fr0/s0)/(fr1/s1),(fr1/s1)/(fr0/s0))*100
    
    return prule
def diff_calculate_mistreatment(pred, y, xs,cond = 1):
    """
        cond = 1 : False negative rate
        cond = -1: False positive rate
    """
    xs = xs.flatten()

    if min(y)==0:
        y_ = (y*2)-1
        pred_ = pred*2-1
    else:
        y_ = y
        pred_ = pred
    assert cond in [-1,1]
    
    fr0 = sum((pred_==-cond)*(y_==cond)*(xs==0))
    fr1 = sum((pred_==-cond)*(y_==cond)*(xs==1))
    s0 = sum((y_==cond)*(xs==0))
    s1 = sum((y_==cond)*(xs==1))

    
    diff = np.abs((fr0/s0)-(fr1/s1))*100
    
    return diff

def calculate_impact(pred,y,xs):
    
    idx_yps0 = (pred==1)*(xs==0)
    idx_yps1 = (pred==1)*(xs==1)
#     idx_yns0 = (pred==-1)*(xs==0)
#     idx_yns1 = (pred==-1)*(xs==1)
    
    s0sum = sum(xs==0)
    s1sum = sum(xs==1)
    
    prule = min((sum(idx_yps1)/s1sum)/(sum(idx_yps0)/s0sum),(sum(idx_yps0)/s0sum)/(sum(idx_yps1)/s1sum))*100
    return prule

def diff_calculate_impact(pred,y,xs):
    
    idx_yps0 = sum((pred==1)*(xs==0))
    idx_yps1 = sum((pred==1)*(xs==1))
#     idx_yns0 = (pred==-1)*(xs==0)
#     idx_yns1 = (pred==-1)*(xs==1)
    
    s0sum = sum(xs==0)
    s1sum = sum(xs==1)
    
    diff = np.abs((idx_yps1/s1sum)-(idx_yps0/s0sum))*100
    return diff

def calculate_prule_clf(pred,y,xs):
    pred = pred.flatten()
    y = y.flatten()
    xs = xs.flatten()
    print("disparate impact: ", calculate_impact(pred,y,xs))
    print("disparate misclassification rate: ", calculate_misclassification(pred,y,xs))
    print("disparate false positive rate:", calculate_mistreatment(pred,y,xs,cond=-1))
    print("disparate false negative rate:", calculate_mistreatment(pred,y,xs,cond=1))

def calculate_odds_clf(pred,y,xs,return_val = False):
    xs = xs.flatten()
    yls = np.unique(y)
    idx_yps0 = (pred==1)*(xs==0)
    idx_yps1 = (pred==1)*(xs==1)
    if return_val:
        prules = {}
    for yl in yls:
        idx_y = y==yl
        s0sum = sum(idx_y*(xs==0))
        s1sum = sum(idx_y*(xs==1))
        prule = min((sum(idx_yps1*idx_y)/s1sum)/(sum(idx_yps0*idx_y)/s0sum),\
                    (sum(idx_yps0*idx_y)/s0sum)/(sum(idx_yps1*idx_y)/s1sum))*100
        if return_val:
            prules[yl] = prule
        if return_val is False:
            print("equalized opportunity for {} : {}".format(yl,prule))
    
    if return_val:
        return prules
def diff_calculate_odds_clf(pred,y,xs,return_val = False):
    xs = xs.flatten()
    yls = np.unique(y)
    idx_yps0 = (pred==1)*(xs==0)
    idx_yps1 = (pred==1)*(xs==1)
    if return_val:
        prules = {}
    for yl in yls:
        idx_y = y==yl
        s0sum = sum(idx_y*(xs==0))
        s1sum = sum(idx_y*(xs==1))
        prule = np.abs((sum(idx_yps1*idx_y)/s1sum)-(sum(idx_yps0*idx_y)/s0sum))*100
        if return_val:
            prules[yl] = prule
        if return_val is False:
            print("equalized opportunity for {} : {}".format(yl,prule))
    
    if return_val:
        return prules
def calculate_parity_reg(pred,y,xs,thrs = None):
    xs = xs.flatten()
    if thrs is None:
        thrs = np.mean(y)
    idx_yps0 = (pred>thrs)*(xs==0)
    idx_yps1 = (pred>thrs)*(xs==1)
    s0sum = sum(xs==0)
    s1sum = sum(xs==1)
    
    prule = min((sum(idx_yps1)/s1sum)/(sum(idx_yps0)/s0sum),(sum(idx_yps0)/s0sum)/(sum(idx_yps1)/s1sum))*100
    print("disparate parity for threshold {}: {}".format(thrs, prule))

def calculate_group_loss(loss_fn, pred, y, xs):
    xs =xs.flatten()
    s0sum = sum(xs==0)
    s1sum = sum(xs==1)
    print("loss function: ", loss_fn.__name__)
    for i in range(2):
        lv = loss_fn(pred[xs==i],y[xs==i])
        print("loss value for group {}: {}".format(i,lv))
    
def l2_loss(pred,y):
    return np.mean((pred-y)**2)

eps = np.finfo(float).eps

def bce_loss(pred_,y):
    return -np.mean(y*np.log(pred_+eps)+(1-y)*np.log(1-pred_+eps))

def calculate_overall_accuracy(pred,y,return_val = False):
    pred = pred.flatten()
    if return_val is False:
        print("Overall accuracy:",np.sum(pred==y)/len(pred))
    else:
        return np.sum(pred==y)/len(pred)*100


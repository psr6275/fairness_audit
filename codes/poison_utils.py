import numpy as np
import torch
import random
import copy
import matplotlib.pyplot as plt

from fairLR_test import *
from save_utils import  load_nparray,save_prediction
from fair_eval import *

def shuffle_array(arr_list, sample_ratio = 0.9, rs = 3):
    n_sam = int(len(arr_list[0])*sample_ratio)
    random.seed(rs)
    r_idxs = list(range(len(arr_list[0])))
    random.shuffle(r_idxs)
    arr_t = []
    for arr in arr_list:
        arr_t.append(copy.deepcopy(arr)[r_idxs][:n_sam])
    return tuple(arr_t)

def poison_Z(xs,pos_rt = 0.1, rs = 3):
    n_pos = int(len(xs)*pos_rt)
    random.seed(rs)
    r_idxs = list(range(len(xs)))
    random.shuffle(r_idxs)
    xsp = copy.deepcopy(xs)
    xsp[r_idxs[:n_pos]] = 1-xs[r_idxs[:n_pos]]
    return xsp

def add_res(res,res_sum):
    for ikey in res:
        if ikey == 'odds':
            for jkey in res[ikey]:
                res_sum[ikey][jkey] +=  res[ikey][jkey]
        else:
            res_sum[ikey] += res[ikey]
    return res_sum
def divide_res(res_sum, rep = 5):
    for ikey in res_sum:
        if ikey =='odds':
            for jkey in res_sum[ikey]:
                    res_sum[ikey][jkey] /=  rep
        else:
            res_sum[ikey] /= rep
    return res_sum
def print_res(res):
    for ikey in res:
        print(ikey,": ",res[ikey])

def test_LR_avg(clf,X_te, y_te,xs_te):
    pred = clf.predict(X_te)
    pred = pred.flatten()
    y_te = y_te.flatten()
    xs_te = xs_te.flatten()
    res = {}
    res['accuary'] = calculate_overall_accuracy(pred,y_te,True)
    res['dispImp'] = calculate_impact(pred,y_te,xs_te)
    res['dispMisclf'] = calculate_misclassification(pred,y_te,xs_te)
    res['dispFPR'] = calculate_mistreatment(pred,y_te,xs_te,cond=-1)
    res['dispFNR'] = calculate_mistreatment(pred,y_te,xs_te,cond=1)
    res['odds'] = calculate_odds_clf(pred,y_te,xs_te,return_val = True)
    return res

def test_LR_avg_diff(clf,X_te, y_te,xs_te):
    pred = clf.predict(X_te)
    pred = pred.flatten()
    y_te = y_te.flatten()
    xs_te = xs_te.flatten()
    res = {}
    res['accuary'] = calculate_overall_accuracy(pred,y_te,True)
    res['dispImp'] = diff_calculate_impact(pred,y_te,xs_te)
    res['dispMisclf'] = diff_calculate_misclassification(pred,y_te,xs_te)
    res['dispFPR'] = diff_calculate_mistreatment(pred,y_te,xs_te,cond=-1)
    res['dispFNR'] = diff_calculate_mistreatment(pred,y_te,xs_te,cond=1)
    res['odds'] = diff_calculate_odds_clf(pred,y_te,xs_te,return_val = True)
    return res

def test_FLR_avg(X_te, y_te,xs_te,load_path = '../results/compas_FLR_model.sm'):
    coef = load_flr(load_path)
    pred = predict_FairLR(X_te,coef)
    pred = pred.flatten()
    y_te = y_te.flatten()
    xs_te = xs_te.flatten()
    res = {}
    res['accuary'] = calculate_overall_accuracy(pred,y_te,True)
    res['dispImp'] = calculate_impact(pred,y_te,xs_te)
    res['dispMisclf'] = calculate_misclassification(pred,y_te,xs_te)
    res['dispFPR'] = calculate_mistreatment(pred,y_te,xs_te,cond=-1)
    res['dispFNR'] = calculate_mistreatment(pred,y_te,xs_te,cond=1)
    res['odds'] = calculate_odds_clf(pred,y_te,xs_te,return_val = True)
    return res

def test_FLR_avg_diff(X_te, y_te,xs_te,load_path = '../results/compas_FLR_model.sm'):
    coef = load_flr(load_path)
    pred = predict_FairLR(X_te,coef)
    pred = pred.flatten()
    y_te = y_te.flatten()
    xs_te = xs_te.flatten()
    res = {}
    res['accuary'] = calculate_overall_accuracy(pred,y_te,True)
    res['dispImp'] = diff_calculate_impact(pred,y_te,xs_te)
    res['dispMisclf'] = diff_calculate_misclassification(pred,y_te,xs_te)
    res['dispFPR'] = diff_calculate_mistreatment(pred,y_te,xs_te,cond=-1)
    res['dispFNR'] = diff_calculate_mistreatment(pred,y_te,xs_te,cond=1)
    res['odds'] = diff_calculate_odds_clf(pred,y_te,xs_te,return_val = True)
    return res


def calculate_gamma(alpha, beta):
    gamma = 2*alpha/(beta*(1-alpha)+alpha)
    return gamma

def calculate_thm(empg, alpha, beta, epsilon, delta=0.05,z_num = 2):
    gamma = calculate_gamma(alpha,beta)
    assert epsilon >2*gamma
    mz = (2/(epsilon-2*gamma-empg)**2)*np.log(2*z_num/delta)
    return mz
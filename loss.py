from __future__ import print_function
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math
from numpy.testing import assert_array_almost_equal
import warnings

warnings.filterwarnings('ignore')

def kl_loss_compute(pred, soft_targets, reduce=True):

    kl = F.kl_div(F.log_softmax(pred, dim=1),F.softmax(soft_targets, dim=1), reduce=False)

    if reduce:
        return torch.mean(torch.sum(kl, dim=1))
    else:
        return torch.sum(kl, 1)
    

def js_loss_compute(pred, soft_targets, reduce=True):
    
    pred_softmax = F.softmax(pred, dim=1)
    targets_softmax = F.softmax(soft_targets, dim=1)
    mean = (pred_softmax + targets_softmax) / 2
    kl_1 = F.kl_div(F.log_softmax(pred, dim=1), mean, reduce=False)
    kl_2 = F.kl_div(F.log_softmax(soft_targets, dim=1), mean, reduce=False)
    js = (kl_1 + kl_2) / 2 
    
    if reduce:
        return torch.mean(torch.sum(js, dim=1))
    else:
        return torch.sum(js, 1)

def loss_ours(y_1, y_2, t, forget_rate, ind, noise_or_not, co_lambda=0.1):

    loss_1 = F.cross_entropy(y_1, t, reduction='none') - co_lambda * js_loss_compute(y_1, y_2,reduce=False)
    ind_1_sorted = np.argsort(loss_1.cpu().data).cuda()
    loss_1_sorted = loss_1[ind_1_sorted]

    loss_2 = F.cross_entropy(y_2, t, reduction='none') - co_lambda * js_loss_compute(y_1, y_2,reduce=False)
    ind_2_sorted = np.argsort(loss_2.cpu().data).cuda()
    loss_2_sorted = loss_2[ind_2_sorted]

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_1_sorted))

    ind_1_update=ind_1_sorted[:num_remember].cpu()
    ind_2_update=ind_2_sorted[:num_remember].cpu()
    if len(ind_1_update) == 0:
        ind_1_update = ind_1_sorted.cpu().numpy()
        ind_2_update = ind_2_sorted.cpu().numpy()
        num_remember = ind_1_update.shape[0]

    pure_ratio_1 = np.sum(noise_or_not[ind[ind_1_sorted.cpu()[:num_remember]]])/float(num_remember)
    pure_ratio_2 = np.sum(noise_or_not[ind[ind_2_sorted.cpu()[:num_remember]]])/float(num_remember)

    loss_1_update = loss_1[ind_2_update]
    loss_2_update = loss_2[ind_1_update]
    
    
    return torch.sum(loss_1_update)/num_remember, torch.sum(loss_2_update)/num_remember, pure_ratio_1, pure_ratio_2


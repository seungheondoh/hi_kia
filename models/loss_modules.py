import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.autograd import Variable
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score, average_precision_score, precision_recall_fscore_support

def accuracy(predict, label):
    predict = predict.detach().cpu().numpy()
    label = label.detach().cpu().numpy()
    y_pred = np.argmax(predict, axis=1)
    y_true = np.argmax(label, axis=1)
    acc = accuracy_score(y_true, y_pred)
    return acc

def get_scores(predict, label):
    predict = predict.detach().cpu().numpy()
    label = label.detach().cpu().numpy()
    y_pred = np.argmax(predict, axis=1)
    y_true = np.argmax(label, axis=1)

    accuracy = accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(label, predict, average='macro')
    pr_auc = average_precision_score(label, predict, average='macro')
    pre_rec_f1_macro = precision_recall_fscore_support(y_true, y_pred, average='macro')
    pre_rec_f1_micro = precision_recall_fscore_support(y_true, y_pred, average='micro')
    return accuracy, roc_auc, pr_auc, pre_rec_f1_macro, pre_rec_f1_micro, y_pred, y_true

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)      # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))       # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()
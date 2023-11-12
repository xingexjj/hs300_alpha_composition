import numpy as np
import torch
from audtorch.metrics.functional import pearsonr

def ICLoss(y_pred, y_true):
    '''
    Calculate pearson correlation coefficient between y_pred and y_true.
    y_pred (torch.FloatTensor([1, n])) : prediction
    y_true (torch.FloatTensor([1, n])) : ground truth
    '''

    return -pearsonr(y_pred, y_true)
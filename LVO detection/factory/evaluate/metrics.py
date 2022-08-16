import numpy as np

from sklearn.metrics import roc_auc_score, f1_score, average_precision_score


# dict key should match name of function
def average_auc(y_true, y_prob, **kwargs):
    return {'average_auc': roc_auc_score(y_true, y_prob, average='macro', multi_class='ovr')}


def auc_mip(y_true, y_prob, **kwargs):
    y_true_mip = (y_true == 0).astype('float')
    y_pred_mip = y_prob[:,0]
    return {'auc_mip': roc_auc_score(y_true_mip, y_pred_mip)}


def auc_thin(y_true, y_prob, **kwargs):
    y_true_thin = (y_true == 4).astype('float')
    y_pred_thin = y_prob[:,4]
    return {'auc_thin': roc_auc_score(y_true_thin, y_pred_thin)}


def auc_elvo(y_true, y_prob, **kwargs):
    y_true_elvo = (y_true != 0).astype('float')
    y_pred_elvo = y_prob[:,1] + y_prob[:,2]
    return {'auc_elvo': roc_auc_score(y_true_elvo, y_pred_elvo)}


def auc_acute(y_true, y_prob, **kwargs):
    y_true_acute = (y_true == 2).astype('float')
    y_pred_acute = y_prob[:,2]
    return {'auc_acute': roc_auc_score(y_true_acute, y_pred_acute)}


def auc_binary(y_true, y_prob, **kwargs):
    y_true_elvo = (y_true == 1).astype('float')
    y_pred_elvo = y_prob[:,1]
    return {'auc_binary': roc_auc_score(y_true_elvo, y_pred_elvo)}


def avp_elvo(y_true, y_prob, **kwargs):
    y_true_elvo = (y_true != 0).astype('float')
    y_pred_elvo = y_prob[:,1] + y_prob[:,2]
    return {'avp_elvo': average_precision_score(y_true_elvo, y_pred_elvo)}


def avp_acute(y_true, y_prob, **kwargs):
    y_true_acute = (y_true == 2).astype('float')
    y_pred_acute = y_prob[:,2]
    return {'avp_acute': average_precision_score(y_true_acute, y_pred_acute)}


def average_f1(y_true, y_prob, **kwargs):
    y_pred = np.argmax(y_prob, axis=1)
    return {'average_f1': f1_score(y_true, y_pred, average='macro')}


def accuracy(y_true, y_prob, **kwargs):
    y_pred = np.argmax(y_prob, axis=1)
    return {'accuracy': np.mean(y_true == y_pred)}
import pickle
import numpy as np
import glob, os.path as osp

from factory.evaluate.metrics import *


def load_pickle(pklfile):
    with open(pklfile, 'rb') as f:
        return pickle.load(f)


def auc_acute_v_normal(y_true, y_pred):
    keep = (y_true == 0) | (y_true == 2)
    y_true = y_true[keep]
    y_pred = y_pred[keep]
    return {'auc_acute_v_normal': roc_auc_score((y_true == 2).astype('float32'), y_pred[:,-1])}


def auc_acute_v_chronic(y_true, y_pred):
    keep = (y_true == 1) | (y_true == 2)
    y_true = y_true[keep]
    y_pred = y_pred[keep]
    return {'auc_acute_v_chronic': roc_auc_score((y_true == 2).astype('float32'), y_pred[:,-1])}


predictions = np.sort(glob.glob(osp.join('../cv-predictions/cv-i3d/*/predictions.pkl')))
folds = [_.split('/')[-2] for _ in predictions]
predictions = [load_pickle(p) for p in predictions]

y_pred, y_true = [], []

for fo, p in zip(folds, predictions):
    y_pred.append(p['y_pred'])
    y_true.append(p['y_true'])
    print('Fold {} //'.format(fo))
    print('AUC ELVO:  {:.4f}'.format(auc_elvo(p['y_true'], p['y_pred'])['auc_elvo']))
    print('AUC ACUTE: {:.4f}'.format(auc_acute(p['y_true'], p['y_pred'])['auc_acute']))
    print('AUC ACUTEvN: {:.4f}'.format(auc_acute_v_normal(p['y_true'], p['y_pred'])['auc_acute_v_normal']))
    print('AUC ACUTEvC: {:.4f}'.format(auc_acute_v_chronic(p['y_true'], p['y_pred'])['auc_acute_v_chronic']))


y_pred = np.vstack(y_pred)
y_pred = y_pred.reshape(-1, y_pred.shape[-1])
y_true = np.concatenate(y_true)
print('Overall //')
print('AUC ELVO:  {:.4f}'.format(auc_elvo(y_true, y_pred)['auc_elvo']))
print('AUC ACUTE: {:.4f}'.format(auc_acute(y_true, y_pred)['auc_acute']))
print('AUC ACUTEvN: {:.4f}'.format(auc_acute_v_normal(y_true, y_pred)['auc_acute_v_normal']))
print('AUC ACUTEvC: {:.4f}'.format(auc_acute_v_chronic(y_true, y_pred)['auc_acute_v_chronic']))

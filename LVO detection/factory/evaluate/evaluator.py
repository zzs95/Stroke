import torch
import pandas as pd
import numpy as np
import os, os.path as osp
import re

from tqdm import tqdm
from .metrics import *
from ..data import cudaify


class Predictor(object):


    def __init__(self,
                 loader,
                 labels_available=True,
                 cuda=True):

        self.loader = loader
        self.labels_available = labels_available
        self.cuda = cuda


    def predict(self, model, criterion, epoch):
        self.epoch = epoch
        y_pred = []
        y_true = []   
        with torch.no_grad():
            losses = []
            for data in tqdm(self.loader, total=len(self.loader)):
                batch, labels = data
                if batch.shape == (1,):
                    y_pred.append(None)
                    continue
                if self.cuda:
                    batch, labels = cudaify(batch, labels)
                output = model(batch)
                if criterion:
                    losses.append(criterion(output, labels).item())
                # Make sure you're using the right final transformation ...
                # softmax vs. sigmoid
                output = torch.softmax(output, dim=1)
                y_pred.append(output.cpu().numpy())
                if self.labels_available:
                    y_true.extend(labels.cpu().numpy())
        stack = True
        for i in y_pred:
            if type(i) == type(None):
                stack = False
                break
        y_pred = np.vstack(y_pred) if stack else y_pred
        y_true = np.asarray(y_true)   
        return y_true, y_pred, losses


class Evaluator(Predictor):


    def __init__(self,
                 loader,
                 metrics,
                 valid_metric,
                 mode,
                 improve_thresh,
                 prefix,
                 save_checkpoint_dir,
                 save_best,
                 early_stopping=np.inf,
                 thresholds=np.arange(0.05, 0.95, 0.05),
                 cuda=True):
        
        super(Evaluator, self).__init__(
            loader=loader, 
            cuda=cuda)

        if type(metrics) is not list: metrics = list(metrics)
        if type(valid_metric) == list:
            for vm in valid_metric: assert vm in metrics
        else:
            assert valid_metric in metrics

        self.loader = loader
        # List of strings corresponding to desired metrics
        # These strings should correspond to function names defined
        # in metrics.py
        self.metrics = metrics
        # valid_metric should be included within metrics
        # This specifies which metric we should track for validation improvement
        self.valid_metric = valid_metric
        # Mode should be one of ['min', 'max']
        # This determines whether a lower (min) or higher (max) 
        # valid_metric is considered to be better
        self.mode = mode
        # This determines by how much the valid_metric needs to improve
        # to be considered an improvement
        self.improve_thresh = improve_thresh
        # Specifies part of the model name
        self.prefix = prefix
        self.save_checkpoint_dir = save_checkpoint_dir
        # save_best = True, overwrite checkpoints if score improves
        # If False, save all checkpoints
        self.save_best = save_best
        self.metrics_file = os.path.join(save_checkpoint_dir, 'metrics.csv')
        if os.path.exists(self.metrics_file): os.system('rm {}'.format(self.metrics_file))
        # How many epochs of no improvement do we wait before stopping training?
        self.early_stopping = early_stopping
        self.stopping = 0
        self.thresholds = thresholds

        self.history = []
        self.epoch = None

        self.reset_best()


    def reset_best(self):
        self.best_model = None
        self.best_score = -np.inf


    def set_logger(self, logger):
        self.logger = logger
        self.print  = self.logger.info


    def validate(self, model, criterion, epoch):
        y_true, y_pred, losses = self.predict(model, criterion, epoch)
        valid_metric = self.calculate_metrics(y_true, y_pred, losses)
        self.save_checkpoint(model, valid_metric)
        return valid_metric


    def generate_metrics_df(self):
        df = pd.concat([pd.DataFrame(d, index=[0]) for d in self.history])
        df.to_csv(self.metrics_file, index=False)


    # Used by Trainer class
    def check_stopping(self):
        return self.stopping >= self.early_stopping


    def check_improvement(self, score):
        # If mode is 'min', make score negative
        # Then, higher score is better (i.e., -0.01 > -0.02)
        multiplier = -1 if self.mode == 'min' else 1
        score = multiplier * score
        improved = score >= (self.best_score + self.improve_thresh)
        if improved:
            self.stopping = 0
            self.best_score = score
        else:
            self.stopping += 1
        return improved


    def save_checkpoint(self, model, valid_metric):
        save_file = '{}_{}_VM-{:.4f}.pth'.format(self.prefix, str(self.epoch).zfill(3), valid_metric).upper()
        save_file = os.path.join(self.save_checkpoint_dir, save_file)
        if self.save_best:
            if self.check_improvement(valid_metric):
                if self.best_model is not None: 
                    os.system('rm {}'.format(self.best_model))
                self.best_model = save_file
                torch.save(model.state_dict(), save_file)
        else:
            torch.save(model.state_dict(), save_file)

 
    def calculate_metrics(self, y_true, y_pred, losses):
        metrics_dict = {}
        metrics_dict['loss'] = np.mean(losses)
        for metric in self.metrics:
            metric = eval(metric)
            metrics_dict.update(metric(y_true, y_pred, thresholds=self.thresholds))
        print_results = 'epoch {epoch} // VALIDATION'.format(epoch=self.epoch)
        if type(self.valid_metric) == list:
            valid_metric = np.mean([metrics_dict[vm] for vm in self.valid_metric])
        else:
            valid_metric = metrics_dict[self.valid_metric]
        metrics_dict['vm'] = valid_metric
        max_str_len = np.max([len(k) for k in metrics_dict.keys()])
        for key in metrics_dict.keys():
            self.print('{key} | {value:.5g}'.format(key=key.ljust(max_str_len), value=metrics_dict[key]))
        metrics_dict['epoch'] = int(self.epoch)
        self.history.append(metrics_dict)
        self.generate_metrics_df()
        return valid_metric



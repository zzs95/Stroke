import argparse
import logging
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import yaml
import copy
import re
import os, os.path as osp

from tqdm import tqdm

try:
    from .factory import set_reproducibility
    from .factory import train as factory_train
    from .factory import evaluate as factory_evaluate
    from .factory import builder 
except:
    from factory import set_reproducibility
    import factory.train as factory_train
    import factory.evaluate as factory_evaluate
    import factory.builder as builder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('mode', type=str) 
    parser.add_argument('--gpu', type=lambda s: [int(_) for _ in s.split(',')] , default=[0])
    parser.add_argument('--num-workers', type=int, default=-1)
    return parser.parse_args()


def create_logger(cfg, mode):
    logfile = osp.join(cfg['evaluation']['params']['save_checkpoint_dir'], 'log_{}.txt'.format(mode))
    if osp.exists(logfile): os.system('rm {}'.format(logfile))
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger('root')
    logger.addHandler(logging.FileHandler(logfile, 'a'))
    return logger


def set_inference_batch_size(cfg):
    if 'evaluation' in cfg.keys() and 'train' in cfg.keys():
        if 'batch_size' not in cfg['evaluation'].keys(): 
            cfg['evaluation']['batch_size'] = 2*cfg['train']['batch_size']

    if 'test' in cfg.keys() and 'train' in cfg.keys():
        if 'batch_size' not in cfg['test'].keys(): 
            cfg['test']['batch_size'] = 2*cfg['train']['batch_size']

    if 'predict' in cfg.keys() and 'train' in cfg.keys():
        if 'batch_size' not in cfg['predict'].keys(): 
            cfg['predict']['batch_size'] = 2*cfg['train']['batch_size']

    return cfg 


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    if args.num_workers > 0:
        if 'transform' not in cfg.keys():
            cfg['transform'] = {}
        cfg['transform']['num_workers'] = args.num_workers

    cfg = set_inference_batch_size(cfg)

    # We will set all the seeds we can, in vain ...
    set_reproducibility(cfg['seed'])
    # Set GPU
    if len(args.gpu) == 1:
        torch.cuda.set_device(args.gpu[0])

    if args.mode == 'predict':
        predict(args, cfg)
        return

    if 'mixup' not in cfg['train']['params'].keys():
        cfg['train']['params']['mixup'] = None

    if 'cutmix' not in cfg['train']['params'].keys():
        cfg['train']['params']['cutmix'] = None

    # Make directory to save checkpoints
    if not osp.exists(cfg['evaluation']['params']['save_checkpoint_dir']): 
        os.makedirs(cfg['evaluation']['params']['save_checkpoint_dir'])

    # Load in labels with CV splits
    df = pd.read_csv(cfg['dataset']['csv_filename'])
    df_labels = np.sort(np.unique(df['label']))
    df_labels_dict = {name : i for i, name in enumerate(df_labels)}
    df['label'] = [df_labels_dict[_] for _ in df['label']]

    if 'chronic' in cfg['dataset'].keys():
        if cfg['dataset']['chronic'] == 'exclude':
            df = df[df['label'] != 1]
            df['label'] = [0 if _ == 0 else 1 for _ in df['label']]
        elif cfg['dataset']['chronic'] == 'binarize':
            df['label'] = [0 if _ == 0 else 1 for _ in df['label']]

    ofold = cfg['dataset']['outer_fold']
    ifold = cfg['dataset']['inner_fold']

    train_df, valid_df, test_df = get_train_valid_test(cfg, df, ofold, ifold)

    logger = create_logger(cfg, args.mode)
    logger.info('Saving to {} ...'.format(cfg['evaluation']['params']['save_checkpoint_dir']))

    if args.mode == 'find_lr':
        cfg['optimizer']['params']['lr'] = cfg['find_lr']['params']['start_lr']
        find_lr(args, cfg, train_df, valid_df)
    elif args.mode == 'train':
        train(args, cfg, train_df, valid_df)
    elif args.mode == 'test':
        test(args, cfg, test_df)


def get_train_valid_test(cfg, df, ofold, ifold):
    # Get train/validation set
    if cfg['train']['outer_only']:
        # valid and test are essentially the same here
        train_df = df[df['outer'] != ofold]
        valid_df = df[df['outer'] == ofold]
        test_df  = df[df['outer'] == ofold]
    else:
        test_df = df[df['outer'] == ofold]
        df = df[df['outer'] != ofold]
        train_df = df[df['inner{}'.format(ofold)] != ifold]
        valid_df = df[df['inner{}'.format(ofold)] == ifold]
    return train_df, valid_df, test_df


def get_invfreq_weights(values, scale=None):
    logger = logging.getLogger('root')
    values, counts = np.unique(values, return_counts=True)
    num_samples = np.sum(counts)
    freqs = counts / float(num_samples)
    max_freq = np.max(freqs)
    invfreqs = max_freq / freqs
    if scale == 'log':
        logger.info('  Log scaling ...') 
        invfreqs = np.log(invfreqs+1)
    elif scale == 'sqrt':
        logger.info('  Square-root scaling ...')
        invfreqs = np.sqrt(invfreqs)
    invfreqs = invfreqs / np.sum(invfreqs)
    return invfreqs


def setup(args, cfg, train_df, valid_df):

    logger = logging.getLogger('root')

    train_images = [osp.join(cfg['dataset']['data_dir'], '{}'.format(_)) for _ in train_df['imgfile']]
    valid_images = [osp.join(cfg['dataset']['data_dir'], '{}'.format(_)) for _ in valid_df['imgfile']]
    train_labels = np.asarray(train_df['label'])
    valid_labels = np.asarray(valid_df['label'])

    train_loader = builder.build_dataloader(cfg, data_info={'imgfiles': train_images, 'labels': train_labels}, mode='train')
    valid_loader = builder.build_dataloader(cfg, data_info={'imgfiles': valid_images, 'labels': valid_labels}, mode='valid')

    # Adjust steps per epoch if necessary (i.e., equal to 0)
    # We assume if gradient accumulation is specified, then the user
    # has already adjusted the steps_per_epoch accordingly in the 
    # config file
    steps_per_epoch = cfg['train']['params']['steps_per_epoch']
    gradient_accmul = cfg['train']['params']['gradient_accumulation']
    if steps_per_epoch == 0:
        cfg['train']['params']['steps_per_epoch'] = len(train_loader)
        # if gradient_accmul > 1:
        #     new_steps_per_epoch = int(cfg['train']['params']['steps_per_epoch'] 
        #                               / gradient_accmul)
        #     cfg['train']['params']['steps_per_epoch'] = new_steps_per_epoch
    

    # Generic build function will work for model/loss
    logger.info('Building [{}] architecture ...'.format(cfg['model']['name']))
    if 'backbone' in cfg['model']['params'].keys():
        logger.info('  Using [{}] backbone ...'.format(cfg['model']['params']['backbone']))
    if 'pretrained' in cfg['model']['params'].keys():
        logger.info('  Pretrained weights : {}'.format(cfg['model']['params']['pretrained']))
    model = builder.build_model(cfg['model']['name'], cfg['model']['params'])
    model = model.train().cuda()

    if cfg['loss']['params'] is None:
        cfg['loss']['params'] = {}

    if 'weight' in cfg['loss']['params']:
        cfg['loss']['params']['weight'] = torch.from_numpy(get_invfreq_weights(train_df['label'], cfg['loss']['params']['weight'])).float().cuda()

    if re.search(r'^OHEM', cfg['loss']['name']):
        cfg['loss']['params']['total_steps'] = cfg['train']['params']['num_epochs'] * cfg['train']['params']['steps_per_epoch']

    criterion = builder.build_loss(cfg['loss']['name'], cfg['loss']['params'])
    optimizer = builder.build_optimizer(
        cfg['optimizer']['name'], 
        model.parameters(), 
        cfg['optimizer']['params'])
    scheduler = builder.build_scheduler(
        cfg['scheduler']['name'], 
        optimizer, 
        cfg=cfg)

    if model.wso:
        model.preprocessor = train_loader.dataset.preprocessor
        train_loader.dataset.preprocessor = None
        valid_loader.dataset.preprocessor = None

    if len(args.gpu) > 1:
        model = nn.DataParallel(model, device_ids=args.gpu)

    return cfg, \
           train_loader, \
           valid_loader, \
           model, \
           optimizer, \
           criterion, \
           scheduler 


def find_lr(args, cfg, train_df, valid_df):

    logger = logging.getLogger('root')

    logger.info('FINDING LR ...')

    cfg, \
    train_loader, \
    valid_loader, \
    model, \
    optimizer, \
    criterion, \
    scheduler = setup(args, cfg, train_df, valid_df)

    finder = factory_train.LRFinder(
        loader=train_loader,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        save_checkpoint_dir=cfg['evaluation']['params']['save_checkpoint_dir'],
        logger=logger,
        gradient_accumulation=cfg['train']['params']['gradient_accumulation'],
        mixup=cfg['train']['params']['mixup'],
        cutmix=cfg['train']['params']['cutmix'],
        amp=cfg['train']['params']['amp'])

    finder.find_lr(**cfg['find_lr']['params'])

    logger.info('Results are saved in : {}'.format(osp.join(finder.save_checkpoint_dir, 'lrfind.csv')))


def train(args, cfg, train_df, valid_df):
    
    logger = logging.getLogger('root')

    logger.info('TRAINING : START')

    logger.info('TRAIN: n={}'.format(len(train_df)))
    logger.info('VALID: n={}'.format(len(valid_df)))

    cfg, \
    train_loader, \
    valid_loader, \
    model, \
    optimizer, \
    criterion, \
    scheduler = setup(args, cfg, train_df, valid_df)

    evaluator = getattr(factory_evaluate, cfg['evaluation']['evaluator'])
    evaluator = evaluator(loader=valid_loader,
        **cfg['evaluation']['params'])

    trainer = getattr(factory_train, cfg['train']['trainer'])
    trainer = trainer(loader=train_loader,
        model=model,
        optimizer=optimizer,
        schedule=scheduler,
        criterion=criterion,
        evaluator=evaluator,
        logger=logger)
    trainer.train(**cfg['train']['params'])


def test(args, cfg, test_df):

    logger = logging.getLogger('root')
    logger.info('TEST : START')

    if 'data_dir' in cfg['test'].keys():
        if cfg['test']['data_dir']: 
            cfg['dataset']['data_dir'] = cfg['test']['data_dir']

    test_images = [osp.join(cfg['dataset']['data_dir'], '{}'.format(_)) for _ in test_df['imgfile']]
    test_labels = np.asarray(test_df['label'])

    test_loader = builder.build_dataloader(cfg, data_info={'imgfiles': test_images, 'labels': test_labels}, mode='test')

    model = builder.build_model(cfg['model']['name'], cfg['model']['params'])
    weights = torch.load(cfg['test']['checkpoint'], map_location=lambda storage, loc: storage)
    if len(args.gpu) == 1: weights = {re.sub(r'^module.', '', k) : v for k,v in weights.items()}
    if len(args.gpu) > 1: model = nn.DataParallel(model, device_ids=args.gpu)
    model.load_state_dict(weights)
    model = model.eval().cuda()

    if model.wso:
        model.preprocessor = test_loader.dataset.preprocessor
        test_loader.dataset.preprocessor = None

    image_ids = list(test_df['series'])
    predictor = getattr(factory_evaluate, cfg['test']['predictor'])
    predictor = predictor(loader=test_loader, **cfg['test']['params'])
    y_true, y_pred, _ = predictor.predict(model, criterion=None, epoch=None)

    if not osp.exists(cfg['test']['save_preds_dir']):
        os.makedirs(cfg['test']['save_preds_dir'])

    with open(osp.join(cfg['test']['save_preds_dir'], 'predictions.pkl'), 'wb') as f:
        pickle.dump({
            'y_true': y_true,
            'y_pred': y_pred,
            'series': image_ids
        }, f)


def predict(args, cfg):

    logger = logging.getLogger('root')
    logger.info('PREDICT : START')

    if 'model_weights' not in cfg.keys() or cfg['model_weights'] is None:
        cfg['model_weights'] = [1.] * len(cfg['model_configs'])

    assert len(cfg['model_weights']) == len(cfg['model_configs'])

    if 'data_dir' in cfg['predict'].keys():
        if cfg['predict']['data_dir']: 
            cfg['dataset']['data_dir'] = cfg['predict']['data_dir']

    df = pd.read_csv(cfg['dataset']['csv_filename'])

    if 'checkpoints' in cfg.keys() and type(cfg['checkpoints']) != type(None):
        assert len(cfg['checkpoints']) == len(cfg['model_configs'])
        assert type(cfg['checkpoints']) == list
        replace_checkpoint_paths = True

    model_configs = []
    for cfg_ind, cfgfile in enumerate(cfg['model_configs']):
        with open(cfgfile) as f:
            model_cfg = yaml.load(f, Loader=yaml.FullLoader)    
        model_cfg['model']['params']['pretrained'] = None
        if replace_checkpoint_paths:
            model_cfg['test']['checkpoint'] = cfg['checkpoints'][cfg_ind]
        model_configs.append(model_cfg)

    def create_model(cfg):
        model = builder.build_model(cfg['model']['name'], cfg['model']['params'])
        weights = torch.load(cfg['test']['checkpoint'], map_location=lambda storage, loc: storage)
        if len(args.gpu) == 1: weights = {re.sub(r'^module.', '', k) : v for k,v in weights.items()}
        if len(args.gpu) > 1: model = nn.DataParallel(model, device_ids=args.gpu)
        model.load_state_dict(weights)
        model = model.eval().cuda()
        return model

    models = [create_model(model_cfg) for model_cfg in model_configs]

    test_images = np.asarray([osp.join(cfg['dataset']['data_dir'], _) for _ in df['series']])
    cfg['predict']['labels_available'] = False
    loader = builder.build_dataloader(cfg, data_info={'imgfiles': test_images, 'labels': np.asarray([0]*len(test_images))}, mode='predict')
    if models[0].wso:
        models[0].preprocessor = loader.dataset.preprocessor
        loader.dataset.preprocessor = None
    image_ids = list(df['series'])
    predictor = getattr(factory_evaluate, cfg['predict']['predictor'])
    predictor = predictor(loader=loader, **cfg['predict']['params'])
    y_pred_list = []
    for m in models:
        _, single_y_pred, _ = predictor.predict(m, criterion=None, epoch=None)
        y_pred_list.append(single_y_pred)

    if not osp.exists(cfg['predict']['save_preds_dir']):
        os.makedirs(cfg['predict']['save_preds_dir'])

    weights = np.asarray(cfg['model_weights']) ; weights /= weights.sum()
    y_pred = np.mean(np.asarray(y_pred_list), axis=0) if len(y_pred_list) > 1 else y_pred_list[0]

    with open(osp.join(cfg['predict']['save_preds_dir'], 'predictions.pkl'), 'wb') as f:
        pickle.dump({
            'y_pred': y_pred,
            'series': image_ids
        }, f)


if __name__ == '__main__':
    main()













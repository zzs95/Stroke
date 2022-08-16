import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.cuda.amp import autocast
except ImportError:
    pass

from torchvision.models.video import r2plus1d_18, mc3_18
from .backbones import *
from .layers import SeparableConv
from .wso import WSO
from .rnn import SpecialRNN


class Net3D(nn.Module):


    def __init__(self, 
                 backbone,
                 pretrained,
                 dropout,
                 num_classes,
                 backbone_params={},
                 wso_params=None):

        super().__init__()

        self.backbone, dim_feats = eval(backbone)(pretrained=pretrained, **backbone_params)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_features=dim_feats, out_features=num_classes)

        if type(wso_params) != type(None):
            self.wso = WSO(**wso_params)
        else:
            self.wso = None


    def _forward(self, x):
        if self.wso:
            x = self.wso(x)
            x = self.preprocessor.preprocess(x, 'torch')

        x = self.backbone(x)
        return self.fc(self.dropout(x))


    def forward(self, x):
        if hasattr(self, '_autocast') and self._autocast and self.training:
            with autocast(): return self._forward(x)
        else:
            return self._forward(x)


class Net2D(nn.Module):


    def __init__(self, 
                 backbone,
                 pretrained,
                 dropout,
                 num_classes,
                 num_input_channels=None,
                 wso_params=None):

        super().__init__()

        if num_input_channels:
            self.backbone, dim_feats = change_num_input_channels(backbone, pretrained, num_input_channels)
            assert len(wso_params['ww']) == 1, 'Changing number of input channels currently only supports 1 WSO channel'
        else:
            self.backbone, dim_feats = eval(backbone)(pretrained=pretrained)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_features=dim_feats, out_features=num_classes)

        if type(wso_params) != type(None):
            self.wso = WSO(**wso_params)
        else:
            self.wso = None


    def forward(self, x):
        if self.wso:
            # x.shape = (B, 1, Z, H, W)
            x = self.wso(x)[:,0]
            x = self.preprocessor.preprocess(x, 'torch')

        x = self.backbone(x)
        return self.fc(self.dropout(x))


class Net2DPool(nn.Module):


    def __init__(self, 
                 backbone,
                 pretrained,
                 dropout,
                 num_classes,
                 pool='max',
                 conv_before_pool=0,
                 seq_len=None,
                 num_input_channels=None,
                 wso_params=None):

        super().__init__()

        if num_input_channels:
            self.backbone, dim_feats = change_num_input_channels(backbone, pretrained, num_input_channels)
            assert len(wso_params['ww']) == 1, 'Changing number of input channels currently only supports 1 WSO channel'
        else:
            self.backbone, dim_feats = eval(backbone)(pretrained=pretrained)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_features=dim_feats, out_features=num_classes)

        if type(wso_params) != type(None):
            self.wso = WSO(**wso_params)
        else:
            self.wso = None

        if conv_before_pool > 0: 
            convs_list = []
            for i in range(conv_before_pool):
                convs_list.append(SeparableConv(nn.Conv1d,
                                                dim_feats,dim_feats, 
                                                kernel_size=3,stride=1,padding=2**i,dilation=2**i,bias=False))
            self.conv = convs_list[0] if len(convs_list) == 1 else nn.Sequential(*convs_list)
        else:
            self.conv = None

        if pool == 'max':
            self.pool = nn.AdaptiveMaxPool1d(1)
        elif pool == 'avg':
            self.pool = nn.AdaptiveAvgPool1d(1)
        elif pool == 'conv':
            assert seq_len != None, 'Must specify `seq_len` if `pool`=`conv`'
            self.pool = nn.Conv1d(seq_len, 1, 1, bias=False)


    def forward(self, x):
        if self.wso:
            # x.shape = (B, 1, Z, H, W)
            x = self.wso(x)
            x = self.preprocessor.preprocess(x, 'torch')
        feats = []
        for i in range(x.size(2)):
            feats.append(self.backbone(x[:,:,i]))
        feats = torch.stack(feats, dim=2)
        if self.conv:
            feats = self.conv(feats)
        if type(self.pool) == nn.Conv1d:
            feats = feats.transpose(1,2)
        feats = self.pool(feats).view(x.size(0), -1)
        return self.fc(self.dropout(feats))


class NetRNN(nn.Module):


    def __init__(self,
                 backbone,
                 num_classes,
                 dropout,
                 pretrained,
                 wso_params=None,
                 rnn='GRU',
                 hidden_size=None,
                 bidirectional=False,
                 num_layers=1):

        super().__init__()
        
        # TODO:
        # I don't like using wild imports and eval ...
        # but I can't figure out how to import the backbones
        self.backbone, dim_feats = eval(backbone)(pretrained=pretrained)

        rnn_module = getattr(nn, rnn.upper())

        if hidden_size is None:
            hidden_size = dim_feats

        self.rnn = rnn_module(dim_feats, 
                              hidden_size, 
                              bidirectional=bidirectional, 
                              num_layers=num_layers, 
                              batch_first=True)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size*2 if bidirectional else hidden_size,
                                num_classes)

        if type(wso_params) != type(None):
            self.wso = WSO(**wso_params)
        else:
            self.wso = None


    def forward(self, x):
        if self.wso:
            # x.shape = (B, 1, Z, H, W)
            x = self.wso(x)
            x = self.preprocessor.preprocess(x, 'torch')
        feats = []
        for i in range(x.size(2)):
            feats.append(self.backbone(x[:,:,i]))
        feats = torch.stack(feats, dim=1)
        out_h, _ = self.rnn(feats)
        out_h = self.dropout(out_h[:,-1])
        return self.fc(out_h)


class NetSpecialRNN(nn.Module):


    def __init__(self,
                 backbone,
                 num_classes,
                 dropout,
                 pretrained,
                 wso_params=None,
                 rnn='GRU',
                 seq_len=64,
                 hidden_size=None):

        super().__init__()
        
        # TODO:
        # I don't like using wild imports and eval ...
        # but I can't figure out how to import the backbones
        self.backbone, dim_feats = eval(backbone)(pretrained=pretrained)

        rnn_module = getattr(nn, rnn.upper())

        if hidden_size is None:
            hidden_size = dim_feats

        self.rnn = SpecialRNN(rnn_module,
                              dim_feats, 
                              hidden_size,
                              num_classes=num_classes,
                              dropout=dropout)

        self.fc = nn.Conv1d(seq_len, 1,1)

        if type(wso_params) != type(None):
            self.wso = WSO(**wso_params)
        else:
            self.wso = None


    def forward(self, x):
        if self.wso:
            # x.shape = (B, 1, Z, H, W)
            x = self.wso(x)
            x = self.preprocessor.preprocess(x, 'torch')
        feats = []
        for i in range(x.size(2)):
            feats.append(self.backbone(x[:,:,i]))
        feats = torch.stack(feats, dim=1)
        out = self.rnn(feats)
        return self.fc(out)[:,0]


import pretrainedmodels 
import pretrainedmodels.utils
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import re

from torchvision.models import video
from .inception_i3d import InceptionV1_I3D
from .efficientnet import EfficientNet
from .resnext_wsl import (
    resnext101_32x8d_wsl  as rx101_32x8, 
    resnext101_32x16d_wsl as rx101_32x16, 
    resnext101_32x32d_wsl as rx101_32x32,
    resnext101_32x48d_wsl as rx101_32x48
)
from .r2plus1d import *
from .slowfast import ResNet_I3D_SlowFast
from .resnet_r3d import ResNet_R3D
from .resnet_i3d import ResNet_I3D
from .senet_3d import get_senet_3d


def resnet50_i3d(pretrained=True):
    model = ResNet_I3D(depth=50)
    if pretrained:
        weights = torch.hub.load_state_dict_from_url('https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmaction/models/kinetics400/i3d_kinetics_rgb_r50_c3d_inflated3x1x1_seg1_f32s2_f32s2-b93cc877.pth')
        weights = {k.replace('backbone.', '') : v for k,v in weights['state_dict'].items() if re.search('backbone', k)}
        model.load_state_dict(weights)
    dim_feats = 2048
    return model, dim_feats        


def se_resnext50_3d(pretrained='imagenet', transition=2, use_pool0=True):
    model = get_senet_3d('se_resnext50_32x4d', pretrained=pretrained, transition=transition)
    if not use_pool0:
        setattr(model.layer0.pool, 'kernel_size', (1, 3, 3))
        setattr(model.layer0.pool, 'stride', (1, 2, 2))
    dim_feats = model.last_linear.in_features
    model.last_linear = pretrainedmodels.utils.Identity()
    return model, dim_feats


def irCSN_r152(pretrained=True):
    model = ResNet_R3D()
    if pretrained:
        weights = torch.hub.load_state_dict_from_url('https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmaction/models/kinetics400/ircsn_kinetics400_se_rgb_r152_f32s2_ig65m_fbai-9d6ed879.pth')
        weights = {k.replace('backbone.', '') : v for k,v in weights.items() if re.search('backbone', k)}
        model.load_state_dict(weights)
    dim_feats = 2048
    return model, dim_feats


def slowfast_r50(pretrained=True, tau=8, alpha=8):
    model = ResNet_I3D_SlowFast(depth=50, tau=tau, alpha=alpha)
    if pretrained:
        weights = torch.hub.load_state_dict_from_url('https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmaction/models/kinetics400/slowfast_kinetics400_se_rgb_r50_4x16_finetune-4623cf03.pth')
        weights = {k.replace('backbone.', '') : v for k,v in weights.items() if re.search('backbone', k)}
        model.load_state_dict(weights)
    #dim_feats = 2048+256
    dim_feats = 2048+2048
    return model, dim_feats


def fastonly_r50(pretrained=True, tau=8, alpha=8):
    model = ResNet_I3D_SlowFast(depth=50, tau=tau, alpha=alpha, fast_only=True)
    if pretrained:
        weights = torch.hub.load_state_dict_from_url('https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmaction/models/kinetics400/slowfast_kinetics400_se_rgb_r50_4x16_finetune-4623cf03.pth')
        weights = {k.replace('backbone.', '') : v for k,v in weights.items() if re.search('backbone', k) and re.search('fast_path', k)}
        model.load_state_dict(weights)
    dim_feats = 256
    return model, dim_feats


def i3d(pretrained=True):
    model = InceptionV1_I3D()
    if pretrained:
        weights = torch.hub.load_state_dict_from_url('https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmaction/models/kinetics400/i3d_kinetics400_se_rgb_inception_v1_seg1_f64s1_imagenet_deepmind-9b8e02b3.pth')
        weights = {k.replace('backbone.', '') : v for k,v in weights.items() if re.search('backbone', k)}
        model.load_state_dict(weights)
    dim_feats = 1024
    return model, dim_feats


def r2plus1d_34(pretrained=True):
    model = r2plus1d_34_32_ig65m(pretrained=pretrained)
    dim_feats = model.fc.in_features
    model.fc = pretrainedmodels.utils.Identity()
    return model, dim_feats


def r2plus1d_18(pretrained=True):
    model = getattr(video, 'r2plus1d_18')(pretrained=pretrained)
    dim_feats = model.fc.in_features
    model.fc = pretrainedmodels.utils.Identity()
    return model, dim_feats


def mc3_18(pretrained=True):
    model = getattr(video, 'mc3_18')(pretrained=pretrained)
    dim_feats = model.fc.in_features
    model.fc = pretrainedmodels.utils.Identity()
    return model, dim_feats


def densenet121(pretrained='imagenet'):
    model = getattr(pretrainedmodels, 'densenet121')(num_classes=1000, pretrained=pretrained) 
    dim_feats = model.last_linear.in_features
    model.features.norm5 = nn.Sequential(nn.AdaptiveAvgPool2d(7), nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
    model.last_linear = pretrainedmodels.utils.Identity()
    return model, dim_feats


def densenet161(pretrained='imagenet'):
    model = getattr(pretrainedmodels, 'densenet161')(num_classes=1000, pretrained=pretrained) 
    dim_feats = model.last_linear.in_features
    model.features.norm5 = nn.Sequential(nn.AdaptiveAvgPool2d(7), nn.BatchNorm2d(2208, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
    model.last_linear = pretrainedmodels.utils.Identity()
    return model, dim_feats


def densenet169(pretrained='imagenet'):
    model = getattr(pretrainedmodels, 'densenet169')(num_classes=1000, pretrained=pretrained) 
    dim_feats = model.last_linear.in_features
    model.features.norm5 = nn.Sequential(nn.AdaptiveAvgPool2d(7), nn.BatchNorm2d(1664, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
    model.last_linear = pretrainedmodels.utils.Identity()
    return model, dim_feats


def generic(name, pretrained):
    model = getattr(pretrainedmodels, name)(num_classes=1000, pretrained=pretrained)
    dim_feats = model.last_linear.in_features
    model.avg_pool = nn.AdaptiveAvgPool2d(1)
    model.last_linear = pretrainedmodels.utils.Identity()
    return model, dim_feats


def resnet34(pretrained='imagenet'):
    return generic('resnet34', pretrained=pretrained)


def resnet50(pretrained='imagenet'):
    return generic('resnet50', pretrained=pretrained)


def resnet101(pretrained='imagenet'):
    return generic('resnet101', pretrained=pretrained)


def resnet152(pretrained='imagenet'):
    return generic('resnet152', pretrained=pretrained)


def se_resnet50(pretrained='imagenet'):
    return generic('se_resnet50', pretrained=pretrained)


def se_resnet101(pretrained='imagenet'):
    return generic('se_resnet101', pretrained=pretrained)


def se_resnet152(pretrained='imagenet'):
    return generic('se_resnet152', pretrained=pretrained)


def se_resnext50(pretrained='imagenet'):
    return generic('se_resnext50_32x4d', pretrained=pretrained)


def se_resnext101(pretrained='imagenet'):
    return generic('se_resnext101_32x4d', pretrained=pretrained)


def inceptionv3(pretrained='imagenet'):
    model, dim_feats = generic('inceptionv3', pretrained=pretrained)
    model.aux_logits = False
    return model, dim_feats


def inceptionv4(pretrained='imagenet'):
    return generic('inceptionv4', pretrained=pretrained)


def inceptionresnetv2(pretrained='imagenet'):
    return generic('inceptionresnetv2', pretrained=pretrained)


def resnext101_wsl(d, pretrained='instagram'):
    model = eval('rx101_32x{}'.format(d))(pretrained=pretrained)
    dim_feats = model.fc.in_features
    model.fc = pretrainedmodels.utils.Identity()
    return model, dim_feats


def resnext101_32x8d_wsl(pretrained='instagram'):
    return resnext101_wsl(8, pretrained=pretrained)

        
def resnext101_32x16d_wsl(pretrained='instagram'):
    return resnext101_wsl(16, pretrained=pretrained)


def resnext101_32x32d_wsl(pretrained='instagram'):
    return resnext101_wsl(32, pretrained=pretrained)


def resnext101_32x48d_wsl(pretrained='instagram'):
    return resnext101_wsl(48, pretrained=pretrained)


def xception(pretrained='imagenet'):
    model = getattr(pretrainedmodels, 'xception')(num_classes=1000, pretrained=pretrained) 
    dim_feats = model.last_linear.in_features
    model.last_linear = pretrainedmodels.utils.Identity()
    return model, dim_feats


def efficientnet(b, pretrained):
    if pretrained == 'imagenet':
        model = EfficientNet.from_pretrained('efficientnet-{}'.format(b))
    elif pretrained is None:
        model = EfficientNet.from_name('efficientnet-{}'.format(b))
    dim_feats = model._fc.in_features
    model._dropout = pretrainedmodels.utils.Identity()
    model._fc = pretrainedmodels.utils.Identity()
    return model, dim_feats


def efficientnet_b0(pretrained='imagenet'):
    return efficientnet('b0', pretrained=pretrained)


def efficientnet_b1(pretrained='imagenet'):
    return efficientnet('b1', pretrained=pretrained)


def efficientnet_b2(pretrained='imagenet'):
    return efficientnet('b2', pretrained=pretrained)


def efficientnet_b3(pretrained='imagenet'):
    return efficientnet('b3', pretrained=pretrained)


def efficientnet_b4(pretrained='imagenet'):
    return efficientnet('b4', pretrained=pretrained)


def efficientnet_b5(pretrained='imagenet'):
    return efficientnet('b5', pretrained=pretrained)


def efficientnet_b6(pretrained='imagenet'):
    return efficientnet('b6', pretrained=pretrained)


def efficientnet_b7(pretrained='imagenet'):
    return efficientnet('b7', pretrained=pretrained)


def change_num_input_channels(name, pretrained, num_channels):
    model, dim_feats = eval(name)(pretrained=pretrained)
    if re.search('efficientnet', name):
        layer_id = '_conv_stem'
    elif re.search('resnet', name):
        layer_id = 'conv1'
    layer = getattr(model, layer_id)
    first_layer_weights = model.state_dict()['{}.weight'.format(layer_id)]
    layer_params = {'in_channels' : num_channels,
                    'out_channels': layer.out_channels,
                    'kernel_size' : layer.kernel_size,
                    'stride':  layer.stride,
                    'dilation': layer.dilation,
                    'padding': layer.padding,
                    'bias': layer.bias}
    setattr(model, layer_id, type(layer)(**layer_params))
    first_layer_weights = np.sum(first_layer_weights.cpu().numpy(), axis=1) / num_channels
    first_layer_weights = np.repeat(np.expand_dims(first_layer_weights, axis=1), num_channels, axis=1)
    model.state_dict()['{}.weight'.format(layer_id)].data.copy_(torch.from_numpy(first_layer_weights))
    return model, dim_feats







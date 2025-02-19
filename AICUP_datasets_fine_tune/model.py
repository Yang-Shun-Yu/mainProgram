import torch.nn as nn
import torch

import os
import torch.nn.functional as F
from collections import OrderedDict
# import DenseNet
from torchvision.models.swin_transformer import swin_b, Swin_B_Weights

# from yolov9.models.yolo import Model
from ultralytics import YOLO

import ibnnet

__all__ = ['make_model', 'IBN_A', 'resnet101_ibn_a', 'resnext101_ibn_a', 'densenet169_ibn_a',
            'se_resnet101_ibn_a', 'swin_reid', 'resnet34_ibn_a']

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

def get_backbone(backbone, pretrained):
    
    # assert backbone in ['resnet_a','resnet_b','resnet', 'resnext', 'seresnet', 'densenet', 
    #                     'resnet34','resnext_b','seresnet_b'], "no such backbone, we only support ['resnet', 'resnext', 'seresnet', 'densenet', 'resnet34]"
    
    # if backbone=='resnet':
    #     pretrained = False

    # if backbone == 'resnet':
    #     return torch.hub.load('XingangPan/IBN-Net', 'resnet101_ibn_a', pretrained=pretrained)
    #     return ibnnet.resnet101_ibn_a(pretrained=False)
    if backbone == 'resnet_a':
        return torch.hub.load('XingangPan/IBN-Net', 'resnet101_ibn_a', pretrained=pretrained)
        return ibnnet.resnet101_ibn_a(pretrained=False)
    
    if backbone == 'resnet_b':
        return torch.hub.load('XingangPan/IBN-Net', 'resnet101_ibn_b', pretrained=pretrained)
        return ibnnet.resnet101_ibn_b(pretrained=False)
    
    if backbone == 'resnext_a':
        return torch.hub.load('XingangPan/IBN-Net', 'resnext101_ibn_a', pretrained=pretrained)
        return ibnnet.resnext101_ibn_a(pretrained=False)

    if backbone == 'seresnet_a':
        return torch.hub.load('XingangPan/IBN-Net', 'se_resnet101_ibn_a', pretrained=pretrained)
        return ibnnet.se_resnet101_ibn_a(pretrained=False)

    if backbone == 'resnet34':
        return torch.hub.load('XingangPan/IBN-Net', 'resnet34_ibn_a', pretrained=pretrained)
    
    if backbone == 'resnext_b':
        return ibnnet.resnext101_ibn_b(pretrained=False)
    
    if backbone == 'seresnet_b':
        return ibnnet.se_resnet101_ibn_b(pretrained=False)



    # if backbone == 'densenet':
    #     return DenseNet.densenet169_ibn_a(pretrained=pretrained)
    
class IBN_A(nn.Module):
    def __init__(self, backbone, pretrained=True, num_classes=576, embedding_dim=2048):
        super().__init__()
        self.backbone = get_backbone(backbone, pretrained=pretrained)

        # the expected embedding space is \mathbb{R}^{2048}. resnet, seresnet, resnext satisfy this automatically
        if backbone == 'densenet':
            self.backbone.classifier = nn.Linear(self.backbone.classifier.in_features, embedding_dim)
        elif backbone == 'resnet34':
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features , embedding_dim)

        else:
            self.backbone.fc = nn.Identity() # pretend the last layer does not exist


        self.bottleneck = nn.BatchNorm1d(embedding_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift

        self.classifier = nn.Linear(embedding_dim, num_classes, bias=False)

        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)


    def forward(self, x):
        f_t = self.backbone(x) # features for triplet loss
        f_i = self.bottleneck(f_t) # features for inference

        out = self.classifier(f_i)  # features for id loss

        return f_t, f_i, out
    
class SwinReID(nn.Module):
    def __init__(self, num_classes, embedding_dim=2048, imagenet_weight=True):
        super().__init__()

        self.swin = swin_b(weights=Swin_B_Weights.IMAGENET1K_V1 if imagenet_weight else None)

        self.swin.head = nn.Linear(self.swin.head.in_features, embedding_dim)
        self.bottleneck = nn.BatchNorm1d(embedding_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift

        self.classifier = nn.Linear(embedding_dim, num_classes, bias=False)

        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)


    def forward(self, x):
        f_t = self.swin(x) # features for triplet loss
        f_i = self.bottleneck(f_t) # features for inference

        out = self.classifier(f_i)  # features for id loss

        return f_t, f_i, out
    
def make_model(backbone, num_classes, embedding_dim=2048):
    print(f'using {backbone} as backbone')

    if backbone == 'swin':
        return SwinReID(num_classes)
    # elif backbone == 'yolo':
    #     return Yolo(num_classes)
    # elif backbone == 'yolo11':
    #     return Yolo11(num_classes)
    
    return IBN_A(backbone=backbone, num_classes=num_classes, embedding_dim=embedding_dim)


# this class is kept for compatibility issue run3, run4 and rerun use this class 
class Resnet101IbnA(nn.Module):
    def __init__(self, num_classes=576):
        from warnings import warn
        warn('Deprecated warning: You should only use this class if you want to load the model trained in older commits. You should use `make_model(backbone, num_classes)` to build the model in newer version.')

        super().__init__()
        self.resnet101_ibn_a = torch.hub.load('XingangPan/IBN-Net', 'resnet101_ibn_a', pretrained=True)
        
        embedding_dim = self.resnet101_ibn_a.fc.in_features
        
        self.resnet101_ibn_a.fc = nn.Identity() # pretend the last layer does not exist



        self.bottleneck = nn.BatchNorm1d(embedding_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift

        self.classifier = nn.Linear(embedding_dim, num_classes, bias=False)

        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)


    def forward(self, x):
        f_t = self.resnet101_ibn_a(x) # features for triplet loss
        f_i = self.bottleneck(f_t) # features for inference

        out = self.classifier(f_i)  # features for id loss

        return f_t, f_i, out

# class Yolo(nn.Module):
#     def __init__(self, num_classes, embedding_dim=2048, pretrained=True):
#         super().__init__()

#         cfg = 'yolov9/models/detect/yolov9-t.yaml'
#         yolo = Model(cfg, ch=3, nc=1)
#         self.backbone_layers = yolo.model[:9] 
#         self.backbone = nn.Sequential(*self.backbone_layers)
#         dummy_input = torch.randn(1, 3, 224, 224)
#         features = self.backbone(dummy_input)
#         features = features.view(features.size(0), -1)
#         backbone_output_dim = features.size(1)
#         self.fc = nn.Linear(backbone_output_dim , embedding_dim)

#         self.bottleneck = nn.BatchNorm1d(embedding_dim)
#         self.bottleneck.bias.requires_grad_(False)  # no shift

#         self.classifier = nn.Linear(embedding_dim, num_classes, bias=False)

#         self.bottleneck.apply(weights_init_kaiming)
#         self.classifier.apply(weights_init_classifier)
#     def forward(self, x):
#         f = self.backbone(x)             # Output shape: (N, C, H, W)
#         f = f.view(f.size(0), -1)        # Flatten to shape: (N, C * H * W)
#         f_t = self.fc(f)                 # Shape: (N, embedding_dim)
#         f_i = self.bottleneck(f_t)       # Shape: (N, embedding_dim)
#         out = self.classifier(f_i)       # Shape: (N, num_classes)
#         return f_t, f_i, out
    
# class Yolo11(nn.Module):
#     def __init__(self, num_classes, embedding_dim=2048):
#         super().__init__()
#         model = YOLO("yolo11n-cls.yaml")  # build a new model from YAML
#         model = YOLO("yolo11n-cls.pt")  # load a pretrained model (recommended for training)
#         model = YOLO("yolo11n-cls.yaml").load("yolo11n-cls.pt")  # build from YAML and transfer weights
#         classification_model = model.model 
#         model_layers = classification_model.model  # This is an nn.Sequential
#         backbone_layers = model_layers[:10]  # Exclude the classification head at index 10
#         self.backbone = nn.Sequential(*backbone_layers)

#         dummy_input = torch.randn(1, 3, 224, 224)
#         features = self.backbone(dummy_input)
#         features = features.view(features.size(0), -1)
#         backbone_output_dim = features.size(1)
#         self.fc = nn.Linear(backbone_output_dim , embedding_dim)

#         self.bottleneck = nn.BatchNorm1d(embedding_dim)
#         self.bottleneck.bias.requires_grad_(False)  # no shift
#         self.classifier = nn.Linear(embedding_dim, num_classes, bias=False)

#         self.bottleneck.apply(weights_init_kaiming)
#         self.classifier.apply(weights_init_classifier)

#     def forward(self, x):
#         f = self.backbone(x)             # Output shape: (N, C, H, W)
#         f = f.view(f.size(0), -1)        # Flatten to shape: (N, C * H * W)
#         f_t = self.fc(f)                 # Shape: (N, embedding_dim)
#         f_i = self.bottleneck(f_t)       # Shape: (N, embedding_dim)
#         out = self.classifier(f_i)       # Shape: (N, num_classes)
#         return f_t, f_i, out